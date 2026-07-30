import contextlib
import os
import pathlib
import subprocess
import sys
import time
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse

import requests
import typer
from rich.markup import escape

from comfy_cli import constants, download_state, tracking, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH
from comfy_cli.file_utils import (
    DownloadCancelled,
    DownloadException,
    _friendly_network_error,
    check_unauthorized,
    download_file,
)
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as print  # context-aware: stderr in JSON mode
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()

workspace_manager = WorkspaceManager()
config_manager = ConfigManager()

_CIVITAI_SUBDOMAIN_SUFFIXES = tuple(f".{h}" for h in constants.CIVITAI_ALLOWED_HOSTS)


model_path_map = {
    "lora": "loras",
    "hypernetwork": "hypernetworks",
    "checkpoint": "checkpoints",
    "textualinversion": "embeddings",
    "controlnet": "controlnet",
}


def get_workspace() -> pathlib.Path:
    return pathlib.Path(workspace_manager.workspace_path)


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    rounded = round(seconds, 1)
    if rounded < 60:
        return f"{rounded:.1f}s"
    minutes, secs = divmod(int(rounded), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def potentially_strip_param_url(path_name: str) -> str:
    return path_name.split("?")[0]


def check_huggingface_url(url: str) -> tuple[bool, str | None, str | None, str | None, str | None]:
    """
    Check if the given URL is a Hugging Face URL and extract relevant information.

    Args:
        url (str): The URL to check.

    Returns:
        Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
            - is_huggingface_url (bool): True if it's a Hugging Face URL, False otherwise.
            - repo_id (Optional[str]): The repository ID if it's a Hugging Face URL, None otherwise.
            - filename (Optional[str]): The filename if present, None otherwise.
            - folder_name (Optional[str]): The folder name if present, None otherwise.
            - branch_name (Optional[str]): The git branch name if present, None otherwise.
    """
    parsed_url = urlparse(url)

    if parsed_url.netloc != "huggingface.co" and parsed_url.netloc != "huggingface.com":
        return False, None, None, None, None

    path_parts = [p for p in parsed_url.path.split("/") if p]

    if len(path_parts) < 5 or (path_parts[2] != "resolve" and path_parts[2] != "blob"):
        return False, None, None, None, None
    repo_id = f"{path_parts[0]}/{path_parts[1]}"
    branch_name = path_parts[3]

    remaining_path = "/".join(path_parts[4:])
    folder_name = os.path.dirname(remaining_path) if "/" in remaining_path else None
    filename = os.path.basename(remaining_path)

    # URL decode the filename
    filename = unquote(filename)

    return True, repo_id, filename, folder_name, branch_name


def check_civitai_url(url: str) -> tuple[bool, bool, int | None, int | None]:
    """
    Returns:
        is_civitai_model_url: True if the url is a civitai *web* model url (e.g. /models/12345)
        is_civitai_api_url: True if the url is a civitai *api* url useful for resolving downloads
        model_id: The model id (for /models/*), else None
        version_id: The version id (for /api/download/models/* or ?modelVersionId=), else None
    """
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host not in constants.CIVITAI_ALLOWED_HOSTS and not host.endswith(_CIVITAI_SUBDOMAIN_SUFFIXES):
            return False, False, None, None
        p_parts = [p for p in parsed.path.split("/") if p]
        query = parse_qs(parsed.query)

        if len(p_parts) >= 4 and p_parts[0] == "api":
            # Case 1: /api/download/models/<version_id>
            # e.g. https://civitai.com/api/download/models/1617665?type=Model&format=SafeTensor
            if p_parts[1] == "download" and p_parts[2] == "models":
                try:
                    version_id = int(p_parts[3])
                    return False, True, None, version_id
                except ValueError:
                    return False, True, None, None

            # Case 2: /api/v1/model-versions/<version_id>
            if p_parts[1] == "v1" and p_parts[2] in ("model-versions", "modelVersions"):
                try:
                    version_id = int(p_parts[3])
                    return False, True, None, version_id
                except ValueError:
                    return False, True, None, None

        # Case 3: /models/<model_id>[/*] with optional ?modelVersionId=<id>
        # e.g. https://civitai.com/models/43331
        #      https://civitai.com/models/43331/majicmix-realistic?modelVersionId=485088
        if len(p_parts) >= 2 and p_parts[0] == "models":
            try:
                model_id = int(p_parts[1])
            except ValueError:
                return False, False, None, None
            version_id = None
            mv = query.get("modelVersionId")
            if mv and len(mv) > 0:
                with contextlib.suppress(ValueError):
                    version_id = int(mv[0])
            if version_id is None:
                mv = query.get("version")
                if mv and len(mv) > 0:
                    with contextlib.suppress(ValueError):
                        version_id = int(mv[0])
            return True, False, model_id, version_id

        return False, False, None, None

    except Exception:
        print("Error parsing CivitAI model URL")
        return False, False, None, None


def request_civitai_model_version_api(version_id: int, headers: dict | None = None):
    # Make a request to the CivitAI API to get the model information
    response = requests.get(
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()  # Raise an error for bad status codes

    model_data = response.json()
    for file in model_data["files"]:
        if file.get("primary", False):  # Assuming we want the primary file
            model_name = file["name"]
            download_url = file["downloadUrl"]
            model_type = model_data["model"]["type"].lower()
            basemodel = model_data["baseModel"].replace(" ", "")
            return model_name, download_url, model_type, basemodel


def request_civitai_model_api(model_id: int, version_id: int = None, headers: dict | None = None):
    # Make a request to the CivitAI API to get the model information
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers, timeout=10)
    response.raise_for_status()  # Raise an error for bad status codes

    model_data = response.json()

    # If version_id is None, use the first version
    if version_id is None:
        version_id = model_data["modelVersions"][0]["id"]

    # Find the version with the specified version_id
    for version in model_data["modelVersions"]:
        if version["id"] == version_id:
            # Get the model name and download URL from the files array
            for file in version["files"]:
                if file.get("primary", False):  # Assuming we want the primary file
                    model_name = file["name"]
                    download_url = file["downloadUrl"]
                    model_type = model_data["type"].lower()
                    basemodel = version["baseModel"].replace(" ", "")
                    return model_name, download_url, model_type, basemodel

    # If the specified version_id is not found, raise an error
    raise ValueError(f"Version ID {version_id} not found for model ID {model_id}")


@app.command(help="Download model file from url")
@tracking.track_command("model")
def download(
    _ctx: typer.Context,
    url: Annotated[
        str,
        typer.Option(help="The URL from which to download the model.", show_default=False),
    ],
    relative_path: Annotated[
        str | None,
        typer.Option(
            help="The relative path from the current workspace to install the model.",
            show_default=True,
        ),
    ] = None,
    filename: Annotated[
        str | None,
        typer.Option(
            help="The filename to save the model.",
            show_default=True,
        ),
    ] = None,
    set_civitai_api_token: Annotated[
        str | None,
        typer.Option(
            "--set-civitai-api-token",
            help="Set the CivitAI API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
    set_hf_api_token: Annotated[
        str | None,
        typer.Option(
            "--set-hf-api-token",
            help="Set the Hugging Face API token to use for model downloading.",
            show_default=False,
        ),
    ] = None,
    downloader: Annotated[
        str | None,
        typer.Option(
            "--downloader",
            help="Download backend: 'httpx' (default) or 'aria2' (requires aria2 RPC server).",
            show_default=False,
        ),
    ] = None,
    background: Annotated[
        bool,
        typer.Option(
            "--background",
            help=(
                "Detach the byte transfer to a background worker and return immediately with a "
                "download id. Poll it with `comfy model download-status <id>`."
            ),
        ),
    ] = False,
):
    if relative_path is not None:
        relative_path = os.path.expanduser(relative_path)

    local_filename = None
    headers = None

    civitai_api_token = config_manager.get_or_override(
        constants.CIVITAI_API_TOKEN_ENV_KEY, constants.CIVITAI_API_TOKEN_KEY, set_civitai_api_token
    )
    hf_api_token = config_manager.get_or_override(
        constants.HF_API_TOKEN_ENV_KEY, constants.HF_API_TOKEN_KEY, set_hf_api_token
    )

    resolved_downloader = downloader or config_manager.get(constants.CONFIG_KEY_DEFAULT_DOWNLOADER) or "httpx"

    is_civitai_model_url, is_civitai_api_url, model_id, version_id = check_civitai_url(url)
    is_huggingface_url, repo_id, hf_filename, hf_folder_name, hf_branch_name = check_huggingface_url(url)

    if is_civitai_model_url or is_civitai_api_url:
        headers = {
            "Content-Type": "application/json",
        }
        if civitai_api_token is not None:
            headers["Authorization"] = f"Bearer {civitai_api_token}"

    if is_civitai_model_url:
        local_filename, url, model_type, basemodel = request_civitai_model_api(model_id, version_id, headers)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    elif is_civitai_api_url:
        local_filename, url, model_type, basemodel = request_civitai_model_version_api(version_id, headers)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    elif is_huggingface_url:
        model_id = "/".join(url.split("/")[-2:])

        local_filename = potentially_strip_param_url(url.split("/")[-1])

        if relative_path is None:
            model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")
            basemodel = ui.prompt_input("Enter base model (e.g. SD1.5, SDXL, ...)", default="")
            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    else:
        print("Model source is unknown")

    if filename is None:
        if local_filename is None:
            local_filename = ui.prompt_input("Enter filename to save model as")
        else:
            local_filename = ui.prompt_input("Enter filename to save model as", default=local_filename)
    else:
        local_filename = filename

    if relative_path is None:
        relative_path = DEFAULT_COMFY_MODEL_PATH

    if local_filename is None:
        raise typer.Exit(code=1)
    if local_filename == "":
        raise DownloadException("Filename cannot be empty")

    local_filepath = get_workspace() / relative_path / local_filename

    if local_filepath.exists():
        print(f"[bold red]File already exists: {local_filepath}[/bold red]")
        return

    start_time = time.monotonic()

    # Every resolution step above (metadata requests, token config, filename,
    # destination-exists) has now run in the foreground, so a bad URL, a missing
    # token, or an already-present file still fails fast and synchronously. Only
    # the byte transfer below is eligible to detach.
    needs_hf_auth = False
    if is_huggingface_url and check_unauthorized(url, headers):
        if hf_api_token is None:
            print(
                f"Unauthorized access to Hugging Face model. Please set the Hugging Face API token using `comfy model download --set-hf-api-token` or via the `{constants.HF_API_TOKEN_ENV_KEY}` environment variable"
            )
            return
        needs_hf_auth = True

    if background:
        _submit_background_download(
            url=url,
            dest=local_filepath,
            downloader=resolved_downloader,
            needs_civitai_auth=bool(is_civitai_model_url or is_civitai_api_url),
            needs_hf_auth=needs_hf_auth,
        )
        return

    if needs_hf_auth:
        try:
            import huggingface_hub
        except ImportError:
            print("huggingface_hub not found. Installing...")
            import subprocess

            from comfy_cli.resolve_python import resolve_workspace_python

            python = resolve_workspace_python(str(get_workspace()))
            subprocess.check_call([python, "-m", "pip", "install", "huggingface_hub"])
            import huggingface_hub

        print(f"Downloading model {model_id} from Hugging Face...")
        output_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            subfolder=hf_folder_name,
            revision=hf_branch_name,
            token=hf_api_token,
            local_dir=get_workspace() / relative_path,
            cache_dir=get_workspace() / relative_path,
        )
        print(f"Model downloaded successfully to: {output_path}")
    else:
        print(f"Start downloading URL: {url} into {local_filepath}")
        try:
            download_file(url, local_filepath, headers, downloader=resolved_downloader)
        except DownloadException as e:
            # escape() so a dynamic error message containing "[/]" or similar
            # rich-markup syntax doesn't trigger MarkupError or get mis-rendered.
            print(f"[bold red]{escape(str(e))}[/bold red]")
            raise typer.Exit(code=1) from None

    elapsed = time.monotonic() - start_time
    print(f"Done in {_format_elapsed(elapsed)}")


# ---------------------------------------------------------------------------
# background downloads: submit, worker, and the poll verbs
# ---------------------------------------------------------------------------


def _civitai_headers() -> dict | None:
    """Rebuild CivitAI request headers from config — never from persisted state."""
    headers = {"Content-Type": "application/json"}
    token = config_manager.get_or_override(constants.CIVITAI_API_TOKEN_ENV_KEY, constants.CIVITAI_API_TOKEN_KEY, None)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _hf_headers() -> dict | None:
    token = config_manager.get_or_override(constants.HF_API_TOKEN_ENV_KEY, constants.HF_API_TOKEN_KEY, None)
    if token is None:
        return None
    return {"Authorization": f"Bearer {token}"}


def _host_allowed(url: str, hosts: tuple[str, ...]) -> bool:
    """True when ``url``'s host is one of ``hosts`` or a subdomain of one."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    return host in hosts or host.endswith(tuple(f".{h}" for h in hosts))


def _worker_headers(state: download_state.DownloadState) -> dict | None:
    """Derive the transfer headers the worker should use.

    The state file deliberately records only *which* credential the resolved URL
    needs; the secret itself is re-read from config here, exactly as the
    foreground ``download()`` does.

    The url is re-checked against the credential's own hosts before the token is
    attached. ``needs_*_auth`` and ``url`` are separate fields of a file on disk,
    so nothing structurally stops them from disagreeing — and a record that says
    "use the CivitAI token" against ``https://attacker.example`` would hand the
    user's bearer token to the attacker. The check costs nothing and makes that
    combination inert.
    """
    if state.needs_civitai_auth:
        headers = _civitai_headers() or {}
        if not _host_allowed(state.url, constants.CIVITAI_ALLOWED_HOSTS):
            headers.pop("Authorization", None)
        return headers
    if state.needs_hf_auth:
        if not _host_allowed(state.url, constants.HF_ALLOWED_HOSTS):
            return None
        return _hf_headers()
    return None


def _spawn_download_worker(state_file: pathlib.Path, log_file: pathlib.Path) -> int:
    """Detach the transfer worker and return its pid.

    stdin is /dev/null and stdout/stderr are *appended* to the download's log so
    a crashed worker leaves a trace. POSIX gets its own session (so the worker
    outlives the terminal and `download-cancel` can signal one process group);
    Windows gets DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP for the equivalent.
    """
    argv = [sys.executable, "-m", "comfy_cli", "model", "_download-worker", "--state", str(state_file)]

    kwargs: dict = {}
    if sys.platform == "win32":
        # Not module-level attributes on POSIX, hence the getattr lookups.
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = detached | new_group
    else:
        kwargs["start_new_session"] = True

    logfh = open(log_file, "ab")
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=logfh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **kwargs,
        )
    finally:
        logfh.close()
    return proc.pid


def _submit_background_download(
    *,
    url: str,
    dest: pathlib.Path,
    downloader: str,
    needs_civitai_auth: bool,
    needs_hf_auth: bool,
) -> None:
    """Write the state file, detach the worker, and emit the submit envelope."""
    renderer = get_renderer()
    workspace = get_workspace()
    dest = pathlib.Path(dest).absolute()

    state = download_state.new(
        url=url,
        dest=str(dest),
        downloader=downloader,
        needs_civitai_auth=needs_civitai_auth,
        needs_hf_auth=needs_hf_auth,
    )

    try:
        # The parent directory has to exist before the worker starts writing, and
        # creating it here means a permission problem surfaces synchronously.
        dest.parent.mkdir(parents=True, exist_ok=True)
        state_file = download_state.write(workspace, state)
        log_file = download_state.log_path(workspace, state.id)
    except OSError as e:
        renderer.error(
            code="download_state_unwritable",
            message=f"Could not create the background download state: {e}",
            hint=f"check that {workspace} is writable, or run without --background",
        )
        raise typer.Exit(code=1) from e

    try:
        pid = _spawn_download_worker(state_file, log_file)
    except OSError as e:
        state.status = "failed"
        state.error = f"could not start the background worker: {e}"
        with contextlib.suppress(OSError):
            download_state.write(workspace, state)
        renderer.error(
            code="download_worker_spawn_failed",
            message=f"Could not start the background download worker: {e}",
            hint="run without --background to download in the foreground",
        )
        raise typer.Exit(code=1) from e

    # Deliberately *not* writing the Popen pid back here. Only the worker
    # records its pid, together with its start time, so the two can never
    # disagree and a pid can never be trusted without its identity proof. The
    # window before the worker's first write is covered instead by
    # `reconcile`'s startup grace (a pidless `starting` record stays `starting`)
    # and by the cancel sentinel (a worker that comes up after `download-cancel`
    # sees the request and exits without downloading) — which also removes the
    # read/write race this write-back used to have with a fast worker.
    on_disk = download_state.read_path(state_file)
    if on_disk is not None:
        state = on_disk

    print(f"Downloading in the background: [cyan]{state.id}[/cyan] → {dest}")
    print(f"Track it with: [cyan]comfy model download-status {state.id}[/cyan]")
    renderer.emit(
        {
            "download_id": state.id,
            "pid": pid,
            "dest": str(dest),
            "total_bytes": state.total_bytes,
            "status": state.status,
        },
        command="model download",
        changed=True,
    )


@app.command("_download-worker", hidden=True)
def _download_worker(
    state_file: Annotated[str, typer.Option("--state", help="Path to the download state file.")],
):
    """Detached worker that performs the byte transfer for one background download.

    Hidden: agents address downloads through `comfy model download-status` /
    `downloads` / `download-cancel`, never by invoking this directly.
    """
    path = pathlib.Path(state_file)
    state = download_state.read_path(path)
    if state is None:
        # The submitter always writes the state file before spawning us, so this
        # only happens if it was deleted underneath us. Nothing to do.
        raise typer.Exit(code=1)

    cancel_marker = download_state.cancel_marker_for(path)

    def cancelled() -> bool:
        return cancel_marker.exists()

    # `download-cancel` may have landed while we were still starting up — it
    # can't signal a process that has no pid on file yet, so the sentinel is how
    # it reaches us. Check before claiming the record, and never write a status
    # after one appears.
    if cancelled() or state.is_terminal:
        if not state.is_terminal:
            state.status = "cancelled"
            state.error = None
            with contextlib.suppress(OSError):
                download_state.write_path(path, state)
        raise typer.Exit(code=0)

    state.pid = os.getpid()
    state.pid_create_time = download_state.process_create_time(os.getpid())
    state.status = "downloading"
    download_state.write_path(path, state)

    last_write = 0.0

    def on_progress(completed: int, total: int | None) -> None:
        nonlocal last_write
        state.completed_bytes = completed
        if total is not None:
            state.total_bytes = total
        now = time.monotonic()
        if now - last_write < download_state.PROGRESS_THROTTLE_S:
            return
        last_write = now
        # Poll for cancellation on the same tick as the progress write, so a
        # cancelled download is never resurrected to `downloading` by a write
        # that raced it.
        if cancelled():
            raise DownloadCancelled()
        with contextlib.suppress(OSError):
            download_state.write_path(path, state)

    try:
        download_file(
            state.url,
            pathlib.Path(state.dest),
            _worker_headers(state),
            downloader=state.downloader,
            progress_callback=on_progress,
        )
    except DownloadCancelled:
        state.status = "cancelled"
        state.error = None
        with contextlib.suppress(OSError):
            pathlib.Path(state.dest).unlink(missing_ok=True)
        state.completed_bytes = 0
        with contextlib.suppress(OSError):
            download_state.write_path(path, state)
        raise typer.Exit(code=0) from None
    except BaseException as e:  # noqa: BLE001 — any failure must reach the state file
        state.status = "failed"
        state.error = _friendly_network_error(e) if isinstance(e, Exception) else f"{type(e).__name__}"
        with contextlib.suppress(OSError):
            download_state.write_path(path, state)
        raise typer.Exit(code=1) from None

    # A transfer that beat the cancel to the finish line stays `completed` and
    # keeps its file — `download-cancel` re-reads this record before deciding
    # what to delete, so both sides agree that a fully-downloaded model is not
    # something to throw away.
    state.status = "completed"
    state.error = None
    actual = pathlib.Path(state.dest)
    try:
        state.completed_bytes = actual.stat().st_size
    except OSError:
        pass
    if state.total_bytes is None:
        state.total_bytes = state.completed_bytes
    with contextlib.suppress(OSError):
        download_state.write_path(path, state)


def _render_download_rows(rows: list[dict]) -> None:
    """Human rendering shared by `download-status` and `downloads`.

    A no-op in JSON/NDJSON mode: `ui.display_table` writes to its own Rich
    console on stdout, which is reserved for the envelope, and a table prepended
    to the JSON would make the output unparseable for the agents this contract
    exists for. They get the same rows in `downloads` / the status payload.
    """
    if get_renderer().is_json():
        return

    def _size(value) -> str:
        if value is None:
            return "?"
        return f"{value / (1024 * 1024):.1f} MB"

    data = [
        (
            row["id"],
            row["status"],
            "—" if row["percent"] is None else f"{row['percent']:.1f}%",
            f"{_size(row['completed_bytes'])} / {_size(row['total_bytes'])}",
            f"{row['elapsed_seconds']:.1f}s",
            row["dest"],
        )
        for row in rows
    ]
    ui.display_table(data, ["ID", "Status", "%", "Bytes", "Elapsed", "Destination"])
    for row in rows:
        if row.get("error"):
            print(f"[bold red]{row['id']}: {escape(str(row['error']))}[/bold red]")


def _reconciled(state: download_state.DownloadState) -> tuple[download_state.DownloadState, bool]:
    """Reconcile ``state`` against reality, persisting a *status* correction.

    Only a status change is written back. Byte counts are re-derived from
    ``stat(dest)`` on every poll anyway, so persisting them buys nothing — and
    would let a poll racing a live worker rewind the file to whatever this
    reader happened to load a moment earlier.
    """
    fresh = download_state.reconcile(state)
    changed = fresh.status != state.status
    if changed:
        with contextlib.suppress(OSError, ValueError):
            download_state.write(get_workspace(), fresh)
    return fresh, changed


@app.command("download-status")
@tracking.track_command("model")
def download_status(
    _ctx: typer.Context,
    download_id: Annotated[str, typer.Argument(help="The download id returned by `download --background`.")],
):
    """Report the progress of one background download."""
    renderer = get_renderer()
    state = download_state.read(get_workspace(), download_id)
    if state is None:
        renderer.error(
            code="download_not_found",
            message=f"No background download with id {download_id!r}.",
            hint="list the known downloads with `comfy model downloads`",
            details={"id": download_id},
        )
        raise typer.Exit(code=1)

    fresh, _ = _reconciled(state)
    payload = download_state.status_payload(fresh)
    _render_download_rows([payload])
    renderer.emit(payload, command="model download-status")


@app.command("downloads")
@tracking.track_command("model")
def downloads(_ctx: typer.Context):
    """List every background download this workspace knows about, newest first."""
    renderer = get_renderer()
    rows = [download_state.status_payload(_reconciled(s)[0]) for s in download_state.list_all(get_workspace())]
    if not rows:
        print("No background downloads found.")
    else:
        _render_download_rows(rows)
    renderer.emit({"total": len(rows), "downloads": rows}, command="model downloads")


@app.command("download-cancel")
@tracking.track_command("model")
def download_cancel(
    _ctx: typer.Context,
    download_id: Annotated[str, typer.Argument(help="The download id returned by `download --background`.")],
):
    """Kill a background download's worker and remove its partial file."""
    renderer = get_renderer()
    workspace = get_workspace()
    state = download_state.read(workspace, download_id)
    if state is None:
        renderer.error(
            code="download_not_found",
            message=f"No background download with id {download_id!r}.",
            hint="list the known downloads with `comfy model downloads`",
            details={"id": download_id},
        )
        raise typer.Exit(code=1)

    # Reconcile before deciding there is anything to cancel: a worker SIGKILLed
    # after the last byte landed but before it could persist `completed` still
    # reads as `downloading`, and cancelling that would delete a finished model.
    # Only a *finished* transfer short-circuits — a dead worker that left a
    # partial behind is exactly what the user is trying to clean up, so it goes
    # through the normal cancel path below.
    if state.status not in download_state.TERMINAL_STATUSES:
        fresh = download_state.reconcile(state)
        if fresh.status == "completed":
            state = fresh
            with contextlib.suppress(OSError, ValueError):
                download_state.write(workspace, state)

    if state.status in download_state.TERMINAL_STATUSES:
        payload = download_state.status_payload(state)
        print(f"Download {download_id} is already {state.status}; nothing to cancel.")
        renderer.emit(payload, command="model download-cancel", changed=False)
        return

    # Sentinel first, then the signal. The sentinel is what a worker that is
    # still starting up (no pid on file yet) — or one that outlives SIGTERM —
    # will see, and once it exists the worker can only write `cancelled`.
    with contextlib.suppress(OSError, ValueError):
        download_state.request_cancel(download_state.cancel_path(workspace, download_id))

    # Re-read before signalling: a worker that was still starting up when we
    # first read has claimed its pid by now, and stopping the wrong (or no)
    # process is how a "cancelled" download keeps writing bytes.
    state = download_state.read(workspace, download_id) or state

    # Escalates to SIGKILL if the worker doesn't go quietly; nothing below may
    # touch the destination file until it is confirmed gone, or the worker would
    # simply re-create what we delete.
    stopped = download_state.stop_worker(state)

    # Re-read the worker's last word before touching the destination. Our copy
    # predates the kill and can still be missing the total the worker learned
    # from the response headers — and that total is the only thing that tells a
    # finished file from a partial one.
    state = download_state.read(workspace, download_id) or state

    removed = False
    partial = pathlib.Path(state.dest)
    size = None
    with contextlib.suppress(OSError):
        size = partial.stat().st_size

    finished = state.status == "completed" or (
        size is not None and state.total_bytes is not None and size >= state.total_bytes
    )
    if finished:
        # The bytes are all there. Keep the file and record what actually
        # happened rather than deleting a complete model out from under the user.
        state.status = "completed"
        state.error = None
        if size is not None:
            state.completed_bytes = size
    else:
        if stopped and size is not None:
            with contextlib.suppress(OSError):
                partial.unlink()
                removed = True
        state.status = "cancelled"
        state.error = None if stopped else "worker may still be running; partial file left in place"
        if removed:
            state.completed_bytes = 0

    with contextlib.suppress(OSError, ValueError):
        download_state.write(workspace, state)

    payload = download_state.status_payload(state)
    if finished:
        print(f"Download [cyan]{download_id}[/cyan] had already finished; kept {partial}.")
    else:
        print(f"Cancelled download [cyan]{download_id}[/cyan].")
    renderer.emit(payload, command="model download-cancel", changed=True)


@app.command(help="Remove one or more downloaded models by name or via interactive selection.")
@tracking.track_command("model")
def remove(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
    model_names: list[str] | None = typer.Option(
        None,
        help="List of model filenames to delete, separated by spaces",
        show_default=False,
    ),
    confirm: bool = typer.Option(
        False,
        help="Confirm for deletion and skip the prompt",
        show_default=False,
    ),
):
    """Remove one or more downloaded models, either by specifying them directly or through an interactive selection."""
    model_dir = get_workspace() / relative_path
    available_models = list_models(model_dir)

    if not available_models:
        typer.echo("No models found to remove.")
        return

    model_dir_resolved = model_dir.resolve()

    to_delete = []
    # Scenario #1: User provided model names to delete
    if model_names:
        # Validate and filter models to delete based on provided names
        missing_models = []
        for name in model_names:
            model_path = (model_dir / name).resolve()
            if not model_path.is_relative_to(model_dir_resolved):
                typer.echo(f"Invalid model path: {name}")
                continue
            if model_path.is_file():
                to_delete.append(model_path)
            else:
                missing_models.append(name)

        if missing_models:
            typer.echo("The following models were not found and cannot be removed: " + ", ".join(missing_models))
            if not to_delete:
                return  # Exit if no valid models were found

    # Scenario #2: User did not provide model names, prompt for selection
    else:
        rel_names = [str(model.relative_to(model_dir)) for model in available_models]
        selections = ui.prompt_multi_select("Select models to delete:", rel_names)
        if not selections:
            typer.echo("No models selected for deletion.")
            return
        to_delete = [model_dir / selection for selection in selections]

    # Confirm deletion
    if to_delete and (
        confirm or ui.prompt_confirm_action("Are you sure you want to delete the selected files?", False)
    ):
        for model_path in to_delete:
            model_path.unlink()
            typer.echo(f"Deleted: {model_path}")
    else:
        typer.echo("Deletion canceled.")


def list_models(path: pathlib.Path) -> list[pathlib.Path]:
    """List all model files recursively in the specified directory."""
    if not path.is_dir():
        return []
    return sorted(f for f in path.rglob("*") if f.is_file())


@app.command("list", help="List the models downloaded into this workspace, as a table.")
@tracking.track_command("model")
def list_command(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
):
    """Display a list of all models currently downloaded in a table format."""
    model_dir = get_workspace() / relative_path
    models = list_models(model_dir)

    if not models:
        typer.echo("No models found.")
        return

    # Prepare data for table display
    data = []
    for model in models:
        rel = model.relative_to(model_dir)
        model_type = str(rel.parent) if len(rel.parts) > 1 else ""
        data.append((model.name, model_type, f"{model.stat().st_size // 1024} KB"))
    column_names = ["Model Name", "Type", "Size"]
    ui.display_table(data, column_names)
