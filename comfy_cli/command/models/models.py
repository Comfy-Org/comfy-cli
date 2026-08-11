import contextlib
import ntpath
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse, urlsplit, urlunsplit

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
    cleanup_partials,
    download_file,
)
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as print  # context-aware: stderr in JSON mode
from comfy_cli.output.sanitize import sanitize_markup
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


def _download_failure(
    *,
    code: str,
    message: str,
    hint: str | None = None,
    details: dict | None = None,
) -> typer.Exit:
    """Emit an ``envelope/1`` error for a `model download` failure and return the
    ``typer.Exit`` the caller should raise.

    Every failure path in :func:`download` funnels through here so a `--json`
    consumer always gets a machine-readable ``error.code`` and a non-zero exit —
    never a bare exit that an envelope-synthesizing wrapper would read as a
    success for a download that never happened.

    In pretty mode ``renderer.error`` renders the red error panel, so call sites
    do not print the message themselves. Rich builds the panel body from
    ``rich.text.Text``, which is literal, so a message containing markup
    metacharacters (``[/]``) neither crashes nor gets swallowed — no ``escape()``
    needed.

    ``code`` is keyword-only on purpose: the registry test
    (``tests/comfy_cli/output/test_error_code_registry.py``) AST-scans for
    ``code="..."`` keywords, so a positional call site would silently drop out of
    the both-ways registry enforcement.
    """
    get_renderer().error(
        code=code,
        message=message,
        hint=hint,
        details=details,
        command="model download",
    )
    return typer.Exit(code=1)


def _scrub_url(url: str) -> str:
    """Drop the query string, fragment and userinfo from *url*.

    CivitAI download links carry the API token as ``?token=`` — ``tracking._scrub_value``
    already strips it before telemetry, and an error envelope is a *louder* channel
    than telemetry (agent/MCP wrappers capture and log stdout verbatim), so the same
    scrubbing applies to anything we put in ``error.details`` or a log line.
    """
    if not isinstance(url, str):
        return url
    try:
        parts = urlsplit(url)
    except ValueError:  # malformed authority (e.g. an unbracketed IPv6 literal)
        return url.partition("?")[0].partition("#")[0]
    if not parts.scheme:
        return url.partition("?")[0].partition("#")[0]
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _reject_unsafe_component(value: str | None, *, label: str, url: str) -> None:
    """Fail the download if *value* is anything but a single, inert path component.

    ``local_filename`` and ``basemodel`` are joined into the destination path, and
    both can come straight off the CivitAI API response (``file["name"]``,
    ``version["baseModel"]``) — which is remote input, accepted without a prompt in
    non-interactive runs. ``pathlib``/``os.path.join`` do not sanitize ``..`` or an
    absolute component, so an unvalidated value writes outside the workspace.

    Subdirectories are still reachable — via ``--relative-path``, which is the
    option that exists for choosing the destination directory — so this rejects a
    traversal without removing the capability.
    """
    if not value:
        return
    unsafe = (
        value in (".", "..")
        or os.path.isabs(value)
        or bool(ntpath.splitdrive(value)[0])
        or "/" in value
        or "\\" in value
    )
    if unsafe:
        raise _download_failure(
            code="invalid_argument",
            message=f"Unsafe {label}: {value!r} is not a plain name (it contains a path separator, a drive, or '..').",
            hint="use `--relative-path <dir>` to choose the destination directory and `--filename <name>` for a plain name",
            details={"url": _scrub_url(url), label: value},
        )


def _resolve_civitai_source(fetch, url: str):
    """Run a CivitAI metadata lookup, converting any failure into an error envelope.

    The lookups raise on HTTP/parse errors and ``request_civitai_model_version_api``
    returns ``None`` when the version carries no primary file; both used to escape
    as an unhandled traceback with no envelope.
    """
    try:
        resolved = fetch()
    except Exception as e:
        raise _download_failure(
            code="download_failed",
            message=f"Could not resolve a downloadable file from the CivitAI URL: {e}",
            hint="check the model/version exists and is public; a private model needs --set-civitai-api-token",
            details={"url": _scrub_url(url), "stage": "resolve"},
        ) from None
    if resolved is None:
        raise _download_failure(
            code="download_failed",
            message="The CivitAI model version has no primary file to download.",
            hint="pick a version that has a primary file, or pass its direct download URL",
            details={"url": _scrub_url(url), "stage": "resolve"},
        )
    return resolved


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
        local_filename, url, model_type, basemodel = _resolve_civitai_source(
            lambda: request_civitai_model_api(model_id, version_id, headers), url
        )
        _reject_unsafe_component(basemodel, label="basemodel", url=url)

        model_path = model_path_map.get(model_type)

        if relative_path is None:
            if model_path is None:
                model_path = ui.prompt_input("Enter model type path (e.g. loras, checkpoints, ...)", default="")

            relative_path = os.path.join(DEFAULT_COMFY_MODEL_PATH, model_path, basemodel)
    elif is_civitai_api_url:
        local_filename, url, model_type, basemodel = _resolve_civitai_source(
            lambda: request_civitai_model_version_api(version_id, headers), url
        )
        _reject_unsafe_component(basemodel, label="basemodel", url=url)

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
        # Neither CivitAI nor Hugging Face — a plain file URL, which IS a supported
        # source: the download proceeds via download_file() below. This is a note,
        # not a failure, so it must not short-circuit.
        # escape() the URL: `print` is Rich's markup-parsing print, and a URL holding
        # markup metacharacters (`[/]`, or an IPv6 literal like `http://[::1]/x`)
        # would otherwise raise MarkupError — an uncaught crash with no envelope,
        # the exact failure mode this command is being hardened against.
        print(f"Model source is unknown; treating {escape(_scrub_url(url))} as a direct file URL")

    if filename is None:
        if local_filename is None:
            local_filename = ui.prompt_input("Enter filename to save model as")
        else:
            local_filename = ui.prompt_input("Enter filename to save model as", default=local_filename)
    else:
        local_filename = filename

    if relative_path is None:
        relative_path = DEFAULT_COMFY_MODEL_PATH

    # None (prompt cancelled / no TTY) and "" (prompting skipped for an agentic
    # caller, which returns the empty default) are the same failure: nothing to
    # save the file as. Both used to end without an envelope — a bare exit 1 and
    # an unhandled DownloadException traceback respectively.
    if not local_filename:
        raise _download_failure(
            code="missing_argument",
            message="Could not determine a filename to save the model as.",
            hint="pass `--filename <name>`",
            details={"url": _scrub_url(url)},
        )

    _reject_unsafe_component(local_filename, label="filename", url=url)

    local_filepath = get_workspace() / relative_path / local_filename

    if local_filepath.exists():
        raise _download_failure(
            code="model_file_exists",
            message=f"File already exists: {local_filepath}",
            hint="pass `--filename` to save under a different name, or remove the existing file",
            details={"path": str(local_filepath)},
        )

    start_time = time.monotonic()

    # Every resolution step above (metadata requests, token config, filename,
    # destination-exists) has now run in the foreground, so a bad URL, a missing
    # token, or an already-present file still fails fast and synchronously. Only
    # the byte transfer below is eligible to detach.
    needs_hf_auth = False
    if is_huggingface_url and check_unauthorized(url, headers):
        if hf_api_token is None:
            raise _download_failure(
                code="hf_unauthorized",
                message="Unauthorized access to the Hugging Face model, and no Hugging Face API token is configured.",
                hint=(
                    "set the token via `comfy model download --set-hf-api-token <token>` or the "
                    f"`{constants.HF_API_TOKEN_ENV_KEY}` environment variable"
                ),
                details={"url": _scrub_url(url), "repo_id": repo_id},
            )
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

            try:
                # NB: this installs into the *workspace* interpreter (pinned by
                # tests/comfy_cli/test_models_python_resolution.py), which is not
                # necessarily the one running this process — so the import below
                # can still fail for e.g. a pipx-installed CLI. That predates this
                # change and is tracked separately; what's new here is that the
                # failure now ends in an envelope instead of a traceback.
                python = resolve_workspace_python(str(get_workspace()))
                subprocess.check_call([python, "-m", "pip", "install", "huggingface_hub"])
                import huggingface_hub
            except Exception as e:
                raise _download_failure(
                    code="download_failed",
                    message=f"`huggingface_hub` is required for Hugging Face downloads and could not be installed: {e}",
                    hint="install it manually (`pip install huggingface_hub`) and retry",
                    details={"url": _scrub_url(url)},
                ) from None

        print(f"Downloading model {escape(str(model_id))} from Hugging Face...")
        try:
            output_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=hf_filename,
                subfolder=hf_folder_name,
                revision=hf_branch_name,
                token=hf_api_token,
                local_dir=get_workspace() / relative_path,
                cache_dir=get_workspace() / relative_path,
            )
        except Exception as e:
            raise _download_failure(
                code="download_failed",
                message=f"Hugging Face download failed: {e}",
                hint="check the repo/revision exists and the token has access to it",
                details={"url": _scrub_url(url), "repo_id": repo_id},
            ) from None

        # `hf_hub_download` names the file after the *repo* path (`hf_filename`)
        # and nests it under `hf_folder_name`, so the resolved `local_filename` was
        # silently ignored on this branch only — the public-HF path below goes
        # through download_file(local_filepath) and honours it. That split made the
        # `local_filepath.exists()` guard above check a path this branch never
        # writes, and made its `model_file_exists` hint ("pass `--filename`")
        # impossible to act on. Gate the move on the *resolved* name rather than on
        # `--filename`: `local_filename` can equally come from the prompt above
        # (whose default is only a suggestion), so keying on `filename is not None`
        # left the same split open via the prompt door.
        if pathlib.Path(output_path) != local_filepath:
            try:
                local_filepath.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output_path), str(local_filepath))
                output_path = str(local_filepath)
            except OSError as e:
                raise _download_failure(
                    code="download_failed",
                    message=f"Downloaded the model but could not save it as {local_filepath}: {e}",
                    hint="check the destination directory is writable, or omit `--filename`",
                    details={"url": _scrub_url(url), "path": str(local_filepath)},
                ) from None

        print(f"Model downloaded successfully to: {escape(str(output_path))}")
    else:
        print(f"Start downloading URL: {escape(_scrub_url(url))} into {escape(str(local_filepath))}")
        try:
            download_file(url, local_filepath, headers, downloader=resolved_downloader)
        except DownloadException as e:
            # `message` is rendered through `Text` (see `_download_failure`), which
            # never parses markup, and `error_panel` sanitizes it (ANSI/control-byte
            # strip) before it reaches the terminal — so a server-chosen message (the
            # 401 branch of guess_status_code_reason echoes one, and aria2 relays its
            # own) can't clear the screen or repaint earlier output.
            raise _download_failure(
                code="download_failed",
                message=str(e),
                details={"url": _scrub_url(url), "path": str(local_filepath)},
            ) from None
        except OSError as e:
            # download_file() converts network failures to DownloadException, but a
            # local filesystem failure (unwritable dir, disk full) still escapes as
            # OSError — which would end the command with a traceback and no envelope.
            raise _download_failure(
                code="download_failed",
                message=f"Could not write the downloaded file to {local_filepath}: {e}",
                hint="check the destination directory exists, is writable, and has free space",
                details={"url": _scrub_url(url), "path": str(local_filepath)},
            ) from None
        except Exception as e:
            # Backstop for the invariant this command promises: a `--json` caller must
            # never get a bare traceback with no envelope. The downloader can still
            # raise something neither branch above names — e.g. a malformed
            # `Content-Length` from a broken proxy used to surface as ValueError out of
            # `int(...)` (now guarded in file_utils, but the class of failure remains).
            raise _download_failure(
                code="download_failed",
                message=f"Download failed unexpectedly ({type(e).__name__}): {e}",
                hint="retry; if it persists, re-run without `--json` for the full traceback",
                details={"url": _scrub_url(url), "path": str(local_filepath)},
            ) from None

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

    # Retire finished records before adding one, so submitting downloads is what
    # bounds the state directory rather than something the user has to remember
    # to do. Purely bookkeeping: never let it fail a submit.
    with contextlib.suppress(Exception):
        download_state.prune(workspace)

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
        # Same split as `download_cancel`: only aria2 wrote to the destination, so
        # only aria2's leftovers are ours to delete. The httpx path raises this
        # from inside the transfer loop, before the rename, and unwinds through
        # `_download_file_httpx`'s own cleanup — the `.part` is already gone and
        # anything at `dest` predates this download.
        if state.downloader == "aria2":
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
            # The state file is written by a detached worker, so this string is
            # server-influenced text read back from disk — sanitize on the way out.
            print(f"[bold red]{row['id']}: {sanitize_markup(row['error'])}[/bold red]")


def _reconciled(state: download_state.DownloadState) -> tuple[download_state.DownloadState, bool]:
    """Reconcile ``state`` against reality, persisting a *status* correction.

    Only a status change is written back. Persisting the byte counts a reader
    derived would let a poll racing a live worker rewind the file to whatever
    this reader happened to load a moment earlier — the worker is the only
    writer of progress, and mid-flight its state file is normally the sole source
    of truth (the destination isn't written until the transfer completes; see
    :func:`download_state.reconcile` for the caveat).
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
    workspace = get_workspace()
    # Trim before listing: this is the verb whose cost — and whose output — grows
    # with every record the directory has ever accumulated.
    with contextlib.suppress(Exception):
        download_state.prune(workspace)
    rows = [download_state.status_payload(_reconciled(s)[0]) for s in download_state.list_all(workspace)]
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
        # Reaching a terminal status is not the same as having reclaimed the disk.
        # A SIGKILLed worker's *first* `download-status` poll persists reconcile's
        # `failed` verdict, and from then on this command short-circuits here — so
        # the only path that sweeps the `.part` file would never run, and multi-GB
        # of it would sit there with no command able to reclaim it (unlike the old
        # truncated file at `dest`, a `.part` is invisible to `model list` and
        # `model remove`). Sweep here too, but only once the worker is confirmed
        # gone: a `cancelled` record can have been written while its worker was
        # still running (that is what the "worker may still be running" error
        # says), and that worker would re-create whatever we removed — or worse,
        # find the temp it is streaming into deleted underneath it.
        reclaimed = 0
        if state.status != "completed" and not download_state.worker_alive(state):
            reclaimed = cleanup_partials(pathlib.Path(state.dest))
            if reclaimed:
                state.completed_bytes = 0
                with contextlib.suppress(OSError, ValueError):
                    download_state.write(workspace, state)
        payload = download_state.status_payload(state)
        if reclaimed:
            print(f"Download {download_id} is already {state.status}; reclaimed its partial file.")
        else:
            print(f"Download {download_id} is already {state.status}; nothing to cancel.")
        renderer.emit(payload, command="model download-cancel", changed=bool(reclaimed))
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
        if stopped:
            # Only aria2 writes straight to the destination (it resumes from its
            # own control file), so only an unfinished *aria2* transfer leaves its
            # bytes here. The httpx path never writes through the destination, so
            # a file sitting at `dest` after an httpx cancel is either one that
            # predates this download — which `download_file` promises survives
            # either way — or a rename that just landed, e.g. an unknown-length
            # transfer (`total_bytes` is None, so the `finished` check above can't
            # see it) whose worker was killed between the rename and persisting
            # `completed`. Deleting it would destroy a complete model this command
            # never wrote.
            if size is not None and state.downloader == "aria2":
                with contextlib.suppress(OSError):
                    partial.unlink()
                    removed = True
            # The httpx downloader streams into a `.part` sibling and only renames
            # onto the destination once the transfer completes, so a worker killed
            # mid-flight leaves its gigabytes *there*, not at `dest`. Without this
            # sweep the cancel would report success and reclaim nothing — the exact
            # hand-cleanup this command exists to spare the user.
            if cleanup_partials(partial):
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
