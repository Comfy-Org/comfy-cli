import contextlib
import errno
import logging
import ntpath
import os
import pathlib
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Annotated
from urllib.parse import parse_qs, unquote, urlparse, urlsplit, urlunsplit

import typer
from rich.markup import escape

from comfy_cli import constants, download_state, tracking, ui
from comfy_cli.command.models import search as models_search_command
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

logger = logging.getLogger(__name__)

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
    failure = typer.Exit(code=1)
    # `typer.Exit` carries an exit code and nothing else, so `str(exc)` is empty.
    # The foreground claim record wants the same text the user just saw in its
    # `error` field; stash it here rather than have every raise site thread the
    # message out by hand.
    failure.comfy_error_message = message
    return failure


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
    # Imported lazily: requests costs ~30ms to import and this module is on
    # the import path of every CLI invocation.
    import requests

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
    import requests  # deferred; see request_civitai_model_version_api

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

    # A destination-exists check cannot see a transfer that is still in flight: on
    # the httpx path a background download streams into a `.part` sibling and only
    # renames onto `dest` at the end, so the destination stays absent for the whole
    # transfer and two submissions for the same model would both pass, both stream
    # a full copy, and the later rename would silently overwrite the earlier.
    #
    # Ordered *before* `exists()` on purpose: `--downloader aria2` writes straight
    # to the destination (it owns its own `.aria2` resume control file), so a live
    # aria2 transfer makes `exists()` true and the caller would get
    # `model_file_exists` with the hint "remove the existing file" — advice that
    # deletes the output a running worker is still writing. The accurate refusal
    # has to win.
    #
    # Scope, precisely: this refuses a submission — foreground or `--background` —
    # into a destination that any live download has claimed, of either kind. It is
    # kept as a *pre-flight* check even though both submit paths now re-scan after
    # writing their own claim, because it is what lets the common sequential case
    # ("I already have this downloading") refuse early, before filename resolution
    # and an HF round trip, and without leaving a record behind. The post-write
    # re-scan is the authoritative gate; this one is the cheap one.
    #
    # Residual, unchanged and unclosable here: the scan only reads *this*
    # workspace's state directory, so two invocations from different workspaces
    # aimed at the same file (via `--relative-path ../..` or an absolute path)
    # consult disjoint directories and cannot see each other. See
    # `_active_download_for`.
    in_flight = _active_download_for(local_filepath)
    if in_flight is not None:
        raise _in_flight_failure(in_flight, local_filepath)

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

    # A foreground transfer claims its destination too, for the same reason the
    # `--background` one does: until it did, it wrote no record and so was
    # invisible to every other invocation — a second foreground run, or a
    # `--background` submit started during one, passed both the scan above and
    # `exists()` and the two transfers landed on the same file. Claim first, then
    # re-scan (`_enforce_claim`), then move bytes; the record is what makes this
    # run visible to whoever comes next, and it shows up in `comfy model
    # downloads` like any other.
    claim = _claim_foreground(url=url, dest=local_filepath, downloader=resolved_downloader)
    if claim is not None:
        _enforce_claim(claim, local_filepath)

    try:
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
                        message=(
                            f"`huggingface_hub` is required for Hugging Face downloads and could not be installed: {e}"
                        ),
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
                # Covers `shutil.Error` too — it derives from `OSError` (Lib/shutil.py:
                # `class Error(OSError)`), so the multi-file/partial-move failures
                # `shutil.move` raises under that name land here and end in an
                # envelope, not a bare traceback. Pinned by
                # `test_hf_move_failure_emits_an_envelope_not_a_traceback`, because
                # this branch has no `except Exception` backstop the way the
                # direct-download one below does.
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
                download_file(
                    url,
                    local_filepath,
                    headers,
                    downloader=resolved_downloader,
                    # Feeds the claim record its byte counts — above all the
                    # total, which is what lets `reconcile` tell a killed-but-
                    # finished run from a killed-mid-flight one. See
                    # `_foreground_progress`.
                    progress_callback=_foreground_progress(claim) if claim is not None else None,
                )
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
    except BaseException as e:
        # `BaseException`, not `Exception`: every failure above raises `typer.Exit`,
        # which derives from `click.exceptions.Exit` — a `BaseException` — so an
        # `Exception` handler would let the ordinary error paths past and leave the
        # record pinned at `downloading` until something reconciled it. This also
        # catches the KeyboardInterrupt the error code's hint tells users to send.
        if claim is not None:
            claim.status = "failed"
            # `_download_failure` stashes the message it rendered; a `typer.Exit`
            # stringifies to "" otherwise.
            claim.error = getattr(e, "comfy_error_message", None) or str(e) or type(e).__name__
        raise
    else:
        if claim is not None:
            claim.status = "completed"
            # Best-effort, and only on the success path where the file is final:
            # one stat makes `download-status`/`downloads` report a real size and
            # 100% for this record instead of a bare `completed` with no bytes.
            with contextlib.suppress(OSError):
                claim.completed_bytes = os.stat(local_filepath).st_size
                claim.total_bytes = claim.completed_bytes
    finally:
        # Reached on both paths, but never on SIGKILL — which is exactly the case
        # `download_state.reconcile` already handles: a record whose pid and
        # pid_create_time no longer match a live process demotes to `failed`, so a
        # hard-killed foreground run self-clears just like a dead worker.
        _persist_record(claim)

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

    # Claim `dest` atomically. Creating the claim file with `O_CREAT | O_EXCL`
    # *is* the decision: the kernel serializes the create, exactly one of any
    # number of simultaneous submitters gets the file, and there is no
    # write-then-re-scan window for the losers to slip through. That window is
    # what the previous guard could only narrow — a re-scan cannot see a claim
    # that has not been written yet, and nothing ordered a competitor's `write`
    # before our scan, so when we scanned first and lost the `_claim_order` tie
    # neither side withdrew and both streamed a full copy (reproduced at 12
    # simultaneous submits; `started_at` is second-resolution, so the tie that
    # makes it likely is the common case rather than an exotic one).
    #
    # The record is written *before* the claim, never after: the claim is only a
    # pointer, and a competitor probing it for liveness must never find it
    # pointing at a record of ours that does not exist yet — it would read as
    # stale and be cleared out from under us.
    #
    # Unchanged non-goals. A *foreground* transfer still writes no claim file —
    # it claims with its state record only — and the claim lives under
    # `get_workspace()`, so two invocations run from different workspaces at one
    # destination still consult different claim directories (the documented
    # cross-workspace residual).
    #
    # Which is why the advisory re-scan survives, and runs *first*, before the
    # claim. It is the only thing that sees a competitor writing no claim file —
    # live foreground transfer, or a background record written by a version that
    # predates this code — and running it before the claim is what keeps it from
    # undoing the claim: after the claim, the records of the submitters we just
    # beat are still on disk for the moment it takes them to withdraw, and a
    # winner that re-scanned then would withdraw against its own losers (every
    # submitter backing off, nothing downloading). It cannot deadlock the other
    # way either: withdrawal here is only ever against a strictly earlier
    # `_claim_order`, so the earliest record never withdraws.
    _enforce_claim(state, dest)

    claim_file = _dest_claim_path(dest)
    if claim_file is not None:
        _acquire_dest_claim(state, dest, claim_file)
    # else: the claims directory is unusable (a read-only state dir, or
    # something sitting where `claims/` should be). Bookkeeping must not turn a
    # download that used to work into an error, so that degrades to the advisory
    # guard above — which is exactly the behavior that shipped before this.

    try:
        pid = _spawn_download_worker(state_file, log_file)
    except OSError as e:
        state.status = "failed"
        state.error = f"could not start the background worker: {e}"
        with contextlib.suppress(OSError):
            download_state.write(workspace, state)
        if claim_file is not None:
            # No worker will ever reach a terminal transition for this record, so
            # nothing else would ever release the claim. It would still read
            # stale (the record is `failed`) and self-clear on the next submit,
            # but leaving it is a needless round of cleanup for the next caller.
            download_state.release_claim(claim_file, owner_id=state.id)
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

    # The submitter took an `O_EXCL` claim on this destination and we are the
    # only thing that reaches a terminal transition for it, so releasing it is
    # ours: everything past the cancel-sentinel check runs under a
    # `try/finally: release()`, so no exit — including an OSError out of the
    # very first `write_path` — can strand the claim. Derived, never carried in
    # the record — the claim is addressed by destination, not by download.
    # `release_claim` no-ops unless the claim still names us, so a claim a later
    # submitter already cleared and replaced is never unlinked from under it.
    #
    # Derived defensively: `state.dest` is read back off disk, so `_dest_key`
    # (`realpath`) is being handed untrusted text and can raise on a shape the
    # submitter's own `dest` never had. Releasing a claim must never be what
    # stops the transfer from recording its result — a claim we cannot address
    # is left behind and self-clears on the next submit, exactly like the SIGKILL
    # case.
    claim_file: pathlib.Path | None
    try:
        claim_file = download_state.claim_marker_for(path, _dest_key(state.dest))
    except (OSError, ValueError):
        claim_file = None

    def release() -> None:
        if claim_file is not None:
            download_state.release_claim(claim_file, owner_id=state.id)

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
        # This is the `download-cancel`-before-the-worker-starts path: cancel
        # itself never touches the claim, the worker boots, sees the sentinel and
        # releases it here on the way out.
        release()
        raise typer.Exit(code=0)

    try:
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
    finally:
        release()


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


def _dest_key(path: pathlib.Path | str) -> str:
    """The comparison key for "the same destination".

    ``realpath``, not ``abspath``: ComfyUI model directories are routinely
    symlinks (``models/loras`` pointing at ``/data/loras``) and get addressed
    both ways, and a lexical normalization leaves those two spellings unequal —
    so both submissions would pass and their transfers would land on the same
    inode. ``realpath`` resolves the symlinks that exist and normalizes the rest,
    which also covers a ``--relative-path`` carrying ``..`` (it is only
    ``expanduser``-ed, never passed through :func:`_reject_unsafe_component`).

    ``normcase`` folds case on Windows; on POSIX it is identity, so a
    case-insensitive macOS volume can still alias two spellings past this — the
    same residual ``local_filepath.exists()`` has.
    """
    return os.path.normcase(os.path.realpath(path))


def _claim_order(state: download_state.DownloadState) -> tuple[str, str]:
    """Total order over competing claims on one destination — advisory only.

    ``started_at`` is second-resolution so ties are common; ``id`` (12 random hex
    chars) breaks them, and because both racers compute the same order over the
    same two records they always agree on who won.

    It no longer decides a background submit. Ordering records can only rank the
    ones a racer happens to *see*, and a racer that scans before the other's
    record lands sees no competitor at all — so two submits could rank each other
    into a pair of winners. What decides a background submit now is the `O_EXCL`
    claim file (:func:`_acquire_dest_claim`), where the kernel's create is the
    order and nobody has to see anybody.

    This order survives for the readers that are still advisory and still want a
    deterministic pick: :func:`_active_download_for`'s winner-pick (one live
    claim reported out of several), and :func:`_enforce_claim`, which is the
    whole guard on the *foreground* path — a foreground transfer writes a record
    but no claim file, so its two racers still have nothing else to agree on.
    """
    return (state.started_at or "", state.id)


def _active_download_for(
    dest: pathlib.Path,
    *,
    exclude_id: str | None = None,
) -> download_state.DownloadState | None:
    """The live download already targeting ``dest``, if any — of either kind.

    Foreground runs write records too (see :func:`_claim_foreground`), so this
    scan now covers foreground-vs-foreground and foreground-vs-background as well
    as the background-vs-background case it was written for. It does not
    distinguish them: a claim is a claim, and the caller words its refusal from
    the winner's ``kind``.

    Reconciles the *matching* records (persisting the correction), so a SIGKILLed
    worker's stale record demotes to ``failed`` here and never wedges the path.
    Records for other destinations are filtered out by ``dest`` first and never
    reconciled: :func:`_reconciled` persists a status correction, and running it
    over every record would make a plain ``comfy model download`` rewrite
    bookkeeping for unrelated downloads — off a stale ``list_all`` snapshot, so a
    worker that completed during the scan could have its ``completed`` record
    overwritten with ``failed``. Reconcile never rewrites ``dest``, so the
    unreconciled value is the right key to filter on.

    The predicate is deliberately "reconciled status is active", *not*
    :func:`download_state.worker_alive`: a just-submitted download sits in
    ``starting`` with no pid for up to :data:`download_state.STARTUP_GRACE_S`
    while its worker's interpreter boots, and ``worker_alive`` reports False for
    a pidless record — so gating on it would pass during exactly the
    near-simultaneous double-submit this guard exists to catch. Reconcile keeps
    both a within-grace ``starting`` record and a live worker blocking, while
    demoting a dead worker's record so it self-clears.

    ``exclude_id`` skips one record: :func:`_submit_background_download` re-runs
    this scan *after* writing its own claim and must not find itself.

    The scan is scoped to ``get_workspace()``'s state directory, and that is the
    residual neither this nor the `O_EXCL` claim file closes — the claim lives in
    the same per-workspace state directory. A destination can sit outside the
    workspace (``--relative-path`` is only ``expanduser``-ed, so it accepts
    ``..`` and absolute paths), so two invocations run against *different*
    workspaces but aimed at the same file consult disjoint
    ``<workspace>/.comfy-downloads`` directories and are invisible to each other
    — both claim, both win, both write. Closing it needs a claim that lives next
    to the destination rather than next to the workspace, or a refusal of
    workspace-escaping ``--relative-path``; both are design calls with their own
    tradeoffs and neither is made here.
    """
    wanted = _dest_key(dest)
    winner: download_state.DownloadState | None = None
    try:
        for state in download_state.list_all(get_workspace()):
            if state.id == exclude_id:
                continue
            if _dest_key(state.dest) != wanted:
                continue
            fresh, _ = _reconciled(state)
            if fresh.status not in download_state.ACTIVE_STATUSES:
                continue
            if winner is None or _claim_order(fresh) < _claim_order(winner):
                winner = fresh
    except (OSError, ValueError):
        # The scan is advisory, so an unreadable state directory — or a corrupt
        # record that trips a lookup mid-loop — must degrade to the pre-guard
        # behavior rather than turn a working download into a traceback. This
        # runs on the foreground path too, which never read the state directory
        # before, so a single bad file must not break every download in the
        # workspace. Whole scan, not just `list_all`: the reconcile of a matching
        # record is just as much part of the advisory read.
        return None
    return winner


def _in_flight_failure(competitor: download_state.DownloadState, dest: pathlib.Path) -> typer.Exit:
    """The ``model_download_in_flight`` refusal, worded for the claim we hit.

    Both halves vary by ``kind``. A background download really can be cancelled
    with ``download-cancel``; a live *foreground* one cannot — that command now
    refuses it, because the recorded pid is a user CLI process sharing the
    terminal's foreground process group — so pointing at it would be advice that
    fails. Ctrl-C in the owning terminal is the honest instruction.
    """
    kind = "foreground" if competitor.is_foreground else "background"
    stop = (
        "interrupt it with Ctrl-C in the terminal running it"
        if competitor.is_foreground
        else f"cancel it with `comfy model download-cancel {competitor.id}`"
    )
    return _download_failure(
        code="model_download_in_flight",
        message=f"A {kind} download ({competitor.id}) is already writing to {dest}.",
        hint=f"track it with `comfy model download-status {competitor.id}`, or {stop}",
        details={
            "path": str(dest),
            "download_id": competitor.id,
            "status": competitor.status,
            "kind": competitor.kind,
        },
    )


def _dest_claim_path(dest: pathlib.Path) -> pathlib.Path | None:
    """Where ``dest``'s claim file lives, or None when claims are unavailable.

    Inside the state directory rather than next to the model file: the state dir
    is already owner-only, and a sidecar in `models/loras` would be something
    users (and `comfy model downloads`) trip over. Keyed by :func:`_dest_key`,
    the same canonical spelling `_active_download_for` compares on, so a symlink
    and its target hash to one claim.

    None rather than a raise when the directory cannot be made: claims are
    bookkeeping, and the caller degrades to the advisory guard.
    """
    try:
        return download_state.claim_path(get_workspace(), _dest_key(dest))
    except (OSError, ValueError):
        return None


def _claim_holder(claim_file: pathlib.Path) -> tuple[str | None, download_state.DownloadState | None]:
    """Resolve a claim file to ``(recorded id, live record)``.

    A claim is live iff its ``download_id`` resolves to a state record whose
    *reconciled* status is still active, or whose worker process is still
    provably running. The first arm delegates the liveness question to the
    record's existing `pid` + `pid_create_time` identity proof and
    `STARTUP_GRACE_S` window: a SIGKILLed worker's record demotes to `failed`
    on reconcile, so its claim reads stale here and the next submitter clears
    it — which is why a SIGKILL needs no sweeper. The second arm covers the
    gap the first cannot: a terminal status does not prove the process exited
    (a cancelled worker is still mid-write until its own terminal transition
    lands, and it releases the claim itself right after), so while
    `worker_alive` still proves the recorded process is that worker, the claim
    is its to release, not ours to clear.

    An unreadable or corrupt claim, and a claim whose record is gone, both come
    back with no live record: stale. The id is still returned when it could be
    read, because the retry path words its refusal from it.
    """
    download_id = download_state.read_claim(claim_file)
    if download_id is None:
        return None, None
    record = download_state.read(get_workspace(), download_id)
    if record is None:
        return download_id, None
    fresh, _ = _reconciled(record)
    if fresh.status not in download_state.ACTIVE_STATUSES and not download_state.worker_alive(fresh):
        return download_id, None
    return download_id, fresh


def _withdraw_record(state: download_state.DownloadState, dest: pathlib.Path, winner_id: str | None) -> None:
    """Take our own just-written record back off disk, having lost the claim.

    Same fallback as :func:`_enforce_claim`'s withdrawal: if the unlink fails,
    a terminal status is the next best thing, because the `starting` record we
    wrote would otherwise read as a live claim to `_active_download_for` and to
    `download-cancel` and refuse every later submission to this destination.
    """
    if download_state.delete(get_workspace(), state.id):
        return
    state.status = "failed"
    state.error = f"withdrew this claim; {winner_id or 'another download'} won {dest}"
    _persist_record(state)


# `os.link` is what makes the claim atomic, and a filesystem without hard links
# (exFAT, FAT32, some network and container mounts) refuses it outright rather
# than transiently. Told apart from a transient failure only to word the warning
# accurately: both degrade identically, because a download that used to work must
# not become an error over bookkeeping.
_CLAIMS_UNSUPPORTED_ERRNOS = frozenset(
    getattr(errno, name) for name in ("ENOTSUP", "EOPNOTSUPP", "EPERM", "ENOSYS") if hasattr(errno, name)
)
# ERROR_INVALID_FUNCTION / ERROR_NOT_SUPPORTED — what Windows returns for
# `CreateHardLinkW` on a volume that has no hard links.
_CLAIMS_UNSUPPORTED_WINERRORS = frozenset((1, 50))

_claims_degraded_reported = False


def _report_claim_degraded(exc: OSError, dest: pathlib.Path) -> None:
    """Warn once per process that destination claims are not atomic here.

    Once, because the alternative is a line per submit for a condition that is a
    property of the filesystem and will not change under a running CLI. It is a
    warning rather than a hard failure for the reason `_dest_claim_path` returns
    None rather than raising: claims are bookkeeping, and the advisory guard
    still catches every competitor it caught before this code existed.
    """
    global _claims_degraded_reported
    if _claims_degraded_reported:
        return
    _claims_degraded_reported = True
    unsupported = exc.errno in _CLAIMS_UNSUPPORTED_ERRNOS or (
        getattr(exc, "winerror", None) in _CLAIMS_UNSUPPORTED_WINERRORS
    )
    why = (
        "this filesystem does not support the hard link that publishes them"
        if unsupported
        else "the claim could not be written"
    )
    logger.warning(
        "atomic destination claims are unavailable (%s: %s); falling back to the "
        "advisory in-flight scan for %s, which cannot arbitrate between simultaneous "
        "submissions",
        why,
        exc,
        dest,
    )


def _acquire_dest_claim(
    state: download_state.DownloadState,
    dest: pathlib.Path,
    claim_file: pathlib.Path,
) -> None:
    """Take the `O_EXCL` claim on ``dest``, or withdraw and refuse.

    Returns None when the claim is ours. Otherwise the destination belongs to
    someone else: our own record comes back off disk (so we leave no phantom
    claim) and the caller gets the usual `model_download_in_flight` refusal.

    A claim we lose to is only decisive while it is *live*. A stale one — its
    record demoted by reconcile, deleted, or the file corrupt — is unlinked and
    the create retried exactly ONCE. Once, not in a loop: a second collision
    means another submitter won the retry race rather than that the claim is
    wedged, and that submitter is a competitor to refuse to, not a lock to keep
    fighting for.
    """
    for attempt in (1, 2):
        try:
            if download_state.acquire_claim(claim_file, download_id=state.id, dest=str(dest)):
                return
        except OSError as e:
            # Anything other than the collision (a read-only state dir, a
            # vanished claims directory, a filesystem with no hard links).
            # Degrade to the advisory guard, exactly as an unavailable claims
            # directory does — but say so first: the advisory guard re-scans
            # rather than arbitrates, so this silently gives up the atomicity
            # this whole path exists to provide, and a destination on such a
            # filesystem can be raced again.
            _report_claim_degraded(e, dest)
            _enforce_claim(state, dest)
            return

        holder_id, holder = _claim_holder(claim_file)
        if holder is not None:
            _withdraw_record(state, dest, holder.id)
            raise _in_flight_failure(holder, dest)

        if attempt == 1:
            # Stale: the claim outlived the download it points at. Clear it and
            # try once more. Another submitter may clear it first and win the
            # create — that is the second pass below, not a problem here.
            #
            # Cleared *conditionally*, by the id we just read: between reading a
            # claim and deciding it is stale sits a state-file read and a
            # `reconcile`, and in that gap its worker may finish, release it, and
            # a fresh submitter take a live claim at the same path. An
            # unconditional unlink would delete that live claim and leave two
            # downloads owning one destination. `release_claim` re-reads and only
            # unlinks while the id still matches, which narrows the window to the
            # compare-and-unlink inside it — it does not close it (the filesystem
            # offers no conditional unlink), but the surviving window no longer
            # spans a reconcile. If the claim did change under us, the retry
            # below collides with the new holder and refuses, which is right.
            if not download_state.release_claim(claim_file, owner_id=holder_id):
                # False for two very different reasons, told apart by a re-read.
                # The claim changing hands under us is the takeover race above —
                # fall through and collide with the new holder. The claim still
                # naming the id we judged stale means the unlink itself failed
                # (a claim file we cannot remove, or a directory sitting at the
                # claim path): retrying would collide with the same corpse
                # forever and report a phantom in-flight download, so name the
                # real problem instead.
                if download_state.read_claim(claim_file) == holder_id:
                    _withdraw_record(state, dest, holder_id)
                    raise _download_failure(
                        code="model_download_claim_unclearable",
                        message=f"A stale download claim on {dest} could not be cleared.",
                        hint=f"remove the claim file at {claim_file} and retry, or check its permissions",
                        details={
                            "path": str(dest),
                            "claim_file": str(claim_file),
                            "download_id": holder_id,
                        },
                    )
            continue

        # Second collision: somebody else took the claim we just cleared. Back
        # off rather than clear theirs too — a retry loop over a contested claim
        # is how two submitters livelock each other. Their id, when the claim was
        # readable, is all we can honestly report: this claim did not resolve to
        # a live record, so quoting a status or kind from one would be inventing
        # it — hence a distinct code rather than a `model_download_in_flight`
        # missing the fields that code documents.
        _withdraw_record(state, dest, holder_id)
        named = f" ({holder_id})" if holder_id else ""
        raise _download_failure(
            code="model_download_claim_contested",
            message=f"Another download{named} claimed {dest} first.",
            hint="check `comfy model downloads`, then retry",
            details={"path": str(dest), "download_id": holder_id},
        )


def _enforce_claim(state: download_state.DownloadState, dest: pathlib.Path) -> None:
    """Re-scan for a competing claim on ``dest`` and withdraw ours if we lost.

    The second half of claim-then-check, shared by both submit paths. The caller
    has already *written* ``state`` — that record is the claim — so this scan sees
    any competitor that also claimed ``dest``, which the pre-flight scan in
    :func:`download` structurally cannot: it runs before the claim exists, with
    the ``--background`` split and (for a Hugging Face url) a whole
    ``check_unauthorized`` round trip between it and the write.

    It is check-then-act and cannot be otherwise — hence the `O_EXCL` claim file
    that now decides the background path (:func:`_acquire_dest_claim`). This
    stays because it is the only guard the *foreground* path has (a foreground
    transfer writes no claim file), and because it is what lets a background
    submit that already holds its claim still notice a live foreground competitor.

    Both racers run this over the same two records and rank them with the same
    :func:`_claim_order`, so they agree on the winner instead of both backing off
    — mutual refusal would wedge the destination and neither download would
    happen. The loser deletes its own record before raising, so it leaves no
    phantom claim behind.

    Raises the ``typer.Exit`` the caller should propagate; returns None when we
    won (or when there was no competitor at all).
    """
    competitor = _active_download_for(dest, exclude_id=state.id)
    if competitor is None or _claim_order(state) < _claim_order(competitor):
        return

    if not download_state.delete(get_workspace(), state.id):
        # The unlink failed (a read-only state directory, a permission change
        # under us). Withdrawing the claim is the whole point of this branch, so
        # falling back to a terminal status is the next best thing: a `failed`
        # record is inert to `_active_download_for` and to `download-cancel`,
        # where the `downloading` one we just wrote would read as a live claim and
        # refuse every later submission to this destination until something
        # reconciled it away.
        state.status = "failed"
        state.error = f"withdrew this claim; {competitor.id} won {dest}"
        _persist_record(state)
    raise _in_flight_failure(competitor, dest)


def _claim_foreground(url: str, dest: pathlib.Path, downloader: str) -> download_state.DownloadState | None:
    """Claim ``dest`` for a transfer about to run in *this* process.

    Before this existed the foreground path wrote no record at all, so it was
    invisible to every other invocation: a second foreground run — or a
    ``--background`` submit started during one — sailed past the destination scan
    and past ``exists()`` (the httpx downloader streams into a ``.part`` sibling,
    so nothing is at ``dest`` until the very end), and the two transfers
    interleaved into one file. With ``--downloader aria2`` they interleave
    literally, since aria2 writes straight to the destination.

    ``pid``/``pid_create_time`` are this CLI process's own, which is what makes
    the record self-clearing: a run that is SIGKILLed never reaches its ``finally``
    to write a terminal status, and :func:`download_state.reconcile` then demotes
    the record exactly as it does for a dead worker. Recording
    ``pid_create_time`` is not optional for that to work — without it
    :func:`download_state.is_worker_process` falls back to matching the *worker's*
    argv marker, which a foreground CLI process does not carry.

    Returns None when the claim could not be persisted, *or* when this process's
    start time could not be read. Both are deliberate degradations to the
    pre-claim behavior rather than failures: the state directory is bookkeeping,
    and an unwritable workspace must not turn a download that used to work into
    an error. (The ``--background`` path *does* fail there, because a detached
    worker has nowhere else to report from.)
    """
    # Absolute, as `_submit_background_download` also takes care to be. `dest` is
    # built from `workspace_manager.workspace_path`, which is not guaranteed
    # absolute, and this string is read back by *other* processes: `_dest_key`
    # resolves it with `realpath`, which anchors a relative path to the reader's
    # cwd, so a relative `dest` would key differently for a reader started
    # elsewhere and the two would not recognize each other's claim.
    # `download_cancel` reads it as a plain path too.
    dest = pathlib.Path(dest).absolute()
    # `_scrub_url`, not `url`: a foreground record is written for its *claim*, and
    # nothing ever reads this field back from one (only the detached worker needs
    # the real url, and it gets its own record). A resolved download url routinely
    # carries a credential — CivitAI links append `?token=`, presigned S3/SAS links
    # carry the signature in the query — so persisting it verbatim would drop a
    # secret into `<workspace>/.comfy-downloads/`, a directory inside the ComfyUI
    # checkout that nothing gitignores, for no reader's benefit.
    state = download_state.new(url=_scrub_url(url), dest=str(dest), downloader=downloader)
    state.kind = "foreground"
    state.status = "downloading"
    state.pid = os.getpid()
    state.pid_create_time = download_state.process_create_time(state.pid)
    if state.pid_create_time is None:
        # Without the start time this claim is unfalsifiable. `worker_alive` falls
        # back to bare pid liveness when `pid_create_time` is None, so once this
        # process exits and the OS recycles its number the record reads *live*
        # forever: `reconcile` never demotes it, every later download to this
        # destination is refused as `model_download_in_flight`, and
        # `download-cancel` refuses it as foreground — leaving hand-deletion of
        # the JSON as the only way out. A claim that cannot retract itself on
        # death is worse than no claim, so degrade to the pre-claim behavior
        # exactly as an unwritable state directory does.
        return None
    try:
        download_state.write(get_workspace(), state)
    except (OSError, ValueError):
        return None
    return state


def _persist_record(state: download_state.DownloadState | None) -> None:
    """Write a download record to disk. Never raises.

    Kind-agnostic, and named that way because it is: the foreground path uses it
    for both the progress writes and the terminal one, and the background
    arbitration path uses it in :func:`_withdraw_record`. The state directory is
    bookkeeping either way, and a download that is otherwise fine must not die
    because a record could not be updated.
    """
    if state is None:
        return
    with contextlib.suppress(OSError, ValueError):
        download_state.write(get_workspace(), state)


def _foreground_progress(state: download_state.DownloadState) -> Callable[[int, int | None], None]:
    """A ``progress_callback`` that keeps a foreground claim's counters current.

    Chiefly for ``total_bytes``, which is what lets a foreground record survive a
    SIGKILL honestly. :func:`download_state.reconcile` resolves a dead ``downloading``
    record to ``completed`` only when ``total_bytes`` is known and the file on disk
    reached it; with no callback the field stayed None for the whole transfer, so a
    run killed in the window between the rename and the ``finally`` below left a
    *complete* model at ``dest`` under a record that read ``failed`` forever — and
    ``download-cancel`` then deleted that finished file as an aria2 partial.

    It also makes the record's progress real. ``comfy model downloads`` and
    ``download-status`` now list foreground downloads, and without this they would
    report 0 bytes and no percent for the entire transfer.

    Writes are throttled to :data:`download_state.PROGRESS_THROTTLE_S`, as the
    worker's are, with one exception: a newly-learned total is written
    immediately. It arrives once, before the first chunk, and it is the field
    that matters if this process is killed a moment later.
    """
    last_write = 0.0

    def on_progress(completed: int, total: int | None) -> None:
        nonlocal last_write
        state.completed_bytes = completed
        learned_total = total is not None and state.total_bytes != total
        if total is not None:
            state.total_bytes = total
        now = time.monotonic()
        if not learned_total and now - last_write < download_state.PROGRESS_THROTTLE_S:
            return
        last_write = now
        _persist_record(state)

    return on_progress


@app.command("download-status")
@tracking.track_command("model")
def download_status(
    _ctx: typer.Context,
    download_id: Annotated[str, typer.Argument(help="The download id returned by `download --background`.")],
):
    """Report the progress of one download, background or foreground."""
    renderer = get_renderer()
    state = download_state.read(get_workspace(), download_id)
    if state is None:
        renderer.error(
            code="download_not_found",
            message=f"No download with id {download_id!r}.",
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
    """List every download this workspace knows about, newest first.

    Covers both kinds: a plain `comfy model download` claims its destination the
    same way a `--background` one does, so it appears here too (`kind` tells them
    apart). That is what makes a second run refuse instead of racing into the
    same file.
    """
    renderer = get_renderer()
    workspace = get_workspace()
    # Trim before listing: this is the verb whose cost — and whose output — grows
    # with every record the directory has ever accumulated.
    with contextlib.suppress(Exception):
        download_state.prune(workspace)
    rows = [download_state.status_payload(_reconciled(s)[0]) for s in download_state.list_all(workspace)]
    if not rows:
        print("No downloads found.")
    else:
        _render_download_rows(rows)
    renderer.emit({"total": len(rows), "downloads": rows}, command="model downloads")


@app.command("download-cancel")
@tracking.track_command("model")
def download_cancel(
    _ctx: typer.Context,
    download_id: Annotated[str, typer.Argument(help="The download id returned by `download --background`.")],
):
    """Kill a background download's worker and remove its partial file.

    Background only, for a live download: a foreground record's pid is a user CLI
    process sharing its terminal's foreground process group, so the `killpg` this
    sends would take out the surrounding shell job. A live foreground download is
    refused with `model_download_foreground_cancel` and the instruction to Ctrl-C
    it in its own terminal; a *dead* one still sweeps here as usual.
    """
    renderer = get_renderer()
    workspace = get_workspace()
    state = download_state.read(workspace, download_id)
    if state is None:
        renderer.error(
            code="download_not_found",
            message=f"No download with id {download_id!r}.",
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

    # Refuse a *live* foreground download before anything is signalled — sentinel
    # included. Everything below this point assumes the recorded pid belongs to a
    # detached worker, which `_spawn_download_worker` gives its own session: that
    # is what makes `kill_worker`'s `os.killpg(os.getpgid(pid), ...)` safe, since
    # the worker is its own process-group leader and the group contains only it
    # and its children. A foreground record's pid is the *user's CLI process*,
    # which shares the terminal's foreground process group — so the same killpg
    # would take out the surrounding shell job (the pipeline, an enclosing
    # script), not just the transfer. There is no narrower signal to send from
    # here, so the honest answer is to tell the user where to press Ctrl-C.
    #
    # A *dead* foreground record still goes down the normal path below: the
    # partial file it left behind is exactly what the user is trying to reclaim,
    # and `stop_worker`/`kill_worker` are inert for a pid that no longer matches
    # its recorded start time.
    if state.is_foreground and download_state.worker_alive(state):
        renderer.error(
            code="model_download_foreground_cancel",
            message=f"Download {download_id} is running in the foreground of another terminal (pid {state.pid}).",
            hint="interrupt it with Ctrl-C in the terminal running it",
            details={"id": download_id, "pid": state.pid, "kind": state.kind, "dest": state.dest},
        )
        raise typer.Exit(code=1)

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


@app.command(
    "list",
    help="List local models on disk (files already downloaded into the current workspace). "
    "For backend/cloud discovery use `comfy model list-folders` / `list-folder`.",
)
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


# The `model` noun owns BOTH the local-filesystem ops defined above
# (download/remove/list) and the backend/cloud discovery leaves
# (list-folders/list-folder/search/show). The discovery leaves are implemented
# on `models_search_command.app`; surface them under `model` by borrowing
# their command registrations (same CommandInfo objects — no logic
# duplication). Done here rather than in `cmdline.py` so the `model` lazy
# subcommand entry there stays a plain `getattr(module, "app")` — it only
# needs to import this module to get the fully-merged tree.
# Mirror the deprecated alias built in `search.py`: carry BOTH the discovery
# leaves and any nested discovery sub-groups so `comfy model …` and the
# `comfy models …` alias stay in lockstep as the discovery tree grows (both
# are empty of sub-groups today). A group callback is deliberately NOT carried
# over here: on the alias it scopes to discovery-only leaves, but `model` also
# owns the local ops (download/remove/list), so mounting discovery's group
# setup on the whole noun would leak it onto those.
app.registered_commands.extend(models_search_command.app.registered_commands)
app.registered_groups.extend(models_search_command.app.registered_groups)
