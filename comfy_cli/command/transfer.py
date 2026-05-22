"""``comfy upload`` / ``comfy download`` — move files between local disk and ComfyUI.

Upload sends local files to the server's input directory (both local and cloud).
Download fetches outputs from completed jobs to the local filesystem.

Pipe-friendly: ``comfy --json run --wait | comfy download`` reads the prompt_id
and output URLs from stdin, avoiding manual extraction.
"""

from __future__ import annotations

import json
import mimetypes
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import typer

from comfy_cli import jobs_state
from comfy_cli.comfy_client import Client, Unauthenticated
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.target import resolve_target


def _default_out_dir() -> str:
    """Return the configured project outputs dir, or ./outputs as fallback."""
    try:
        from comfy_cli.config_manager import ConfigManager
        from comfy_cli.constants import CONFIG_KEY_DEFAULT_PROJECT_DIR

        project = ConfigManager().get(CONFIG_KEY_DEFAULT_PROJECT_DIR)
        if project:
            from pathlib import Path

            d = Path(project) / "outputs"
            if d.is_dir():
                return str(d)
    except Exception:  # noqa: BLE001
        pass
    return "./outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(target: Any) -> dict[str, str]:
    """Build auth headers for a target (cloud only)."""
    headers: dict[str, str] = {}
    if target.is_cloud:
        if target.api_key:
            headers["X-API-Key"] = target.api_key
        elif target.auth_token:
            headers["Authorization"] = f"Bearer {target.auth_token}"
    return headers


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def execute_upload(
    files: list[str],
    *,
    where: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Upload one or more local files to the ComfyUI server's input directory.

    Returns the list of server-side filenames (the ``name`` field from each
    upload response).
    """
    renderer = get_renderer()
    target = resolve_target(where=where)

    uploads: list[dict[str, Any]] = []
    cloud_names: list[str] = []

    for filepath in files:
        path = Path(filepath)
        if not path.is_file():
            renderer.error(
                code="upload_failed",
                message=f"File not found: {filepath}",
                hint="check the file path and try again",
                details={"filename": filepath},
            )
            raise typer.Exit(code=1)

        filename = path.name
        file_data = path.read_bytes()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # Build multipart/form-data body
        boundary = uuid.uuid4().hex
        body = b""
        # -- overwrite field
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        body += (b"true" if overwrite else b"false") + b"\r\n"
        # -- file field
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        body += file_data
        body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        url = target.url("upload/image")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        for hdr, val in _auth_headers(target).items():
            req.add_header(hdr, val)

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            status = e.code
            renderer.error(
                code="upload_failed",
                message=f"Failed to upload {filename}: HTTP {status}",
                hint="check the file exists and the server is reachable",
                details={"status": status, "filename": filename},
            )
            raise typer.Exit(code=1)

        cloud_name = result.get("name", filename)
        subfolder = result.get("subfolder", "")
        file_type = result.get("type", "input")

        cloud_names.append(cloud_name)
        uploads.append(
            {
                "local_path": str(path.resolve()),
                "cloud_name": cloud_name,
                "subfolder": subfolder,
                "type": file_type,
            }
        )
        pprint(f"✓ uploaded {filename} → {cloud_name} ({file_type})")

    renderer.emit({"uploads": uploads}, command="upload")
    return cloud_names


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def execute_download(
    prompt_id: str | None = None,
    *,
    out_dir: str | None = None,
    where: str | None = None,
    url_only: bool = False,
) -> list[str]:
    """Download all outputs from a completed job to a local directory.

    Supports piped input: ``comfy --json run --wait | comfy download``.
    Returns the list of saved file paths.
    """
    renderer = get_renderer()
    piped_urls: list[str] = []

    # -- Try reading from stdin if prompt_id wasn't given explicitly ----------
    if prompt_id is None and not sys.stdin.isatty():
        try:
            envelope = json.load(sys.stdin)
        except (json.JSONDecodeError, ValueError):
            envelope = {}
        prompt_id = envelope.get("data", {}).get("prompt_id")
        piped_urls = envelope.get("data", {}).get("outputs", []) or []

    if not prompt_id:
        renderer.error(
            code="download_no_prompt",
            message="No prompt_id provided",
            hint=("pass a prompt_id argument, or pipe the output of 'comfy --json run --wait' into this command"),
        )
        raise typer.Exit(code=1)

    target = resolve_target(where=where)

    # -- Resolve output URLs --------------------------------------------------
    output_urls: list[str] = []

    if piped_urls:
        output_urls = list(piped_urls)
    else:
        # Try the on-disk state file first
        state = jobs_state.read(prompt_id)
        if state is not None and state.outputs:
            output_urls = list(state.outputs)
        else:
            # Fall back to querying the API
            try:
                client = Client(target)
                record = client.get_history(prompt_id)
                if record is not None:
                    output_urls = client.extract_output_urls(record)
                else:
                    renderer.error(
                        code="download_job_not_found",
                        message=f"Job {prompt_id} not found in state files or API",
                        hint="check the prompt_id and ensure the job has completed",
                        details={"prompt_id": prompt_id},
                    )
                    raise typer.Exit(code=1)
            except Unauthenticated:
                renderer.error(
                    code="download_job_not_found",
                    message=f"Job {prompt_id} not found in local state files and cloud auth is missing",
                    hint="run 'comfy cloud login' or check the prompt_id",
                    details={"prompt_id": prompt_id},
                )
                raise typer.Exit(code=1)

    if not output_urls:
        renderer.error(
            code="download_no_outputs",
            message=f"Job {prompt_id} has no outputs yet",
            hint="wait for the job to complete before downloading",
            details={"prompt_id": prompt_id},
        )
        raise typer.Exit(code=1)

    # -- URL-only mode: emit URLs without downloading --------------------------
    if url_only:
        renderer.emit(
            {
                "prompt_id": prompt_id,
                "urls": output_urls,
            },
            command="download",
        )
        return output_urls

    # -- Download each URL ----------------------------------------------------
    dest = Path(out_dir or _default_out_dir())
    dest.mkdir(parents=True, exist_ok=True)

    auth_hdrs = _auth_headers(target)
    saved_files: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    short_id = prompt_id[:8]

    for idx, url in enumerate(output_urls):
        # Derive extension from the URL's filename query param
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        remote_name = qs.get("filename", ["output.png"])[0]
        ext = Path(remote_name).suffix or ".png"
        local_name = f"{short_id}_{idx:03d}{ext}"
        local_path = dest / local_name

        req = urllib.request.Request(url)
        for hdr, val in auth_hdrs.items():
            req.add_header(hdr, val)

        try:
            # Use default urlopen — follows redirects (needed for signed
            # storage URLs on cloud).
            with urllib.request.urlopen(req) as resp:
                with open(local_path, "wb") as fp:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fp.write(chunk)
        except urllib.error.HTTPError as e:
            renderer.error(
                code="download_failed",
                message=f"Failed to download output {idx}: HTTP {e.code}",
                hint="check that the job completed successfully and the server is reachable",
                details={"status": e.code, "url": url, "index": idx},
            )
            raise typer.Exit(code=1)

        file_size = local_path.stat().st_size
        saved_files.append(
            {
                "url": url,
                "path": str(local_path.resolve()),
                "size": file_size,
            }
        )
        saved_paths.append(str(local_path.resolve()))

    pprint(f"✓ downloaded {len(saved_files)} file(s) to {dest}")

    # Show inline previews for human users (skipped in JSON/agent mode)
    if renderer.is_pretty():
        from comfy_cli.output.preview import preview

        for sf in saved_files:
            preview(sf["path"])

    renderer.emit(
        {
            "prompt_id": prompt_id,
            "out_dir": str(dest.resolve()),
            "files": saved_files,
        },
        command="download",
    )
    return saved_paths
