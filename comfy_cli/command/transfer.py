"""``comfy upload`` / ``comfy download`` — move files between local disk and ComfyUI.

Upload sends local files to the server's input directory (both local and cloud).
Download fetches outputs from completed jobs to the local filesystem.

Pipe-friendly: ``comfy --json run --wait | comfy download`` reads the prompt_id
and output URLs from stdin, avoiding manual extraction.
"""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import typer

from comfy_cli import jobs_state
from comfy_cli.comfy_client import Client, Unauthenticated, extract_output_entries
from comfy_cli.http import NoRedirectHandler
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.target import resolve_target


def _default_out_dir() -> str:
    """Return the governing project/1 root's ``outputs/`` dir, else the
    legacy ``default_project_dir`` config key's outputs dir, else ./outputs."""
    # project/1 convention first: downloads land in <root>/outputs by
    # contract (execute_download mkdirs the dest, so it need not exist yet).
    try:
        from comfy_cli.project import find_project

        p = find_project()
        if p is not None:
            return str(p.root / "outputs")
    except Exception:  # noqa: BLE001
        pass
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


# Stripped on every download redirect so auth never crosses origins.
_AUTH_HEADERS_TO_STRIP = frozenset({"authorization", "x-api-key", "x-comfy-api-key", "cookie"})
_MAX_REDIRECTS = 5


class _DownloadRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects but strip auth headers — cloud's `/api/view` 302s to
    a signed GCS URL where the signature is the auth."""

    max_redirections = _MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        scheme = urllib.parse.urlsplit(newurl).scheme
        if scheme not in ("http", "https"):
            raise urllib.error.HTTPError(
                req.full_url, code, f"refusing redirect to non-HTTP scheme: {scheme}", headers, fp
            )
        for src in (new_req.headers, new_req.unredirected_hdrs):
            for key in list(src.keys()):
                if key.lower() in _AUTH_HEADERS_TO_STRIP:
                    del src[key]
        return new_req


_TRANSFER_OPENER = urllib.request.build_opener(NoRedirectHandler("redirect refused (auth leak prevention)"))
_DOWNLOAD_OPENER = urllib.request.build_opener(_DownloadRedirectHandler())


def _sanitize_multipart_filename(name: str) -> str:
    """Escape a filename for use in Content-Disposition per RFC 7578.

    Strips characters that break multipart framing (quotes, backslashes,
    carriage returns, newlines) to prevent header injection.
    """
    return re.sub(r'["\\\r\n]', "_", name)


def _assert_download_url(url: str) -> None:
    """Reject download URLs that aren't http(s) to prevent SSRF."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"refusing to download from non-HTTP URL: {url}")


def _sanitize_item_name(item: str) -> str:
    """A filesystem-safe token for an item key used in download filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", item) or "item"


def _collision_safe_path(path: Path) -> Path:
    """Never overwrite an existing download: ``name.ext`` → ``name.1.ext``,
    ``name.2.ext``, … (deterministic, first free slot).

    A retry fan-out reusing the same item ids re-downloads into the same
    out-dir, and the per-job counters restart at 000 — without this, attempt
    2 silently clobbers attempt 1. Symlinks (including dangling ones) count
    as taken so the suffix walk can never be steered into writing through
    one.
    """
    if not path.exists() and not path.is_symlink():
        return path
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem}.{n}{path.suffix}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        n += 1


def _annotate_output_urls(output_urls: list[str], state) -> list[tuple[str | None, str | None]]:
    """Per output URL: ``(node_id, item)`` provenance, ``None`` when unknown.

    Joins URLs back to the state file's final history ``record`` on the
    (filename, subfolder, type) query-param triple — the same triple
    ``Client.view_url`` encodes — so matching survives base_url drift between
    submit and download. ``node_id -> item`` comes from the compose
    ``item_map`` (a node belongs to an item when it appears in the item's
    ``nodes`` list or is its ``save_node``). Without a record everything is
    ``(None, None)`` and the caller falls back to legacy naming.
    """
    if state is None or not isinstance(state.record, dict):
        return [(None, None)] * len(output_urls)

    key_to_node: dict[tuple[str, str, str], str] = {}
    for entry in extract_output_entries(state.record):
        key_to_node.setdefault((entry["filename"], entry["subfolder"], entry["type"]), entry["node_id"])

    node_to_item: dict[str, str] = {}
    for item, entry in (state.item_map or {}).items():
        if not isinstance(entry, dict):
            continue
        members = list(entry.get("nodes") or [])
        if entry.get("save_node") is not None:
            members.append(entry["save_node"])
        for node_id in members:
            node_to_item[str(node_id)] = str(item)

    annotations: list[tuple[str | None, str | None]] = []
    for url in output_urls:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        key = (
            qs.get("filename", [""])[0],
            qs.get("subfolder", [""])[0],
            qs.get("type", ["output"])[0],
        )
        node_id = key_to_node.get(key)
        item = node_to_item.get(node_id) if node_id is not None else None
        annotations.append((node_id, item))
    return annotations


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _upload_file(path: Path, target: Any, *, overwrite: bool) -> dict:
    """POST one local file to ``target``'s ``/upload/image`` endpoint.

    This is the ONLY ingestion path the CLI uses — files always travel over
    the server's HTTP API, never by writing into a ComfyUI install's folders.
    Returns the server's parsed JSON response (``name``/``subfolder``/``type``).
    Raises ``urllib.error.HTTPError`` on a non-2xx response; callers own the
    envelope/error rendering.
    """
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
    safe_filename = _sanitize_multipart_filename(filename)
    body += f'Content-Disposition: form-data; name="image"; filename="{safe_filename}"\r\n'.encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += file_data
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    url = target.url("upload/image")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    for hdr, val in _auth_headers(target).items():
        req.add_header(hdr, val)

    with _TRANSFER_OPENER.open(req) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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
        file_size = path.stat().st_size
        max_upload = 2 * 1024 * 1024 * 1024  # 2 GB safety cap
        if file_size > max_upload:
            renderer.error(
                code="upload_failed",
                message=f"File too large: {file_size} bytes (limit {max_upload})",
                hint="compress or resize the file before uploading",
                details={"filename": filepath, "size": file_size, "limit": max_upload},
            )
            raise typer.Exit(code=1)
        try:
            result = _upload_file(path, target, overwrite=overwrite)
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
        # Human progress line is pretty-mode-only: machine consumers read the
        # envelope, and stdout must stay pure JSON for `| jq` pipelines.
        if renderer.is_pretty():
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
    # `comfy --json run --wait | comfy download` is the documented pipe
    # pattern, so this must survive whatever the upstream wrote: an error
    # envelope (`"data": null`), a non-envelope JSON value, or non-JSON
    # garbage — never a traceback.
    if prompt_id is None and not sys.stdin.isatty():
        try:
            envelope = json.load(sys.stdin)
        except (json.JSONDecodeError, ValueError):
            envelope = {}
        if not isinstance(envelope, dict):
            envelope = {}
        data = envelope.get("data")
        if not isinstance(data, dict):
            data = {}
        if envelope.get("ok") is False:
            # Upstream command failed — surface its error (and prompt_id when
            # it carried one, e.g. a --wait poll failure on a still-running
            # job) so the caller can recover instead of guessing.
            upstream_error = envelope.get("error") if isinstance(envelope.get("error"), dict) else None
            upstream_details = (upstream_error or {}).get("details")
            if not isinstance(upstream_details, dict):
                upstream_details = {}
            upstream_prompt_id = data.get("prompt_id") or upstream_details.get("prompt_id")
            details: dict[str, Any] = {"upstream_command": envelope.get("command")}
            if upstream_error is not None:
                details["upstream_error"] = upstream_error
            if upstream_prompt_id:
                details["prompt_id"] = upstream_prompt_id
            renderer.error(
                code="download_no_prompt",
                message="Piped envelope reports a failed upstream command (ok=false) — nothing to download",
                hint=(
                    f"the upstream job may still exist; check `comfy jobs status {upstream_prompt_id}` "
                    f"and retry `comfy download {upstream_prompt_id}`"
                    if upstream_prompt_id
                    else "fix the upstream failure (see details.upstream_error), then re-run the pipe"
                ),
                details=details,
            )
            raise typer.Exit(code=1)
        prompt_id = data.get("prompt_id")
        piped_urls = data.get("outputs") or []

    if not prompt_id:
        renderer.error(
            code="download_no_prompt",
            message="No prompt_id provided",
            hint=("pass a prompt_id argument, or pipe the output of 'comfy --json run --wait' into this command"),
        )
        raise typer.Exit(code=1)

    target = resolve_target(where=where)

    # -- Resolve output URLs --------------------------------------------------
    # The state file is read regardless of the URL source: its `record` +
    # `item_map` (when present) drive item-aware file naming below.
    state = jobs_state.read(prompt_id)
    output_urls: list[str] = []

    if piped_urls:
        output_urls = list(piped_urls)
    else:
        # Try the on-disk state file first
        if state is not None and state.outputs:
            output_urls = list(state.outputs)
        else:
            # Fall back to querying the API. Download is an observer command —
            # often running in concurrent retry loops — so it must never clear
            # the shared OAuth session on a fatal refresh error.
            try:
                client = Client(target, clear_session_on_auth_failure=False)
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
            except Unauthenticated as exc:
                renderer.error(
                    code="download_job_not_found",
                    message=f"Job {prompt_id} not found in local state files and cloud auth is missing",
                    hint="run 'comfy cloud login' or check the prompt_id",
                    details={"prompt_id": prompt_id},
                )
                raise typer.Exit(code=1) from exc

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

    # Provenance per URL (node id + compose foreach item) from the state
    # file's record/item_map. Item-mapped outputs are named `<item>_<nnn>`
    # with a PER-ITEM counter; everything else keeps `<prompt8>_<idx>`.
    annotations = _annotate_output_urls(output_urls, state)
    item_counters: dict[str, int] = {}

    for idx, url in enumerate(output_urls):
        # Derive extension from the URL's filename query param
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        remote_name = qs.get("filename", ["output.png"])[0]
        ext = Path(remote_name).suffix or ".png"
        node_id, item = annotations[idx]
        if item is not None:
            safe_item = _sanitize_item_name(item)
            n = item_counters.get(safe_item, 0)
            item_counters[safe_item] = n + 1
            local_name = f"{safe_item}_{n:03d}{ext}"
        else:
            local_name = f"{short_id}_{idx:03d}{ext}"
        # Suffix deterministically instead of overwriting a prior attempt.
        local_path = _collision_safe_path(dest / local_name)

        try:
            _assert_download_url(url)
        except ValueError as e:
            renderer.error(
                code="download_failed",
                message=str(e),
                hint="output URLs should be http or https",
                details={"url": url, "index": idx},
            )
            raise typer.Exit(code=1)

        # Refuse to overwrite symlinks (could be pointed at arbitrary files).
        if local_path.is_symlink():
            renderer.error(
                code="download_failed",
                message=f"Refusing to write to symlink: {local_path}",
                hint="remove the symlink and retry",
                details={"path": str(local_path), "index": idx},
            )
            raise typer.Exit(code=1)

        req = urllib.request.Request(url)
        for hdr, val in auth_hdrs.items():
            req.add_header(hdr, val)

        max_download = 10 * 1024 * 1024 * 1024  # 10 GB safety cap
        try:
            with _DOWNLOAD_OPENER.open(req) as resp:
                total = 0
                with open(local_path, "wb") as fp:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_download:
                            raise ValueError(f"download exceeds {max_download} byte safety limit")
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
        entry: dict[str, Any] = {
            "url": url,
            "path": str(local_path.resolve()),
            "size": file_size,
        }
        # Optional provenance keys — present only when known (no nulls).
        if node_id is not None:
            entry["node_id"] = node_id
        if item is not None:
            entry["item"] = item
        saved_files.append(entry)
        saved_paths.append(str(local_path.resolve()))

    # Human progress line + inline previews are pretty-mode-only: machine
    # consumers read the envelope, and `comfy --json download | jq` requires
    # stdout to carry nothing but JSON (envelope as the last line).
    if renderer.is_pretty():
        pprint(f"✓ downloaded {len(saved_files)} file(s) to {dest}")

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
