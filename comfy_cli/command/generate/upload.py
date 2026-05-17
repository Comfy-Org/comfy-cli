"""Upload reference assets via ``/customers/storage``.

The cloud endpoint issues short-lived signed URLs:

1. POST ``/customers/storage`` with ``{file_name, content_type, file_hash?}`` →
   ``{upload_url, download_url, expires_at, existing_file}``.
2. If ``existing_file`` is true the server already has a hash-match — skip the
   PUT and reuse ``download_url``. Otherwise PUT the bytes to ``upload_url``
   with the same ``Content-Type`` header.
3. ``download_url`` is what downstream model calls reference; it's a signed URL
   that expires after 24 hours.

This module also exposes a small helper that takes either a local path or a
remote ``http(s)://`` URL — remote URLs are re-hosted by downloading and then
running the same flow.
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import httpx

from comfy_cli.command.generate import client, spec

_DEFAULT_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class UploadResult:
    url: str  # the signed download_url to feed into downstream model calls
    expires_at: str | None  # ISO 8601 timestamp from the server
    existing_file: bool  # True when the server returned a hash-match (no upload)


def _guess_content_type(name: str) -> str:
    ctype, _ = mimetypes.guess_type(name)
    return ctype or _DEFAULT_CONTENT_TYPE


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request_signed_url(
    file_name: str,
    content_type: str,
    file_hash: str,
    api_key: str,
) -> dict:
    """POST /customers/storage and return the parsed response dict."""
    url = spec.base_url() + "/customers/storage"
    body = {"file_name": file_name, "content_type": content_type, "file_hash": file_hash}
    headers = client._auth_headers(api_key, {"Content-Type": "application/json"})
    resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
    client.raise_for_status(resp)
    try:
        return resp.json()
    except ValueError as e:
        # Surface the same way other parse errors do — bare ValueError would
        # leak a traceback into CLI output.
        raise client.ApiError(resp.status_code, resp.text or str(e), "Storage response was not valid JSON") from e


def _put_bytes(upload_url: str, data: bytes, content_type: str) -> None:
    """PUT the raw bytes to the signed URL. No auth header — the URL is signed."""
    with httpx.Client(timeout=120.0, follow_redirects=False) as c:
        r = c.put(upload_url, content=data, headers={"Content-Type": content_type})
        if r.status_code >= 400:
            raise client.ApiError(r.status_code, r.text, f"Upload to signed URL failed (HTTP {r.status_code})")


def upload_bytes(data: bytes, file_name: str, api_key: str, content_type: str | None = None) -> UploadResult:
    """Upload raw bytes and return the signed download URL. Hash-based dedup is
    handled transparently — if the server already has these bytes, ``existing_file``
    is True and no PUT happens."""
    ctype = content_type or _guess_content_type(file_name)
    file_hash = _sha256_hex(data)
    signed = _request_signed_url(file_name=file_name, content_type=ctype, file_hash=file_hash, api_key=api_key)
    if not signed.get("existing_file"):
        upload_url = signed.get("upload_url")
        if not upload_url:
            raise client.ApiError(0, str(signed), "Server response missing upload_url")
        _put_bytes(upload_url, data, ctype)
    download_url = signed.get("download_url")
    if not download_url:
        raise client.ApiError(0, str(signed), "Server response missing download_url")
    return UploadResult(
        url=download_url,
        expires_at=signed.get("expires_at"),
        existing_file=bool(signed.get("existing_file", False)),
    )


def upload_path(path: Path | str, api_key: str) -> UploadResult:
    p = Path(path).expanduser()
    if not p.is_file():
        raise client.ApiError(0, "", f"File not found: {p}")
    try:
        data = p.read_bytes()
    except OSError as e:
        raise client.ApiError(0, "", f"Unable to read file: {p} ({e})") from e
    return upload_bytes(data, file_name=p.name, api_key=api_key)


def upload_remote_url(url: str, api_key: str) -> UploadResult:
    """Re-host a remote http(s) URL through /customers/storage so it ends up on
    Comfy's CDN. Mirrors the genmedia behavior of accepting URLs to `upload`."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        data = r.content
        # Prefer the server's Content-Type; fall back to URL extension.
        ctype = r.headers.get("content-type", "").split(";", 1)[0].strip() or _guess_content_type(url)
        # Pick a filename from the URL path, defaulting to a hash-based name.
        suffix = Path(url.split("?", 1)[0]).name or _sha256_hex(data)[:12]
    return upload_bytes(data, file_name=suffix, api_key=api_key, content_type=ctype)


def upload_target(target: str | Path, api_key: str) -> UploadResult:
    """Accept either a local file path or a remote URL; return the hosted URL."""
    s = str(target)
    if s.startswith(("http://", "https://")):
        return upload_remote_url(s, api_key=api_key)
    return upload_path(s, api_key=api_key)
