"""HTTP client for the Comfy cloud API.

A thin wrapper around httpx that:
- attaches ``Authorization: Bearer $COMFY_API_KEY`` to every request,
- targets ``$COMFY_API_BASE_URL`` (defaulting to ``https://api.comfy.org``),
- splits a request payload into JSON or multipart based on the endpoint's
  declared content-type, streaming any ``format: binary`` fields as files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from comfy_cli.command.generate import spec
from comfy_cli.command.generate.schema import FlagDef


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, message: str | None = None) -> None:
        super().__init__(message or f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def resolve_api_key(explicit: str | None = None) -> str:
    """Order: explicit flag → COMFY_API_KEY env var. Raise if neither set."""
    key = explicit.strip() if isinstance(explicit, str) and explicit.strip() else os.environ.get("COMFY_API_KEY", "")
    key = key.strip()
    if not key:
        raise ApiError(
            401,
            "",
            "No API key. Pass --api-key or set COMFY_API_KEY in your environment. "
            "Generate one at https://platform.comfy.org/api-keys.",
        )
    return key


def _split_payload(
    values: dict[str, Any], flags: list[FlagDef], content_type: str
) -> tuple[dict[str, Any] | None, list[tuple[str, Any]] | None, dict[str, Any] | None]:
    """Return (json_body, multipart_files, multipart_data).

    For JSON endpoints: json_body is the dict, others are None.
    For multipart: files is a list of (field_name, (filename, fileobj, mime)) tuples
    and data is the non-file form fields (stringified or JSON-encoded as needed).
    """
    flag_by_name = {f.name: f for f in flags}
    if content_type != "multipart/form-data":
        return values, None, None

    files: list[tuple[str, Any]] = []
    data: dict[str, Any] = {}
    for name, value in values.items():
        flag = flag_by_name.get(name)
        if flag and flag.kind == "binary":
            path = Path(value) if not isinstance(value, Path) else value
            if not path.is_file():
                raise ApiError(0, "", f"--{name}: file not found: {path}")
            files.append((name, (path.name, path.open("rb"), "application/octet-stream")))
        elif flag and flag.kind == "array" and flag.item_kind == "binary":
            for p in value:
                p = Path(p) if not isinstance(p, Path) else p
                if not p.is_file():
                    raise ApiError(0, "", f"--{name}: file not found: {p}")
                files.append((name, (p.name, p.open("rb"), "application/octet-stream")))
        elif flag and flag.kind in ("object", "array"):
            # Multipart form fields are scalar — JSON-encode complex values.
            import json as _json

            data[name] = _json.dumps(value)
        elif flag and flag.kind == "boolean":
            data[name] = "true" if value else "false"
        else:
            data[name] = str(value)
    return None, files, data


def _auth_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    # The server accepts two key types on different headers:
    #   - "comfyui-..." API keys → X-API-Key (validated by sha256 lookup)
    #   - Firebase ID tokens     → Authorization: Bearer (validated as a JWT)
    # See comfy-api server/middleware/authentication/comfy_firebase_auth.go.
    headers = {"User-Agent": "comfy-cli/api", "X-Comfy-Env": "comfy-cli"}
    if api_key.startswith("comfyui-"):
        headers["X-API-Key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def send_request(
    endpoint: spec.Endpoint,
    values: dict[str, Any],
    flags: list[FlagDef],
    api_key: str,
    timeout: float = 120.0,
) -> httpx.Response:
    """Send the initial request for `endpoint` with the given typed values."""
    url = spec.base_url() + endpoint.path
    json_body, files, data = _split_payload(values, flags, endpoint.request_content_type)
    headers = _auth_headers(api_key)
    try:
        if endpoint.method.lower() == "get":
            return httpx.get(url, params=values, headers=headers, timeout=timeout)
        if endpoint.request_content_type == "application/json":
            return httpx.post(url, json=json_body, headers=headers, timeout=timeout)
        return httpx.post(url, files=files, data=data, headers=headers, timeout=timeout)
    finally:
        # Ensure file handles from multipart are closed even on httpx errors.
        if files:
            for _name, payload in files:
                fileobj = payload[1]
                try:
                    fileobj.close()
                except Exception:  # noqa: BLE001
                    pass


def get(url: str, api_key: str, timeout: float = 60.0) -> httpx.Response:
    """GET helper for polling sibling endpoints and downloading result URLs."""
    if url.startswith("/"):
        url = spec.base_url() + url
    return httpx.get(url, headers=_auth_headers(api_key), timeout=timeout)


def download_bytes(url: str, timeout: float = 120.0) -> bytes:
    """Fetch result media. These URLs are usually pre-signed and not Comfy-hosted,
    so we don't send the Comfy bearer token."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        body = resp.json()
        import json as _json

        body_str = _json.dumps(body, indent=2)
    except Exception:  # noqa: BLE001
        body_str = resp.text
    raise ApiError(resp.status_code, body_str)
