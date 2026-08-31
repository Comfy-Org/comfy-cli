"""V2 data-plane asset resolution with hash-first deduplication."""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError, assert_safe_deploy_url
from comfy_cli.hashing import blake3_file
from comfy_cli.http import (
    ResponseTooLarge,
    no_redirect_urlopen,
    read_capped,
    request_json,
    target_auth_headers,
)
from comfy_cli.target import Target

_MAX_JSON: Final = 5 * 1024 * 1024
_UPLOAD_TIMEOUT: Final = 60.0
UPLOAD_CHUNK_BYTES: Final = 1024 * 1024

_ERROR_RULES: Final = {
    ("probe", 401): {"code": "deploy_not_signed_in", "message": "the asset probe is not authenticated"},
    ("probe", 500): {"code": "deploy_server_error", "message": "the asset probe failed"},
    ("from_hash", 401): {"code": "deploy_not_signed_in", "message": "the asset mint is not authenticated"},
    ("from_hash", 403): {"code": "deploy_forbidden", "message": "the asset mint is forbidden"},
    ("from_hash", 500): {"code": "deploy_server_error", "message": "the asset mint failed"},
    ("upload", 401): {"code": "deploy_not_signed_in", "message": "the asset upload is not authenticated"},
    ("upload", 403): {"code": "deploy_forbidden", "message": "the asset upload is forbidden"},
    ("upload", 409): {
        "code": "deploy_asset_upload_failed",
        "message": "the server-computed asset hash did not match the local hash",
    },
    ("upload", 422): {"code": "deploy_bad_request", "message": "the asset upload is invalid"},
    ("upload", 500): {"code": "deploy_server_error", "message": "the asset upload failed"},
}
_DEFAULT_ERROR: Final = {"code": "deploy_server_error", "message": "the asset request failed"}


@dataclass(frozen=True, slots=True)
class AssetResolveRequest:
    local_path: Path
    file_path: str
    upload_enabled: bool = True


@dataclass(frozen=True, slots=True)
class AssetResolveResult:
    asset: JsonObject
    uploaded: int
    deduped: int
    bytes: int


@dataclass(frozen=True, slots=True)
class _MultipartBody:
    """A body whose declared length and streamed bytes are one observation.

    ``file_size`` comes from the *open handle*, and the body streams from that
    same handle bounded to exactly that many bytes. The two used to be
    independent looks at the filesystem — a ``stat()`` when the request was
    built, an ``open()`` whenever urllib got round to consuming the body — so a
    file rewritten in between announced a ``Content-Length`` that did not
    describe what followed it.
    """

    file: BinaryIO
    prefix: bytes
    suffix: bytes
    file_size: int

    @property
    def content_length(self) -> int:
        return len(self.prefix) + self.file_size + len(self.suffix)

    def __iter__(self) -> Iterator[bytes]:
        yield self.prefix
        self.file.seek(0)
        remaining = self.file_size
        while remaining > 0:
            chunk = self.file.read(min(UPLOAD_CHUNK_BYTES, remaining))
            if not chunk:
                raise DeployAPIError(
                    code="deploy_asset_upload_failed",
                    message="the asset shrank while it was being uploaded",
                    details={"expected_bytes": self.file_size, "missing_bytes": remaining},
                )
            remaining -= len(chunk)
            yield chunk
        yield self.suffix


def _multipart_field(boundary: str, name: str, value: str) -> bytes:
    return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()


def _multipart_request(
    target: Target,
    request: AssetResolveRequest,
    expected_hash: str,
    file: BinaryIO,
) -> tuple[urllib.request.Request, int]:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(request.local_path.name)[0] or "application/octet-stream"
    fields = (
        ("content_type", content_type),
        ("file_path", request.file_path),
        ("expected_hash", expected_hash),
        ("tags", "input"),
    )
    safe_filename = request.local_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    prefix = b"".join(_multipart_field(boundary, name, value) for name, value in fields)
    prefix += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    file_size = os.fstat(file.fileno()).st_size
    body = _MultipartBody(file, prefix, suffix, file_size)
    upload = urllib.request.Request(target.url("assets"), data=body, method="POST")
    upload.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    upload.add_header("Content-Length", str(body.content_length))
    for name, value in target_auth_headers(target).items():
        upload.add_header(name, value)
    return upload, file_size


def _server_code(error: urllib.error.HTTPError, url: str) -> str | None:
    try:
        raw = read_capped(error, url, max_bytes=_MAX_JSON)
    except ResponseTooLarge:
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    payload = parsed.get("error")
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    return code if isinstance(code, str) else None


def _mapped_error(operation: str, error: urllib.error.HTTPError, url: str) -> DeployAPIError:
    status = error.code
    lookup_status = 500 if 500 <= status <= 599 else status
    rule = _ERROR_RULES.get((operation, lookup_status), _DEFAULT_ERROR)
    details: JsonObject = {"operation": operation, "http_status": status}
    server_code = _server_code(error, url)
    if server_code is not None:
        details["server_code"] = server_code
    return DeployAPIError(rule["code"], rule["message"], status=status, details=details)


def _invalid_response(operation: str, status: int) -> DeployAPIError:
    return DeployAPIError(
        code="deploy_server_error",
        message=f"the data plane returned an invalid asset {operation} response",
        status=status,
        details={"operation": operation, "http_status": status},
    )


class DeployAssetClient:
    def __init__(self, base_url: str, token: str) -> None:
        resolved_url = base_url.rstrip("/")
        assert_safe_deploy_url(resolved_url, source="the deployment endpointUrl")
        self.target = Target(kind="cloud", base_url=resolved_url, path_prefix="/api/v2", auth_token=token)

    def resolve_asset(self, request: AssetResolveRequest) -> AssetResolveResult:
        expected_hash = blake3_file(request.local_path)
        encoded_hash = urllib.parse.quote(expected_hash, safe="")
        probe_url = self.target.url("assets", "by-hash", encoded_hash)
        try:
            status, _ = request_json(
                probe_url,
                self.target,
                method="HEAD",
                max_bytes=_MAX_JSON,
            )
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return self._resolve_miss(request, expected_hash)
            raise _mapped_error("probe", error, probe_url) from error
        except (ResponseTooLarge, TimeoutError, urllib.error.URLError) as error:
            raise DeployAPIError(
                code="deploy_server_error",
                message="the asset probe failed",
                details={"operation": "probe"},
            ) from error
        if status != 200:
            raise _invalid_response("probe", status)

        mint_url = self.target.url("assets", "from-hash")
        body: JsonObject = {"hash": expected_hash, "file_path": request.file_path}
        try:
            status, parsed = request_json(
                mint_url,
                self.target,
                method="POST",
                body=body,
                max_bytes=_MAX_JSON,
            )
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return self._resolve_miss(request, expected_hash)
            raise _mapped_error("from_hash", error, mint_url) from error
        except (ResponseTooLarge, TimeoutError, urllib.error.URLError) as error:
            raise DeployAPIError(
                code="deploy_server_error",
                message="the asset mint failed",
                details={"operation": "from_hash"},
            ) from error
        if status in {200, 201} and isinstance(parsed, dict):
            return AssetResolveResult(asset=parsed, uploaded=0, deduped=1, bytes=0)
        raise _invalid_response("mint", status)

    def _resolve_miss(self, request: AssetResolveRequest, expected_hash: str) -> AssetResolveResult:
        if not request.upload_enabled:
            raise DeployAPIError(
                code="deploy_asset_missing",
                message="the asset is not already present and uploads are disabled",
                details={"file_path": request.file_path, "hash": expected_hash},
            )
        # The handle stays open across the urlopen because urllib consumes the
        # body iterator lazily, after the request is built.
        with request.local_path.open("rb") as file:
            upload, file_size = _multipart_request(self.target, request, expected_hash, file)
            try:
                with no_redirect_urlopen(upload, timeout=_UPLOAD_TIMEOUT) as response:
                    status = response.status
                    raw = read_capped(response, upload.full_url, max_bytes=_MAX_JSON)
            except urllib.error.HTTPError as error:
                raise _mapped_error("upload", error, upload.full_url) from error
            except (ResponseTooLarge, TimeoutError, urllib.error.URLError) as error:
                raise DeployAPIError(
                    code="deploy_asset_upload_failed",
                    message="the asset upload did not complete",
                    details={"file_path": request.file_path, "hash": expected_hash},
                ) from error
        try:
            parsed = json.loads(raw) if raw else None
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            parsed = None
        if status in {200, 201} and isinstance(parsed, dict):
            return AssetResolveResult(asset=parsed, uploaded=1, deduped=0, bytes=file_size)
        raise _invalid_response("upload", status)
