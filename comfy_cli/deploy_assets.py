"""V2 data-plane asset resolution with hash-first deduplication."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Final

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.credentials import DataPlaneToken
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
    """One local file to turn into a data-plane asset.

    ``allowed_hashes`` is ``--allow-upload``: when set, a file the deployment
    does not already hold is uploaded only if its hash is listed. A person
    approves the exact files a ``--no-upload`` refusal listed, by hash, so a
    same-named file from another root, or a file edited since, is refused
    rather than sent. An approved upload reads one private snapshot, checked
    against the approved hash, so a file rewritten during the probe or the
    upload cannot send bytes nobody approved.
    """

    local_path: Path
    file_path: str
    upload_enabled: bool = True
    allowed_hashes: frozenset[str] | None = None


class DeployAssetMissingError(DeployAPIError):
    """A file the deployment lacks, which this run may not upload.

    ``details`` names it in full (the absolute ``local_path``, its ``bytes``
    and ``hash``) so a refusal can serve as the exact upload plan to approve.
    """

    def __init__(self, request: AssetResolveRequest, expected_hash: str, size: int) -> None:
        # ``local_path`` is always the file the scanner picked, never a snapshot.
        reason = "uploads are disabled" if not request.upload_enabled else "its hash is not one --allow-upload names"
        super().__init__(
            code="deploy_asset_missing",
            message=f"the asset is not already present and {reason}",
            details={
                "file_path": request.file_path,
                "hash": expected_hash,
                "local_path": str(request.local_path),
                "bytes": size,
            },
        )


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
    def __init__(self, base_url: str, token: str | DataPlaneToken) -> None:
        resolved_url = base_url.rstrip("/")
        assert_safe_deploy_url(resolved_url, source="the deployment endpointUrl")
        self._token = token
        self._target = Target(kind="cloud", base_url=resolved_url, path_prefix="/api/v2")

    @property
    def target(self) -> Target:
        """The endpoint with the bearer as of now: hashing and uploading many
        inputs can outlast the sign-in token the run started with."""
        token = self._token if isinstance(self._token, str) else self._token.current()
        return replace(self._target, auth_token=token)

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
        approved = request.allowed_hashes is None or expected_hash in request.allowed_hashes
        if not request.upload_enabled or not approved:
            raise DeployAssetMissingError(request, expected_hash, request.local_path.stat().st_size)
        if request.allowed_hashes is None:
            return self._upload(request, request.local_path, expected_hash)
        # An approval binds bytes, not a path: the path can be rewritten after
        # the hash above. Copy it once, check the copy, and send only the copy.
        # The copy is made only here, on a miss, so a file Cloud already holds
        # never needs temporary disk space.
        with _snapshot(request) as snapshot:
            snapshot_hash = blake3_file(snapshot)
            if snapshot_hash != expected_hash:
                raise DeployAssetMissingError(request, snapshot_hash, snapshot.stat().st_size)
            return self._upload(request, snapshot, snapshot_hash)

    def _upload(self, request: AssetResolveRequest, source: Path, expected_hash: str) -> AssetResolveResult:
        # The handle stays open across the urlopen because urllib consumes the
        # body iterator lazily, after the request is built.
        with source.open("rb") as file:
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


_SNAPSHOT_PREFIX = "comfy-cli-asset-"


def _snapshot_error(request: AssetResolveRequest, error: OSError) -> DeployAPIError:
    """The copy could not be made (a full disk, say): a local failure, not the server's."""
    reason = error.strerror or str(error)
    # Not ``tempfile.gettempdir()``: when no temp folder is usable, asking again
    # raises the same error and this report never reaches the person.
    copy = str(error.filename) if error.filename else ""
    temp_dir = os.path.dirname(copy) if os.path.basename(copy).startswith(_SNAPSHOT_PREFIX) else tempfile.tempdir
    return DeployAPIError(
        code="deploy_asset_upload_failed",
        message=f"could not copy {request.local_path} to a private temporary file to upload it: {reason}",
        details={
            "file_path": request.file_path,
            "local_path": str(request.local_path),
            "temp_dir": temp_dir,
            "reason": reason,
        },
        hint="nothing was uploaded; free space in `details.temp_dir` or set TMPDIR to a folder with room, then retry",
    )


@contextmanager
def _snapshot(request: AssetResolveRequest) -> Iterator[Path]:
    """A private copy of the request's file in the system temp folder, removed on every exit.

    ``mkstemp`` creates it readable by this user only. The ``comfy-cli-`` prefix
    names the tool that left it, should a crash skip the ``finally``. A failure
    to make the copy is reported as such, naming the file; an error raised while
    the copy is in use passes through unchanged.
    """
    try:
        descriptor, name = tempfile.mkstemp(prefix=_SNAPSHOT_PREFIX)
    except OSError as error:
        raise _snapshot_error(request, error) from error
    snapshot = Path(name)
    try:
        try:
            with os.fdopen(descriptor, "wb") as copy, request.local_path.open("rb") as original:
                shutil.copyfileobj(original, copy, UPLOAD_CHUNK_BYTES)
        except OSError as error:
            raise _snapshot_error(request, error) from error
        yield snapshot
    finally:
        with suppress(OSError):
            snapshot.unlink(missing_ok=True)
