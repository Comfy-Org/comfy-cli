from __future__ import annotations

import errno
import http.client
import io
import json
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

import pytest
from typing_extensions import assert_never

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.deploy_assets import (
    UPLOAD_CHUNK_BYTES,
    AssetResolveRequest,
    DeployAssetClient,
)
from comfy_cli.target import Target

_BASE_URL = "https://dep-1.run.comfy.app"
_HASH = "blake3:6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
_ASSET: JsonObject = {
    "id": "asset-1",
    "hash": _HASH,
    "size_bytes": 3,
    "content_type": "image/png",
    "file_path": "inputs/input.png",
    "created_new": False,
    "created_at": "2026-08-23T00:00:00Z",
    "url": "https://storage.example/asset-1",
    "url_expires_at": "2026-08-23T00:05:00Z",
    "expires_at": None,
    "job_id": None,
}

_JsonOutcome: TypeAlias = tuple[int, JsonObject | None] | BaseException


@runtime_checkable
class _ChunkBody(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...


class _JsonTransport:
    def __init__(self, events: list[tuple[str, str]], *outcomes: _JsonOutcome) -> None:
        self.events = events
        self.outcomes = iter(outcomes)
        self.calls: list[tuple[str, str, JsonObject | None, Target]] = []

    def __call__(
        self,
        url: str,
        target: Target,
        *,
        method: str = "GET",
        body: JsonObject | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_bytes: int,
    ) -> tuple[int, JsonObject | None]:
        del headers, timeout, max_bytes
        self.events.append((method, url))
        self.calls.append((method, url, body, target))
        outcome = next(self.outcomes)
        match outcome:
            case BaseException():
                raise outcome
            case (status, parsed):
                return status, parsed
            case unreachable:
                assert_never(unreachable)


class _Response:
    def __init__(self, status: int, payload: JsonObject) -> None:
        self.status = status
        self._body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)


class _MultipartTransport:
    def __init__(self, events: list[tuple[str, str]], *, status: int = 201) -> None:
        self.events = events
        self.status = status
        self.requests: list[urllib.request.Request] = []
        self.chunks: list[bytes] = []

    def __call__(self, request: urllib.request.Request, *, timeout: float) -> _Response:
        del timeout
        self.events.append((request.get_method(), request.full_url))
        self.requests.append(request)
        body = request.data
        assert isinstance(body, _ChunkBody)
        self.chunks = list(body)
        return _Response(self.status, _ASSET)


def _http_error(status: int, code: str) -> urllib.error.HTTPError:
    payload = {"error": {"code": code, "message": "server prose"}}
    return urllib.error.HTTPError(
        _BASE_URL,
        status,
        "server prose",
        http.client.HTTPMessage(),
        io.BytesIO(json.dumps(payload).encode()),
    )


def _request(
    path: Path, *, upload_enabled: bool = True, allowed_hashes: frozenset[str] | None = None
) -> AssetResolveRequest:
    return AssetResolveRequest(
        local_path=path, file_path="inputs/input.png", upload_enabled=upload_enabled, allowed_hashes=allowed_hashes
    )


def test_dedupe_hit_issues_head_then_from_hash_and_constructs_no_multipart(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    transport = _JsonTransport(events, (200, None), (201, _ASSET))
    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", transport)

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("dedupe hit constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    result = DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path))

    # Then
    encoded_hash = _HASH.replace(":", "%3A")
    assert events == [
        ("HEAD", f"{_BASE_URL}/api/v2/assets/by-hash/{encoded_hash}"),
        ("POST", f"{_BASE_URL}/api/v2/assets/from-hash"),
    ]
    assert transport.calls[1][2] == {"hash": _HASH, "file_path": "inputs/input.png"}
    assert transport.calls[0][3].auth_token == "jwt-token"
    assert result.asset["id"] == "asset-1"
    assert (result.uploaded, result.deduped, result.bytes) == (0, 1, 0)


def test_dedupe_miss_streams_one_multipart_with_expected_hash(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    transport = _JsonTransport(events, _http_error(404, "blob_not_found"))
    multipart = _MultipartTransport(events)
    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", transport)
    monkeypatch.setattr("comfy_cli.deploy_assets.no_redirect_urlopen", multipart)

    # When
    result = DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path))

    # Then
    assert events == [
        ("HEAD", f"{_BASE_URL}/api/v2/assets/by-hash/{_HASH.replace(':', '%3A')}"),
        ("POST", f"{_BASE_URL}/api/v2/assets"),
    ]
    request = multipart.requests[0]
    body = b"".join(multipart.chunks)
    assert b'name="expected_hash"\r\n\r\n' + _HASH.encode() in body
    assert b'name="tags"\r\n\r\ninput\r\n' in body
    assert b'name="content_type"\r\n\r\nimage/png\r\n' in body
    assert request.get_header("Authorization") == "Bearer jwt-token"
    content_length = request.get_header("Content-length")
    assert content_length is not None
    assert int(content_length) == sum(map(len, multipart.chunks))
    assert (result.uploaded, result.deduped, result.bytes) == (1, 0, 3)


def test_multipart_file_bytes_are_yielded_in_bounded_chunks(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "large.bin"
    path.write_bytes(b"z" * (UPLOAD_CHUNK_BYTES * 2 + 17))
    events: list[tuple[str, str]] = []
    multipart = _MultipartTransport(events)
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )
    monkeypatch.setattr("comfy_cli.deploy_assets.no_redirect_urlopen", multipart)

    # When
    DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path))

    # Then
    file_chunks = multipart.chunks[1:-1]
    assert [len(chunk) for chunk in file_chunks] == [UPLOAD_CHUNK_BYTES, UPLOAD_CHUNK_BYTES, 17]
    assert max(map(len, multipart.chunks)) <= UPLOAD_CHUNK_BYTES


def test_hash_mismatch_maps_to_asset_upload_failed(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.no_redirect_urlopen",
        lambda request, *, timeout: (_ for _ in ()).throw(_http_error(409, "hash_mismatch")),
    )

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path))

    # Then
    assert exc_info.value.code == "deploy_asset_upload_failed"
    assert exc_info.value.details["server_code"] == "hash_mismatch"


def test_no_upload_miss_maps_to_asset_missing_without_multipart(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("--no-upload constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, upload_enabled=False))

    # Then
    assert exc_info.value.code == "deploy_asset_missing"
    assert events == [("HEAD", f"{_BASE_URL}/api/v2/assets/by-hash/{_HASH.replace(':', '%3A')}")]


def test_no_upload_miss_names_the_full_local_path_and_size(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport([], _http_error(404, "blob_not_found")),
    )

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, upload_enabled=False))

    # Then
    assert exc_info.value.details == {
        "file_path": "inputs/input.png",
        "hash": _HASH,
        "local_path": str(path),
        "bytes": 3,
    }


def test_a_miss_whose_hash_was_not_allowed_is_refused_without_multipart(tmp_path, monkeypatch):
    # Given a person approved other bytes than these
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("an unapproved file constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(
            _request(path, allowed_hashes=frozenset({"blake3:" + "0" * 64}))
        )

    # Then
    assert exc_info.value.code == "deploy_asset_missing"
    assert exc_info.value.details["local_path"] == str(path)
    assert exc_info.value.details["hash"] == _HASH
    assert events == [("HEAD", f"{_BASE_URL}/api/v2/assets/by-hash/{_HASH.replace(':', '%3A')}")]


def test_a_miss_whose_hash_was_allowed_still_uploads(tmp_path, monkeypatch):
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )
    monkeypatch.setattr("comfy_cli.deploy_assets.no_redirect_urlopen", _MultipartTransport(events))

    # When
    result = DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    assert events[-1] == ("POST", f"{_BASE_URL}/api/v2/assets")
    assert (result.uploaded, result.deduped, result.bytes) == (1, 0, 3)


def test_an_approved_file_rewritten_during_the_probe_is_refused_with_the_new_hash(tmp_path, monkeypatch):
    """Given an approved file, When it is rewritten while the hash probe is out,
    Then the snapshot taken on the miss holds the new bytes, its hash is not the
    approved one, nothing is sent, and the snapshot is gone."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))

    def probe_then_rewrite(url: str, *_args, **_kwargs):
        path.write_bytes(b"replaced while the probe was out")
        raise _http_error(404, "blob_not_found")

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("rewritten bytes constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", probe_then_rewrite)
    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    assert exc_info.value.code == "deploy_asset_missing"
    assert exc_info.value.details["hash"] != _HASH
    assert exc_info.value.details["bytes"] == len(b"replaced while the probe was out")
    assert list(private.iterdir()) == []


def test_an_approved_file_rewritten_during_the_upload_sends_only_the_approved_bytes(tmp_path, monkeypatch):
    """Given an approved file, When it is rewritten after the snapshot is taken,
    Then the upload streams the approved bytes from the snapshot, never the new
    ones, and the snapshot is gone afterwards."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    events: list[tuple[str, str]] = []
    multipart = _MultipartTransport(events)

    def rewrite_then_send(request: urllib.request.Request, *, timeout: float) -> _Response:
        path.write_bytes(b"replaced before the body was read")
        return multipart(request, timeout=timeout)

    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport(events, _http_error(404, "blob_not_found")),
    )
    monkeypatch.setattr("comfy_cli.deploy_assets.no_redirect_urlopen", rewrite_then_send)

    # When
    result = DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    body = b"".join(multipart.chunks)
    assert b"replaced before the body was read" not in body
    assert b"\r\n\r\nabc\r\n--" in body
    assert (result.uploaded, result.bytes) == (1, 3)
    assert list(private.iterdir()) == []


def test_an_asset_cloud_already_holds_makes_no_temporary_copy(tmp_path, monkeypatch):
    """Given `--allow-upload`, When Cloud already holds the bytes, Then the asset
    is minted from its hash and no temporary copy is made, so a large file needs
    no temporary disk space."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", _JsonTransport([], (200, None), (201, _ASSET)))

    def mkstemp_tripwire(*_args, **_kwargs):
        raise AssertionError("an asset Cloud already holds made a temporary copy")

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp_tripwire)

    # When
    result = DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    assert (result.uploaded, result.deduped) == (0, 1)


def test_a_snapshot_that_cannot_be_written_is_a_local_error_naming_the_file(tmp_path, monkeypatch):
    """Given an approved upload, When the temporary copy cannot be written (a
    full disk), Then the error names the file and the temp folder, is not a
    server error, nothing is sent, and the partial copy is gone."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport([], _http_error(404, "blob_not_found")),
    )

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("a failed snapshot constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets.shutil.copyfileobj", disk_full)
    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    error = exc_info.value
    assert error.code == "deploy_asset_upload_failed"
    assert str(path) in str(error)
    assert error.details["local_path"] == str(path)
    assert error.details["temp_dir"] == str(private)
    assert error.details["reason"] == "No space left on device"
    assert list(private.iterdir()) == []


def test_no_usable_temp_folder_is_still_reported_as_a_local_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Given an approved upload Cloud lacks and no temp folder Python can use (a
    full disk before one was ever chosen), When it is resolved, Then the error is
    the local copy failure naming the file, not an OSError the run reports as a
    server error."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport([], _http_error(404, "blob_not_found")),
    )

    def no_temp_folder(*_args, **_kwargs):
        raise FileNotFoundError(errno.ENOENT, "No usable temporary directory found")

    monkeypatch.setattr("comfy_cli.deploy_assets.tempfile.mkstemp", no_temp_folder)
    monkeypatch.setattr("comfy_cli.deploy_assets.tempfile.gettempdir", no_temp_folder)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    error = exc_info.value
    assert error.code == "deploy_asset_upload_failed"
    assert error.details["local_path"] == str(path)
    assert error.details["temp_dir"] is None


def test_an_approved_path_now_holding_other_bytes_is_refused_with_their_hash(tmp_path, monkeypatch):
    """Given a hash approved for `abc`, When the file now holds other bytes,
    Then nothing is sent, the refusal names the new hash so the person can be
    asked again, and the snapshot is gone."""
    # Given
    path = tmp_path / "input.png"
    path.write_bytes(b"other bytes")
    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    monkeypatch.setattr(
        "comfy_cli.deploy_assets.request_json",
        _JsonTransport([], _http_error(404, "blob_not_found")),
    )

    def multipart_tripwire(*_args, **_kwargs):
        raise AssertionError("a file holding unapproved bytes constructed a multipart request")

    monkeypatch.setattr("comfy_cli.deploy_assets._multipart_request", multipart_tripwire)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployAssetClient(_BASE_URL, "jwt-token").resolve_asset(_request(path, allowed_hashes=frozenset({_HASH})))

    # Then
    assert exc_info.value.code == "deploy_asset_missing"
    assert exc_info.value.details["local_path"] == str(path)
    assert exc_info.value.details["bytes"] == len(b"other bytes")
    assert exc_info.value.details["hash"] != _HASH
    assert list(private.iterdir()) == []
