from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, TypeAlias, assert_never, runtime_checkable

import pytest

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


def _request(path: Path, *, upload_enabled: bool = True) -> AssetResolveRequest:
    return AssetResolveRequest(local_path=path, file_path="inputs/input.png", upload_enabled=upload_enabled)


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
