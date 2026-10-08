from __future__ import annotations

import io
import urllib.error
import urllib.request
from http.client import HTTPMessage
from pathlib import Path

import pytest

from comfy_cli import deploy_download
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.output.renderer import Renderer

_ENDPOINT = "https://dep-id.run.comfy.app"
_STORAGE = frozenset({"https://storage.googleapis.com"})


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://api.runpod.ai/v2/xyz",
        "https://dep-id.attacker.example",
        "https://dep-id.attacker.run.comfy.app",
        "https://dep-id.run.comfy.app/jobs",
        "https://dep-id.run.comfy.app?next=evil",
    ],
)
def test_endpoint_validation_rejects_every_untrusted_shape(endpoint_url: str) -> None:
    # Given / When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.validate_endpoint_origin("dep-id", endpoint_url)

    # Then
    assert exc_info.value.code == "deploy_endpoint_unknown"


@pytest.mark.parametrize(
    ("endpoint_url", "expected"),
    [
        ("https://dep-id.run.comfy.app", "https://dep-id.run.comfy.app"),
        ("https://dep-id.stg.run.comfy.app/", "https://dep-id.stg.run.comfy.app"),
    ],
)
def test_endpoint_validation_accepts_both_builtin_gateway_suffixes(endpoint_url: str, expected: str) -> None:
    # Given / When / Then
    assert deploy_download.validate_endpoint_origin("dep-id", endpoint_url) == expected


def test_endpoint_validation_rejects_empty_suffix_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("COMFY_DEPLOY_HOST_SUFFIXES", "")

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.validate_endpoint_origin("dep-id", _ENDPOINT)

    # Then
    assert exc_info.value.code == "deploy_endpoint_unknown"
    assert "COMFY_DEPLOY_HOST_SUFFIXES" in str(exc_info.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.1/output",
        "https://[::1]/output",
        "https://169.254.169.254/latest/meta-data",
        "https://private.internal/output",
    ],
    ids=["ipv4-private", "ipv6-loopback", "metadata", "dns-private"],
)
def test_storage_allowlist_rejects_private_destinations_before_open(url: str) -> None:
    # Given / When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.validate_output_url(url, _ENDPOINT, _STORAGE)

    # Then
    assert exc_info.value.code == "deploy_endpoint_unknown"
    assert "COMFY_DEPLOY_STORAGE_ORIGINS" in str(exc_info.value)


def test_storage_origin_configuration_is_exact_and_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("COMFY_DEPLOY_STORAGE_ORIGINS", "")

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.configured_storage_origins()

    # Then
    assert exc_info.value.code == "deploy_endpoint_unknown"
    assert "COMFY_DEPLOY_STORAGE_ORIGINS" in str(exc_info.value)


def test_output_origin_outside_the_allowlist_is_rejected_before_streaming() -> None:
    # Given / When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.validate_output_url("https://cdn.attacker.example/out.png", _ENDPOINT, _STORAGE)

    # Then
    assert exc_info.value.code == "deploy_endpoint_unknown"
    assert exc_info.value.details["origin"] == "https://cdn.attacker.example"


def test_public_storage_to_private_redirect_is_rejected_before_the_second_request() -> None:
    # Given
    request = urllib.request.Request("https://storage.googleapis.com/bucket/signed")
    handler = deploy_download.DeploymentRedirectHandler(frozenset({_ENDPOINT, *_STORAGE}))

    # When
    with pytest.raises(urllib.error.URLError) as exc_info:
        handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            "https://10.0.0.1/stolen",
        )

    # Then
    assert "COMFY_DEPLOY_STORAGE_ORIGINS" in str(exc_info.value.reason)


def test_redirect_chain_never_restores_the_jwt_when_it_returns_to_the_endpoint() -> None:
    # Given
    initial = urllib.request.Request(f"{_ENDPOINT}/api/v2/assets/asset-1")
    initial.add_header("Authorization", "Bearer cloud-secret")
    handler = deploy_download.DeploymentRedirectHandler(frozenset({_ENDPOINT, *_STORAGE}))

    # When
    storage_request = handler.redirect_request(
        initial,
        io.BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        "https://storage.googleapis.com/bucket/signed",
    )
    assert storage_request is not None
    return_request = handler.redirect_request(
        storage_request,
        io.BytesIO(),
        302,
        "Found",
        HTTPMessage(),
        f"{_ENDPOINT}/api/v2/assets/asset-1",
    )

    # Then
    assert return_request is not None
    assert storage_request.get_header("Authorization") is None
    assert return_request.get_header("Authorization") is None


def _output(name: str, url: str, *, asset_id: str = "asset-1") -> JsonObject:
    return {
        "node_id": "9",
        "name": name,
        "type": "image",
        "id": asset_id,
        "url": url,
    }


def test_download_call_site_sends_jwt_only_to_the_endpoint_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    requests: list[tuple[str, dict[str, str]]] = []

    def record(url: str, _idx: int, path: Path, headers: dict[str, str], _renderer: Renderer, *, opener) -> None:
        requests.append((url, headers))
        path.write_bytes(b"output")

    monkeypatch.setattr("comfy_cli.deploy_download._stream_http_one", record)
    outputs = (
        _output("proxy.png", f"{_ENDPOINT}/api/v2/assets/proxy"),
        _output("signed.png", "https://storage.googleapis.com/bucket/signed", asset_id="asset-2"),
    )
    request = deploy_download.OutputDownloadRequest(outputs, _ENDPOINT, "cloud-secret", tmp_path)

    # When
    deploy_download.download_job_outputs(request, Renderer())

    # Then
    assert requests == [
        (f"{_ENDPOINT}/api/v2/assets/proxy", {"Authorization": "Bearer cloud-secret"}),
        ("https://storage.googleapis.com/bucket/signed", {}),
    ]


@pytest.mark.parametrize("name", ["../../etc/passwd", "/etc/passwd", "C:\\Windows\\win.ini", ""])
def test_output_name_is_validated_before_directory_components_are_stripped(name: str, tmp_path: Path) -> None:
    # Given / When
    with pytest.raises(DeployAPIError):
        deploy_download.safe_output_path(tmp_path, name)


@pytest.mark.parametrize("field", ["node_id", "type", "id"])
def test_a_malformed_output_is_rejected_before_any_bytes_reach_the_disk(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    streamed: list[str] = []

    def record(url: str, _idx: int, path: Path, _headers: dict[str, str], _renderer: Renderer, *, opener) -> None:
        streamed.append(url)
        path.write_bytes(b"output")

    monkeypatch.setattr("comfy_cli.deploy_download._stream_http_one", record)
    malformed = dict(_output("image.png", f"{_ENDPOINT}/api/v2/assets/proxy"))
    del malformed[field]
    output_dir = tmp_path / "outputs"
    request = deploy_download.OutputDownloadRequest((malformed,), _ENDPOINT, "cloud-secret", output_dir)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        deploy_download.download_job_outputs(request, Renderer())

    # Then
    assert exc_info.value.details["field"] == field
    assert streamed == []
    assert not output_dir.exists()


def test_output_collision_is_disambiguated_without_overwriting(tmp_path: Path) -> None:
    # Given
    existing = tmp_path / "image.png"
    existing.write_bytes(b"original")

    # When
    selected = deploy_download.safe_output_path(tmp_path, "nested/image.png")

    # Then
    assert selected == tmp_path.resolve() / "image.1.png"
    assert existing.read_bytes() == b"original"
