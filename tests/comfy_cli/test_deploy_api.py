from __future__ import annotations

import http.client
import inspect
import io
import json
import urllib.error
import urllib.parse
from typing import Any

import pytest
from deploy_api_cases import (
    BASE_URL as _BASE_URL,
)
from deploy_api_cases import (
    COMPUTE as _COMPUTE,
)
from deploy_api_cases import (
    MAX_JSON as _MAX_JSON,
)
from deploy_api_cases import (
    STATUS_CASES as _STATUS_CASES,
)
from deploy_api_cases import (
    WIRES as _WIRES,
)
from deploy_api_cases import (
    StatusCase,
    Wire,
)

from comfy_cli import deploy_api
from comfy_cli import http as http_mod
from comfy_cli.deploy_api import DeployAPIError, DeployAuthError, DeployClient
from comfy_cli.http import ResponseTooLarge


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        url,
        target,
        *,
        method="GET",
        body=None,
        timeout=30.0,
        max_bytes,
        headers=None,
    ):
        self.calls.append(
            {
                "url": url,
                "target": target,
                "method": method,
                "body": body,
                "max_bytes": max_bytes,
                "headers": headers,
            }
        )
        return 200, {
            "id": "dep-1",
            "status": "ready",
            "deployments": [],
            "hasMore": False,
            "total": 0,
            "events": [],
            "comfyuiLog": "",
            "regions": [],
        }


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", rec)
    return rec


@pytest.mark.parametrize("wire", _WIRES, ids=lambda wire: wire.operation)
def test_each_method_emits_the_openapi_request(recorder, wire: Wire):
    # Given
    client = DeployClient(_BASE_URL, "jwt-token")

    # When
    getattr(client, wire.method_name)(*wire.args, **wire.kwargs)

    # Then
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert (call["method"], call["url"], call["body"]) == (wire.http_method, wire.url, wire.body)
    assert call["headers"] == wire.headers
    assert call["max_bytes"] == wire.max_bytes
    assert call["target"] is client.target
    assert client.target.path_prefix == "/v1" and client.target.auth_token == "jwt-token"


def test_create_attaches_the_idempotency_key_through_the_real_authed_http_path(monkeypatch):
    # Given
    seen = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return b'{"id":"dep-1","status":"queued","statusUrl":"/v1/deployments/dep-1"}'

    def open_request(req, timeout=None):
        seen.append(req)
        return Response()

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", open_request)

    # When
    DeployClient(_BASE_URL, "jwt-token").create_deployment("v1", _COMPUTE, idempotency_key="idem-1")

    # Then
    assert seen[0].get_header("Idempotency-key") == "idem-1"
    assert seen[0].get_header("Authorization") == "Bearer jwt-token"


def _invoke(client: DeployClient, operation: str) -> Any:
    wire = next(row for row in _WIRES if row.operation == operation)
    return getattr(client, wire.method_name)(*wire.args, **wire.kwargs)


def test_base_url_resolution_prefers_explicit_then_environment_then_default(monkeypatch):
    # Given / When
    monkeypatch.setenv("COMFY_DEPLOY_URL", "https://env.test/deploy/")
    explicit = DeployClient("https://flag.test/deploy/", "token")
    environment = DeployClient(None, "token")
    monkeypatch.delenv("COMFY_DEPLOY_URL")
    default = DeployClient(None, "token")

    # Then
    assert explicit.target.base_url == "https://flag.test/deploy"
    assert environment.target.base_url == "https://env.test/deploy"
    assert default.target.base_url == deploy_api.DEFAULT_DEPLOY_URL


def test_from_session_refreshes_and_uses_the_resolved_url(monkeypatch):
    # Given
    seen: list[bool] = []
    session = type("Session", (), {"access_token": "fresh-jwt"})()
    monkeypatch.setattr(
        "comfy_cli.deploy_api.credentials.get_session", lambda *, refresh: seen.append(refresh) or session
    )

    # When
    client = DeployClient.from_session("https://flag.test/deploy")

    # Then
    assert seen == [True]
    assert client.target.auth_token == "fresh-jwt"


def test_from_session_without_a_jwt_raises_the_registered_auth_error(monkeypatch):
    # Given
    monkeypatch.setattr("comfy_cli.deploy_api.credentials.get_session", lambda *, refresh: None)

    # When / Then
    with pytest.raises(DeployAuthError) as exc_info:
        DeployClient.from_session()
    assert exc_info.value.code == "deploy_not_signed_in"


def test_plaintext_non_loopback_is_rejected_before_session_or_header_construction(monkeypatch):
    # Given
    session_calls: list[bool] = []
    header_calls: list[bool] = []
    monkeypatch.setattr(
        "comfy_cli.deploy_api.credentials.get_session", lambda *, refresh: session_calls.append(refresh)
    )
    monkeypatch.setattr("comfy_cli.http.target_auth_headers", lambda target: header_calls.append(True) or {})

    # When / Then
    with pytest.raises(ValueError, match="non-https"):
        DeployClient.from_session("http://example.com/deploy")
    assert session_calls == []
    assert header_calls == []


class _HTTPFailure:
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        self.calls = 0

    def __call__(self, url, target, **kwargs):
        self.calls += 1
        payload = io.BytesIO(json.dumps({"error": "SERVER_CODE", "message": self.message}).encode())
        raise urllib.error.HTTPError(url, self.status, self.message, http.client.HTTPMessage(), payload)


@pytest.mark.parametrize(
    "case",
    _STATUS_CASES,
    ids=lambda case: f"{case.operation}-{case.status}-{case.code}",
)
def test_every_operation_status_pair_has_the_required_mapping(monkeypatch, case: StatusCase):
    # Given
    failure = _HTTPFailure(case.status, case.message)
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", failure)
    client = DeployClient(_BASE_URL, "jwt-token")

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        _invoke(client, case.operation)

    # Then
    assert exc_info.value.code == case.code
    assert exc_info.value.status == case.status
    assert case.message in str(exc_info.value)
    assert failure.calls == 1


@pytest.mark.parametrize("operation", ["create", "scale", "start"])
def test_structural_400_is_bad_request_not_compute_unavailable(monkeypatch, operation: str):
    # Given
    failure = _HTTPFailure(400, "computeConfig.min must be >= 0")
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", failure)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        _invoke(DeployClient(_BASE_URL, "jwt-token"), operation)

    # Then
    assert exc_info.value.code == "deploy_bad_request"


@pytest.mark.parametrize("operation", ["scale", "start"])
def test_nonspecial_409_is_conflict_and_keeps_the_current_status(monkeypatch, operation: str):
    # Given
    failure = _HTTPFailure(409, "current status is stopping")
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", failure)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        _invoke(DeployClient(_BASE_URL, "jwt-token"), operation)

    # Then
    assert exc_info.value.code == "deploy_conflict"
    assert "stopping" in str(exc_info.value)


@pytest.mark.parametrize(
    ("compute_config", "idempotency_key"),
    [
        (
            {**_COMPUTE, "min": -1},
            "idem",
        ),
        (
            {**_COMPUTE, "max": 0},
            "idem",
        ),
        (
            {**_COMPUTE, "min": 2, "max": 1},
            "idem",
        ),
        (_COMPUTE, "x" * 256),
    ],
)
def test_create_validates_structural_400_causes_before_the_request(recorder, compute_config, idempotency_key):
    # Given
    client = DeployClient(_BASE_URL, "jwt-token")

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        client.create_deployment("v1", compute_config, idempotency_key=idempotency_key)

    # Then
    assert exc_info.value.code == "deploy_bad_request"
    assert recorder.calls == []


@pytest.mark.parametrize("wire", _WIRES, ids=lambda wire: wire.operation)
def test_a_redirect_from_every_control_plane_endpoint_is_not_followed(monkeypatch, wire: Wire):
    # Given
    opens: list[str] = []

    def redirect(req, timeout=None):
        opens.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 302, "redirect refused", http.client.HTTPMessage(), io.BytesIO())

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", redirect)
    client = DeployClient(_BASE_URL, "jwt-token")

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        getattr(client, wire.method_name)(*wire.args, **wire.kwargs)

    # Then
    assert exc_info.value.status == 302
    assert opens == [wire.url]


def test_an_oversized_control_plane_response_raises_response_too_large(monkeypatch):
    # Given
    body = b"x" * (_MAX_JSON + 1)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return body[:count]

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", lambda req, timeout=None: Response())

    # When / Then
    with pytest.raises(ResponseTooLarge):
        DeployClient(_BASE_URL, "jwt-token").get_deployment("dep-1")


def test_iter_deployments_is_lazy_and_walks_three_pages(monkeypatch):
    # Given
    calls: list[str] = []
    pages = {
        "": {"deployments": [{"id": "dep-1"}], "nextCursor": "c1", "hasMore": True, "total": 3},
        "c1": {"deployments": [{"id": "dep-2"}], "nextCursor": "c2", "hasMore": True, "total": 3},
        "c2": {"deployments": [{"id": "dep-3"}], "hasMore": False, "total": 3},
    }

    def page_request(url, target, **kwargs):
        after = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("after", [""])[0]
        calls.append(after)
        return 200, pages[after]

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", page_request)
    client = DeployClient(_BASE_URL, "jwt-token")

    # When
    iterator = client.iter_deployments(limit=1)
    assert calls == []
    yielded = list(iterator)

    # Then
    assert calls == ["", "c1", "c2"]
    assert [page["deployments"][0]["id"] for page in yielded] == ["dep-1", "dep-2", "dep-3"]


def test_list_all_deployments_collects_all_three_pages_on_top_of_the_iterator(monkeypatch):
    # Given
    pages = iter(
        [
            {"deployments": [{"id": "dep-1"}]},
            {"deployments": [{"id": "dep-2"}]},
            {"deployments": [{"id": "dep-3"}]},
        ]
    )
    client = DeployClient(_BASE_URL, "jwt-token")
    monkeypatch.setattr(client, "iter_deployments", lambda **kwargs: pages)

    # When
    deployments = client.list_all_deployments()

    # Then
    assert [row["id"] for row in deployments] == ["dep-1", "dep-2", "dep-3"]


def test_public_pagination_primitives_do_not_expose_the_raw_cursor():
    # Given / When
    signatures = [
        inspect.signature(DeployClient.list_deployments),
        inspect.signature(DeployClient.iter_deployments),
        inspect.signature(DeployClient.list_all_deployments),
    ]

    # Then
    assert all("after" not in signature.parameters for signature in signatures)
