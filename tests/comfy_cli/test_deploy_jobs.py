from __future__ import annotations

import http.client
import io
import json
import urllib.error
from typing import Any

import pytest

from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.deploy_jobs import _MAX_SERVER_MESSAGE, DeployJobClient, JobSubmitRequest

_BASE_URL = "https://dep-1.run.comfy.app"
_WORKFLOW = {"1": {"class_type": "KSampler", "inputs": {}}}
_JOB = {"id": "job-1", "status": "queued"}
_KEY = "idem-1"


class _Transport:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = iter(outcomes)
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
                "headers": headers,
                "max_bytes": max_bytes,
            }
        )
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _ControlPlane:
    def __init__(self, status: str = "provisioning") -> None:
        self.status = status
        self.calls: list[str] = []

    def get_deployment(self, deployment_id: str, /) -> dict[str, Any]:
        self.calls.append(deployment_id)
        return {"id": deployment_id, "status": self.status}


def _error(status: int, code: str, *, message: str = "server prose", details: dict | None = None):
    payload = {"error": {"code": code, "message": message, "details": details}}
    return urllib.error.HTTPError(
        _BASE_URL,
        status,
        message,
        http.client.HTTPMessage(),
        io.BytesIO(json.dumps(payload).encode()),
    )


def _retryable_error(code: str, seconds: str = "2"):
    error = _error(429, code)
    error.headers["Retry-After"] = seconds
    return error


def _request(partner_credential: tuple[str, str] | None = None) -> JobSubmitRequest:
    return JobSubmitRequest(
        workflow=_WORKFLOW,
        idempotency_key=_KEY,
        deployment_id="dep-1",
        partner_credential=partner_credential,
    )


def test_success_issues_exactly_one_post_without_empty_extra_data(monkeypatch):
    # Given
    transport = _Transport((201, _JOB))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)
    client = DeployJobClient(_BASE_URL, "jwt-token")

    # When
    job = client.submit_job(_request(), _ControlPlane())

    # Then
    assert job["id"] == "job-1"
    assert len(transport.calls) == 1
    assert transport.calls[0]["method"] == "POST"
    assert transport.calls[0]["url"] == f"{_BASE_URL}/api/v2/jobs"
    assert transport.calls[0]["body"] == {"workflow": _WORKFLOW}
    assert transport.calls[0]["headers"] == {"Idempotency-Key": _KEY}
    assert transport.calls[0]["target"].auth_token == "jwt-token"


@pytest.mark.parametrize(
    "field",
    ["api_key_comfy_org", "auth_token_comfy_org"],
)
def test_the_resolved_credential_is_forwarded_under_its_own_field(monkeypatch, field):
    """An OAuth session and an API key are different credentials and ride different keys."""
    # Given
    transport = _Transport((201, _JOB))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request((field, "comfy-secret")), _ControlPlane())

    # Then
    assert transport.calls[0]["body"] == {
        "workflow": _WORKFLOW,
        "extra_data": {field: "comfy-secret"},
    }


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timed out"), urllib.error.URLError("connection refused")],
    ids=["timeout", "connection-error"],
)
def test_unknown_transport_outcome_is_never_retried(monkeypatch, failure):
    # Given
    transport = _Transport(failure)
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == "deploy_job_submit_unknown"
    assert len(transport.calls) == 1
    message = str(exc_info.value).lower()
    assert "job may exist" in message and "no way to find" in message
    assert "poll your" not in message and "list your" not in message


def test_5xx_is_the_same_single_attempt_unknown_outcome(monkeypatch):
    # Given
    transport = _Transport(_error(503, "upstream_error"))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == "deploy_job_submit_unknown"
    assert len(transport.calls) == 1


def test_deployment_not_ready_retries_same_key_then_refreshes_status(monkeypatch):
    # Given
    transport = _Transport(*[_retryable_error("deployment_not_ready") for _ in range(3)])
    sleeps: list[float] = []
    control = _ControlPlane(status="failed")
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)
    client = DeployJobClient(_BASE_URL, "jwt-token", sleep=sleeps.append)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        client.submit_job(_request(), control)

    # Then
    assert exc_info.value.code == "deploy_not_ready"
    assert exc_info.value.details["status"] == "failed"
    assert control.calls == ["dep-1"]
    assert len(transport.calls) == 3 and sleeps == [2.0, 2.0]
    assert {call["headers"]["Idempotency-Key"] for call in transport.calls} == {_KEY}
    assert str(exc_info.value) == "server prose"


def test_queue_full_retries_same_key_then_maps_rate_limit(monkeypatch):
    # Given
    transport = _Transport(*[_retryable_error("queue_full", "3") for _ in range(3)])
    sleeps: list[float] = []
    control = _ControlPlane()
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token", sleep=sleeps.append).submit_job(_request(), control)

    # Then
    assert exc_info.value.code == "deploy_rate_limited"
    assert len(transport.calls) == 3 and sleeps == [3.0, 3.0]
    assert control.calls == []
    assert {call["headers"]["Idempotency-Key"] for call in transport.calls} == {_KEY}


def test_an_extravagant_retry_after_is_clamped_rather_than_obeyed(monkeypatch):
    """A `Retry-After` is a hint, not custody of the process.

    Obeyed literally, `86400` parks a foreground `submit_job` for a day with no
    output. `deploy_events` already clamps its own; this is the same ceiling.
    """
    # Given a server asking for a day of backoff
    transport = _Transport(*[_retryable_error("queue_full", "86400") for _ in range(3)])
    sleeps: list[float] = []
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError):
        DeployJobClient(_BASE_URL, "jwt-token", sleep=sleeps.append).submit_job(_request(), _ControlPlane())

    # Then it still retries, but bounded
    assert sleeps == [10.0, 10.0]


@pytest.mark.parametrize(
    ("server_code", "client_code"),
    [("deployment_not_ready", "deploy_not_ready"), ("queue_full", "deploy_rate_limited")],
)
def test_429_without_retry_after_is_terminal(monkeypatch, server_code, client_code):
    # Given
    transport = _Transport(_error(429, server_code))
    control = _ControlPlane()
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), control)

    # Then
    assert exc_info.value.code == client_code
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("server_code", "client_code"),
    [
        ("deployment_stopped", "deploy_conflict"),
        ("idempotency_key_reuse", "deploy_idempotency_reuse"),
        # Without its own row this fell through to the generic 422 and told the
        # reader the workflow was invalid, when the workflow is fine and an
        # input it names is not reachable.
        ("missing_asset", "deploy_asset_missing"),
    ],
)
def test_terminal_422_is_never_retried(monkeypatch, server_code, client_code):
    # Given
    transport = _Transport(_error(422, server_code))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == client_code
    assert len(transport.calls) == 1


def test_invalid_workflow_carries_node_errors_through(monkeypatch):
    # Given
    node_errors = {"12": [{"field": "model", "reason": "missing_input"}]}
    transport = _Transport(_error(422, "invalid_workflow", details={"node_errors": node_errors}))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == "deploy_workflow_invalid"
    assert exc_info.value.details["node_errors"] == node_errors
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("status", "server_code", "client_code"),
    [
        (402, "insufficient_credits", "deploy_payment_required"),
        (403, "forbidden", "deploy_forbidden"),
        (404, "not_found", "deploy_not_found"),
    ],
)
def test_other_4xx_rows_are_terminal(monkeypatch, status, server_code, client_code):
    # Given
    transport = _Transport(_error(status, server_code))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == client_code
    assert len(transport.calls) == 1


def test_configured_api_key_never_appears_in_an_error(monkeypatch):
    # Given
    secret = "comfy-secret-never-leak"
    transport = _Transport(_error(403, "forbidden", message=secret, details={"echo": secret}))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(("api_key_comfy_org", secret)), _ControlPlane())

    # Then
    assert secret not in str(exc_info.value)
    assert secret not in json.dumps(exc_info.value.details)


def _json_body(secret: str) -> str:
    """A well-formed body echoing the secret, escaped as `json.dumps` sees fit."""
    return json.dumps({"error": {"code": "invalid_workflow", "message": f"rejected {secret}", "details": {}}})


def _unicode_escaped_body(secret: str) -> str:
    """The same echo, with the secret written as `\\u` escapes."""
    escaped = "".join(f"\\u{ord(character):04x}" for character in secret)
    return '{"error":{"code":"invalid_workflow","message":"rejected ' + escaped + '","details":{}}}'


@pytest.mark.parametrize(
    ("secret", "build_body"),
    [
        pytest.param("comfyui-supersecret", _json_body, id="plain-echo"),
        pytest.param('a"b-secret', _json_body, id="secret-containing-a-quote"),
        pytest.param("back\\slash-secret", _json_body, id="secret-containing-a-backslash"),
        pytest.param("comfyui-supersecret", _unicode_escaped_body, id="server-echoes-it-unicode-escaped"),
    ],
)
def test_a_credential_is_redacted_however_the_server_encoded_it(monkeypatch, secret, build_body):
    """Given a server that echoes the credential, When it is parsed, Then it is redacted.

    Redacting the raw document text only catches an echo whose bytes match the
    credential literally. A `\\u`-escaped one, or a secret carrying a quote or a
    backslash, is restored intact by `json.loads` and would otherwise reach the
    rendered envelope.
    """
    # Given
    error = urllib.error.HTTPError(f"{_BASE_URL}/jobs", 422, "err", {}, io.BytesIO(build_body(secret).encode()))
    transport = _Transport(error)
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(("api_key_comfy_org", secret)), _ControlPlane())

    # Then
    assert secret not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)


def test_plaintext_non_loopback_endpoint_is_rejected_before_transport(monkeypatch):
    # Given
    transport = _Transport((201, _JOB))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When / Then
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient("http://attacker.example", "jwt-token")
    assert exc_info.value.code == "deploy_insecure_url"
    assert "non-https" in str(exc_info.value)
    assert transport.calls == []


def test_server_explanation_replaces_the_canned_message(monkeypatch):
    # Given
    explanation = (
        'the asset reference on node "1" is not on a file input this deployment stages; '
        "set it on a supported loader node's file field"
    )
    transport = _Transport(_error(422, "invalid_workflow", message=explanation))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.code == "deploy_workflow_invalid"
    assert str(exc_info.value) == explanation
    assert exc_info.value.details["server_code"] == "invalid_workflow"


@pytest.mark.parametrize("message", ["", "   ", None, 42])
def test_canned_message_survives_a_server_that_explains_nothing(monkeypatch, message):
    # Given
    transport = _Transport(_error(422, "invalid_workflow", message=message))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert str(exc_info.value) == "the workflow is invalid"


def test_an_unbounded_server_message_is_truncated(monkeypatch):
    # Given
    transport = _Transport(_error(422, "invalid_workflow", message="x" * 10_000))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    rendered = str(exc_info.value)
    assert len(rendered) == _MAX_SERVER_MESSAGE + 1 and rendered.endswith("…")


def test_node_errors_survive_a_server_code_the_client_does_not_enumerate(monkeypatch):
    # Given
    node_errors = {"3": [{"field": "image", "reason": "missing_input"}]}
    transport = _Transport(_error(422, "some_future_code", details={"node_errors": node_errors}))
    monkeypatch.setattr("comfy_cli.deploy_jobs.request_json", transport)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        DeployJobClient(_BASE_URL, "jwt-token").submit_job(_request(), _ControlPlane())

    # Then
    assert exc_info.value.details["node_errors"] == node_errors
