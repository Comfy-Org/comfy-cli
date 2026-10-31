"""Unit tests for the shared cloud error → envelope mapper."""

from __future__ import annotations

import io
import urllib.error

import pytest
import typer

from comfy_cli.command._cloud_errors import _MAX_ERROR_BODY_BYTES, handle_cloud_http_error


class _FakeRenderer:
    """Captures the single ``renderer.error`` call the mapper is expected to make."""

    def __init__(self):
        self.calls: list[dict] = []

    def error(self, **kwargs):
        self.calls.append(kwargs)


def _http_error(code: int, body: bytes = b'{"error":"boom"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://cloud.example.com/x", code, "err", {}, io.BytesIO(body))


def _handle(renderer, e: Exception) -> typer.Exit:
    return handle_cloud_http_error(
        renderer,
        e,
        operation="cancel",
        not_found_code="prompt_not_found",
        not_found_message="no cloud job with id 'p1'",
        not_found_hint="check `comfy jobs ls --where cloud`",
        id_label="prompt_id",
        resource_id="p1",
    )


@pytest.mark.parametrize("code", [401, 403, 500])
def test_error_body_read_is_capped(code: int):
    """``base_url`` is env-configurable, so a hostile endpoint returning a huge
    error body must not be buffered unbounded into memory."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(code, b"A" * (5 * 1024 * 1024)))

    body = renderer.calls[0]["details"]["body"]
    assert len(body) <= _MAX_ERROR_BODY_BYTES
    assert body == "A" * _MAX_ERROR_BODY_BYTES


class _ExplodingBody(io.BytesIO):
    def read(self, *args):
        raise ConnectionResetError("stream reset mid-read")


@pytest.mark.parametrize("code", [401, 403, 500])
def test_body_read_failure_still_emits_envelope(code: int):
    """A reset/truncated body stream must degrade to an empty body, not escape
    as an unhandled traceback in place of the structured envelope."""
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", code, "err", {}, _ExplodingBody(b"ignored"))

    exit_exc = _handle(renderer, e)

    assert isinstance(exit_exc, typer.Exit)
    assert len(renderer.calls) == 1
    assert renderer.calls[0]["details"]["body"] == ""


def test_404_does_not_consume_body():
    """The 404 envelope discards the body, so it should not read the stream."""
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", 404, "err", {}, _ExplodingBody(b"ignored"))

    _handle(renderer, e)

    call = renderer.calls[0]
    assert call["code"] == "prompt_not_found"
    assert "body" not in call["details"]


@pytest.mark.parametrize("code", [401, 403])
def test_unauthorized_details_carry_body_and_resource_id(code: int):
    """A 403 for a non-auth reason (forbidden resource, quota, finished job)
    must not lose the server's explanation or the id being operated on."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(code, b'{"error":"quota exceeded"}'))

    call = renderer.calls[0]
    assert call["code"] == "cloud_unauthorized"
    assert call["details"]["status"] == code
    assert "quota exceeded" in call["details"]["body"]
    assert call["details"]["prompt_id"] == "p1"
    assert call["details"]["operation"] == "cancel"


def test_401_hint_points_at_relogin():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(401))
    assert renderer.calls[0]["hint"] == "re-run `comfy cloud login`"


def test_403_hint_is_softened():
    """403 is not unambiguously an auth failure, so the hint must not assert
    re-login as the remediation the way 401's does."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403))

    hint = renderer.calls[0]["hint"]
    assert "details.body" in hint
    assert hint != "re-run `comfy cloud login`"


def test_url_error_surfaces_network_hint():
    renderer = _FakeRenderer()
    _handle(renderer, urllib.error.URLError("connection refused"))

    call = renderer.calls[0]
    assert call["code"] == "cloud_http_error"
    assert "comfy cloud whoami" in call["hint"]


# --- 429: throttled, not rejected --------------------------------------------
#
# `comfy run` used to report a 429 as the generic `cloud_http_error` with a
# hint to "check the workflow is valid" — sending the agent off to "fix" a valid
# workflow. A 429 gets its own code and a retry hint, and carries the server's
# Retry-After when it sent one.


def _http_error_with_headers(code: int, headers: dict, body: bytes = b'{"error":"slow down"}'):
    return urllib.error.HTTPError("https://cloud.example.com/x", code, "Too Many Requests", headers, io.BytesIO(body))


def test_429_is_cloud_rate_limited_with_retry_after():
    renderer = _FakeRenderer()
    exit_exc = _handle(renderer, _http_error_with_headers(429, {"Retry-After": "30"}))

    assert isinstance(exit_exc, typer.Exit)
    call = renderer.calls[0]
    assert call["code"] == "cloud_rate_limited"
    assert call["details"]["status"] == 429
    assert call["details"]["retry_after"] == 30
    assert call["details"]["operation"] == "cancel"
    assert call["details"]["prompt_id"] == "p1"
    assert "slow down" in call["details"]["body"]
    assert "retry" in call["hint"].lower()


def test_429_without_retry_after_omits_it():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error_with_headers(429, {}))

    call = renderer.calls[0]
    assert call["code"] == "cloud_rate_limited"
    assert "retry_after" not in call["details"]


@pytest.mark.parametrize("code", [400, 409, 500, 502])
def test_other_statuses_stay_cloud_http_error(code: int):
    renderer = _FakeRenderer()
    _handle(renderer, _http_error_with_headers(code, {"Retry-After": "30"}))

    call = renderer.calls[0]
    assert call["code"] == "cloud_http_error"
    assert call["details"]["status"] == code
    assert "retry_after" not in call["details"]


def test_429_http_date_retry_after_becomes_seconds():
    """Retry-After may be an HTTP-date instead of delta-seconds (RFC 9110 §10.2.3)."""
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    renderer = _FakeRenderer()
    _handle(renderer, _http_error_with_headers(429, {"Retry-After": when}))

    retry_after = renderer.calls[0]["details"]["retry_after"]
    assert 100 <= retry_after <= 121


def test_429_http_date_in_the_past_is_zero():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error_with_headers(429, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert renderer.calls[0]["details"]["retry_after"] == 0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-5"])
def test_429_invalid_retry_after_is_dropped(value: str):
    """A non-finite or negative delay must not reach the envelope: the renderer
    would emit bare NaN/Infinity (not strict JSON), or a negative wait hint."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error_with_headers(429, {"Retry-After": value}))

    call = renderer.calls[0]
    assert call["code"] == "cloud_rate_limited"
    assert "retry_after" not in call["details"]
    assert value not in call["hint"]


# --- 429 hint stays honest about what a 429 proves ---------------------------
#
# HTTP 429 says the server is throttling; it does not by itself prove the
# request had no effect. The hint must not claim "not rejected" or tell the
# caller to repeat a request unconditionally, since repeating a submit that did
# go through would create a second job.


def test_429_hint_does_not_claim_the_request_had_no_effect():
    from comfy_cli.command._cloud_errors import rate_limited_error

    err = rate_limited_error("save", 5.0, {})
    hint = err["hint"].lower()
    assert "not rejected" not in hint
    assert "unchanged" not in hint
    assert "5s" in hint
    # A request that creates or changes something gets checked before a retry.
    assert "check" in hint


def test_429_hint_takes_a_caller_specific_next_step():
    from comfy_cli.command._cloud_errors import rate_limited_error

    err = rate_limited_error("job status", None, {"prompt_id": "p"}, next_step="the watcher keeps polling")
    assert err["hint"].endswith("the watcher keeps polling")
    assert "a few seconds" in err["hint"]
    assert err["details"] == {"prompt_id": "p", "status": 429}


def test_registry_scopes_cloud_rate_limited_to_cloud():
    """Local `comfy run` reports a 429 from its own server as `client_error`
    (`details.status` 429); `cloud_rate_limited` is the Cloud-only code, and the
    registry says so, so a consumer does not expect it from a local target."""
    from comfy_cli import error_codes

    by_code = {c.code: c for c in error_codes.REGISTRY}
    rate_limited = by_code["cloud_rate_limited"]
    assert "Cloud only" in rate_limited.meaning
    assert "client_error" in rate_limited.meaning
    assert "429" in by_code["client_error"].meaning
    hint = rate_limited.hint.lower()
    assert "unchanged" not in hint
    assert "comfy jobs ls" in hint


# A plan refusal (the account's plan does not allow the run) is HTTP 402 with a
# typed body ({"error": {"type": ..., "message": ...}}). The cloud submit
# endpoint used to send the same refusals as 429, told apart from throttling
# only by the type. Reporting either as `cloud_rate_limited` ("wait, then
# retry") sends an agent into retries that can never succeed; both get their own
# non-retryable code carrying the server's message.


def _emit_submit(body: str, status: int = 429):
    from comfy_cli.command._cloud_errors import emit_status_error

    renderer = _FakeRenderer()
    emit_status_error(
        renderer,
        status=status,
        retry_after=None,
        operation="submit",
        message=f"Cloud server rejected the workflow (HTTP {status}): Too Many Requests",
        hint="check the workflow is valid",
        details={"status": status, "body": body},
        rate_limited_next_step="check `comfy jobs ls --where cloud` for this job before re-running",
    )
    return renderer.calls[0]


@pytest.mark.parametrize(
    "error_type",
    [
        "FREE_TIER_EXHAUSTED",
        "FREE_TIER_NOT_ALLOWED",
        "PAYMENT_REQUIRED",
        "CLOUD_SUBSCRIPTION_REQUIRED",
        "PARTNER_NODE_PAYMENT_REQUIRED",
        "MODEL_PAYMENT_REQUIRED",
    ],
)
def test_429_plan_refusal_is_cloud_payment_required(error_type: str):
    server_message = "A cloud subscription is required to queue workflows."
    call = _emit_submit(f'{{"error":{{"type":"{error_type}","message":"{server_message}"}}}}')

    assert call["code"] == "cloud_payment_required"
    assert server_message in call["message"]
    assert call["details"]["reason"] == error_type
    assert call["details"]["status"] == 429
    assert server_message in call["details"]["body"]
    hint = call["hint"].lower()
    assert "retry" in hint and "not" in hint, "the hint must say retrying will not help"
    assert "comfy jobs ls" not in hint, "a refused submit queued nothing, so there is no job to look for"


@pytest.mark.parametrize(
    "body",
    [
        '{"error":{"type":"QUEUE_LIMIT","message":"Maximum queued jobs limit reached (10 jobs in this workspace)"}}',
        '{"error":{"type":"FREE_TIER_UNAVAILABLE","message":"Free-tier is temporarily unavailable. Please try again shortly."}}',
        '{"error":"slow down"}',
        '{"error":{"type":[]}}',
        '{"error":{"type":{"nested":"CLOUD_SUBSCRIPTION_REQUIRED"}}}',
        '["error"]',
        "not json at all",
        "",
    ],
)
def test_other_429_bodies_stay_cloud_rate_limited(body: str):
    call = _emit_submit(body)
    assert call["code"] == "cloud_rate_limited"
    assert "comfy jobs ls" in call["hint"]


@pytest.mark.parametrize("error_type", ["FREE_TIER_EXHAUSTED", "CLOUD_SUBSCRIPTION_REQUIRED", "SOME_FUTURE_TYPE"])
def test_402_is_cloud_payment_required_whatever_the_type(error_type: str):
    server_message = "You've used all your free generations. Upgrade to keep creating."
    call = _emit_submit(f'{{"error":{{"type":"{error_type}","message":"{server_message}"}}}}', status=402)

    assert call["code"] == "cloud_payment_required"
    assert server_message in call["message"]
    assert "HTTP 402" in call["message"]
    assert call["details"]["status"] == 402
    assert call["details"]["reason"] == error_type
    assert "comfy jobs ls" not in call["hint"]


@pytest.mark.parametrize("body", ["", "not json at all", '{"error":"no credits"}', '{"error":{"type":[]}}'])
def test_402_without_a_typed_body_is_still_cloud_payment_required(body: str):
    call = _emit_submit(body, status=402)

    assert call["code"] == "cloud_payment_required"
    assert call["details"]["status"] == 402
    assert "reason" not in call["details"]
    assert "HTTP 402" in call["message"]


def test_registry_lists_cloud_payment_required():
    from comfy_cli import error_codes

    by_code = {c.code: c for c in error_codes.REGISTRY}
    entry = by_code["cloud_payment_required"]
    assert "429" in entry.meaning
    assert "retry" in entry.hint.lower()


# --- 403 insufficient_scope → re-login guidance ------------------------------

_SCOPE_HEADERS = {"WWW-Authenticate": 'Bearer error="insufficient_scope", scope="comfy-cloud:secrets:write"'}
_SCOPE_BODY = b'{"message":"insufficient_scope: comfy-cloud:secrets:write"}'


def test_403_insufficient_scope_header_says_relogin():
    """The enforced scope gate answers RFC 6750 insufficient_scope; refresh
    cannot widen a grant, so the envelope must point at a fresh login."""
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/api/secrets", 403, "Forbidden", _SCOPE_HEADERS, io.BytesIO(_SCOPE_BODY))

    exit_exc = _handle(renderer, e)

    assert exit_exc.exit_code == 1
    (call,) = renderer.calls
    assert call["code"] == "cloud_unauthorized"
    assert "predates a permission change" in call["message"]
    assert "comfy cloud login" in call["message"]
    assert call["details"]["reason"] == "insufficient_scope"
    assert call["details"]["required_scope"] == "comfy-cloud:secrets:write"
    assert call["details"]["operation"] == "cancel"


def test_403_insufficient_scope_body_only_still_detected():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, _SCOPE_BODY))

    (call,) = renderer.calls
    assert call["details"]["reason"] == "insufficient_scope"
    # The documented body form names the scope too.
    assert call["details"]["required_scope"] == "comfy-cloud:secrets:write"


@pytest.mark.parametrize(
    "body",
    [
        b"insufficient_scope: comfy-cloud:agent:write",
        b"insufficient_scope",
        b'{"error":"insufficient_scope"}',
        b'{"error":{"code":"insufficient_scope: comfy-cloud:agent:write"}}',
    ],
)
def test_403_documented_body_forms_are_detected(body: bytes):
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, body))

    (call,) = renderer.calls
    assert call["details"]["reason"] == "insufficient_scope"


@pytest.mark.parametrize(
    "body",
    [
        b'{"message":"forbidden: this is not an insufficient_scope problem, the workspace is suspended"}',
        b"Access denied. (Hint: insufficient_scope errors are reported separately.)",
        b'{"message":"validation failed","details":"node text: insufficient_scope"}',
    ],
)
def test_403_body_merely_mentioning_token_keeps_generic_hint(body: bytes):
    """A body that only mentions the token in prose, or in a non-error field
    (e.g. reflected workflow text), is not an insufficient_scope denial."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, body))

    (call,) = renderer.calls
    assert call["message"] == "HTTP 403 during cancel"
    assert "reason" not in call["details"]


@pytest.mark.parametrize(
    "www_auth",
    [
        'Bearer error = "insufficient_scope", scope="comfy-cloud:secrets:write"',
        "Bearer error=insufficient_scope, scope=comfy-cloud:secrets:write",
        'Bearer realm="api", error_description="scope=\\"wrong\\"", error="insufficient_scope", '
        'scope="comfy-cloud:secrets:write"',
    ],
)
def test_403_header_challenge_parsing(www_auth: str):
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", 403, "Forbidden", {"WWW-Authenticate": www_auth}, io.BytesIO(b"denied"))

    _handle(renderer, e)

    (call,) = renderer.calls
    assert call["details"]["reason"] == "insufficient_scope"
    assert call["details"]["required_scope"] == "comfy-cloud:secrets:write"


def test_403_scope_in_second_challenge_is_found():
    import email.message

    headers = email.message.Message()
    headers["WWW-Authenticate"] = 'Basic realm="x"'
    headers["WWW-Authenticate"] = 'Bearer error="insufficient_scope", scope="comfy-cloud:agent:write"'
    renderer = _FakeRenderer()
    _handle(renderer, urllib.error.HTTPError("https://x/y", 403, "Forbidden", headers, io.BytesIO(b"")))

    (call,) = renderer.calls
    assert call["details"]["required_scope"] == "comfy-cloud:agent:write"


def test_403_non_string_header_value_does_not_crash():
    class _OddHeaders:
        def get(self, name):
            return 42

    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", 403, "Forbidden", _OddHeaders(), io.BytesIO(b'{"message":"forbidden"}'))

    _handle(renderer, e)

    (call,) = renderer.calls
    assert call["message"] == "HTTP 403 during cancel"


@pytest.mark.parametrize("scope", ["x" * 5000, "comfy-cloud:secrets:write\x1b[31m"])
def test_required_scope_is_bounded_and_sanitized(scope: str):
    renderer = _FakeRenderer()
    headers = {"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{scope}"'}
    _handle(renderer, urllib.error.HTTPError("https://x/y", 403, "Forbidden", headers, io.BytesIO(b"")))

    (call,) = renderer.calls
    assert call["details"]["reason"] == "insufficient_scope"
    assert "required_scope" not in call["details"]


def test_insufficient_scope_hint_mentions_scope_override():
    """A COMFY_CLOUD_SCOPES / cloud_scopes override is sent verbatim, so a
    re-login alone would request the same narrow grant."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, _SCOPE_BODY))

    hint = renderer.calls[0]["hint"]
    assert "COMFY_CLOUD_SCOPES" in hint
    assert "cloud_scopes" in hint


def test_plain_403_keeps_generic_hint():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, b'{"message":"forbidden"}'))

    (call,) = renderer.calls
    assert call["message"] == "HTTP 403 during cancel"
    assert "reason" not in call["details"]


def test_401_mentioning_insufficient_scope_is_not_a_scope_error():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(401, _SCOPE_BODY))

    (call,) = renderer.calls
    assert call["message"] == "HTTP 401 during cancel"


def test_emit_status_error_maps_insufficient_scope():
    """``run`` submit routes its HTTPError through ``emit_status_error``."""
    from comfy_cli.command._cloud_errors import emit_status_error

    renderer = _FakeRenderer()
    emit_status_error(
        renderer,
        status=403,
        retry_after=None,
        operation="submit",
        message="Cloud server rejected the workflow (HTTP 403)",
        hint="check the workflow",
        details={"status": 403, "body": _SCOPE_BODY.decode()},
    )

    (call,) = renderer.calls
    assert call["code"] == "cloud_unauthorized"
    assert call["details"]["reason"] == "insufficient_scope"


def test_emit_status_error_poll_detects_scope_without_body_in_details():
    """``run``'s poll keeps ``details`` to status/prompt_id, so it passes the
    body and ``WWW-Authenticate`` separately for detection."""
    from comfy_cli.command._cloud_errors import emit_status_error

    renderer = _FakeRenderer()
    emit_status_error(
        renderer,
        status=403,
        retry_after=None,
        operation="poll",
        message="Cloud server error while polling (HTTP 403)",
        hint=None,
        details={"status": 403, "prompt_id": "p1"},
        scope_body="denied",
        www_authenticate='Bearer error="insufficient_scope", scope="comfy-cloud:jobs:read"',
    )

    (call,) = renderer.calls
    assert call["code"] == "cloud_unauthorized"
    assert call["details"]["prompt_id"] == "p1"
    assert call["details"]["required_scope"] == "comfy-cloud:jobs:read"
    assert "body" not in call["details"]


def test_comfy_client_http_error_carries_www_authenticate():
    from unittest import mock

    from comfy_cli.comfy_client import Client, HTTPError
    from comfy_cli.target import Target

    client = Client(Target(kind="local", base_url="https://cloud.example.com"))
    err = urllib.error.HTTPError(
        "https://cloud.example.com/api/x", 403, "Forbidden", _SCOPE_HEADERS, io.BytesIO(_SCOPE_BODY)
    )
    with mock.patch("comfy_cli.comfy_client._OPENER.open", side_effect=err):
        with pytest.raises(HTTPError) as excinfo:
            client._request("POST", ["api", "prompt"], body={})

    assert excinfo.value.www_authenticate == (_SCOPE_HEADERS["WWW-Authenticate"],)


def test_comfy_client_http_error_keeps_every_www_authenticate_value():
    """A Bearer challenge in a later header field must still reach detection."""
    import email.message
    from unittest import mock

    from comfy_cli.comfy_client import Client, HTTPError
    from comfy_cli.target import Target

    headers = email.message.Message()
    headers["WWW-Authenticate"] = 'Basic realm="x"'
    headers["WWW-Authenticate"] = 'Bearer error="insufficient_scope", scope="comfy-cloud:agent:write"'
    client = Client(Target(kind="local", base_url="https://cloud.example.com"))
    err = urllib.error.HTTPError("https://cloud.example.com/api/x", 403, "Forbidden", headers, io.BytesIO(b"denied"))
    with mock.patch("comfy_cli.comfy_client._OPENER.open", side_effect=err):
        with pytest.raises(HTTPError) as excinfo:
            client._request("POST", ["api", "prompt"], body={})

    assert excinfo.value.www_authenticate == (
        'Basic realm="x"',
        'Bearer error="insufficient_scope", scope="comfy-cloud:agent:write"',
    )
    from comfy_cli.command._cloud_errors import insufficient_scope_error

    found = insufficient_scope_error(403, excinfo.value.body, excinfo.value.www_authenticate)
    assert found["details"]["required_scope"] == "comfy-cloud:agent:write"


@pytest.mark.parametrize(
    "www_auth, required_scope",
    [
        # The scope comes from the challenge that carries the error, not an earlier one.
        ('Bearer realm="a", scope="old", Bearer error="insufficient_scope", scope="new"', "new"),
        ('Basic realm="x", Bearer error=insufficient_scope, scope=comfy-cloud:agent:write', "comfy-cloud:agent:write"),
    ],
)
def test_403_scope_is_read_from_the_matching_bearer_challenge(www_auth: str, required_scope: str):
    from comfy_cli.command._cloud_errors import insufficient_scope_error

    found = insufficient_scope_error(403, "", {"WWW-Authenticate": www_auth})

    assert found["details"]["required_scope"] == required_scope


def test_403_insufficient_scope_on_a_non_bearer_challenge_is_ignored():
    from comfy_cli.command._cloud_errors import insufficient_scope_error

    headers = {"WWW-Authenticate": 'Basic error="insufficient_scope", scope="comfy-cloud:agent:write"'}

    assert insufficient_scope_error(403, '{"message":"forbidden"}', headers) is None


def test_cloud_http_handler_maps_insufficient_scope():
    from comfy_cli.command.cloud_http import handle_cloud_http_error as workflow_handler

    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/api/x", 403, "Forbidden", _SCOPE_HEADERS, io.BytesIO(_SCOPE_BODY))

    exit_exc = workflow_handler(renderer, e, operation="save")

    assert exit_exc.exit_code == 1
    (call,) = renderer.calls
    assert call["code"] == "cloud_unauthorized"
    assert call["details"]["required_scope"] == "comfy-cloud:secrets:write"
    # The server's own explanation stays inspectable.
    assert call["details"]["body"] == _SCOPE_BODY.decode()


# A JSON body whose `insufficient_scope` field sits past the 1000-byte display cap.
_LONG_SCOPE_BODY = (
    b'{"detail":"' + b"x" * (2 * _MAX_ERROR_BODY_BYTES) + b'","error":"insufficient_scope: comfy-cloud:agent:write"}'
)


def test_403_scope_field_past_display_cap_is_detected():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403, _LONG_SCOPE_BODY))

    (call,) = renderer.calls
    assert call["details"]["reason"] == "insufficient_scope"
    assert call["details"]["required_scope"] == "comfy-cloud:agent:write"
    # Only the rendered body is capped.
    assert call["details"]["body"] == _LONG_SCOPE_BODY[:_MAX_ERROR_BODY_BYTES].decode()


def test_cloud_http_handler_detects_scope_field_past_display_cap():
    from comfy_cli.command.cloud_http import handle_cloud_http_error as workflow_handler

    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/api/x", 403, "Forbidden", {}, io.BytesIO(_LONG_SCOPE_BODY))

    workflow_handler(renderer, e, operation="save")

    (call,) = renderer.calls
    assert call["details"]["required_scope"] == "comfy-cloud:agent:write"
    assert call["details"]["body"] == _LONG_SCOPE_BODY[:_MAX_ERROR_BODY_BYTES].decode()


def test_unauthorized_body_read_is_bounded_for_detection():
    from comfy_cli.command._cloud_errors import _MAX_SCOPE_BODY_BYTES, read_unauthorized_body

    e = _http_error(403, b"A" * (5 * 1024 * 1024))

    scope_body, body = read_unauthorized_body(e)

    assert len(scope_body) == _MAX_SCOPE_BODY_BYTES
    assert body == "A" * _MAX_ERROR_BODY_BYTES
