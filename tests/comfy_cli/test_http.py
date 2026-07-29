import http.client
import json
import urllib.error
import urllib.request

import pytest

from comfy_cli.http import NoRedirectHandler, ResponseTooLarge, request_json, target_auth_headers
from comfy_cli.target import Target


def _call(handler, method_name, code=302):
    req = urllib.request.Request("https://example.com/thing")
    headers = http.client.HTTPMessage()
    method = getattr(handler, method_name)
    method(req, None, code, "Found", headers)


@pytest.mark.parametrize(
    "method_name",
    ["http_error_301", "http_error_302", "http_error_303", "http_error_307", "http_error_308"],
)
def test_refuses_every_redirect_status(method_name):
    handler = NoRedirectHandler()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _call(handler, method_name, code=308)
    err = exc_info.value
    assert err.code == 308  # status code is preserved
    assert str(err.reason) == "redirect refused"  # default message
    assert err.url == "https://example.com/thing"


def test_default_message():
    handler = NoRedirectHandler()
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _call(handler, "http_error_301", code=301)
    assert exc_info.value.code == 301
    assert str(exc_info.value.reason) == "redirect refused"


def test_custom_message_passthrough():
    handler = NoRedirectHandler("redirect refused (auth leak prevention)")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _call(handler, "http_error_302", code=302)
    assert str(exc_info.value.reason) == "redirect refused (auth leak prevention)"
    assert exc_info.value.code == 302


def test_target_auth_headers_local_attaches_nothing_even_with_creds():
    """The security property this builder exists to enforce: a local Target
    NEVER contributes auth headers, even if stray credentials are set on it
    (the exact misuse the ``is_cloud`` gate defends against)."""
    target = Target(
        kind="local",
        base_url="http://127.0.0.1:8188",
        auth_token="stray",
        api_key="stray",
    )
    assert target_auth_headers(target) == {}


def test_target_auth_headers_cloud_api_key_only():
    target = Target(kind="cloud", base_url="https://cloud.example", api_key="k")
    assert target_auth_headers(target) == {"X-API-Key": "k"}


def test_target_auth_headers_cloud_auth_token_only():
    target = Target(kind="cloud", base_url="https://cloud.example", auth_token="t")
    assert target_auth_headers(target) == {"Authorization": "Bearer t"}


def test_target_auth_headers_cloud_both_api_key_wins():
    target = Target(kind="cloud", base_url="https://cloud.example", auth_token="t", api_key="k")
    assert target_auth_headers(target) == {"X-API-Key": "k"}


# ---------------------------------------------------------------------------
# request_json — the shared capped-read JSON helper
# ---------------------------------------------------------------------------


def _fake_resp(body: bytes, status: int = 200):
    """Minimal urlopen-compatible response. ``read(n)`` truncates like the real one."""

    class _Resp:
        def __init__(self):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n: int | None = None):
            return body if n is None else body[:n]

    return _Resp()


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, payload, status: int = 200):
    """Route every urlopen call to ``payload`` (bytes) and record the Requests seen."""
    seen: list[urllib.request.Request] = []

    def _fake(req, timeout=None):
        seen.append(req)
        if isinstance(payload, Exception):
            raise payload
        return _fake_resp(payload, status)

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return seen


@pytest.fixture
def cloud_target():
    return Target(kind="cloud", base_url="https://cloud.example", path_prefix="/api", api_key="test-api-key")


@pytest.fixture
def local_target():
    # Stray credentials on purpose: a local target must never emit them.
    return Target(kind="local", base_url="http://127.0.0.1:8188", path_prefix="", api_key="stray", auth_token="stray")


def test_request_json_oversize_raises_response_too_large(monkeypatch, cloud_target):
    # An oversize body must NOT masquerade as an unparseable one (which would
    # silently degrade to ``None`` and look like an empty response).
    _patch_urlopen(monkeypatch, b'{"data": []}')
    with pytest.raises(ResponseTooLarge) as exc_info:
        request_json("https://cloud.example/api/thing", cloud_target, max_bytes=4)
    # The message is interpolated into search's envelope, so it must stay descriptive.
    msg = str(exc_info.value)
    assert "https://cloud.example/api/thing" in msg
    assert "4" in msg


def test_request_json_body_exactly_at_cap_still_parses(monkeypatch, cloud_target):
    # Boundary: len(raw) == cap is a *complete* body, not a truncated one.
    body = b'{"a": 1}'
    _patch_urlopen(monkeypatch, body)
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=len(body)) == (200, {"a": 1})


def test_request_json_empty_body_returns_none(monkeypatch, cloud_target):
    _patch_urlopen(monkeypatch, b"", status=204)
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024) == (204, None)


def test_request_json_non_utf8_body_returns_none(monkeypatch, cloud_target):
    # UnicodeDecodeError is a ValueError but *not* a JSONDecodeError; if it were
    # not named in the except clause it would escape request_json uncaught.
    _patch_urlopen(monkeypatch, b"\xff\xfe\x00not json")
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024) == (200, None)


def test_request_json_unparseable_body_returns_none(monkeypatch, cloud_target):
    _patch_urlopen(monkeypatch, b"<html>not json</html>")
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024) == (200, None)


def test_request_json_list_body_parses(monkeypatch, cloud_target):
    # Both model-listing endpoints return top-level JSON arrays.
    _patch_urlopen(monkeypatch, b'["checkpoints", "loras"]')
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024) == (
        200,
        ["checkpoints", "loras"],
    )


def test_request_json_get_sends_no_body_and_no_content_type(monkeypatch, cloud_target):
    seen = _patch_urlopen(monkeypatch, b"{}")
    request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024)
    req = seen[0]
    assert req.get_method() == "GET"
    assert req.data is None
    assert req.get_header("Content-type") is None


def test_request_json_post_attaches_json_body_and_content_type(monkeypatch, cloud_target):
    seen = _patch_urlopen(monkeypatch, b"{}", status=201)
    status, _ = request_json(
        "https://cloud.example/api/thing", cloud_target, method="POST", body={"name": "wf"}, max_bytes=1024
    )
    assert status == 201
    req = seen[0]
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"name": "wf"}
    assert req.get_header("Content-type") == "application/json"


def test_request_json_attaches_auth_headers_for_cloud(monkeypatch, cloud_target):
    seen = _patch_urlopen(monkeypatch, b"{}")
    request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024)
    # urllib title-cases header names on the Request object.
    assert seen[0].get_header("X-api-key") == "test-api-key"


def test_request_json_attaches_no_auth_headers_for_local(monkeypatch, local_target):
    # Same defense-in-depth gate as target_auth_headers: a local target never
    # gets a credential, so a stray token can't leak to a plaintext server.
    seen = _patch_urlopen(monkeypatch, b"{}")
    request_json("http://127.0.0.1:8188/models", local_target, max_bytes=1024)
    req = seen[0]
    # Assert on the whole header bag, not two named keys: a named-key check
    # would pass vacuously if urllib ever changed how it cases header names.
    assert req.headers == {}


def test_request_json_raises_urllib_errors_verbatim(monkeypatch, cloud_target):
    # Callers map these to envelope codes themselves, so they must not be swallowed.
    err = urllib.error.HTTPError("https://cloud.example/api/thing", 503, "boom", http.client.HTTPMessage(), None)
    _patch_urlopen(monkeypatch, err)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024)
    assert exc_info.value.code == 503


def test_request_json_max_bytes_is_keyword_required(cloud_target):
    # No default: each caller keeps owning its own cap constant.
    with pytest.raises(TypeError):
        request_json("https://cloud.example/api/thing", cloud_target)
