import http.client
import json
import types
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

import comfy_cli.http as http_mod
from comfy_cli.http import (
    NoRedirectHandler,
    ResponseTooLarge,
    authed_urlopen,
    build_authed_request,
    no_redirect_urlopen,
    request_json,
    target_auth_headers,
)
from comfy_cli.target import Target


def _target(*, api_key=None, auth_token=None, is_cloud=True):
    """Minimal stand-in for a resolved Target — build_authed_request reads
    ``.api_key``, ``.auth_token`` and ``.is_cloud`` (via target_auth_headers)."""
    return types.SimpleNamespace(api_key=api_key, auth_token=auth_token, is_cloud=is_cloud)


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


# ---------------------------------------------------------------------------
# build_authed_request
# ---------------------------------------------------------------------------


def test_oauth_wins_over_api_key():
    """When both credentials are set, Authorization is attached and X-API-Key is
    not — the same OAuth-first precedence as ``resolve_target`` and as the
    ``extra_data`` credential ``submit_prompt`` sends, so the header and the
    body can never name different identities."""
    req = build_authed_request("https://x/thing", _target(api_key="k", auth_token="t"))
    assert req.get_header("Authorization") == "Bearer t"
    assert req.get_header("X-api-key") is None


def test_bearer_when_only_auth_token():
    req = build_authed_request("https://x/thing", _target(auth_token="t"))
    assert req.get_header("Authorization") == "Bearer t"
    assert req.get_header("X-api-key") is None


def test_no_auth_header_when_uncredentialed():
    """A local (uncredentialed) target gets no credential header."""
    req = build_authed_request("https://x/thing", _target())
    assert req.get_header("Authorization") is None
    assert req.get_header("X-api-key") is None


def test_no_auth_header_for_local_target_even_with_stray_creds():
    """A local Target must never carry a credential, even if one is stray-set —
    build_authed_request delegates to target_auth_headers's is_cloud gate
    rather than attaching headers unconditionally."""
    req = build_authed_request("http://127.0.0.1:8188/thing", _target(api_key="k", auth_token="t", is_cloud=False))
    assert req.get_header("Authorization") is None
    assert req.get_header("X-api-key") is None


def test_content_type_passthrough():
    req = build_authed_request("https://x/thing", _target(), content_type="application/json")
    assert req.get_header("Content-type") == "application/json"


def test_no_content_type_header_by_default():
    req = build_authed_request("https://x/thing", _target())
    assert req.get_header("Content-type") is None


def test_method_and_data_passthrough():
    req = build_authed_request("https://x/thing", _target(), method="POST", data=b"payload")
    assert req.get_method() == "POST"
    assert req.data == b"payload"


# ---------------------------------------------------------------------------
# authed_urlopen
# ---------------------------------------------------------------------------


def test_authed_urlopen_uses_no_redirect_opener():
    """authed_urlopen opens through the shared _AUTHED_OPENER, passing the
    built Request and timeout through."""
    sentinel = object()
    with patch.object(http_mod._AUTHED_OPENER, "open", return_value=sentinel) as opened:
        result = authed_urlopen("https://x/thing", _target(api_key="k"), method="POST", data=b"d", timeout=15)
    assert result is sentinel
    req = opened.call_args.args[0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == "https://x/thing"
    assert req.get_header("X-api-key") == "k"
    assert req.get_method() == "POST"
    assert opened.call_args.kwargs["timeout"] == 15


def test_authed_urlopen_propagates_refused_redirect():
    """A 30x refused by the opener surfaces as HTTPError to the caller instead
    of following the redirect with credentials attached."""
    err = urllib.error.HTTPError("https://x/thing", 302, "redirect refused", http.client.HTTPMessage(), None)
    with patch.object(http_mod._AUTHED_OPENER, "open", side_effect=err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            authed_urlopen("https://x/thing", _target(auth_token="t"))
    assert exc_info.value.code == 302


def test_migrated_caller_propagates_refused_redirect():
    """End-to-end: a migrated caller (models search's _http_get_json) surfaces a
    refused-redirect HTTPError rather than swallowing or following it."""
    from comfy_cli.command.models.search import _http_get_json

    err = urllib.error.HTTPError("https://x/api/assets", 302, "redirect refused", http.client.HTTPMessage(), None)
    with patch.object(http_mod._AUTHED_OPENER, "open", side_effect=err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _http_get_json("https://x/api/assets", _target(api_key="k"))
    assert exc_info.value.code == 302


# ---------------------------------------------------------------------------
# no_redirect_urlopen
# ---------------------------------------------------------------------------


def test_no_redirect_urlopen_uses_no_redirect_opener():
    """A prepared Request goes through _AUTHED_OPENER untouched — no header is
    attached, since this helper's callers carry their credential themselves."""
    sentinel = object()
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=b"{}", method="POST")
    with patch.object(http_mod._AUTHED_OPENER, "open", return_value=sentinel) as opened:
        result = no_redirect_urlopen(req, timeout=15)
    assert result is sentinel
    assert opened.call_args.args[0] is req
    assert opened.call_args.kwargs["timeout"] == 15


def test_no_redirect_urlopen_propagates_refused_redirect():
    """The /prompt submit embeds a credential in its body, so a 30x must raise
    rather than be followed — we don't lean on urllib dropping the body."""
    err = urllib.error.HTTPError(
        "http://127.0.0.1:8188/prompt", 307, "redirect refused", http.client.HTTPMessage(), None
    )
    with patch.object(http_mod._AUTHED_OPENER, "open", side_effect=err):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            no_redirect_urlopen(urllib.request.Request("http://127.0.0.1:8188/prompt"))
    assert exc_info.value.code == 307


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi"])
def test_authed_opener_refuses_non_http_schemes(url, tmp_path):
    """The credential-carrying opener is pinned to http(s). ``build_opener``
    would install File/FTP/Data handlers, so a caller steered into a local-file
    or ftp URL could exfiltrate or read unintended content; those schemes must
    fall through to UnknownHandler instead."""
    with pytest.raises(urllib.error.URLError) as exc_info:
        authed_urlopen(url, _target(api_key="k"))
    assert "unknown url type" in str(exc_info.value.reason)


def test_authed_opener_handler_set():
    """Only http(s)-relevant handlers are installed — no File/FTP/Data."""
    names = {type(h).__name__ for h in http_mod._AUTHED_OPENER.handlers}
    assert {"HTTPHandler", "HTTPSHandler", "NoRedirectHandler"} <= names
    assert not names & {"FileHandler", "FTPHandler", "DataHandler"}


# ---------------------------------------------------------------------------
# target_auth_headers
# ---------------------------------------------------------------------------


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


def test_target_auth_headers_cloud_both_oauth_wins():
    """Unreachable in production (``resolve_target`` resolves one credential),
    but pinned so the tie-break stays OAuth-first: an API-key header here would
    disagree with the OAuth identity ``submit_prompt`` puts in ``extra_data``
    and would make the resulting 401 unrefreshable."""
    target = Target(kind="cloud", base_url="https://cloud.example", auth_token="t", api_key="k")
    assert target_auth_headers(target) == {"Authorization": "Bearer t"}


def test_target_auth_headers_cloud_uncredentialed_is_empty():
    """A cloud Target with no credential contributes no headers — callers add
    nothing rather than an empty/``None`` credential header."""
    target = Target(kind="cloud", base_url="https://cloud.example")
    assert target_auth_headers(target) == {}


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
    """Route every ``request_json`` call to ``payload`` (bytes) and record the Requests seen.

    ``request_json`` opens through the shared ``_AUTHED_OPENER`` (built with
    ``NoRedirectHandler``), not the bare ``urllib.request.urlopen`` function,
    so the fake must patch the opener's ``open`` method.
    """
    seen: list[urllib.request.Request] = []

    def _fake(req, timeout=None):
        seen.append(req)
        if isinstance(payload, Exception):
            raise payload
        return _fake_resp(payload, status)

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", _fake)
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


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_request_json_rejects_non_positive_max_bytes(cloud_target, max_bytes):
    with pytest.raises(ValueError, match="max_bytes"):
        request_json("https://cloud.example/api/thing", cloud_target, max_bytes=max_bytes)


def test_request_json_opens_via_shared_opener_with_no_redirect_handler():
    # A 30x with an authenticated request in flight must not be followed —
    # NoRedirectHandler on the shared opener is what refuses it. Assert the
    # opener request_json actually uses carries that handler, so a future
    # revert to the bare default opener (which follows redirects and copies
    # headers onto the new request) is caught here rather than in production.
    assert any(isinstance(h, NoRedirectHandler) for h in http_mod._AUTHED_OPENER.handlers)


def test_request_json_refuses_plaintext_http_for_cloud_target(monkeypatch, cloud_target):
    # A credential must never go out over cleartext HTTP to a non-loopback
    # host — even if some misconfiguration points COMFY_CLOUD_BASE_URL at
    # http://. No urlopen call should happen at all in this case.
    seen = _patch_urlopen(monkeypatch, b"{}")
    with pytest.raises(ValueError, match="non-https"):
        request_json("http://cloud.example/api/thing", cloud_target, max_bytes=1024)
    assert seen == []


def test_request_json_allows_plaintext_http_for_loopback(monkeypatch, local_target):
    # Loopback has no network to sniff, so plaintext is fine — this is the
    # normal case for local ComfyUI.
    _patch_urlopen(monkeypatch, b"{}")
    assert request_json("http://127.0.0.1:8188/models", local_target, max_bytes=1024) == (200, {})


def test_request_json_no_auth_headers_skips_https_gate(monkeypatch):
    # A cloud target with no credentials at all (e.g. logged out) attaches no
    # headers, so there's nothing to protect — the https/loopback gate must
    # not block that request.
    target = Target(kind="cloud", base_url="http://cloud.example", path_prefix="/api")
    _patch_urlopen(monkeypatch, b"{}")
    assert request_json("http://cloud.example/api/thing", target, max_bytes=1024) == (200, {})


def test_request_json_recursion_error_returns_none(monkeypatch, cloud_target):
    # A pathologically nested body can exhaust the interpreter stack in
    # json.loads; RecursionError is neither JSONDecodeError nor
    # UnicodeDecodeError, so it needs naming here or it escapes uncaught like
    # the other two. Forced directly (rather than via real deep nesting,
    # which CPython's C-accelerated decoder tolerates to very large depths)
    # so this test stays fast and deterministic.
    _patch_urlopen(monkeypatch, b'{"a": 1}')
    monkeypatch.setattr(http_mod.json, "loads", lambda *a, **kw: (_ for _ in ()).throw(RecursionError()))
    assert request_json("https://cloud.example/api/thing", cloud_target, max_bytes=1024) == (200, None)
