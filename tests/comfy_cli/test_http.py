import http.client
import json
import pathlib
import ssl
import sys
import types
import urllib.error
import urllib.request
from unittest.mock import patch

import certifi
import pytest

import comfy_cli.http as http_mod
from comfy_cli.http import (
    DEFAULT_HTTP_TIMEOUT,
    DOWNLOAD_TIMEOUT,
    MAX_RESPONSE_BYTES,
    NoRedirectHandler,
    ResponseTooLarge,
    authed_urlopen,
    build_authed_request,
    no_redirect_urlopen,
    read_capped,
    request_json,
    target_auth_headers,
)
from comfy_cli.target import Target


def _target(*, api_key=None, auth_token=None, is_cloud=True):
    """Minimal stand-in for a resolved Target — build_authed_request reads
    ``.api_key``, ``.auth_token`` and ``.is_cloud`` (via target_auth_headers)."""
    return types.SimpleNamespace(api_key=api_key, auth_token=auth_token, is_cloud=is_cloud)


def test_default_http_timeout_is_a_positive_scalar():
    assert isinstance(DEFAULT_HTTP_TIMEOUT, int | float)
    assert DEFAULT_HTTP_TIMEOUT > 0


def test_download_timeout_is_a_connect_read_tuple():
    # A (connect, read) tuple so connect failures fail fast while a legitimately
    # long transfer is not capped (requests applies read timeout per socket read).
    assert isinstance(DOWNLOAD_TIMEOUT, tuple)
    assert len(DOWNLOAD_TIMEOUT) == 2
    connect, read = DOWNLOAD_TIMEOUT
    assert connect > 0 and read > 0


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


# ---------------------------------------------------------------------------
# read_capped — the shared bounded-read primitive
# ---------------------------------------------------------------------------


def test_read_capped_returns_a_body_under_the_cap():
    assert read_capped(_fake_resp(b"hello"), "https://example.com/x", max_bytes=1024) == b"hello"


def test_read_capped_body_exactly_at_cap_is_complete_not_truncated():
    # Boundary: len(raw) == cap is a *complete* body. The primitive reads cap+1
    # precisely so this case is distinguishable from a truncated one.
    body = b"0123456789"
    assert read_capped(_fake_resp(body), "https://example.com/x", max_bytes=len(body)) == body


def test_read_capped_one_byte_over_the_cap_raises():
    with pytest.raises(ResponseTooLarge):
        read_capped(_fake_resp(b"0123456789"), "https://example.com/x", max_bytes=9)


def test_read_capped_message_names_the_url_and_the_cap():
    # Callers interpolate this into user-facing envelopes, so it must stay
    # descriptive enough to tell a user *which* endpoint misbehaved.
    with pytest.raises(ResponseTooLarge) as exc_info:
        read_capped(_fake_resp(b"x" * 100), "https://example.com/gallery.json", max_bytes=4)
    msg = str(exc_info.value)
    assert "https://example.com/gallery.json" in msg
    assert "4" in msg


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_read_capped_rejects_non_positive_max_bytes(max_bytes):
    with pytest.raises(ValueError):
        read_capped(_fake_resp(b"x"), "https://example.com/x", max_bytes=max_bytes)


def test_read_capped_default_cap_is_the_shared_constant():
    # An unbounded read is the bug this primitive exists to remove, so the
    # default must be a real ceiling, not None/0.
    assert MAX_RESPONSE_BYTES == 64 * 1024 * 1024
    reads: list[int] = []

    class _Recording:
        def read(self, n):
            reads.append(n)
            return b"body"

    assert read_capped(_Recording(), "https://example.com/x") == b"body"
    assert reads == [MAX_RESPONSE_BYTES + 1]


def test_read_capped_asks_for_exactly_one_byte_past_the_cap():
    reads: list[int] = []

    class _Recording:
        def read(self, n):
            reads.append(n)
            return b"ab"

    read_capped(_Recording(), "https://example.com/x", max_bytes=8)
    assert reads == [9]


# --- the CA trust store -------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_trust_store_cache():
    """`trust_store`/`ssl_context` are process-wide caches, so every case must
    resolve the environment it set rather than the first one any test saw."""
    http_mod.trust_store.cache_clear()
    http_mod.ssl_context.cache_clear()
    yield
    http_mod.trust_store.cache_clear()
    http_mod.ssl_context.cache_clear()


def _no_env(monkeypatch):
    for name in (http_mod.CA_FILE_ENV_VAR, http_mod.CA_DIR_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


def test_an_explicit_bundle_outranks_certifi(tmp_path, monkeypatch):
    """Given SSL_CERT_FILE, When the store resolves, Then the named bundle wins.

    The documented escape hatch has to be real: an explicitly supplied context
    bypasses the env vars OpenSSL would have read for itself.
    """
    # Given
    bundle = tmp_path / "corporate.pem"
    bundle.write_bytes(pathlib.Path(certifi.where()).read_bytes())
    monkeypatch.setenv(http_mod.CA_FILE_ENV_VAR, str(bundle))
    monkeypatch.delenv(http_mod.CA_DIR_ENV_VAR, raising=False)

    # When
    store = http_mod.trust_store()

    # Then
    assert store.cafile == str(bundle)
    assert store.error is None
    assert store.pinned is True, "an operator named it, so a load failure is fatal"
    assert http_mod.ssl_context().verify_mode is ssl.CERT_REQUIRED


def test_certifi_is_used_when_nothing_is_configured(monkeypatch):
    """Given no override, When the store resolves, Then certifi supplies the CAs."""
    # Given
    _no_env(monkeypatch)

    # When
    store = http_mod.trust_store()

    # Then
    assert store.cafile == certifi.where()
    assert "certifi" in store.source
    assert store.pinned is False


def test_a_system_bundle_covers_a_machine_without_certifi(tmp_path, monkeypatch):
    """Given no certifi, When a system bundle exists, Then it is used.

    The failure this exists for: a distribution whose Python has neither certifi
    nor a compiled-in OpenSSL trust path, where every https call died with
    CERTIFICATE_VERIFY_FAILED while `curl` to the same host succeeded.
    """
    # Given
    _no_env(monkeypatch)
    system = tmp_path / "ca-certificates.crt"
    system.write_text("", encoding="utf-8")
    monkeypatch.setattr(http_mod, "_SYSTEM_CA_FILES", (str(system),))
    monkeypatch.setitem(sys.modules, "certifi", None)

    # When
    store = http_mod.trust_store()

    # Then
    assert store.cafile == str(system)
    assert "system bundle" in store.source


def test_no_store_anywhere_falls_back_to_the_interpreter(monkeypatch):
    """Given nothing at all, When the store resolves, Then the interpreter decides."""
    # Given
    _no_env(monkeypatch)
    monkeypatch.setattr(http_mod, "_SYSTEM_CA_FILES", ())
    monkeypatch.setitem(sys.modules, "certifi", None)

    # When
    store = http_mod.trust_store()

    # Then
    assert store.cafile is None and store.capath is None
    assert store.error is None


def test_an_unusable_override_fails_closed_rather_than_silently_substituting(tmp_path, monkeypatch):
    """Given a bogus SSL_CERT_FILE, When a context is built, Then it trusts nothing.

    The caller named a trust root on purpose. Quietly verifying against certifi
    instead would answer a question nobody asked, and would hide a typo in the
    one setting an operator uses to pin a corporate CA.
    """
    # Given
    missing = tmp_path / "nope.pem"
    monkeypatch.setenv(http_mod.CA_FILE_ENV_VAR, str(missing))
    monkeypatch.delenv(http_mod.CA_DIR_ENV_VAR, raising=False)

    # When
    store = http_mod.trust_store()

    # Then
    assert store.error is not None and str(missing) in store.error
    assert http_mod.ssl_context().cert_store_stats()["x509_ca"] == 0
    hint = http_mod.tls_trust_hint()
    assert str(missing) in hint and http_mod.CA_FILE_ENV_VAR in hint


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(ssl.SSLCertVerificationError("unable to get local issuer certificate"), True, id="bare"),
        pytest.param(
            urllib.error.URLError(ssl.SSLCertVerificationError("unable to get local issuer certificate")),
            True,
            id="wrapped-in-urlerror",
        ),
        pytest.param(
            urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1017)"),
            True,
            id="reason-string-only",
        ),
        pytest.param(urllib.error.URLError("connection refused"), False, id="unrelated-transport"),
        pytest.param(TimeoutError("timed out"), False, id="timeout"),
    ],
)
def test_verification_failures_are_told_apart_from_other_transport_errors(error, expected):
    """Given a transport error, When classified, Then only trust failures match.

    The failure arrives wrapped and does not always keep its class, so the chain
    is walked and the OpenSSL reason string counts as evidence.
    """
    # Given / When / Then
    assert http_mod.tls_verification_failed(error) is expected


def test_the_classifier_terminates_on_a_self_referential_chain():
    """Given a cyclic cause chain, When classified, Then it returns rather than hangs."""
    # Given
    first = urllib.error.URLError("outer")
    second = urllib.error.URLError("inner")
    first.reason = second
    second.reason = first

    # When / Then
    assert http_mod.tls_verification_failed(first) is False


def test_every_opener_verifies_against_the_shared_context(monkeypatch):
    """Given an opener, When it is built, Then its https handler carries the context."""
    # Given
    _no_env(monkeypatch)

    # When
    opener = http_mod.build_http_only_opener()

    # Then
    handlers = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
    assert len(handlers) == 1
    assert handlers[0]._context is http_mod.ssl_context()


def _spy_on_default_context(monkeypatch) -> list[dict]:
    """Record the kwargs `ssl.create_default_context` is built with.

    The distinction under test is which CPython branch runs, and it is not
    visible in the resulting store on a machine whose platform default is empty
    — which is exactly the machine this bug was found on. `cafile=` given means
    `load_default_certs()` is skipped; absent means it runs.
    """
    calls: list[dict] = []
    real = ssl.create_default_context

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", spy)
    return calls


def test_certifi_is_added_to_the_default_roots_not_substituted_for_them(tmp_path, monkeypatch):
    """Given an extra bundle, When the context is built, Then the OS roots survive.

    `create_default_context(cafile=...)` and `load_default_certs()` are EXCLUSIVE
    branches in CPython, so supplying certifi as `cafile` narrows the trust store
    to certifi alone. That breaks every environment whose CA lives only in the OS
    store — a TLS-inspecting corporate proxy, a private root installed with
    `update-ca-certificates`, a Windows root pushed by policy — while the bug
    being fixed is the opposite one, an interpreter whose default store is empty.
    The two sources have to add up.
    """
    # Given
    extra = tmp_path / "extra-ca.pem"
    extra.write_bytes(pathlib.Path(certifi.where()).read_bytes())
    added = extra.read_text().count("-----BEGIN CERTIFICATE-----")
    _no_env(monkeypatch)
    monkeypatch.setattr(http_mod, "_SYSTEM_CA_FILES", ())
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: str(extra)))
    calls = _spy_on_default_context(monkeypatch)

    # When
    context = http_mod.ssl_context()

    # Then
    assert calls == [{}], "the platform default roots must be loaded, so no cafile/capath is passed"
    assert context.cert_store_stats()["x509_ca"] >= added, "and the extra bundle is loaded on top of them"


def test_an_explicit_bundle_is_loaded_without_overriding_the_ca_directory(tmp_path, monkeypatch):
    """Given only SSL_CERT_FILE, When the context is built, Then OpenSSL's rule holds.

    `SSL_CERT_FILE` overrides the default CA *file* and leaves the default CA
    *directory* alone; only setting both replaces everything. `load_default_certs`
    already applies exactly that rule, so the context has to be built WITHOUT
    `cafile=` — passing it takes the exclusive branch and discards the directory
    too, which is stricter than the variable has ever meant.

    The explicit load on top is what makes a Windows build, whose
    `load_default_certs` ignores the variables entirely, honour them as well.
    """
    # Given
    only = tmp_path / "only.pem"
    only.write_bytes(pathlib.Path(certifi.where()).read_bytes())
    monkeypatch.setenv(http_mod.CA_FILE_ENV_VAR, str(only))
    monkeypatch.delenv(http_mod.CA_DIR_ENV_VAR, raising=False)
    calls = _spy_on_default_context(monkeypatch)

    # When
    context = http_mod.ssl_context()

    # Then
    assert {} in calls, "the context itself must be built with no cafile, so OpenSSL applies its own rule"
    assert context.cert_store_stats()["x509_ca"] >= only.read_text().count("-----BEGIN CERTIFICATE-----")


def test_an_unloadable_supplement_falls_through_to_the_platform_roots(tmp_path, monkeypatch):
    """Given a bundle that exists but will not parse, When built, Then the roots remain.

    `os.path.isfile` proves existence, not loadability — an empty, truncated or
    unreadable bundle raises from `load_verify_locations`. These openers are
    built at MODULE scope, so that surfaced as a bare traceback out of
    `import comfy_cli.http`, before any renderer existed to turn it into an
    envelope: the exact unparseable non-envelope failure this work exists to
    remove. A supplement that will not load must be skipped, not fatal.

    The platform default is stubbed to a NON-EMPTY store on purpose. A host whose
    real default is empty — the very kind this change exists for — cannot tell
    "fell through to the platform roots" from "trusts nothing" by counting, and a
    context that trusts nothing also reports CERT_REQUIRED.
    """
    # Given
    broken = tmp_path / "broken.pem"
    broken.write_text("")
    _no_env(monkeypatch)
    monkeypatch.setattr(http_mod, "_SYSTEM_CA_FILES", ())
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: str(broken)))

    real = ssl.create_default_context
    populated = len(pathlib.Path(certifi.where()).read_text().split("-----BEGIN CERTIFICATE-----")) - 1

    def platform_with_roots(*args, **kwargs):
        context = real(*args, **kwargs)
        if not kwargs.get("cafile") and not kwargs.get("capath"):
            context.load_verify_locations(cafile=certifi.where())
        return context

    monkeypatch.setattr(ssl, "create_default_context", platform_with_roots)

    # When
    context = http_mod.ssl_context()

    # Then
    assert context.cert_store_stats()["x509_ca"] == populated, (
        "the platform roots must survive a supplement that would not load"
    )


def test_an_unloadable_PINNED_store_still_fails_closed(tmp_path, monkeypatch):
    """Given a pinned store that will not load, When built, Then nothing is trusted.

    The other side of the same guard: the caller named this root on purpose, so
    falling through to the platform's would verify against something they did
    not choose.
    """
    # Given
    broken = tmp_path / "broken.pem"
    broken.write_text("")
    monkeypatch.setenv(http_mod.CA_FILE_ENV_VAR, str(broken))
    monkeypatch.delenv(http_mod.CA_DIR_ENV_VAR, raising=False)

    # When / Then
    assert http_mod.ssl_context().cert_store_stats()["x509_ca"] == 0
