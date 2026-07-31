import http.client
import types
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

import comfy_cli.http as http_mod
from comfy_cli.http import (
    DEFAULT_HTTP_TIMEOUT,
    DOWNLOAD_TIMEOUT,
    NoRedirectHandler,
    authed_urlopen,
    build_authed_request,
    no_redirect_urlopen,
    target_auth_headers,
)
from comfy_cli.target import Target


def _target(*, api_key=None, auth_token=None, is_cloud=True):
    """Minimal stand-in for a resolved Target — build_authed_request reads
    ``.api_key``, ``.auth_token`` and ``.is_cloud`` (via target_auth_headers)."""
    return types.SimpleNamespace(api_key=api_key, auth_token=auth_token, is_cloud=is_cloud)


def test_default_http_timeout_is_a_positive_scalar():
    assert isinstance(DEFAULT_HTTP_TIMEOUT, (int, float))
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
