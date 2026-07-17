import http.client
import types
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

import comfy_cli.http as http_mod
from comfy_cli.http import NoRedirectHandler, authed_urlopen, build_authed_request


def _target(*, api_key=None, auth_token=None):
    """Minimal stand-in for a resolved Target — build_authed_request only reads
    ``.api_key`` and ``.auth_token``."""
    return types.SimpleNamespace(api_key=api_key, auth_token=auth_token)


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


def test_api_key_wins_over_auth_token():
    """When both credentials are set, X-API-Key is attached and Authorization is not."""
    req = build_authed_request("https://x/thing", _target(api_key="k", auth_token="t"))
    assert req.get_header("X-api-key") == "k"
    assert req.get_header("Authorization") is None


def test_bearer_when_only_auth_token():
    req = build_authed_request("https://x/thing", _target(auth_token="t"))
    assert req.get_header("Authorization") == "Bearer t"
    assert req.get_header("X-api-key") is None


def test_no_auth_header_when_uncredentialed():
    """A local (uncredentialed) target gets no credential header."""
    req = build_authed_request("https://x/thing", _target())
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
