import http.client
import urllib.error
import urllib.request

import pytest

from comfy_cli.http import DEFAULT_HTTP_TIMEOUT, DOWNLOAD_TIMEOUT, NoRedirectHandler


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
