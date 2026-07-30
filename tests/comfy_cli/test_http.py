import http.client
import urllib.error
import urllib.request

import pytest

from comfy_cli.http import NoRedirectHandler, target_auth_headers
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
