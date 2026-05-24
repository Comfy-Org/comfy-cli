"""Tests for the download redirect handler.

Cloud's ``/api/view`` returns 302 to a short-lived signed GCS URL. The
download path follows the redirect but **must** strip auth headers
(X-API-Key, Authorization, Cookie) before the follow-up request — the
signed URL's signature is the auth, and dragging our credential along
would leak it to storage.
"""

from __future__ import annotations

import io
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from comfy_cli.command.transfer import (
    _AUTH_HEADERS_TO_STRIP,
    _DOWNLOAD_OPENER,
    _DownloadRedirectHandler,
)


def _make_request(url: str, **headers) -> urllib.request.Request:
    req = urllib.request.Request(url)
    for k, v in headers.items():
        req.add_header(k.replace("_", "-"), v)
    return req


class TestRedirectStripping:
    """Unit tests against the handler's redirect_request() method directly."""

    def test_strips_api_key_on_redirect(self):
        handler = _DownloadRedirectHandler()
        req = _make_request("https://cloud.example.com/api/view?filename=x.png", X_API_Key="secret")
        new_req = handler.redirect_request(
            req,
            fp=io.BytesIO(),
            code=302,
            msg="Found",
            headers={"Location": "https://storage.googleapis.com/signed?sig=abc"},
            newurl="https://storage.googleapis.com/signed?sig=abc",
        )
        # The follow-up request must NOT carry the X-API-Key.
        all_headers = {**new_req.headers, **new_req.unredirected_hdrs}
        for k in all_headers:
            assert k.lower() not in _AUTH_HEADERS_TO_STRIP, f"leaked header: {k}"

    def test_strips_authorization_on_redirect(self):
        handler = _DownloadRedirectHandler()
        req = _make_request("https://cloud.example.com/api/view", Authorization="Bearer eyJhbG…")
        new_req = handler.redirect_request(
            req,
            fp=io.BytesIO(),
            code=302,
            msg="Found",
            headers={"Location": "https://storage.googleapis.com/signed"},
            newurl="https://storage.googleapis.com/signed",
        )
        all_headers = {**new_req.headers, **new_req.unredirected_hdrs}
        assert "Authorization" not in all_headers
        assert "authorization" not in {k.lower() for k in all_headers}

    def test_preserves_non_auth_headers(self):
        handler = _DownloadRedirectHandler()
        req = _make_request("https://cloud.example.com/api/view", X_API_Key="secret", User_Agent="comfy-cli")
        new_req = handler.redirect_request(
            req,
            fp=io.BytesIO(),
            code=302,
            msg="Found",
            headers={"Location": "https://storage.googleapis.com/signed"},
            newurl="https://storage.googleapis.com/signed",
        )
        # User-Agent should survive; X-API-Key should not.
        all_headers = {k.lower(): v for k, v in {**new_req.headers, **new_req.unredirected_hdrs}.items()}
        assert all_headers.get("user-agent") == "comfy-cli"
        assert "x-api-key" not in all_headers

    def test_rejects_non_http_redirect_target(self):
        handler = _DownloadRedirectHandler()
        req = _make_request("https://cloud.example.com/api/view", X_API_Key="secret")
        with pytest.raises(urllib.error.HTTPError) as exc:
            handler.redirect_request(
                req,
                fp=io.BytesIO(),
                code=302,
                msg="Found",
                headers={"Location": "file:///etc/passwd"},
                newurl="file:///etc/passwd",
            )
        assert "non-HTTP" in str(exc.value) or "scheme" in str(exc.value)


# ---------------------------------------------------------------------------
# Integration test against a real HTTP server.
#
# Spin up a thread-local server that 302s /api/view -> /signed.png and serves
# a tiny PNG at /signed. The test asserts the follow-up request carries no
# auth header.
# ---------------------------------------------------------------------------


_PNG_BODY = b"\x89PNG\r\n\x1a\nfake-png-body-for-test"


class _RedirectServer(BaseHTTPRequestHandler):
    received_headers: list[dict[str, str]] = []

    def log_message(self, format, *args):  # noqa: A002 — silence stderr
        pass

    def do_GET(self):
        type(self).received_headers.append(dict(self.headers))
        if self.path.startswith("/api/view"):
            # 302 redirect to the "signed" URL on the same host (avoids needing
            # a second port; the signing fiction is just for the test).
            self.send_response(302)
            self.send_header("Location", "/signed")
            self.end_headers()
        elif self.path == "/signed":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PNG_BODY)))
            self.end_headers()
            self.wfile.write(_PNG_BODY)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def redirect_server():
    _RedirectServer.received_headers = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectServer)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()


def test_download_opener_follows_redirect_and_strips_auth(redirect_server):
    """End-to-end: the opener follows the 302 but the auth header doesn't go with it."""
    port = redirect_server
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/view?filename=x.png")
    req.add_header("X-API-Key", "super-secret-key")

    with _DOWNLOAD_OPENER.open(req, timeout=5) as resp:
        body = resp.read()

    assert body == _PNG_BODY

    # Two requests reached the server: the initial /api/view (with auth) and
    # the follow-up /signed (without auth).
    headers = _RedirectServer.received_headers
    assert len(headers) == 2
    # First request keeps the credential.
    assert headers[0].get("X-Api-Key") == "super-secret-key" or headers[0].get("X-API-Key") == "super-secret-key"
    # Second request DOES NOT carry it.
    h2_lower = {k.lower(): v for k, v in headers[1].items()}
    assert "x-api-key" not in h2_lower, f"auth leaked to redirect target: {headers[1]}"
    assert "authorization" not in h2_lower
