"""Scheme-pinning guards for every credential-carrying urllib opener.

``urllib.request.build_opener()`` implicitly installs ``FileHandler``,
``FTPHandler`` and ``DataHandler``. An opener that attaches ``Authorization``/
``X-API-Key`` must not speak those schemes: a caller steered into a ``file://``
or ``ftp://`` URL could read unintended local content through an opener that
carries credentials. These tests mirror ``test_http.py``'s checks on the shared
``_AUTHED_OPENER`` for each of the other openers.
"""

import urllib.error
import urllib.request

import pytest

from comfy_cli import comfy_client, http
from comfy_cli.cloud import oauth
from comfy_cli.command import transfer
from comfy_cli.cql import engine, loader

NON_HTTP_URLS = ["file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi"]

# Every opener the CLI builds, including _DOWNLOAD_OPENER: it deliberately
# follows redirects (its own handler strips auth headers and re-checks the
# scheme), but it should be scheme-pinned like the rest.
OPENERS = [
    ("http._AUTHED_OPENER", http._AUTHED_OPENER),
    ("comfy_client._OPENER", comfy_client._OPENER),
    ("cql.engine._opener", engine._opener),
    ("cql.loader._LOADER_OPENER", loader._LOADER_OPENER),
    ("cloud.oauth._OAUTH_OPENER", oauth._OAUTH_OPENER),
    ("transfer._TRANSFER_OPENER", transfer._TRANSFER_OPENER),
    ("transfer._DOWNLOAD_OPENER", transfer._DOWNLOAD_OPENER),
]
OPENER_IDS = [name for name, _ in OPENERS]


@pytest.mark.parametrize("opener", [o for _, o in OPENERS], ids=OPENER_IDS)
@pytest.mark.parametrize("url", NON_HTTP_URLS)
def test_opener_refuses_non_http_schemes(opener, url):
    """Non-http(s) schemes fall through to UnknownHandler."""
    with pytest.raises(urllib.error.URLError) as exc_info:
        opener.open(url)
    assert "unknown url type" in str(exc_info.value.reason)


@pytest.mark.parametrize("opener", [o for _, o in OPENERS], ids=OPENER_IDS)
def test_opener_handler_set(opener):
    """Only http(s)-relevant handlers are installed — no File/FTP/Data."""
    names = {type(h).__name__ for h in opener.handlers}
    assert {"HTTPHandler", "HTTPSHandler", "UnknownHandler"} <= names
    assert not names & {"FileHandler", "FTPHandler", "DataHandler"}


@pytest.mark.parametrize(
    ("name", "opener"),
    [(n, o) for n, o in OPENERS if n != "transfer._DOWNLOAD_OPENER"],
    ids=[n for n, _ in OPENERS if n != "transfer._DOWNLOAD_OPENER"],
)
def test_credentialed_openers_refuse_redirects(name, opener):
    """Every opener that carries credentials keeps its NoRedirectHandler."""
    assert any(isinstance(h, http.NoRedirectHandler) for h in opener.handlers)


def test_download_opener_still_follows_redirects():
    """_DOWNLOAD_OPENER's redirect handler is intentional — scheme-pinning it
    must not turn it into a no-redirect opener."""
    assert any(isinstance(h, transfer._DownloadRedirectHandler) for h in transfer._DOWNLOAD_OPENER.handlers)
    assert not any(isinstance(h, http.NoRedirectHandler) for h in transfer._DOWNLOAD_OPENER.handlers)


def test_build_http_only_opener_installs_caller_handlers():
    handler = http.NoRedirectHandler("custom message")
    opener = http.build_http_only_opener(handler)
    assert handler in opener.handlers


def test_build_http_only_opener_matches_build_opener_proxy_support(monkeypatch):
    """Proxy support is preserved: ProxyHandler registers exactly the schemes
    build_opener's own ProxyHandler would for the same environment."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")

    def proxies(opener):
        return next(h.proxies for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler))

    assert proxies(http.build_http_only_opener()) == proxies(urllib.request.build_opener())
    assert "https" in proxies(http.build_http_only_opener())
