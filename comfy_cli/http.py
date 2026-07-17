"""Shared HTTP helpers with an auth-leak-safe redirect policy."""

import urllib.error
import urllib.request


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow HTTP redirects.

    Following a redirect with ``Authorization: Bearer …`` / ``X-API-Key`` in
    flight risks replaying the credential at the redirect target (auth leak /
    SSRF). None of our authenticated endpoints redirect under normal
    operation; a 30x is almost certainly a misconfiguration or an attack, so
    we surface it as a clear ``HTTPError`` instead of following it.

    ``message`` is parameterizable so call sites can keep their own wording.
    """

    def __init__(self, message: str = "redirect refused"):
        super().__init__()
        self._message = message

    def http_error_301(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, self._message, headers, fp)

    http_error_302 = http_error_303 = http_error_307 = http_error_308 = http_error_301


def _http_only_proxy_handler() -> urllib.request.ProxyHandler:
    """A ProxyHandler that can only proxy http(s).

    ``ProxyHandler()`` defaults to ``getproxies()`` and registers a
    ``<scheme>_open`` method for *every* entry it finds, so an ``ftp_proxy`` in
    the environment would give the opener an ``ftp_open`` — servicing
    ``ftp://`` through the proxy and stepping straight past the
    ``UnknownHandler`` that ``build_http_only_opener`` relies on. Filtering the
    map to http(s) keeps real proxy support (``proxy_bypass``/``no_proxy`` read
    the environment directly, not this dict) while leaving non-http schemes
    with nowhere to go.
    """
    proxies = {scheme: url for scheme, url in urllib.request.getproxies().items() if scheme in ("http", "https")}
    return urllib.request.ProxyHandler(proxies)


def build_http_only_opener(*handlers: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """Build an opener that speaks http(s) and nothing else.

    ``build_opener()`` would also install ``FileHandler``/``FTPHandler``/
    ``DataHandler``. Our call sites build their URLs from a trusted
    ``target.base_url``, so that isn't reachable today, but these openers
    attach credentials — pinning them to http(s) means a future caller can't
    be steered into a ``file://``, ``ftp://`` or ``data:`` fetch. Unknown
    schemes fall to ``UnknownHandler``, which raises
    ``URLError("unknown url type")``.

    ``handlers`` are the caller's own additions (e.g. a redirect policy). As in
    ``build_opener``, a caller-supplied handler replaces the default it
    subclasses rather than being appended behind it. Note that no redirect
    handler is installed unless the caller passes one, so a bare opener
    surfaces a 30x as an ``HTTPError`` rather than following it.
    """
    defaults = [
        (urllib.request.ProxyHandler, _http_only_proxy_handler),
        (urllib.request.HTTPHandler, urllib.request.HTTPHandler),
        (urllib.request.HTTPDefaultErrorHandler, urllib.request.HTTPDefaultErrorHandler),
        (urllib.request.HTTPErrorProcessor, urllib.request.HTTPErrorProcessor),
        (urllib.request.UnknownHandler, urllib.request.UnknownHandler),
    ]
    # urllib.request only defines HTTPSHandler on an SSL-capable build; naming it
    # unconditionally would blow up at import time on one without.
    if hasattr(urllib.request, "HTTPSHandler"):
        defaults.append((urllib.request.HTTPSHandler, urllib.request.HTTPSHandler))

    opener = urllib.request.OpenerDirector()
    for klass, factory in defaults:
        if not any(isinstance(handler, klass) for handler in handlers):
            opener.add_handler(factory())
    for handler in handlers:
        opener.add_handler(handler)
    return opener


_AUTHED_OPENER = build_http_only_opener(NoRedirectHandler())

# The uncredentialed fetches — the template gallery on raw.githubusercontent.com
# and the REST calls against a local ``http://{host}:{port}`` ComfyUI server.
# These reached for ``urllib.request.urlopen()``, i.e. the global default
# opener, which also speaks ``file://``, ``ftp://`` and ``data:``. Nothing here
# attaches a credential header, so unlike ``_AUTHED_OPENER`` there is no
# redirect-replay exposure and ``HTTPRedirectHandler`` is installed explicitly
# to keep the redirect-following those call sites have always had. What the
# pinning buys is that a URL which stops being trusted — a gallery URL that
# becomes configurable, say — still can't be steered into a local-file read.
_PLAIN_OPENER = build_http_only_opener(urllib.request.HTTPRedirectHandler())


def plain_urlopen(url, *, timeout: float = 30.0):
    """Open an uncredentialed request via the http(s)-only shared opener.

    ``url`` is a full URL or a prepared ``Request``. Redirects are followed, as
    they were when these call sites used the global default opener.
    """
    return _PLAIN_OPENER.open(url, timeout=timeout)


def build_authed_request(
    url: str,
    target,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
) -> urllib.request.Request:
    """Build a urllib Request carrying the target's credential header.

    api_key wins over auth_token; the policy layer (resolve_target) populates
    at most one, so this is a mechanic, not a policy choice. No header is
    attached for an uncredentialed (local) target.
    """
    req = urllib.request.Request(url, data=data, method=method)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


def authed_urlopen(
    url: str,
    target,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
):
    """Open an authed request via the no-redirect opener. A 30x raises
    HTTPError instead of replaying credentials at the redirect target."""
    req = build_authed_request(url, target, method=method, data=data, content_type=content_type)
    return _AUTHED_OPENER.open(req, timeout=timeout)
