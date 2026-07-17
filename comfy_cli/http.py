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


def _build_authed_opener() -> urllib.request.OpenerDirector:
    """Build the credential-carrying opener with http/https handlers only.

    ``build_opener()`` would also install ``FileHandler``/``FTPHandler``. Every
    call site builds its URL from a trusted ``target.base_url``, so that isn't
    reachable today, but this is the opener that attaches credentials — pinning
    it to http(s) means a future caller can't be steered into a ``file://`` or
    ``ftp://`` fetch. Unknown schemes fall to ``UnknownHandler``, which raises
    ``URLError("unknown url type")``.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        NoRedirectHandler(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


_AUTHED_OPENER = _build_authed_opener()


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
