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
