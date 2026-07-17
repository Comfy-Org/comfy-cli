"""Shared HTTP helpers: default timeouts and an auth-leak-safe redirect policy."""

import urllib.error
import urllib.request

# Default timeout (seconds) for plain, non-streaming API calls. Without an
# explicit timeout ``requests`` blocks forever on a stalled peer; this makes such
# a call fail fast with a typed ``requests.Timeout`` instead of hanging the CLI.
DEFAULT_HTTP_TIMEOUT = 30.0

# Timeout for streaming downloads and large uploads, as a (connect, read) tuple.
# ``requests`` applies the read timeout per socket read rather than to the whole
# transfer, so this caps how long we wait to *start* connecting/receiving without
# capping a legitimately long transfer.
DOWNLOAD_TIMEOUT = (10.0, 60.0)


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
