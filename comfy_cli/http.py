"""Shared HTTP helpers: default timeouts and an auth-leak-safe redirect policy."""

import json
import urllib.error
import urllib.parse
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

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def assert_safe_url(url: str) -> None:
    """Reject plaintext HTTP for non-loopback hosts.

    Anything carrying a credential (``X-API-Key`` / Bearer token) over the
    wire must be HTTPS unless the host is a loopback address (where there's
    no network to sniff).
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"refusing to send request to non-https, non-loopback URL: {url} "
        "(set COMFY_CLOUD_BASE_URL to an https:// endpoint)"
    )


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


def target_auth_headers(target) -> dict[str, str]:
    """Auth headers for a routing Target — cloud only.

    Local ComfyUI has no auth; refusing to attach credentials to a non-cloud
    target means a stray token on a local Target can never leak to a
    plaintext server (same defense-in-depth gate as comfy_client.Client).

    ``resolve_target`` resolves a single credential, so at most one of
    api_key / auth_token is ever populated and the tie-break is unreachable in
    production. It is still ordered OAuth-first, to match every other
    credential decision in the codebase: ``resolve_target``'s own precedence
    (a live session beats API keys, which are on a deprecation path), the
    ``auth_token_comfy_org`` credential ``submit_prompt`` injects into
    ``extra_data`` for partner-API nodes, and ``Client._try_refresh_token``,
    which can only self-heal a 401 when the request rode an OAuth token. A
    lone api_key-first branch here would be the one place that disagrees:
    given both credentials it would authenticate at the gateway as the API
    key while handing partner-API nodes the OAuth identity, and the resulting
    401 could never refresh.
    """
    headers: dict[str, str] = {}
    if target.is_cloud:
        if target.auth_token:
            headers["Authorization"] = f"Bearer {target.auth_token}"
        elif target.api_key:
            headers["X-API-Key"] = target.api_key
    return headers


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


def no_redirect_urlopen(url, *, timeout: float = 30.0):
    """Open a prepared credential-bearing ``Request`` without following redirects.

    ``authed_urlopen`` covers the common case where the credential rides a
    header we attach ourselves. This is the escape hatch for a request whose
    credential the caller has already placed somewhere we can't build — the
    ``/prompt`` submit carries ``api_key_comfy_org`` inside its JSON body — and
    which therefore wants the same no-redirect policy without the header
    mechanics.
    """
    return _AUTHED_OPENER.open(url, timeout=timeout)


def build_authed_request(
    url: str,
    target,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
) -> urllib.request.Request:
    """Build a urllib Request carrying the target's credential header.

    Delegates to ``target_auth_headers`` for the header itself — that's the
    single ``is_cloud`` gate every other call site goes through, so a local
    Target can't carry a stray credential into this path either.
    """
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in target_auth_headers(target).items():
        req.add_header(k, v)
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


class ResponseTooLarge(Exception):
    """A response exceeded the caller's byte cap — refuse to truncate."""


# Default ceiling for a single response body buffered whole into memory.
#
# 64 MiB, picked for this primitive rather than inherited from whichever call
# site happened to be nearest. Two numbers bracket it. The largest legitimate
# body any of these readers sees is ``/api/object_info`` at roughly 9 MB on
# cloud (the template gallery index and the ComfyUI ``/queue`` + ``/history``
# payloads are orders of magnitude smaller), so 64 MiB is ~7x headroom over the
# real ceiling and cannot plausibly reject a well-behaved server. At the other
# end it is a body a laptop can hold twice over without swapping, which is what
# a cap is for: the failure mode being bounded here is a server that streams
# without end, not one that is merely chatty.
#
# It also matches the cap the already-bounded call sites converged on
# independently (``models/search``, both ``workflow`` readers), so the CLI has
# one number rather than five. ``cql.loader`` keeps its own, larger
# ``MAX_INPUT_BYTES`` — that one bounds a user-supplied object_info dump, a
# different thing being measured.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def read_capped(resp, url: str, *, max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read a whole response body, refusing to buffer more than ``max_bytes``.

    The shared bounded-read primitive: an unbounded ``resp.read()`` lets a
    misbehaving or hostile server decide how much of the CLI's memory to
    consume, so every reader that buffers a body whole goes through here.

    Reads one byte past the cap so a body that exactly fills it is still
    recognizable as complete, and raises :class:`ResponseTooLarge` rather than
    returning a silently truncated body — a truncated JSON body would surface
    as a confusing parse error, and a truncated file body would be written to
    disk as if it were whole.

    ``url`` is interpolated into the error message; call sites pass the URL
    they requested, since ``resp`` alone may not name the host after a
    redirect. Callers that need a different ceiling pass ``max_bytes``; see
    ``MAX_RESPONSE_BYTES`` above for why the default is what it is.
    """
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ResponseTooLarge(f"response from {url} exceeds {max_bytes} byte cap")
    return raw


def request_json(
    url: str,
    target,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: float = 30.0,
    max_bytes: int,
) -> tuple[int, dict | list | None]:
    """Authed HTTP call returning (status, parsed_json_or_none).

    Raises urllib errors verbatim so callers can map them to envelope codes,
    and ``ResponseTooLarge`` when the body exceeds ``max_bytes`` — an oversize
    body must not masquerade as an unparseable one. An empty or unparseable
    (bad JSON / non-UTF-8 / too-deeply-nested) body parses to ``None``;
    ``UnicodeDecodeError`` is a ``ValueError`` but *not* a ``JSONDecodeError``,
    so it needs naming here or it escapes uncaught.

    ``max_bytes`` is keyword-required with no default so every caller keeps
    owning its own cap constant. Auth headers never go out over the wire
    without this: redirects are refused via the shared no-redirect opener (a
    30x can't replay the credential at another host), and the URL itself
    must be HTTPS or loopback before a credential is attached.
    """
    # ``read_capped`` re-checks this, but only after the request has gone out;
    # validating up front means a bad cap costs no network round-trip.
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
    headers = target_auth_headers(target)
    if headers:
        assert_safe_url(url)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with _AUTHED_OPENER.open(req, timeout=timeout) as resp:
        status = resp.status
        # ``read_capped``'s message names the URL and the cap — search
        # interpolates it into its envelope, so it must stay descriptive.
        raw = read_capped(resp, url, max_bytes=max_bytes)
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return status, None
