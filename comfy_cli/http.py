"""Shared HTTP helpers with an auth-leak-safe redirect policy."""

import json
import urllib.error
import urllib.parse
import urllib.request

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
    ``resolve_target`` populates at most one of api_key / auth_token, so the
    precedence branch is mechanics, not policy.
    """
    headers: dict[str, str] = {}
    if target.is_cloud:
        if target.api_key:
            headers["X-API-Key"] = target.api_key
        elif target.auth_token:
            headers["Authorization"] = f"Bearer {target.auth_token}"
    return headers


class ResponseTooLarge(Exception):
    """A response exceeded the caller's byte cap — refuse to truncate."""


_OPENER = urllib.request.build_opener(NoRedirectHandler())


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
    without this: redirects are refused via ``NoRedirectHandler`` (a 30x
    can't replay the credential at another host), and the URL itself must be
    HTTPS or loopback before a credential is attached.
    """
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
    with _OPENER.open(req, timeout=timeout) as resp:
        status = resp.status
        # Read one byte past the cap so a full body is distinguishable from a truncated one.
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        # Keep the message descriptive: search interpolates it into its envelope.
        raise ResponseTooLarge(f"response from {url} exceeds {max_bytes} byte cap")
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return status, None
