"""OAuth 2.1 Authorization Code + PKCE flow for the comfy CLI.

The cloud server (services/ingest/server/implementation/oauth/...) only
supports two grant types — ``authorization_code`` and ``refresh_token`` —
and requires PKCE with the S256 challenge method, plus a ``resource``
parameter on every authorize request. There's no device-code flow.

The CLI flow is therefore:

  1. POST /oauth/register  — Dynamic Client Registration (RFC 7591). One-shot:
     cache the returned ``client_id`` so subsequent logins reuse it.
  2. Generate PKCE pair (code_verifier + code_challenge=S256).
  3. Start a localhost HTTP server on a random 127.0.0.1:<port>.
  4. Open the browser to GET /oauth/authorize?... with our redirect_uri
     pointing at the local server.
  5. User logs in / consents on the cloud frontend.
  6. Cloud redirects to http://127.0.0.1:<port>/callback?code=...&state=...
  7. POST /oauth/token  with grant_type=authorization_code + the
     code_verifier. Receive { access_token, refresh_token, expires_in, ... }.

Refresh:

  8. POST /oauth/token  with grant_type=refresh_token. Server rotates the
     refresh token; we always replace what we stored.

No tokens or codes are logged. The browser URL is printed for users who can't
auto-open (SSH, headless), with a clear caveat that it includes the PKCE
challenge but not any secret.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import socket
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.error import HTTPError, URLError

from comfy_cli.cloud import (
    CALLBACK_PATH,
    CLIENT_ID,
    CLIENT_NAME,
    get_base_url,
)
from comfy_cli.http import NoRedirectHandler

# ---------------------------------------------------------------------------
# Error types — caller maps these to renderer.error(code=...) codes.
# ---------------------------------------------------------------------------


class OAuthError(Exception):
    """Base for OAuth flow failures. ``code`` is the error envelope code."""

    code: str = "oauth_failed"

    def __init__(self, message: str, *, hint: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.hint = hint
        self.details = details or {}


class OAuthRegisterError(OAuthError):
    code = "oauth_register_failed"


class OAuthAuthorizeError(OAuthError):
    code = "oauth_authorize_failed"


class OAuthTokenError(OAuthError):
    code = "oauth_token_failed"


class OAuthRefreshError(OAuthError):
    code = "oauth_refresh_failed"


class OAuthCancelled(OAuthError):
    code = "oauth_cancelled"


class OAuthTimeout(OAuthError):
    code = "oauth_timeout"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOOPBACK_HOST = "127.0.0.1"
# Path must match the path registered for the first-party client. See the
# note in comfy_cli/cloud/__init__.py about RFC 8252 §7.3 port-variance.
_CALLBACK_PATH = CALLBACK_PATH
_AUTH_DEFAULT_TIMEOUT_S = 300  # 5 minutes for the user to click through
_HTTP_TIMEOUT_S = 30  # network timeout for token/register POSTs


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


def _b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge).

    The cloud server requires S256 and a 43-char base64url challenge (see
    request.go's ErrInvalidCodeChallenge). A 32-byte verifier yields exactly
    that after SHA-256 + base64url-no-pad.
    """
    verifier = _b64url_no_pad(secrets.token_bytes(32))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _b64url_no_pad(digest)
    assert len(challenge) == 43, "PKCE S256 challenge must be 43 chars"
    return verifier, challenge


def generate_state() -> str:
    """CSRF token threaded through the redirect. Required by the server."""
    return _b64url_no_pad(secrets.token_bytes(16))


# ---------------------------------------------------------------------------
# Dynamic Client Registration (RFC 7591)
# ---------------------------------------------------------------------------


@dataclass
class RegisteredClient:
    client_id: str
    client_name: str
    redirect_uris: tuple[str, ...]
    issued_at: int | None = None


def register_client(
    *,
    base_url: str | None = None,
    client_name: str = CLIENT_NAME,
    redirect_uris: tuple[str, ...] = (f"http://127.0.0.1:0{CALLBACK_PATH}",),
) -> RegisteredClient:
    base_url = base_url or get_base_url()
    """Register the CLI as a public native client via DCR.

    The redirect_uris field must be present at registration; for a CLI we
    register a generic loopback placeholder. The actual port is decided at
    login time and validated by the server policy (loopback-with-any-port).
    """
    body = {
        "redirect_uris": list(redirect_uris),
        "application_type": "native",
        "client_name": client_name,
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    try:
        resp = _post_json(f"{base_url}/oauth/register", body)
    except _HTTPFail as e:
        raise OAuthRegisterError(
            f"failed to register CLI as an OAuth client at {base_url}: {e}",
            hint="check that the cloud server is reachable and accepts public-client registration",
            details={"status": e.status, "body": e.body},
        ) from None
    if "client_id" not in resp:
        raise OAuthRegisterError(
            "oauth/register response did not include client_id",
            details={"response": resp},
        )
    return RegisteredClient(
        client_id=resp["client_id"],
        client_name=resp.get("client_name", client_name),
        redirect_uris=tuple(resp.get("redirect_uris", redirect_uris)),
        issued_at=resp.get("client_id_issued_at"),
    )


# ---------------------------------------------------------------------------
# Localhost callback server
# ---------------------------------------------------------------------------


@dataclass
class _CallbackCapture:
    """Thread-shared bucket the HTTP handler fills, that ``run_oauth_login``
    blocks on."""

    code: str | None = None
    state: str | None = None
    error: str | None = None
    error_description: str | None = None
    received_event: threading.Event = field(default_factory=threading.Event)


def _build_handler(
    *,
    expected_state: str,
    capture: _CallbackCapture,
    success_html: str,
    failure_html: str,
) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):  # noqa: D401 - http server
        # Suppress access-log noise to stderr; we have our own logging.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: ARG002
            pass

        def do_GET(self) -> None:  # noqa: N802 (stdlib name)
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path != _CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            error = qs.get("error", [None])[0]
            error_description = qs.get("error_description", [None])[0]
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]

            html_body: str
            if error or not code or state != expected_state:
                capture.error = error or "missing_code_or_state_mismatch"
                capture.error_description = error_description
                html_body = failure_html
                self.send_response(400)
            else:
                capture.code = code
                capture.state = state
                html_body = success_html
                self.send_response(200)
            payload = html_body.encode("utf-8")
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # No caching — these are one-shot pages.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except OSError:
                pass
            finally:
                capture.received_event.set()

    return CallbackHandler


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((_LOOPBACK_HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


_SUCCESS_HTML = """<!doctype html>
<html><head><title>comfy CLI — signed in</title>
<style>
body{font:14px/1.5 -apple-system, system-ui, sans-serif;color:#222;background:#fafafa;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:32px 40px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)}
h1{font-size:18px;margin:0 0 8px}
p{margin:4px 0;color:#555}
.ok{color:#0a7f29;font-weight:600}
.dim{color:#888;font-size:12px;margin-top:16px}
</style></head><body>
<div class="card">
<h1><span class="ok">✓</span> Signed in to Comfy Cloud</h1>
<p>You can close this tab and return to the terminal.</p>
<p class="dim">comfy CLI · __HOST__</p>
</div></body></html>"""

_FAILURE_HTML = """<!doctype html>
<html><head><title>comfy CLI — sign-in failed</title>
<style>
body{font:14px/1.5 -apple-system, system-ui, sans-serif;color:#222;background:#fafafa;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#fff;border:1px solid #f0c0c0;border-radius:8px;padding:32px 40px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04)}
h1{font-size:18px;margin:0 0 8px;color:#b32020}
p{margin:4px 0;color:#555}
.dim{color:#888;font-size:12px;margin-top:16px}
</style></head><body>
<div class="card">
<h1>✗ Sign-in failed</h1>
<p>Return to the terminal for details.</p>
<p class="dim">comfy CLI</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Token result
# ---------------------------------------------------------------------------


@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_in: int | None
    expires_at: int | None  # absolute epoch seconds
    scope: str | None

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }


def _token_set_from_response(resp: dict) -> TokenSet:
    access = resp.get("access_token")
    if not access:
        raise OAuthTokenError(
            "token response missing access_token",
            details={"response": _redact_token_response(resp)},
        )
    expires_in = resp.get("expires_in")
    expires_at = int(time.time()) + int(expires_in) if isinstance(expires_in, (int, float)) else None
    return TokenSet(
        access_token=access,
        refresh_token=resp.get("refresh_token"),
        token_type=resp.get("token_type", "Bearer"),
        expires_in=expires_in,
        expires_at=expires_at,
        scope=resp.get("scope"),
    )


def _redact_token_response(resp: dict) -> dict:
    """Strip access_token / refresh_token before stuffing into error details."""
    redacted = dict(resp)
    for k in ("access_token", "refresh_token"):
        if k in redacted and isinstance(redacted[k], str):
            v = redacted[k]
            redacted[k] = (v[:6] + "…") if len(v) > 6 else "…"
    return redacted


# ---------------------------------------------------------------------------
# The full flow
# ---------------------------------------------------------------------------


@dataclass
class LoginResult:
    tokens: TokenSet
    client_id: str
    base_url: str
    resource: str
    scope: str
    redirect_uri: str

    def to_storage_record(self) -> dict:
        return {
            "kind": "oauth",
            "base_url": self.base_url,
            "resource": self.resource,
            "client_id": self.client_id,
            "scope": self.scope,
            "tokens": self.tokens.to_dict(),
        }


def run_login(
    *,
    base_url: str | None = None,
    resource: str | None = None,
    scopes: tuple[str, ...] | None = None,
    client_id: str = CLIENT_ID,
    client_name: str = CLIENT_NAME,
    open_browser: bool = True,
    timeout_s: float = _AUTH_DEFAULT_TIMEOUT_S,
    on_url_ready: Any = None,
    register_if_missing: bool = False,
) -> LoginResult:
    """Run the full Authorization Code + PKCE flow.

    ``client_id`` defaults to the first-party ``comfy-cli`` provisioned
    in the cloud's seed migration. Pass ``register_if_missing=True`` to fall
    back to RFC 7591 Dynamic Client Registration if the first-party client is
    rejected (e.g., on a dev backend that hasn't been seeded).

    ``on_url_ready`` is an optional callback invoked with the authorize URL
    once it's been constructed but before the browser is opened. Used by the
    pretty-mode renderer to print the URL for headless users.
    """
    from comfy_cli.cloud import get_resource_url, get_scopes

    base_url = base_url or get_base_url()
    resource = resource or get_resource_url()
    scopes = scopes or get_scopes()

    if not client_id and not register_if_missing:
        raise OAuthAuthorizeError(
            "no client_id and register_if_missing=False",
            hint="pass an explicit client_id or set register_if_missing=True",
        )
    if not client_id and register_if_missing:
        registered = register_client(base_url=base_url, client_name=client_name)
        client_id = registered.client_id

    # 2. PKCE + state.
    verifier, challenge = generate_pkce_pair()
    state = generate_state()

    # 3. Stand up the loopback server before we send the URL. Bind to port 0
    #    and read back the OS-assigned port to avoid a pick-then-bind TOCTOU race.
    capture = _CallbackCapture()
    success_html = _SUCCESS_HTML.replace("__HOST__", urllib.parse.urlsplit(base_url).netloc or base_url)
    handler_cls = _build_handler(
        expected_state=state,
        capture=capture,
        success_html=success_html,
        failure_html=_FAILURE_HTML,
    )
    server = http.server.HTTPServer((_LOOPBACK_HOST, 0), handler_cls)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = int(server.server_address[1])
    redirect_uri = f"http://{_LOOPBACK_HOST}:{port}{_CALLBACK_PATH}"
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        # 4. Build authorize URL.
        authorize_url = _build_authorize_url(
            base_url=base_url,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            state=state,
            challenge=challenge,
            resource=resource,
        )
        if on_url_ready is not None:
            try:
                on_url_ready(authorize_url)
            except Exception:  # noqa: BLE001 — callback errors must not break login
                pass

        if open_browser:
            # webbrowser.open returns True if it *thinks* it succeeded — but we
            # still let the user click the link manually if they need to.
            try:
                webbrowser.open(authorize_url, new=2, autoraise=True)
            except Exception:  # noqa: BLE001
                pass

        # 5. Wait for callback (or timeout).
        got = capture.received_event.wait(timeout=timeout_s)
        if not got:
            raise OAuthTimeout(
                f"timed out waiting for browser callback after {int(timeout_s)}s",
                hint="re-run `comfy auth login` and complete the sign-in in your browser",
            )
        if capture.error or not capture.code:
            raise OAuthAuthorizeError(
                f"authorization failed: {capture.error or 'no code returned'}",
                hint="re-run `comfy auth login` and check for typos or browser blockers",
                details={
                    "oauth_error": capture.error,
                    "oauth_error_description": capture.error_description,
                },
            )
    finally:
        # Best-effort cleanup; the daemon thread will exit after handle_request returns.
        try:
            server.server_close()
        except OSError:
            pass

    # 6. Exchange code for tokens. resource= must be echoed on the token
    # request (RFC 8707) so the issuer can audience-bind the resulting JWT.
    tokens = exchange_code(
        base_url=base_url,
        client_id=client_id,
        code=capture.code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
        resource=resource,
    )

    return LoginResult(
        tokens=tokens,
        client_id=client_id,
        base_url=base_url,
        resource=resource,
        scope=" ".join(scopes),
        redirect_uri=redirect_uri,
    )


def _build_authorize_url(
    *,
    base_url: str,
    client_id: str,
    redirect_uri: str,
    scopes: tuple[str, ...],
    state: str,
    challenge: str,
    resource: str,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    return f"{base_url}/oauth/authorize?{urllib.parse.urlencode(params)}"


def exchange_code(
    *,
    base_url: str,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    resource: str | None = None,
) -> TokenSet:
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    # RFC 8707 resource indicator: must be echoed on the token request so the
    # issuer can mint an audience-bound token. Skipping it makes
    # audience-enforcing servers reject the exchange.
    if resource:
        body["resource"] = resource
    try:
        resp = _post_form(f"{base_url}/oauth/token", body)
    except _HTTPFail as e:
        raise OAuthTokenError(
            f"token exchange failed: {e}",
            hint="re-run `comfy auth login` to start a fresh authorization",
            details={"status": e.status, "body": e.body},
        ) from None
    return _token_set_from_response(resp)


def refresh_tokens(
    *,
    base_url: str,
    client_id: str,
    refresh_token: str,
    resource: str | None = None,
    scopes: tuple[str, ...] | None = None,
) -> TokenSet:
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if resource:
        body["resource"] = resource
    if scopes:
        body["scope"] = " ".join(scopes)
    try:
        resp = _post_form(f"{base_url}/oauth/token", body)
    except _HTTPFail as e:
        raise OAuthRefreshError(
            f"refresh failed: {e}",
            hint="run `comfy auth login` to sign in again",
            details={"status": e.status, "body": e.body},
        ) from None
    return _token_set_from_response(resp)


# Upper bound on how long a waiter blocks for the refresh lock. The holder
# only keeps it for one token POST (``_HTTP_TIMEOUT_S`` = 30s), so this leaves
# headroom for that round-trip plus the read-modify-write. If it's ever
# exceeded (a wedged peer) we fall back to whatever is persisted rather than
# racing an in-flight refresh.
_REFRESH_LOCK_TIMEOUT_S = 45.0

# OAuth2 token-endpoint error codes that mean the refresh-token *family* is
# truly dead — only a fresh ``comfy cloud login`` can recover, so we clear the
# stored session. Everything else is deliberately NOT fatal (see below).
_FATAL_TOKEN_ERROR_CODES = ("invalid_grant", "invalid_token")


def _is_fatal_token_error(exc: OAuthRefreshError) -> bool:
    """True only when a refresh failure means the whole token *family* is dead.

    The auth server rotates refresh tokens and does reuse detection: a spent
    (or replayed) refresh token comes back as ``invalid_grant`` ("refresh
    token reuse detected") and the server invalidates the entire family. That,
    and an explicit ``invalid_token`` / reuse signal, are the only failures that
    justify wiping the login.

    This is gated on the *error code in the response body*, NOT the bare HTTP
    status. A previous version treated every 400/401 as fatal, which logged the
    user out on recoverable failures: a 401 ``invalid_client`` (the
    dynamically-registered client was GC'd server-side → re-register, don't
    re-login) or a 400 ``invalid_request`` / ``invalid_scope`` (a config/audience
    bug) would needlessly destroy a perfectly good session. A status of 0
    (network/URLError) or a 5xx is transient and likewise never fatal.

    Unknown / unparseable 4xx is treated as NON-fatal so a single odd response
    can't wipe the login: the reactive path retries at most once and then
    surfaces the 401, and the proactive path falls back to the local expiry
    check — neither can loop on a dead token, so keeping the session is safe.
    """
    blob = f"{exc.details.get('body', '')} {exc}".lower()
    if "reuse" in blob:
        return True
    return any(code in blob for code in _FATAL_TOKEN_ERROR_CODES)


# The reason from the most recent fatal refresh, stashed per-thread so the
# caller (which only sees ``ensure_fresh_session`` return ``None``) can surface
# the auth server's *actual* error_description instead of a canned guess. The
# refresh and the caller that reports it run in the same thread/call-stack, so a
# thread-local is the right scope — and it can't leak a stale reason across
# unrelated commands the way a module global would.
_last_fatal = threading.local()


def _describe_token_error(exc: OAuthRefreshError) -> str | None:
    """Pull a short, human-facing reason out of an RFC 6749 §5.2 token-error
    body — e.g. ``invalid_grant: workspace membership lost``. Returns ``None``
    when the body carries nothing useful (network error, empty body)."""
    body = exc.details.get("body", "")
    err = desc = None
    if isinstance(body, str) and body.strip():
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            err = data.get("error")
            desc = data.get("error_description")
    if err and desc:
        return f"{err}: {desc}"
    if err:
        return str(err)
    if desc:
        return str(desc)
    return None


def _record_fatal_refresh(exc: OAuthRefreshError) -> None:
    _last_fatal.reason = _describe_token_error(exc)


def take_last_fatal_refresh_reason() -> str | None:
    """Return (and clear) the reason recorded by the most recent fatal refresh
    on this thread. One-shot: a second call returns ``None`` so a stale reason
    can never attach to an unrelated later failure."""
    reason = getattr(_last_fatal, "reason", None)
    _last_fatal.reason = None
    return reason


def ensure_fresh_session(*, leeway_s: int = 60, force: bool = False, allow_clear: bool = True):
    """Return the stored cloud session, refreshing it first when the access
    token is expired (or within ``leeway_s`` of expiring) and a refresh token
    is available.

    The access-token lifetime is set by the auth server (short by design);
    this spends the long-lived refresh token to keep the *session* alive, so a
    user who keeps working within the refresh window never has to re-run
    ``comfy cloud login``. Best-effort: on a *transient* refresh failure it
    returns the existing (stale) session so the caller's own expiry check still
    fires. Returns ``None`` when there is no session at all — or when a refresh
    hit reuse-detection / ``invalid_grant`` (the family is dead, so the stored
    session is cleared and the caller surfaces the ``cloud_unauthorized``
    "run ``comfy cloud login``" guidance).

    ``force=True`` refreshes regardless of the *local* expiry check. Use it on
    the reactive path after a server 401 (which is authoritative): the access
    token has been rejected even if our clock — skewed, or with no
    ``expires_at`` recorded — still thinks it is valid. The refresh token is
    still required; without one this is a no-op.

    ``allow_clear=False`` keeps the stored session on disk even when a refresh
    hits a fatal token error (reuse-detection / ``invalid_grant``). Background,
    read-mostly callers (the detached ``comfy run`` watcher) pass this so a
    single transient blip — or a *spurious* invalid_grant from a token replay —
    can never wipe the login out from under the foreground command that owns the
    session lifecycle. The refresh still returns ``None`` so the caller knows it
    could not freshen the token; it just declines to destroy the shared session.

    Concurrency: the auth server rotates the refresh token on every refresh and
    invalidates the old one. A parallel ``comfy run`` fan-out (plus background
    ``jobs watch`` processes) would otherwise have N processes each load the
    same stored token and POST refresh concurrently — the first wins, the rest
    replay a consumed token and trip reuse-detection, killing the whole family.
    To prevent that, the *decide → POST → persist* sequence runs under a
    cross-process file lock (the same lock the secret store uses) with a
    double-check after acquisition: if a peer already rotated the token while
    we waited, we use the freshly persisted one instead of spending ours again.
    """
    from comfy_cli.auth import store as auth_store

    session = auth_store.get_cloud_session()
    if session is None or not session.refresh_token:
        return session
    if not force and not session.is_expired(leeway_s=leeway_s):
        # Fast path: token is comfortably valid. No lock, no network — keeps
        # the proactive ``resolve_target`` leg cheap on the common case.
        return session

    # A refresh is indicated. Serialize the whole decide→refresh→persist across
    # processes so concurrent callers coalesce into a single network refresh
    # and a rotated refresh token is never replayed by a second waiter.
    from comfy_cli import locking

    observed_refresh_token = session.refresh_token
    try:
        with locking.file_lock(auth_store.lock_path(), timeout=_REFRESH_LOCK_TIMEOUT_S):
            return _locked_refresh(
                auth_store,
                observed_refresh_token=observed_refresh_token,
                leeway_s=leeway_s,
                force=force,
                allow_clear=allow_clear,
            )
    except TimeoutError:
        # A peer has held the lock longer than a refresh should take. Don't
        # race it (a refresh outside the lock could replay a token a peer is
        # mid-rotation on) and don't re-read the store here — get_cloud_session
        # re-acquires this same lock with no timeout and could block past our
        # budget. Return the pre-lock session; the caller's own expiry check
        # decides, and at worst one stale-token 401 surfaces — the session is
        # preserved, never wiped.
        return session


def _locked_refresh(auth_store, *, observed_refresh_token: str, leeway_s: int, force: bool, allow_clear: bool = True):
    """Decide → POST → persist, executed while holding the secrets lock.

    Re-reads the persisted session first (double-checked locking): if a peer
    rotated the refresh token while we waited for the lock, we adopt the new
    one rather than spending the now-consumed token we observed before locking.
    """
    session = auth_store.get_cloud_session()
    if session is None or not session.refresh_token:
        return session

    # Double-check after acquiring the lock. A peer may have refreshed while we
    # blocked: if the stored refresh token rotated away from what we observed
    # before locking, the family already advanced — use it, never re-send the
    # consumed token (that is exactly what trips reuse-detection).
    if session.refresh_token != observed_refresh_token:
        return session
    if not force and not session.is_expired(leeway_s=leeway_s):
        # Peer advanced the access token's expiry without us noticing a token
        # rotation (e.g. same token, fresh enough now). Nothing to do.
        return session

    from comfy_cli.cloud import CLIENT_ID, get_resource_url

    try:
        new_tokens = refresh_tokens(
            base_url=session.base_url,
            client_id=session.client_id or CLIENT_ID,
            refresh_token=session.refresh_token,
            resource=session.resource or get_resource_url(),
        )
    except OAuthRefreshError as e:
        if _is_fatal_token_error(e):
            # Reuse-detected / invalid_grant: as far as the token endpoint is
            # concerned the family is dead. Drop the stored tokens so they're
            # never replayed in a loop, and signal the caller (via ``None``) to
            # surface the cloud_unauthorized guidance.
            #
            # Record the server's actual error_description (e.g. "workspace
            # membership lost", "resource state changed", "refresh token reuse
            # detected") so the caller can report *why* instead of a canned
            # guess — these look identical to the user otherwise but point at
            # very different root causes.
            #
            # ``allow_clear=False`` (background watcher) declines to wipe the
            # shared session: the foreground command owns its lifecycle, and a
            # spurious invalid_grant from a transient blip must not log the user
            # off mid-run. We still return ``None`` so the watcher knows the
            # refresh did not produce a usable token.
            _record_fatal_refresh(e)
            if allow_clear:
                auth_store.clear_cloud_session()
            return None
        return session  # transient (network) — keep the stale session
    except Exception:  # noqa: BLE001 — any other error is best-effort
        return session

    # Atomic persist of the rotated family before the lock is released, so the
    # next process reads the new tokens (save_cloud_session writes tmp+rename).
    #
    # This server rotates the refresh token on every successful refresh and does
    # reuse detection — the whole lock design above relies on that. So once the
    # refresh succeeds the token we sent is *spent*; we must NOT fall back to
    # re-persisting it if the response somehow omits a new one (an old
    # ``... or session.refresh_token`` fallback). Re-saving the consumed token
    # would guarantee an ``invalid_grant`` (and a forced logout) on the very
    # next use. Persisting ``None`` instead keeps the user authenticated on the
    # fresh access token and degrades to a clean re-login when it expires,
    # rather than tripping reuse-detection mid-use.
    _last_fatal.reason = None  # success — drop any reason from an earlier attempt
    return auth_store.save_cloud_session(
        base_url=session.base_url,
        resource=session.resource,
        client_id=session.client_id,
        scope=session.scope,
        access_token=new_tokens.access_token,
        refresh_token=new_tokens.refresh_token,
        token_type=new_tokens.token_type,
        expires_at=new_tokens.expires_at,
    )


# ---------------------------------------------------------------------------
# Tiny HTTP helpers (stdlib only — no external deps)
# ---------------------------------------------------------------------------


class _HTTPFail(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


def _post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    return _send_and_parse(req)


def _post_form(url: str, body: dict) -> dict:
    data = urllib.parse.urlencode(body).encode("ascii")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    return _send_and_parse(req)


def _assert_https_or_loopback(url: str) -> None:
    """OAuth carries client_secrets and authorization codes — refuse cleartext.

    Loopback is exempt (no wire to sniff); everything else must be https.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return
    raise _HTTPFail(0, f"refusing plaintext HTTP for OAuth endpoint: {url}")


_OAUTH_OPENER = urllib.request.build_opener(NoRedirectHandler())


def _send_and_parse(req: urllib.request.Request) -> dict:
    _assert_https_or_loopback(req.full_url)
    try:
        with _OAUTH_OPENER.open(req, timeout=_HTTP_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace") or "{}"
            return json.loads(raw)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise _HTTPFail(e.code, body) from None
    except URLError as e:
        raise _HTTPFail(0, str(e)) from None
    except json.JSONDecodeError as e:
        raise _HTTPFail(200, f"non-JSON response body: {e}") from None
