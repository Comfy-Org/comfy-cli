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
            self.wfile.write(payload)
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

    # 3. Stand up the loopback server before we send the URL.
    port = _pick_free_port()
    redirect_uri = f"http://{_LOOPBACK_HOST}:{port}{_CALLBACK_PATH}"
    capture = _CallbackCapture()
    success_html = _SUCCESS_HTML.replace("__HOST__", urllib.parse.urlsplit(base_url).netloc or base_url)
    handler_cls = _build_handler(
        expected_state=state,
        capture=capture,
        success_html=success_html,
        failure_html=_FAILURE_HTML,
    )
    server = http.server.HTTPServer((_LOOPBACK_HOST, port), handler_cls)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


# Refuse redirects: an evil 302 from the token endpoint to attacker.example
# would replay the verifier + code at the redirect target.
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def http_error_301(self, req, fp, code, msg, headers):
        raise HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_303 = http_error_307 = http_error_308 = http_error_301


_OAUTH_OPENER = urllib.request.build_opener(_NoRedirect())


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
