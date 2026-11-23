"""The renewable partner-node token a signed-in local run hands to ComfyUI.

The sign-in's access token lasts fifteen minutes and ComfyUI cannot renew it. A
partner-node token is minted from the sign-in, lasts about 90 minutes, and ComfyUI
renews it through comfy-api for its session's seven days. One token is cached next
to the stored sign-in, for that sign-in only, so runs share one server session.

Callers hold the secrets lock (``auth_store.lock_path()``) around :func:`current`.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import time
import urllib.error
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from comfy_cli.auth import store as auth_store

if TYPE_CHECKING:
    from collections.abc import Callable

    from comfy_cli.auth.store import CloudSession

AUDIENCE = "comfy-partner-node"
RENEW_WITHIN_S = 45 * 60
RETRY_MINT_AFTER_S = 10 * 60
_STILL_VALID_S = 60
_SIGN_IN_LIFETIME_S = 15 * 60
_LATEST_EPOCH_S = 10**11
_HTTP_TIMEOUT_S = 15.0
_MAX_RESPONSE_BYTES = 64 * 1024
_COMFY_API_BASES = {
    "https://cloud.comfy.org": "https://api.comfy.org",
    "https://stagingcloud.comfy.org": "https://stagingapi.comfy.org",
    "https://testcloud.comfy.org": "https://testapi.comfy.org",
}

_log = logging.getLogger(__name__)


class PartnerTokenError(Exception):
    def __init__(self, status: int, code: str | None = None) -> None:
        super().__init__(f"HTTP {status} {code}" if code else f"HTTP {status}")
        self.status = status
        self.code = code


@dataclass(frozen=True)
class _Token:
    value: str
    sid: str
    expires_at: float
    lifetime: float | None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"_Token(value=***, expires_at={self.expires_at!r})"

    def left(self) -> float:
        return self.expires_at - time.time()

    def shorter_than(self, other: _Token) -> bool:
        """Issued for less time than ``other``: renewal stopped at its session's end."""
        if self.lifetime is None or other.lifetime is None:
            return False
        return self.lifetime < other.lifetime - _STILL_VALID_S


def _claims(token: object) -> dict[str, Any] | None:
    parts = token.split(".") if isinstance(token, str) else []
    if len(parts) != 3:
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except (ValueError, RecursionError):
        return None
    return claims if isinstance(claims, dict) else None


def _parse(token: object) -> _Token | None:
    claims = _claims(token)
    if claims is None:
        return None
    audience = claims.get("aud")
    sid, exp = claims.get("sid"), claims.get("exp")
    if AUDIENCE not in (audience if isinstance(audience, list) else [audience]):
        return None
    if not isinstance(sid, str) or not sid or not _epoch(exp):
        return None
    iat = claims.get("iat")
    lifetime = float(exp - iat) if _epoch(iat) else None
    return _Token(token, sid, float(exp), lifetime)


def _epoch(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < _LATEST_EPOCH_S


def login_key(session: CloudSession | None) -> dict[str, str] | None:
    """The sign-in a cached token belongs to, or ``None`` when this sign-in gets no partner-node token."""
    if session is None or session.base_url not in _COMFY_API_BASES:
        return None
    claims = _claims(session.access_token) or {}
    key = {
        "base_url": session.base_url,
        "client": claims.get("client_id"),
        "user": claims.get("sub"),
        "workspace": claims.get("workspace_id"),
        "family": claims.get("fam"),
    }
    return key if all(isinstance(value, str) and value for value in key.values()) else None


def _error_code(error: urllib.error.HTTPError) -> str | None:
    try:
        body = json.loads(error.read(4096) or b"null")
    except (OSError, http.client.HTTPException, ValueError, RecursionError):
        return None
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if isinstance(code, str):
        return code
    nested = body.get("error")
    return nested.get("type") if isinstance(nested, dict) and isinstance(nested.get("type"), str) else None


def _post(url: str, bearer: str, body: dict | None) -> dict | None:
    from comfy_cli.http import ResponseTooLarge, request_json
    from comfy_cli.target import Target

    target = Target(kind="cloud", base_url=url, auth_token=bearer)
    try:
        _status, payload = request_json(
            url, target, method="POST", body=body, timeout=_HTTP_TIMEOUT_S, max_bytes=_MAX_RESPONSE_BYTES
        )
    except urllib.error.HTTPError as error:
        raise PartnerTokenError(error.code, _error_code(error)) from None
    except (OSError, http.client.HTTPException, ResponseTooLarge, ValueError) as error:
        raise PartnerTokenError(0, type(error).__name__) from None
    return payload if isinstance(payload, dict) else None


def _mint(session: CloudSession) -> _Token:
    payload = _post(f"{session.base_url}/api/auth/token", session.access_token, {"resource": "partner-node"})
    token = _parse((payload or {}).get("token"))
    if token is None:
        raise PartnerTokenError(200, "unusable_token")
    return token


def _renew(base_url: str, held: _Token) -> _Token:
    payload = _post(f"{_COMFY_API_BASES[base_url]}/auth/partner-node/renew", held.value, None)
    token = _parse((payload or {}).get("token"))
    if token is None or token.sid != held.sid:
        raise PartnerTokenError(200, "unusable_token")
    return token


def _record(key: dict[str, str]) -> dict[str, Any]:
    record = auth_store.get_partner_node_token()
    return record if isinstance(record, dict) and record.get("login") == key else {"login": key}


def _held(key: dict[str, str]) -> _Token | None:
    return _parse(_record(key).get("token"))


def _mint_failed_lately(key: dict[str, str]) -> bool:
    failed_at = _record(key).get("mint_failed_at")
    return _epoch(failed_at) and 0 <= time.time() - failed_at < RETRY_MINT_AFTER_S


def _keep(key: dict[str, str], token: _Token) -> None:
    auth_store.save_partner_node_token({"login": key, "token": token.value})


def current(session: CloudSession, sign_in: Callable[[], CloudSession | None]) -> str | None:
    """The partner-node token for ``session``'s next run, or ``None`` to send the sign-in token instead.

    The cached token is sent while it has ``RENEW_WITHIN_S`` left and renewed
    after that. When renewal fails, it is still sent while it outlives a fresh
    sign-in token. A new one is minted when renewal is refused (401/403), stops
    at the session's end, or fails for a shorter-lived token; a 429 never mints.
    No mint is made into a store that cannot be written, since it would be lost
    and made again on every run, nor within ``RETRY_MINT_AFTER_S`` of a failed
    mint for this sign-in. ``sign_in`` returns the live sign-in a mint is made
    from.
    """
    key = login_key(session)
    if key is None:
        return None
    held = _held(key)
    capped = None
    if held is not None:
        if held.left() >= RENEW_WITHIN_S:
            return held.value
        try:
            renewed = _renew(session.base_url, held)
        except PartnerTokenError as error:
            _log.info("partner-node token renewal failed: %s", error)
            if error.status in (401, 403):
                auth_store.save_partner_node_token({k: v for k, v in _record(key).items() if k != "token"})
            elif held.left() > _SIGN_IN_LIFETIME_S:
                return held.value
            elif error.status == 429:
                return None
        else:
            if not renewed.shorter_than(held):
                _keep(key, renewed)
                return renewed.value
            capped = renewed
    if _mint_failed_lately(key):
        _log.info("not minting a partner-node token: a mint failed in the last %d minutes", RETRY_MINT_AFTER_S // 60)
        return _outlasting_a_sign_in(capped)
    live = sign_in()
    if live is None or live.is_expired() or login_key(live) != key:
        return _outlasting_a_sign_in(capped)
    try:
        auth_store.rewrite()
    except OSError as error:
        _log.info("not minting a partner-node token the store cannot keep: %s", type(error).__name__)
        return _outlasting_a_sign_in(capped)
    try:
        minted = _mint(live)
    except PartnerTokenError as error:
        _log.info("partner-node token mint failed: %s", error)
        auth_store.save_partner_node_token({**_record(key), "mint_failed_at": time.time()})
        return _outlasting_a_sign_in(capped)
    _keep(key, minted)
    return minted.value


def _outlasting_a_sign_in(token: _Token | None) -> str | None:
    return token.value if token is not None and token.left() > _SIGN_IN_LIFETIME_S else None


def revoke_replaced(record: object) -> str | None:
    """Revoke the cached token of a sign-in being replaced, and only its own session.

    The sign-in's other sessions, its Cloud jobs among them, keep renewing: only
    ``comfy cloud logout`` ends them. ``"token"`` or ``"failed"``; ``None`` when no
    token was cached, and then no request is made.
    """
    record = record if isinstance(record, dict) else None
    login = record.get("login") if record is not None else None
    held = _parse(record.get("token")) if record is not None else None
    base_url = login.get("base_url") if isinstance(login, dict) else None
    if held is None or base_url not in _COMFY_API_BASES:
        return None
    return "token" if _revoke(base_url, held.value) else "failed"


def warn_if_replaced_token_kept(renderer: Any, revocation: str | None) -> None:
    if revocation == "failed":
        renderer.warn(
            "Could not revoke the partner-node token of the replaced sign-in. A local ComfyUI that ran "
            "with it can keep renewing it for up to 7 days; restart that ComfyUI to drop it."
        )


def _revoke(base_url: str, bearer: str) -> bool:
    try:
        _post(f"{base_url}/api/auth/token/revoke", bearer, None)
    except PartnerTokenError as error:
        _log.info("partner-node token revoke failed: %s", error)
        return False
    return True


def _access_token(session: CloudSession) -> str | None:
    """A live access token for ``session``, refreshed in memory only."""
    if session.access_token and not session.is_expired():
        return session.access_token
    if not session.refresh_token:
        return None
    from comfy_cli.cloud import CLIENT_ID, oauth

    try:
        tokens = oauth.refresh_tokens(
            base_url=session.base_url,
            client_id=session.client_id or CLIENT_ID,
            refresh_token=session.refresh_token,
            resource=session.resource or None,
            timeout=_HTTP_TIMEOUT_S,
        )
    except Exception as error:  # noqa: BLE001 — a revoke after sign-out is best-effort
        _log.info("could not refresh the sign-in to revoke its partner-node tokens: %s", error)
        return None
    return tokens.access_token


def revoke_sessions(session: CloudSession | None, record: object) -> str | None:
    """Revoke every partner-node session of a sign-in that logout dropped from the store.

    Called after the store dropped ``session`` and its cached ``record``, so an
    interrupted or slow revoke never leaves it signed in. The sessions of Cloud
    jobs submitted with the sign-in are among those revoked. The sign-in's access
    token, refreshed in memory when it has expired, revokes every session of that
    sign-in. A refused refresh, even ``invalid_grant``, does not prove the sign-in
    ended, so a cached token it does not cover still revokes its own session.

    ``"sign_in"``: every session of the sign-in is revoked.
    ``"token"``: only the cached token's own session was revoked.
    ``"failed"``: nothing could be revoked. ``None``: there was no sign-in or token.
    """
    record = record if isinstance(record, dict) else None
    login = record.get("login") if record is not None else None
    held = _parse(record.get("token")) if record is not None else None
    tried = False
    if session is not None and session.base_url in _COMFY_API_BASES:
        tried = True
        bearer = _access_token(session)
        if bearer is not None and _revoke(session.base_url, bearer) and (record is None or login == login_key(session)):
            return "sign_in"
    base_url = login.get("base_url") if isinstance(login, dict) else None
    if held is not None and base_url in _COMFY_API_BASES:
        tried = True
        if _revoke(base_url, held.value):
            return "token"
    return "failed" if tried else None
