"""Plaintext (Phase 4) secret store for comfy-cli.

API:

    list_records() -> list[AuthRecord]
    get(provider) -> AuthRecord | None
    set(provider, key) -> AuthRecord    # creates or updates
    remove(provider) -> bool            # True if removed, False if absent

Locking uses :mod:`comfy_cli.locking` so concurrent ``auth set`` calls don't
clobber each other.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli import constants, locking
from comfy_cli.utils import get_os

# Comfy Cloud is *not* listed here: it is authenticated via OAuth (see
# `cloud_session` below and `comfy_cli.cloud.oauth`). Only providers whose
# native authentication is an API key remain.
SUPPORTED_PROVIDERS = ("civitai", "huggingface")


@dataclass(frozen=True)
class AuthRecord:
    provider: str
    key: str
    updated_at: str

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if redact:
            d["key"] = _redact(self.key)
            d["key_redacted"] = True
        else:
            d["key_redacted"] = False
        return d


def _redact(key: str) -> str:
    if len(key) <= 16:
        return "***"
    return f"{key[:4]}…{key[-4:]}"


SECRETS_PATH_ENV = "COMFY_SECRETS_PATH"


def secrets_path() -> Path:
    override = os.environ.get(SECRETS_PATH_ENV)
    if override:
        return Path(override)
    base = Path(constants.DEFAULT_CONFIG[get_os()])
    return base / "secrets.json"


def lock_path() -> Path:
    """Stable sidecar path that all secret-store mutations serialize on.

    The lock MUST NOT be the data file itself: ``_write_all`` persists via
    ``os.replace(tmp, secrets.json)``, which swaps in a *new inode* on every
    write. ``fcntl.flock`` is bound to the open file description (the inode),
    so if processes locked ``secrets.json`` directly, a holder mid-refresh
    would be flocking the now-unlinked old inode while a newcomer opens — and
    flocks — the freshly-renamed inode. The two would run the critical section
    simultaneously, replay the same rotated refresh token, and trip the auth
    server's reuse-detection (wiping the whole token family). A dedicated
    ``secrets.json.lock`` file is created once and never replaced, so every
    process serializes on one stable inode regardless of how many times the
    data file is atomically rewritten underneath it.
    """
    p = secrets_path()
    return p.with_name(p.name + ".lock")


_EMPTY: dict[str, Any] = {"providers": {}, "cloud_session": None}


def _read_all(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"providers": {}, "cloud_session": None}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {"providers": {}, "cloud_session": None}
    if not raw.strip():
        return {"providers": {}, "cloud_session": None}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Don't blow up on a corrupt store; return empty and let `set` rewrite.
        return {"providers": {}, "cloud_session": None}
    providers = data.get("providers") if isinstance(data, dict) else None
    cloud_session = data.get("cloud_session") if isinstance(data, dict) else None
    return {
        "providers": providers if isinstance(providers, dict) else {},
        "cloud_session": cloud_session if isinstance(cloud_session, dict) else None,
    }


def _write_all(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write with 0600 mode from inception.

    The tmp file is opened with ``O_CREAT|O_EXCL`` and explicit mode 0o600 so
    the secrets are never world-readable, even briefly, even on systems with a
    permissive umask. A unique tmp name (`<basename>.<pid>.<rand>.tmp`) avoids
    collisions with stale tmp files from killed processes.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Best-effort tighten parent directory if it already existed permissively.
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    # Unique tmp name per writer; survives kill -9 (the orphan is harmless).
    tmp_suffix = f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
    tmp = path.with_suffix(path.suffix + tmp_suffix)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # Open-with-mode is atomic w.r.t. permissions on POSIX; on Windows the
    # mode bit is mostly cosmetic but we still set it for consistency.
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        # Clean up the tmp file if we didn't make it to os.replace.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Final defensive chmod in case the inode is being reused with looser perms.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def list_records() -> list[AuthRecord]:
    path = secrets_path()
    with locking.file_lock(lock_path()):
        data = _read_all(path)
    out: list[AuthRecord] = []
    for name, body in data["providers"].items():
        if not isinstance(body, dict):
            continue
        key = body.get("key")
        if not isinstance(key, str):
            continue
        out.append(AuthRecord(provider=name, key=key, updated_at=str(body.get("updated_at", ""))))
    return sorted(out, key=lambda r: r.provider)


def get(provider: str) -> AuthRecord | None:
    path = secrets_path()
    with locking.file_lock(lock_path()):
        data = _read_all(path)
    body = data["providers"].get(provider)
    if not isinstance(body, dict):
        return None
    key = body.get("key")
    if not isinstance(key, str) or not key:
        return None
    return AuthRecord(provider=provider, key=key, updated_at=str(body.get("updated_at", "")))


_SAFE_PROVIDER = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def set(provider: str, key: str) -> AuthRecord:
    if not provider:
        raise ValueError("provider name is required")
    if not _SAFE_PROVIDER.match(provider):
        raise ValueError(f"invalid provider name: {provider!r} (must be alphanumeric/dash/underscore, max 64 chars)")
    if not key:
        raise ValueError("key cannot be empty")
    path = secrets_path()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with locking.file_lock(lock_path()):
        data = _read_all(path)
        data["providers"][provider] = {"key": key, "updated_at": now}
        _write_all(path, data)
    return AuthRecord(provider=provider, key=key, updated_at=now)


def remove(provider: str) -> bool:
    path = secrets_path()
    with locking.file_lock(lock_path()):
        data = _read_all(path)
        if provider not in data["providers"]:
            return False
        del data["providers"][provider]
        _write_all(path, data)
    return True


def known_providers() -> Iterable[str]:
    return SUPPORTED_PROVIDERS


# ---------------------------------------------------------------------------
# Cloud OAuth session
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudSession:
    base_url: str
    resource: str
    client_id: str
    scope: str
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_at: int | None  # absolute epoch seconds
    saved_at: str

    def is_expired(self, *, leeway_s: int = 30) -> bool:
        """True if the access token is past (or within `leeway_s` of) expiry."""
        if self.expires_at is None:
            return False
        import time as _time

        return _time.time() + leeway_s >= self.expires_at

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        data = {
            "base_url": self.base_url,
            "resource": self.resource,
            "client_id": self.client_id,
            "scope": self.scope,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "saved_at": self.saved_at,
        }
        if redact:
            data["access_token"] = _redact(self.access_token) if self.access_token else None
            data["refresh_token"] = _redact(self.refresh_token) if self.refresh_token else None
            data["tokens_redacted"] = True
        else:
            data["access_token"] = self.access_token
            data["refresh_token"] = self.refresh_token
            data["tokens_redacted"] = False
        return data


def _session_from_dict(body: dict[str, Any]) -> CloudSession | None:
    if not isinstance(body, dict):
        return None
    tokens = body.get("tokens")
    if not isinstance(tokens, dict):
        return None
    access = tokens.get("access_token")
    if not isinstance(access, str) or not access:
        return None
    refresh = tokens.get("refresh_token") if isinstance(tokens.get("refresh_token"), str) else None
    return CloudSession(
        base_url=str(body.get("base_url", "")),
        resource=str(body.get("resource", "")),
        client_id=str(body.get("client_id", "")),
        scope=str(body.get("scope", "")),
        access_token=access,
        refresh_token=refresh,
        token_type=str(tokens.get("token_type", "Bearer")),
        expires_at=tokens.get("expires_at") if isinstance(tokens.get("expires_at"), int) else None,
        saved_at=str(body.get("saved_at", "")),
    )


def get_cloud_session() -> CloudSession | None:
    path = secrets_path()
    with locking.file_lock(lock_path()):
        data = _read_all(path)
    session = data.get("cloud_session")
    if not isinstance(session, dict):
        return None
    return _session_from_dict(session)


def save_cloud_session(
    *,
    base_url: str,
    resource: str,
    client_id: str,
    scope: str,
    access_token: str,
    refresh_token: str | None,
    token_type: str,
    expires_at: int | None,
) -> CloudSession:
    if not access_token:
        raise ValueError("access_token cannot be empty")
    path = secrets_path()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "base_url": base_url,
        "resource": resource,
        "client_id": client_id,
        "scope": scope,
        "saved_at": now,
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": token_type,
            "expires_at": expires_at,
        },
    }
    with locking.file_lock(lock_path()):
        data = _read_all(path)
        data["cloud_session"] = payload
        _write_all(path, data)
    return CloudSession(
        base_url=base_url,
        resource=resource,
        client_id=client_id,
        scope=scope,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type=token_type,
        expires_at=expires_at,
        saved_at=now,
    )


def clear_cloud_session() -> bool:
    path = secrets_path()
    with locking.file_lock(lock_path()):
        data = _read_all(path)
        if data.get("cloud_session") is None:
            return False
        data["cloud_session"] = None
        _write_all(path, data)
    return True
