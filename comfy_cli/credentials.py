"""The ONE credential resolver for Comfy Cloud / partner-API auth.

Historically four call sites each hand-rolled an OAuth-first credential
chain (cloud target resolution, local partner-node injection, the generate
partner-API proxy, and ``cloud whoami``) and they drifted. They all share a
single precedence order — only small per-site knobs differ — so the chain
lives here exactly once:

    explicit flag → live OAuth session → purpose env var → stored key

Two *purposes* exist and their credentials are NOT interchangeable:

- ``"cloud"``   — the Comfy Cloud platform API (Bearer / ``X-API-Key`` on
  cloud.comfy.org). Env var: ``COMFY_CLOUD_API_KEY``. Values are passed
  verbatim (no stripping), matching the historical target-resolution chain.
- ``"partner"`` — partner-API nodes / the partner proxy
  (``api_key_comfy_org``). Env var: ``COMFY_API_KEY``. Ambient values are
  whitespace-stripped and whitespace-only values are treated as absent,
  matching the historical ``comfy generate`` chain.

Both purposes fall back to the same stored key (provider
``comfy-cloud-api-key``, persisted via the hidden ``comfy cloud set-key``).

This module is also the only sanctioned *read* gateway to the OAuth session
(:func:`get_session`) and the ambient API key (:func:`find_api_key`) — a
ratchet test (``tests/comfy_cli/test_credentials.py``) rejects any new
direct ``os.environ``/``get_cloud_session``/``ensure_fresh_session`` reads
elsewhere in the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from comfy_cli.auth.store import CloudSession

# Provider name under which a Comfy Cloud API key is persisted in the auth
# store. Hidden / testing-only path; the canonical sign-in is OAuth.
# (Re-exported by ``comfy_cli.target`` for back-compat.)
CLOUD_API_KEY_PROVIDER = "comfy-cloud-api-key"

Purpose = Literal["cloud", "partner"]

# purpose → (env var, stored-key provider, strip ambient values?)
_PURPOSES: dict[str, tuple[str, str, bool]] = {
    "cloud": ("COMFY_CLOUD_API_KEY", CLOUD_API_KEY_PROVIDER, False),
    "partner": ("COMFY_API_KEY", CLOUD_API_KEY_PROVIDER, True),
}


@dataclass(frozen=True)
class Credential:
    kind: Literal["oauth", "api_key"]
    value: str
    source: str  # "flag" | "session" | "env:<VAR>" | "stored:<provider>"

    # Never leak the secret into logs / pytest failure dumps.
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Credential(kind={self.kind!r}, value=***, source={self.source!r})"


def get_session(*, refresh: bool = True, force: bool = False, allow_clear: bool = True) -> CloudSession | None:
    """Read the stored Comfy Cloud OAuth session.

    ``refresh=True`` goes through ``ensure_fresh_session`` (spends the
    refresh token when the access token is expired / near expiry);
    ``refresh=False`` reads the store as-is, possibly returning an expired
    session — callers that only display state, or that must never touch the
    network, want this.

    ``force=True`` (implies the refresh path) refreshes unconditionally,
    ignoring the local expiry check. Reserved for the *reactive* path: after a
    server 401, the token is known-rejected even if our clock disagrees.

    ``allow_clear=False`` forwards to ``ensure_fresh_session`` so a fatal
    refresh failure does NOT clear the stored session. Background watchers pass
    this — they are read-mostly and must never log the user off the shared
    session; only foreground, user-driven commands own that lifecycle.
    """
    if refresh or force:
        from comfy_cli.cloud import oauth

        return oauth.ensure_fresh_session(force=force, allow_clear=allow_clear)
    from comfy_cli.auth import store as auth_store

    return auth_store.get_cloud_session()


def find_api_key(*, purpose: Purpose) -> Credential | None:
    """Locate an ambient API key for ``purpose``: env var → stored key.

    Ignores any OAuth session — use this for presence checks (e.g. whoami's
    "API key present but outranked by the session" note) and as the tail of
    :func:`resolve_cloud_credential`.
    """
    import os

    env_var, provider, strip = _PURPOSES[purpose]

    env_value = os.environ.get(env_var)
    if env_value is not None:
        candidate = env_value.strip() if strip else env_value
        if candidate:
            return Credential(kind="api_key", value=candidate, source=f"env:{env_var}")

    from comfy_cli.auth import store as auth_store

    record = auth_store.get(provider)
    stored = getattr(record, "key", None) if record is not None else None
    if stored:
        candidate = stored.strip() if strip else stored
        if candidate:
            return Credential(kind="api_key", value=candidate, source=f"stored:{provider}")

    return None


def resolve_cloud_credential(
    *,
    purpose: Purpose,
    explicit: str | None = None,
    base_url: str | None = None,
    refresh: bool = True,
) -> Credential | None:
    """Resolve the active credential for ``purpose``, or ``None``.

    Precedence (OAuth-first — API keys are on a deprecation path; only a
    deliberate per-call flag outranks a live session):

    1. ``explicit`` flag value (whitespace-stripped; blank → ignored).
    2. Live (non-expired) OAuth session. Refreshed first when
       ``refresh=True``; read as-is when ``refresh=False``. When
       ``base_url`` is given, a session minted for a *different* base URL is
       skipped (replay-guard: never send a token to a host the user didn't
       authenticate against).
    3. The purpose's env var (``COMFY_CLOUD_API_KEY`` / ``COMFY_API_KEY``).
    4. The stored ``comfy-cloud-api-key`` key (``comfy cloud set-key``).
    """
    explicit_key = explicit.strip() if isinstance(explicit, str) else ""
    if explicit_key:
        return Credential(kind="api_key", value=explicit_key, source="flag")

    session = get_session(refresh=refresh)
    if (
        session is not None
        and not session.is_expired()
        and session.access_token
        and (base_url is None or session.base_url == base_url)
    ):
        return Credential(kind="oauth", value=session.access_token, source="session")

    return find_api_key(purpose=purpose)
