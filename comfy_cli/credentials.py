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
``comfy-cloud-api-key``, persisted via ``comfy cloud set-key``).

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
# store. Testing-only path; the canonical sign-in is OAuth.
# (Re-exported by ``comfy_cli.target`` for back-compat.)
CLOUD_API_KEY_PROVIDER = "comfy-cloud-api-key"

# Env var carrying a pre-obtained Comfy Cloud Bearer token (a Firebase/Cloud
# JWT). Unlike ``COMFY_CLOUD_API_KEY`` (sent as ``X-API-Key``), this is sent as
# ``Authorization: Bearer``. It exists so a trusted caller that already holds
# the user's validated token — e.g. the cloud agent forwarding the request's
# ``X-Comfy-Token`` — can authenticate as that user without an interactive
# ``comfy cloud login`` session. It is NOT refreshed client-side; the server
# validates it at request time (and a 401 surfaces normally).
CLOUD_BEARER_ENV_VAR = "COMFY_CLOUD_AUTH_TOKEN"

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


def cloud_bearer_env_token() -> str | None:
    """Return a forwarded Comfy Cloud Bearer token from the environment, or None.

    Reads ``COMFY_CLOUD_AUTH_TOKEN`` (see :data:`CLOUD_BEARER_ENV_VAR`). Cloud-only
    — it authenticates as the token's user via ``Authorization: Bearer``.
    """
    import os

    tok = os.environ.get(CLOUD_BEARER_ENV_VAR)
    return tok.strip() if tok and tok.strip() else None


def resolve_cloud_credential(
    *,
    purpose: Purpose,
    explicit: str | None = None,
    base_url: str | None = None,
    refresh: bool = True,
    allow_clear: bool = True,
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
    3. (cloud only) A forwarded Bearer token in ``COMFY_CLOUD_AUTH_TOKEN``,
       sent as ``Authorization: Bearer`` — the trusted-caller path (see
       :data:`CLOUD_BEARER_ENV_VAR`).
    4. The purpose's env var (``COMFY_CLOUD_API_KEY`` / ``COMFY_API_KEY``).
    5. The stored ``comfy-cloud-api-key`` key (``comfy cloud set-key``).

    ``allow_clear=False`` is forwarded into :func:`get_session` (and on to
    ``ensure_fresh_session``) so a fatal refresh error on THIS call does not
    wipe the shared stored session. Best-effort, read-mostly callers that must
    never log the user off the shared session (e.g. the local partner-node
    injector) pass this; a failed refresh then just falls through to the
    env/stored-key tail instead of destroying the login.
    """
    explicit_key = explicit.strip() if isinstance(explicit, str) else ""
    if explicit_key:
        return Credential(kind="api_key", value=explicit_key, source="flag")

    session = get_session(refresh=refresh, allow_clear=allow_clear)
    if (
        session is not None
        and not session.is_expired()
        and session.access_token
        and (base_url is None or session.base_url == base_url)
    ):
        return Credential(kind="oauth", value=session.access_token, source="session")

    if purpose == "cloud":
        env_bearer = cloud_bearer_env_token()
        if env_bearer:
            return Credential(kind="oauth", value=env_bearer, source=f"env:{CLOUD_BEARER_ENV_VAR}")

    return find_api_key(purpose=purpose)


def resolve_partner_credential() -> tuple[str, str] | None:
    """The ``extra_data`` credential partner-API nodes authenticate with.

    Returns ``(field, value)`` — ``auth_token_comfy_org`` for a session,
    ``api_key_comfy_org`` for a key — or ``None`` when nothing is configured.
    Shared by the local exec and ``deploy run`` submit paths, so it consults
    every source either one already read. In precedence order:

    1. a live OAuth session;
    2. ``COMFY_CLOUD_AUTH_TOKEN`` (:data:`CLOUD_BEARER_ENV_VAR`), forwarded by a
       trusted caller instead of an interactive login;
    3. ``COMFY_API_KEY``, then ``COMFY_CLOUD_API_KEY``;
    4. the stored ``comfy cloud set-key`` key.

    Best-effort: the session is refreshed when possible, but ``allow_clear=False``
    keeps a fatal refresh error from logging the user out from under a foreground
    command, and any unexpected error falls through to a network-free read of the
    env and stored keys rather than aborting the run.
    """
    try:
        cloud = resolve_cloud_credential(purpose="cloud", refresh=True, allow_clear=False)
    except Exception:  # noqa: BLE001 — best-effort: never abort a run on a refresh hiccup
        cloud = resolve_cloud_credential(purpose="cloud", refresh=False, allow_clear=False)
    # A request-scoped credential outranks every ambient key, or the run
    # authenticates as the wrong account and spends its credits.
    if cloud is not None and cloud.kind == "oauth":
        return ("auth_token_comfy_org", cloud.value)
    key_credential = _partner_api_key()
    if key_credential is None:
        return None
    return ("api_key_comfy_org", key_credential.value)


def _partner_api_key() -> Credential | None:
    """The API key half of :func:`resolve_partner_credential`: env vars, then stored.

    Both env vars are read here rather than through two :func:`find_api_key`
    calls: that function checks *its* env var and then the stored key, so the
    first call would let machine state win over the second env var.
    """
    import os

    # Stripped, and whitespace-only treated as absent: the "cloud" env var passes
    # ambient values verbatim, and padding would reach a partner auth header.
    for env_var in (_PURPOSES["partner"][0], _PURPOSES["cloud"][0]):
        value = (os.environ.get(env_var) or "").strip()
        if value:
            return Credential(kind="api_key", value=value, source=f"env:{env_var}")
    return find_api_key(purpose="partner")
