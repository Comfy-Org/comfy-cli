"""The ONE credential resolver for Comfy Cloud / partner-API auth.

Historically four call sites each hand-rolled an OAuth-first credential
chain (cloud target resolution, local partner-node injection, the generate
partner-API proxy, and ``cloud whoami``) and they drifted. They all share a
single precedence order — only small per-site knobs differ — so the chain
lives here exactly once:

    explicit flag → credential handed in by env → live OAuth session → stored key

A credential in the environment is one the caller handed in for this run, so it
outranks a ``comfy cloud login`` left on the machine (possibly for another
account or workspace), and that sign-in is then neither refreshed nor
rewritten. Empty or whitespace-only env values count as unset, so with nothing
in the environment the stored sign-in is used exactly as before.
:func:`env_credential` is the one reader of those env vars.

``comfy build`` and ``comfy deploy`` are the one exception, key-first, because
comfy-builder and comfy-deploy swap a key at ingest on every request: see
:func:`platform_api_key`.

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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from comfy_cli.auth.store import CloudSession

# Provider name under which a Comfy Cloud API key is persisted in the auth
# store. Testing-only path; the canonical sign-in is OAuth.
# (Re-exported by ``comfy_cli.target`` for back-compat.)
CLOUD_API_KEY_PROVIDER = "comfy-cloud-api-key"

_log = logging.getLogger(__name__)

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


def refreshed_access_token(rejected: str | None, *, allow_clear: bool = True) -> str | None:
    """Force the shared refresh after a server 401, and return the new access token.

    Returns ``None`` when the refresh produced no token or the same one the
    server just refused, so the caller surfaces the 401 rather than retrying a
    token already known to fail. A fatal refresh clears the stored session, and
    the 401 then carries the usual sign-in guidance.
    """
    session = get_session(refresh=True, force=True, allow_clear=allow_clear)
    if session is None or not session.access_token or session.access_token == rejected:
        return None
    return session.access_token


class DataPlaneToken:
    """The bearer one long command sends to a deployment endpoint, kept fresh.

    The sign-in's access token lasts fifteen minutes. A command that read it once
    and sent it for a whole job (`comfy deploy run --wait`) was refused by the
    deployment as soon as the token it started with expired, though the job
    itself ran on: whichever run was in flight at each expiry lost its result.

    ``current`` re-reads the stored sign-in before each request, refreshing it
    when it is within a minute of expiring; ``after_rejection`` forces the shared
    refresh after a 401. A workspace API key, or a token handed to the client
    directly, does not expire here and is sent unchanged (``refreshes=False``).
    """

    def __init__(self, value: str, *, refreshes: bool, base_url: str | None = None, allow_clear: bool = True) -> None:
        self._value = value
        self._refreshes = refreshes
        # A sign-in to another environment, made while this command runs, is
        # never picked up in its place.
        self._base_url = base_url
        self._allow_clear = allow_clear

    @classmethod
    def for_target_token(cls, value: str, *, base_url: str, allow_clear: bool) -> DataPlaneToken:
        """Refresh ``value`` only if it is the stored sign-in's token.

        A Target does not say where its token came from, and a token injected
        through :data:`CLOUD_BEARER_ENV_VAR` must be sent unchanged even with a
        sign-in stored beside it.
        """
        if env_credential("cloud") is not None:
            # The target was built from the handed-in credential, even if its
            # value happens to equal the stored sign-in's token.
            return cls(value, refreshes=False, base_url=base_url, allow_clear=allow_clear)
        stored = get_session(refresh=False)
        is_sign_in = stored is not None and stored.access_token == value and stored.base_url == base_url
        return cls(value, refreshes=is_sign_in, base_url=base_url, allow_clear=allow_clear)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DataPlaneToken(value=***, refreshes={self._refreshes!r})"

    def _same_environment(self, session: CloudSession | None) -> bool:
        return session is not None and (self._base_url is None or session.base_url == self._base_url)

    def current(self) -> str:
        """The token to send now; the one already held when the store cannot be read or written."""
        if not self._refreshes:
            return self._value
        try:
            # The environment is checked before refreshing, so a sign-in another
            # shell stored elsewhere is never spent (or cleared) on this command's behalf.
            if self._same_environment(get_session(refresh=False)):
                session = get_session(refresh=True, allow_clear=self._allow_clear)
                if session is not None and session.access_token and self._same_environment(session):
                    self._value = session.access_token
        except OSError as error:
            # A job already running must not be abandoned over the sign-in store:
            # the token held may still be accepted, and a 401 reports itself.
            _log.warning("could not re-read the stored sign-in; keeping the current token: %s", error)
        return self._value

    def after_rejection(self) -> bool:
        """Swap in a refreshed token after a 401; ``False`` when there is none to retry with."""
        if not self._refreshes:
            return False
        try:
            if not self._same_environment(get_session(refresh=False)):
                return False
            token = refreshed_access_token(self._value, allow_clear=self._allow_clear)
        except OSError as error:
            _log.warning("could not refresh the stored sign-in after a 401: %s", error)
            return False
        if token is None:
            return False
        self._value = token
        return True


def _env_value(env_var: str, *, strip: bool = True) -> str | None:
    """``env_var``'s value, or ``None`` when it is unset, empty or whitespace-only."""
    import os

    value = os.environ.get(env_var)
    if value is None or not value.strip():
        return None
    return value.strip() if strip else value


def env_credential(purpose: Purpose) -> Credential | None:
    """The credential handed in through the environment for ``purpose``, or ``None``.

    - ``"cloud"``: ``COMFY_CLOUD_AUTH_TOKEN`` (a Cloud JWT, sent as Bearer), then
      ``COMFY_CLOUD_API_KEY`` (a workspace key, sent as ``X-API-Key``). A caller
      sets one or the other; if both are set the token wins.
    - ``"partner"``: ``COMFY_API_KEY``.

    Whatever this returns outranks a stored sign-in in every resolver here, and
    while it is set the stored sign-in is never refreshed. Empty or
    whitespace-only values count as unset.
    """
    if purpose == "cloud":
        token = cloud_bearer_env_token()
        if token:
            return Credential(kind="oauth", value=token, source=f"env:{CLOUD_BEARER_ENV_VAR}")
    env_var, _provider, strip = _PURPOSES[purpose]
    value = _env_value(env_var, strip=strip)
    if value:
        return Credential(kind="api_key", value=value, source=f"env:{env_var}")
    return None


def find_api_key(*, purpose: Purpose) -> Credential | None:
    """Locate an ambient API key for ``purpose``: env var → stored key.

    Ignores any OAuth session — use this for presence checks (e.g. whoami's
    "API key present but unused" note).
    """
    env_var, provider, strip = _PURPOSES[purpose]

    env_value = _env_value(env_var, strip=strip)
    if env_value:
        return Credential(kind="api_key", value=env_value, source=f"env:{env_var}")

    return _stored_api_key(provider, strip=strip)


def _stored_api_key(provider: str, *, strip: bool) -> Credential | None:
    """The key ``comfy cloud set-key`` saved, or ``None``."""
    from comfy_cli.auth import store as auth_store

    record = auth_store.get(provider)
    stored = getattr(record, "key", None) if record is not None else None
    if stored:
        candidate = stored.strip() if strip else stored
        if candidate:
            return Credential(kind="api_key", value=candidate, source=f"stored:{provider}")

    return None


def platform_api_key() -> Credential | None:
    """The workspace API key ``comfy build`` and ``comfy deploy`` send in place of a sign-in, or ``None``.

    comfy-builder and comfy-deploy take the key in ``X-API-Key`` and swap it at
    ingest, so a CI job holding only a key needs no sign-in. In precedence order:

    1. ``COMFY_CLOUD_API_KEY``, whatever sign-in is stored, since a caller sets it
       for the run in hand;
    2. the stored ``comfy cloud set-key`` key, only when no sign-in is stored,
       because set-key saved it as a fallback and it must not move a signed-in
       person into the key's workspace.

    ``COMFY_CLOUD_AUTH_TOKEN`` is not a key; :func:`platform_env_token` covers it
    and the clients check it first. Values are stripped, since they go into a
    header. Never refreshes the sign-in.
    """
    env_var = _PURPOSES["cloud"][0]
    env_value = _env_value(env_var)
    if env_value:
        return Credential(kind="api_key", value=env_value, source=f"env:{env_var}")
    if get_session(refresh=False) is not None:
        return None

    return _stored_api_key(CLOUD_API_KEY_PROVIDER, strip=True)


def platform_env_token() -> str | None:
    """The Cloud JWT handed in through ``COMFY_CLOUD_AUTH_TOKEN`` for ``comfy build`` / ``comfy deploy``.

    Checked before :func:`platform_api_key` and the stored sign-in, the same
    order :func:`env_credential` gives every other cloud call. Sent unchanged:
    it is never refreshed here, and a 401 surfaces to the caller.
    """
    return cloud_bearer_env_token()


def refused_key_hint(key: Credential) -> str:
    """What to do once a server refuses a :func:`platform_api_key`, naming where the key came from."""
    if key.source.startswith("env:"):
        where = f"the workspace API key in {key.source.removeprefix('env:')}"
    else:
        where = "the workspace API key saved by `comfy cloud set-key`"
    return f"{where} was refused; replace it with a valid key"


def cloud_bearer_env_token() -> str | None:
    """Return a forwarded Comfy Cloud Bearer token from the environment, or None.

    Reads ``COMFY_CLOUD_AUTH_TOKEN`` (see :data:`CLOUD_BEARER_ENV_VAR`). Cloud-only
    — it authenticates as the token's user via ``Authorization: Bearer``.
    """
    return _env_value(CLOUD_BEARER_ENV_VAR)


def resolve_cloud_credential(
    *,
    purpose: Purpose,
    explicit: str | None = None,
    base_url: str | None = None,
    refresh: bool = True,
    allow_clear: bool = True,
) -> Credential | None:
    """Resolve the active credential for ``purpose``, or ``None``.

    Precedence:

    1. ``explicit`` flag value (whitespace-stripped; blank → ignored).
    2. The credential handed in through the environment (:func:`env_credential`):
       for ``"cloud"`` a Bearer token in ``COMFY_CLOUD_AUTH_TOKEN``, then
       ``COMFY_CLOUD_API_KEY``; for ``"partner"`` ``COMFY_API_KEY``. When one is
       set the stored session is not read, so it is never refreshed or rewritten.
    3. Live (non-expired) OAuth session. Refreshed first when
       ``refresh=True``; read as-is when ``refresh=False``. When
       ``base_url`` is given, a session minted for a *different* base URL is
       skipped (replay-guard: never send a token to a host the user didn't
       authenticate against).
    4. The stored ``comfy-cloud-api-key`` key (``comfy cloud set-key``).

    Empty or whitespace-only env values count as unset, so without one the
    order is the flag, the session, then the stored key, as it always was.

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

    handed_in = env_credential(purpose)
    if handed_in is not None:
        return handed_in

    session = get_session(refresh=refresh, allow_clear=allow_clear)
    if (
        session is not None
        and not session.is_expired()
        and session.access_token
        and (base_url is None or session.base_url == base_url)
    ):
        return Credential(kind="oauth", value=session.access_token, source="session")

    _env_var, provider, strip = _PURPOSES[purpose]
    return _stored_api_key(provider, strip=strip)


def resolve_partner_credential() -> tuple[str, str] | None:
    """The ``extra_data`` credential partner-API nodes authenticate with.

    Returns ``(field, value)`` — ``auth_token_comfy_org`` for a session,
    ``api_key_comfy_org`` for a key — or ``None`` when nothing is configured.
    Shared by the local exec and ``deploy run`` submit paths, so it consults
    every source either one already read. In precedence order:

    1. ``COMFY_CLOUD_AUTH_TOKEN`` (:data:`CLOUD_BEARER_ENV_VAR`), handed in by the
       caller for this run;
    2. ``COMFY_API_KEY``, then ``COMFY_CLOUD_API_KEY``;
    3. a live OAuth session;
    4. the stored ``comfy cloud set-key`` key.

    When 1 or 2 is set the session is not read, refreshed or rewritten. Empty
    or whitespace-only env values count as unset.

    Best-effort: the session is refreshed when possible, but ``allow_clear=False``
    keeps a fatal refresh error from logging the user out from under a foreground
    command, and any unexpected error falls through to a network-free read of the
    env and stored keys rather than aborting the run.
    """
    handed_in = _partner_env_credential()
    if handed_in is not None:
        field = "auth_token_comfy_org" if handed_in.kind == "oauth" else "api_key_comfy_org"
        return (field, handed_in.value)
    try:
        cloud = resolve_cloud_credential(purpose="cloud", refresh=True, allow_clear=False)
    except Exception:  # noqa: BLE001 — best-effort: never abort a run on a refresh hiccup
        cloud = resolve_cloud_credential(purpose="cloud", refresh=False, allow_clear=False)
    if cloud is not None and cloud.kind == "oauth":
        return ("auth_token_comfy_org", cloud.value)
    key_credential = find_api_key(purpose="partner")
    if key_credential is None:
        return None
    return ("api_key_comfy_org", key_credential.value)


def keyed_partner_credential(workspace_key: str) -> tuple[str, str]:
    """The partner-node credential for a job running on ``workspace_key``.

    The job runs in the key's workspace, so its partner nodes take that key
    rather than a sign-in's token, unless ``COMFY_API_KEY`` names a partner key
    on purpose. Network-free, unlike :func:`resolve_partner_credential`.
    """
    import os

    partner = (os.environ.get(_PURPOSES["partner"][0]) or "").strip()
    return ("api_key_comfy_org", partner or workspace_key)


def _partner_env_credential() -> Credential | None:
    """The env half of :func:`resolve_partner_credential`, in its order.

    The token comes first: it is request-scoped, and an ambient key ahead of it
    would authenticate the run as the wrong account and spend its credits. Key
    values are stripped, since padding would reach a partner auth header.
    """
    token = cloud_bearer_env_token()
    if token:
        return Credential(kind="oauth", value=token, source=f"env:{CLOUD_BEARER_ENV_VAR}")
    for env_var in (_PURPOSES["partner"][0], _PURPOSES["cloud"][0]):
        value = _env_value(env_var)
        if value:
            return Credential(kind="api_key", value=value, source=f"env:{env_var}")
    return None
