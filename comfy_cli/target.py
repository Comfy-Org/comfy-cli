"""Routing target for ComfyUI HTTP traffic.

A ``Target`` is the small bundle of facts that distinguishes "talk to a local
ComfyUI on 127.0.0.1:8188" from "talk to Comfy Cloud on
fe-pr-12159.testenvs.comfy.org": a base URL, an optional path prefix
(``/api``), and an optional Bearer token. Everything else flows through the
unified :class:`comfy_cli.comfy_client.Client`.

A few thin differences remain (the cloud history endpoint is ``history_v2``,
queue inspection works differently, no WebSocket on cloud) — those are
captured as fields on the Target so the client can branch on them rather than
duplicating call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Back-compat re-export: the provider name now lives with the credential
# resolver. Hidden / testing-only path; the canonical sign-in is OAuth.
from comfy_cli.credentials import CLOUD_API_KEY_PROVIDER as CLOUD_API_KEY_PROVIDER


@dataclass(frozen=True)
class Target:
    """Where to send ComfyUI HTTP requests."""

    kind: str  # "local" | "cloud"
    base_url: str  # e.g. "http://127.0.0.1:8188" or "https://fe-pr-12159.testenvs.comfy.org"
    path_prefix: str = ""  # "" for local, "/api" for cloud
    history_path: str = "history"  # "history" for local, "history_v2" for cloud
    jobs_path: str | None = None  # "jobs" for cloud, None for local (uses queue+history)
    # auth_token (OAuth Bearer) and api_key (X-API-Key) are both repr-suppressed
    # so logger.debug + pytest failure dumps can't leak credentials.
    auth_token: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    host: str | None = None  # only for local — preserved for back-compat
    port: int | None = None  # only for local

    @property
    def is_cloud(self) -> bool:
        return self.kind == "cloud"

    def url(self, *parts: str) -> str:
        """Build a fully-qualified URL, applying the path prefix."""
        joined = "/".join(p.strip("/") for p in parts if p)
        if self.path_prefix:
            return f"{self.base_url}{self.path_prefix}/{joined}"
        return f"{self.base_url}/{joined}"


def resolve_target(
    *, where: str | None, host: str | None = None, port: int | None = None, config: Any = None
) -> Target:
    """Pick a Target based on ``--where`` plus host/port (for local).

    Precedence honoring the global routing mode: explicit ``where`` arg >
    ``COMFY_WHERE`` env var > persisted ``where_default`` config key >
    auto-detect (``cloud`` if credentials exist, else ``local``).
    The ``config`` arg is honored if passed; otherwise
    we instantiate a ``ConfigManager`` to read the persisted key.
    """
    from comfy_cli import where as where_module

    config_value: str | None = None
    if config is not None:
        config_value = config.get(where_module.CONFIG_KEY_WHERE_DEFAULT)
    else:
        try:
            from comfy_cli.config_manager import ConfigManager

            config_value = ConfigManager().get(where_module.CONFIG_KEY_WHERE_DEFAULT)
        except Exception:  # noqa: BLE001 — never break target resolution on a bad config
            pass

    decision = where_module.resolve(flag=where, config_value=config_value)

    if decision.target is where_module.WhereTarget.CLOUD:
        from comfy_cli.cloud import get_base_url
        from comfy_cli.credentials import resolve_cloud_credential

        # OAuth-first: a live session is preferred over API keys (which are on
        # a deprecation path). Precedence: OAuth session > env var > stored key.
        # ``base_url=`` arms the replay-guard: a session minted for a different
        # base_url (or expired) is ignored so credentials are never replayed to
        # a host the user didn't authenticate against; the API-key fallback or
        # a downstream "unauthenticated" error takes over. ``refresh=True`` is
        # the *proactive* leg: a near-expiry access token is refreshed before
        # it's attached, so the user keeps working across the ~1h token
        # lifetime without ever hitting a 401. The refresh is a cheap no-op
        # when the token isn't near expiry (no network), and the client/loader
        # still force-refresh *reactively* on a server 401 as a backstop.
        base_url = get_base_url()
        cred = resolve_cloud_credential(purpose="cloud", base_url=base_url, refresh=True)
        token = cred.value if cred is not None and cred.kind == "oauth" else None
        api_key = cred.value if cred is not None and cred.kind == "api_key" else None

        return Target(
            kind="cloud",
            base_url=base_url,
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token=token,
            api_key=api_key,
        )

    # Local — keep the existing host/port resolution semantics. host/port may
    # be None here; callers fill those in from their config_manager.
    resolved_host = host or "127.0.0.1"
    resolved_port = int(port or 8188)
    # Bracket IPv6 literals so the URL is well-formed (RFC 3986 §3.2.2).
    url_host = f"[{resolved_host}]" if ":" in resolved_host and not resolved_host.startswith("[") else resolved_host
    return Target(
        kind="local",
        base_url=f"http://{url_host}:{resolved_port}",
        path_prefix="",
        history_path="history",
        jobs_path=None,
        auth_token=None,
        host=resolved_host,
        port=resolved_port,
    )
