"""``--where`` resolution: pick the backend that handles a routed command.

Targets:

- ``local``  — run against the local ComfyUI server.
- ``cloud``  — talk to Comfy Cloud over HTTPS.

Precedence for ``--where``:

1. Explicit ``--where`` flag.
2. ``COMFY_WHERE`` environment variable.
3. ``defaults.where`` in the governing project/1 ``comfy.yaml``
   (see :mod:`comfy_cli.project`).
4. ``where_default`` in the config file.
5. Auto-detect: ``cloud`` if any cloud credential is configured
   (API key env/store, or active OAuth session), else ``local``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from comfy_cli.cancellation import get_token  # noqa: F401  — re-exported indirectly


class WhereTarget(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


CLOUD_PROVIDER = "comfy-cloud"
ENV_DEFAULT = "COMFY_WHERE"
CONFIG_KEY_WHERE_DEFAULT = "where_default"


@dataclass
class WhereResolution:
    target: WhereTarget
    source: str  # "flag" | "env" | "project" | "config" | "default"


# Sentinel: "caller didn't say" — resolve() then looks up the governing
# project itself, so the existing call sites get project routing without
# changes. Pass ``project_value=None`` to disable the lookup explicitly
# (tests / deliberately project-unaware callers).
_UNSET: Any = object()


def resolve(
    *,
    flag: str | None = None,
    env: Mapping[str, str] | None = None,
    config_value: str | None = None,
    project_value: str | None = _UNSET,
) -> WhereResolution:
    """Pick the target. Invalid values raise ``ValueError`` with a clear message."""
    e = env if env is not None else os.environ
    if flag:
        return WhereResolution(target=_parse(flag), source="flag")
    env_choice = e.get(ENV_DEFAULT)
    if env_choice:
        return WhereResolution(target=_parse(env_choice), source="env")
    if project_value is _UNSET:
        project_value = _project_where_default()
    if project_value:
        return WhereResolution(target=_parse(project_value), source="project")
    if config_value:
        return WhereResolution(target=_parse(config_value), source="config")

    # Auto-detect: if cloud credentials are configured, default to cloud.
    if _has_cloud_credentials():
        return WhereResolution(target=WhereTarget.CLOUD, source="auto")
    return WhereResolution(target=WhereTarget.LOCAL, source="default")


def _project_where_default() -> str | None:
    """``defaults.where`` from the project/1 ``comfy.yaml`` governing cwd, if
    any. Discovery itself never raises (see :mod:`comfy_cli.project`); a
    present-but-invalid value is parsed by the caller like any other source."""
    # Lazy import: keep `where` cheap for the common no-project path and
    # avoid import cycles.
    from comfy_cli.project import find_project

    project = find_project()
    if project is None:
        return None
    defaults = project.config.get("defaults")
    value = defaults.get("where") if isinstance(defaults, dict) else None
    return value if isinstance(value, str) and value else None


def _has_cloud_credentials() -> bool:
    """Return True if any cloud auth path is configured (API key or OAuth).

    Presence check, not resolution: an *expired* session still counts (cloud
    is clearly the configured backend; preflight surfaces the expiry), so this
    deliberately doesn't use ``resolve_cloud_credential``.
    """
    from comfy_cli.credentials import find_api_key, get_session

    if find_api_key(purpose="cloud") is not None:
        return True
    return get_session(refresh=False) is not None


def _parse(value: str) -> WhereTarget:
    norm = value.strip().lower()
    try:
        return WhereTarget(norm)
    except ValueError as exc:
        raise ValueError(f"invalid --where value {value!r}: expected one of {[t.value for t in WhereTarget]}") from exc


# ---- cloud client (local-only stub) ---------------------------------------


@dataclass
class CloudError:
    code: str  # "cloud_not_configured" | "cloud_unauthorized" | "cloud_unavailable"
    message: str
    hint: str | None
    details: dict[str, Any]


def cloud_preflight() -> CloudError | None:
    """Return an error envelope payload if the cloud path can't proceed.

    Accepts either auth path:
      - ``COMFY_CLOUD_API_KEY`` env var, OR
      - persisted ``comfy-cloud-api-key`` provider record, OR
      - active OAuth session (valid + non-expired).

    Failure modes:
      - Nothing configured     → ``cloud_not_configured``
      - OAuth session expired  → ``cloud_unauthorized``
    """
    from comfy_cli.credentials import find_api_key, get_session

    # API key path — no expiry check, key is either valid or it isn't (server
    # tells us at request time).
    if find_api_key(purpose="cloud") is not None:
        return None

    # Proactively refresh an expired-but-refreshable session so work doesn't
    # die just because the short-lived access token lapsed between commands.
    session = get_session(refresh=True)
    if session is None:
        return CloudError(
            code="cloud_not_configured",
            message="Not signed in to Comfy Cloud.",
            hint="run: comfy cloud login (or set COMFY_CLOUD_API_KEY for hidden testing path)",
            details={"provider": CLOUD_PROVIDER},
        )
    if session.is_expired():
        return CloudError(
            code="cloud_unauthorized",
            message="Comfy Cloud session has expired.",
            hint="run: comfy cloud login",
            details={
                "provider": CLOUD_PROVIDER,
                "expires_at": session.expires_at,
            },
        )
    return None
