"""First-run onboarding nudge.

A brand-new user who runs a command before configuring anything has no obvious
path to the ``comfy setup`` wizard. This surfaces a single, one-time, one-line
nudge toward it — gated so it never fires for configured users, never repeats,
and never touches the machine (JSON) contract.
"""

from __future__ import annotations

from comfy_cli.config_manager import ConfigManager

# Config flag (config.ini) marking that we've nudged this install once.
SETUP_NUDGED_KEY = "setup_nudged"

NUDGE_TEXT = (
    "👋 New here? Run `comfy setup` to get going — it picks local/cloud, signs you in, and installs the agent skills."
)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def should_nudge_setup(*, signed_in: bool, config: ConfigManager | None = None) -> bool:
    """True only on a genuine first run: not signed in and not nudged before.

    Side-effect free — call :func:`mark_setup_nudged` after actually showing it.
    """
    if signed_in:
        return False
    cfg = config or ConfigManager()
    return not _truthy(cfg.get(SETUP_NUDGED_KEY))


def mark_setup_nudged(*, config: ConfigManager | None = None) -> None:
    """Record that the one-time nudge has been shown so it never repeats."""
    (config or ConfigManager()).set(SETUP_NUDGED_KEY, "1")
