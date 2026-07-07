"""First-run setup nudge: fire once, only for unconfigured users."""

from __future__ import annotations

from comfy_cli import onboarding
from comfy_cli.config_manager import ConfigManager


def test_signed_in_users_are_never_nudged():
    assert onboarding.should_nudge_setup(signed_in=True) is False


def test_first_run_nudges_then_marks_and_stays_quiet():
    cfg = ConfigManager()
    # Fresh (isolated) config + not signed in -> nudge.
    assert onboarding.should_nudge_setup(signed_in=False, config=cfg) is True
    onboarding.mark_setup_nudged(config=cfg)
    # Marked -> never again.
    assert onboarding.should_nudge_setup(signed_in=False, config=cfg) is False
    # Still suppressed for a signed-in user regardless.
    assert onboarding.should_nudge_setup(signed_in=True, config=cfg) is False
