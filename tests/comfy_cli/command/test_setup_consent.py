"""Tests for the telemetry-consent step of ``comfy setup``.

The consent *logic* (env opt-out, init_tracking, the global lazy prompt) is
covered in test_tracking.py. Here we pin the wizard step's branching:
env opt-out and an already-recorded choice never re-ask or re-write, a
non-interactive run leaves the flag UNSET, and an interactive answer is
forwarded verbatim to init_tracking.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import questionary

from comfy_cli import tracking
from comfy_cli.command import setup as setup_inner
from comfy_cli.config_manager import ConfigManager


@pytest.fixture
def consent_spies(monkeypatch):
    """init_tracking spy + a recorded-consent value controller."""
    init_spy = MagicMock()
    monkeypatch.setattr(tracking, "init_tracking", init_spy)

    confirm_spy = MagicMock()
    monkeypatch.setattr(questionary, "confirm", confirm_spy)

    def set_recorded(value):
        monkeypatch.setattr(ConfigManager(), "get_bool", lambda _k: value)

    def set_answer(value):
        confirm_spy.return_value = types.SimpleNamespace(ask=lambda: value)

    return types.SimpleNamespace(init=init_spy, confirm=confirm_spy, set_recorded=set_recorded, set_answer=set_answer)


def test_env_optout_never_writes_or_asks(monkeypatch, consent_spies):
    monkeypatch.setattr(tracking, "_telemetry_disabled_by_env", lambda: True)
    setup_inner._do_consent(None, non_interactive=False)
    consent_spies.init.assert_not_called()
    consent_spies.confirm.assert_not_called()


def test_already_recorded_is_reported_not_reasked(monkeypatch, consent_spies):
    monkeypatch.setattr(tracking, "_telemetry_disabled_by_env", lambda: False)
    consent_spies.set_recorded(True)
    setup_inner._do_consent(None, non_interactive=False)
    consent_spies.init.assert_not_called()
    consent_spies.confirm.assert_not_called()


def test_non_interactive_leaves_flag_unset(monkeypatch, consent_spies):
    monkeypatch.setattr(tracking, "_telemetry_disabled_by_env", lambda: False)
    consent_spies.set_recorded(None)
    setup_inner._do_consent(None, non_interactive=True)
    consent_spies.init.assert_not_called()
    consent_spies.confirm.assert_not_called()


def test_interactive_yes_enables(monkeypatch, consent_spies):
    monkeypatch.setattr(tracking, "_telemetry_disabled_by_env", lambda: False)
    consent_spies.set_recorded(None)
    consent_spies.set_answer(True)
    setup_inner._do_consent(None, non_interactive=False)
    consent_spies.init.assert_called_once_with(True)


def test_interactive_no_disables(monkeypatch, consent_spies):
    monkeypatch.setattr(tracking, "_telemetry_disabled_by_env", lambda: False)
    consent_spies.set_recorded(None)
    consent_spies.set_answer(False)
    setup_inner._do_consent(None, non_interactive=False)
    consent_spies.init.assert_called_once_with(False)
