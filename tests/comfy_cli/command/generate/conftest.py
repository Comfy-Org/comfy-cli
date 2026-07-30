"""Shared fixtures for the ``comfy generate`` test package."""

import pytest


@pytest.fixture(autouse=True)
def _auto_confirm_spend():
    """Pre-authorize the credit-spend gate (BE-4103) for generate tests.

    Most tests in this package exercise behavior *downstream* of the consent
    interlock, and under CliRunner there is no TTY — without this, every
    execution-path invocation would fail closed before reaching the code
    under test. The gate's own tests override this fixture by name in
    ``test_spend_gate.py`` to put the gate back in force.

    ConfigManager is a process-wide singleton, so the teardown removes the
    in-memory key rather than relying on the per-test config-dir isolation
    alone (the file is isolated; the loaded configparser is not).
    """
    from comfy_cli import constants
    from comfy_cli.config_manager import ConfigManager

    cm = ConfigManager()
    cm.set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "true")
    yield
    cm.config.remove_option("DEFAULT", constants.CONFIG_KEY_SPEND_AUTO_CONFIRM)
