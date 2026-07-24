"""Spend-gate tests for ``comfy generate`` (BE-4103).

A generation call spends Comfy credits, so the proxy call sits behind a
consent interlock: interactive TTY runs prompt first, ``--yes`` or the
persisted ``spend.auto_confirm`` config bypass the prompt, and ``--json`` /
non-TTY runs with neither **fail closed** — error out, spend nothing, never
hang on a prompt no one can answer.
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from comfy_cli import constants
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import app as gen_app
from comfy_cli.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def _auto_confirm_spend():
    """Override the package conftest's pre-authorization: these tests
    exercise the gate itself, so it starts (and ends) cleared."""
    cm = ConfigManager()
    cm.config.remove_option("DEFAULT", constants.CONFIG_KEY_SPEND_AUTO_CONFIRM)
    yield
    cm.config.remove_option("DEFAULT", constants.CONFIG_KEY_SPEND_AUTO_CONFIRM)


@pytest.fixture(autouse=True)
def disable_tracking_prompt(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "comfyui-test")
    return "comfyui-test"


@pytest.fixture
def post_spy(monkeypatch):
    """Record every proxy POST; each recorded call is a would-be credit spend."""
    calls: list = []

    def _post(*a, **kw):
        calls.append((a, kw))
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})

    monkeypatch.setattr(gen_app.client.httpx, "post", _post)
    return calls


@pytest.fixture
def interactive_tty(monkeypatch):
    """Simulate a human at a terminal — CliRunner's piped stdin is never a TTY."""
    monkeypatch.setattr(gen_app, "_stdin_is_tty", lambda: True)


# ─── Fail-closed: no consent, no TTY → error, nothing spent ───────────────


def test_json_without_consent_fails_closed_and_spends_nothing(runner, api_key, post_spy):
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 1
    assert post_spy == []
    payload = json.loads(r.stdout)
    assert payload["code"] == "spend_consent_required"
    assert "--yes" in payload["error"]


def test_pretty_non_tty_without_consent_fails_closed(runner, api_key, post_spy):
    # No --json, but stdin is a pipe (CliRunner): never hang on a prompt.
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])
    assert r.exit_code == 1
    assert post_spy == []
    assert "spends Comfy credits" in r.stdout
    assert "--yes" in r.stdout


def test_async_submit_is_gated_too(runner, api_key, post_spy):
    # --async still spends on submit — the gate must cover it.
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1", "--async", "--json"],
    )
    assert r.exit_code == 1
    assert post_spy == []


def test_gate_runs_before_auth_resolution(runner, post_spy):
    # No COMFY_API_KEY in env: consent still decides first, so an
    # unauthenticated machine caller sees the consent error, not an auth one,
    # and no OAuth refresh / network happens without consent.
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 1
    assert post_spy == []
    assert json.loads(r.stdout)["code"] == "spend_consent_required"


# ─── Bypasses: --yes flag and spend.auto_confirm config ───────────────────


def test_json_with_yes_proceeds(runner, api_key, post_spy):
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json", "--yes"])
    assert r.exit_code == 0, r.stdout
    assert len(post_spy) == 1


def test_json_with_auto_confirm_config_proceeds(runner, api_key, post_spy):
    ConfigManager().set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "true")
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 0, r.stdout
    assert len(post_spy) == 1


def test_auto_confirm_false_still_fails_closed(runner, api_key, post_spy):
    ConfigManager().set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "false")
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 1
    assert post_spy == []


def test_auto_confirm_garbage_value_fails_closed(runner, api_key, post_spy):
    # A corrupt config value must never silently authorize spending.
    ConfigManager().set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "banana")
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 1
    assert post_spy == []


# ─── Interactive TTY: prompt before spending ──────────────────────────────


def test_interactive_prompt_accept_proceeds(runner, api_key, post_spy, interactive_tty):
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"], input="y\n")
    assert r.exit_code == 0, r.stdout
    assert len(post_spy) == 1
    assert "spends Comfy credits" in r.stdout


def test_interactive_prompt_decline_spends_nothing(runner, api_key, post_spy, interactive_tty):
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"], input="n\n")
    assert r.exit_code == 1
    assert post_spy == []
    assert "no credits were spent" in r.stdout


def test_interactive_prompt_default_is_no(runner, api_key, post_spy, interactive_tty):
    # Bare Enter must not spend.
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"], input="\n")
    assert r.exit_code == 1
    assert post_spy == []


# ─── Ungated paths: nothing that spends ───────────────────────────────────


def test_emit_workflow_is_not_gated(runner, post_spy, tmp_path):
    # --emit-workflow writes a local artifact, calls no proxy, spends nothing.
    out = tmp_path / "wf.json"
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--emit-workflow", str(out)])
    assert r.exit_code == 0, r.stdout
    assert out.exists()
    assert post_spy == []


def test_schema_error_surfaces_before_consent(runner, api_key, post_spy):
    # Bad args fail on validation, not on consent — no prompt, no spend.
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "abc", "--height", "1"])
    assert r.exit_code == 1
    assert post_spy == []
    assert "spend" not in r.stdout.lower()


# ─── Lifecycle: consent failures emit generate:error(kind=consent) ────────


def test_consent_failure_emits_error_kind_consent(runner, api_key, post_spy, monkeypatch):
    events: list[tuple[str, dict]] = []

    def _record(event_name, properties=None, *, mixpanel_name=None):
        events.append((event_name, dict(properties or {})))

    monkeypatch.setattr("comfy_cli.tracking.track_event", _record)
    monkeypatch.setattr("comfy_cli.command.generate.app.tracking.track_event", _record)
    monkeypatch.setattr("comfy_cli.cmdline.tracking.track_event", _record)

    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 1
    names = [n for n, _ in events]
    assert names.count("generate:start") == 1
    err_props = [p for n, p in events if n == "generate:error"]
    assert len(err_props) == 1
    assert err_props[0]["error_kind"] == "consent"
    assert "generate:success" not in names


# ─── `comfy generate consent` action ──────────────────────────────────────


def test_consent_show_defaults_to_false(runner):
    r = runner.invoke(cli_app, ["generate", "consent", "--json"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload == {"spend_auto_confirm": False, "action": "show"}


def test_consent_always_persists_and_unlocks_generate(runner, api_key, post_spy):
    r = runner.invoke(cli_app, ["generate", "consent", "always", "--json"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["spend_auto_confirm"] is True
    # Persisted to the config file on disk, not just in memory.
    from pathlib import Path

    cfg_text = Path(ConfigManager().get_config_file_path()).read_text()
    assert "spend.auto_confirm = true" in cfg_text

    r2 = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r2.exit_code == 0, r2.stdout
    assert len(post_spy) == 1


def test_consent_ask_reverts(runner, api_key, post_spy):
    runner.invoke(cli_app, ["generate", "consent", "always"])
    r = runner.invoke(cli_app, ["generate", "consent", "ask", "--json"])
    assert r.exit_code == 0
    assert json.loads(r.stdout)["spend_auto_confirm"] is False

    r2 = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r2.exit_code == 1
    assert post_spy == []


def test_consent_show_pretty_output(runner):
    r = runner.invoke(cli_app, ["generate", "consent"])
    assert r.exit_code == 0
    assert "spend.auto_confirm: false" in r.stdout


def test_consent_unknown_action_errors(runner):
    r = runner.invoke(cli_app, ["generate", "consent", "sometimes"])
    assert r.exit_code == 1
    assert "Unknown consent action" in r.stdout
