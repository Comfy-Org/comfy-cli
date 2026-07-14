"""Envelope contract tests for `comfy stop` and `comfy launch --background`.

These pin the agent-facing contract that GitHub issue #509 was about: the
global `--json` flag used to be accepted but ignored by these two lifecycle
commands, so a successful stop/launch printed human text and emitted no
envelope/1 document. Programmatic callers (the local MCP `_run_comfy`, CI)
then read a success as a malformed/failed response.

We assert:
- JSON mode emits a single, schema-valid envelope on both success and the
  in-band error paths, with the shapes the issue specifies.
- Pretty (non-JSON) mode is unchanged — no envelope on stdout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import jsonschema
import pytest
import typer

from comfy_cli import cmdline
from comfy_cli.caller import Caller
from comfy_cli.command import launch as launch_mod
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas"


def _force_renderer(mode: OutputMode) -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=(mode is OutputMode.JSON),
    )
    r.mode = mode
    set_renderer(r)
    return r


def _last_envelope(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected an envelope on stdout"
    return json.loads(out.splitlines()[-1])


def _validator_for(name: str) -> jsonschema.Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    store: dict = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        if s.get("$id"):
            store[s["$id"]] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def _validate(envelope: dict, data_schema: str) -> None:
    _validator_for("envelope.json").validate(envelope)
    if envelope["ok"]:
        _validator_for(data_schema).validate(envelope["data"])


class _FakeConfig:
    """Stand-in for the ConfigManager singleton used by `stop`."""

    def __init__(self, background):
        self.background = background
        self.config = {"DEFAULT": {}}
        if background is not None:
            # `stop` gates on the raw config key before reading `.background`.
            self.config["DEFAULT"]["background"] = repr(background)
        self.removed = False

    def remove_background(self):
        self.removed = True


# --------------------------------------------------------------------------
# stop
# --------------------------------------------------------------------------


class TestStopJson:
    def test_success_emits_stop_envelope(self, monkeypatch, capsys):
        fake = _FakeConfig(("127.0.0.1", 8188, 4242))
        monkeypatch.setattr(cmdline, "ConfigManager", lambda: fake)
        monkeypatch.setattr(cmdline.utils, "kill_all", lambda pid: True)
        _force_renderer(OutputMode.JSON)

        cmdline.stop()

        env = _last_envelope(capsys)
        _validate(env, "stop.json")
        assert env["ok"] is True
        assert env["command"] == "stop"
        assert env["where"] == "local"
        assert env["changed"] is True
        assert env["data"] == {"host": "127.0.0.1", "port": 8188, "stopped": True}
        assert fake.removed is True

    def test_kill_failure_still_emits_envelope_stopped_false(self, monkeypatch, capsys):
        fake = _FakeConfig(("127.0.0.1", 8188, 4242))
        monkeypatch.setattr(cmdline, "ConfigManager", lambda: fake)
        monkeypatch.setattr(cmdline.utils, "kill_all", lambda pid: False)
        _force_renderer(OutputMode.JSON)

        cmdline.stop()

        env = _last_envelope(capsys)
        _validate(env, "stop.json")
        assert env["ok"] is True
        assert env["data"]["stopped"] is False
        assert env["changed"] is False
        # The stale background entry is cleared regardless of the kill outcome.
        assert fake.removed is True

    def test_no_background_is_structured_error(self, monkeypatch, capsys):
        fake = _FakeConfig(None)
        monkeypatch.setattr(cmdline, "ConfigManager", lambda: fake)
        monkeypatch.setattr(
            cmdline.utils, "kill_all", lambda pid: pytest.fail("kill_all must not run with nothing to stop")
        )
        _force_renderer(OutputMode.JSON)

        with pytest.raises(typer.Exit) as ei:
            cmdline.stop()
        assert ei.value.exit_code == 1

        env = _last_envelope(capsys)
        _validator_for("envelope.json").validate(env)
        assert env["ok"] is False
        assert env["command"] == "stop"
        assert env["error"]["code"] == "no_background"

    def test_pretty_mode_unchanged_no_envelope(self, monkeypatch, capsys):
        fake = _FakeConfig(("127.0.0.1", 8188, 4242))
        monkeypatch.setattr(cmdline, "ConfigManager", lambda: fake)
        monkeypatch.setattr(cmdline.utils, "kill_all", lambda pid: True)
        _force_renderer(OutputMode.PRETTY)

        cmdline.stop()

        out = capsys.readouterr().out
        # Human line still prints to stdout; crucially, no JSON envelope does.
        assert "Background ComfyUI is stopped" in out
        assert "envelope/1" not in out


# --------------------------------------------------------------------------
# launch --background
# --------------------------------------------------------------------------


class _StopExit(Exception):
    """Sentinel raised in place of os._exit so the monitor loop unwinds."""

    def __init__(self, code):
        self.code = code


def _drive_monitor_with_success_line(monkeypatch, tmp_path, mode: OutputMode):
    """Run `launch_and_monitor` end-to-end with a fake child that writes the
    success marker to the logfile, so the real success branch (config write +
    envelope emit + os._exit) executes."""
    monkeypatch.chdir(tmp_path)

    class _FakePopen:
        def __init__(self, cmd, stdout=None, **kwargs):
            self.pid = 4242
            # The real child writes to this logfile fd; emulate the success
            # banner the monitor tails for.
            stdout.write("Launching ComfyUI from: /ws\nTo see the GUI go to: http://127.0.0.1:8188\n")
            stdout.flush()

        def poll(self):
            # None keeps the reader loop consuming lines; the success handler
            # fires (and raises via the patched os._exit) before EOF matters.
            return None

    monkeypatch.setattr(launch_mod.subprocess, "Popen", _FakePopen)

    def _fake_exit(code):
        raise _StopExit(code)

    monkeypatch.setattr(launch_mod.os, "_exit", _fake_exit)
    _force_renderer(mode)

    with pytest.raises(_StopExit) as ei:
        asyncio.run(launch_mod.launch_and_monitor(["comfy", "launch"], "127.0.0.1", 8188))
    return ei.value


class TestLaunchBackgroundJson:
    def test_success_emits_launch_envelope(self, monkeypatch, tmp_path, capsys):
        exit_exc = _drive_monitor_with_success_line(monkeypatch, tmp_path, OutputMode.JSON)
        assert exit_exc.code == 0

        env = _last_envelope(capsys)
        _validate(env, "launch.json")
        assert env["ok"] is True
        assert env["command"] == "launch"
        assert env["where"] == "local"
        assert env["changed"] is True
        assert env["data"] == {"host": "127.0.0.1", "port": 8188, "pid": 4242, "background": True}

    def test_success_pretty_mode_writes_no_envelope(self, monkeypatch, tmp_path, capsys):
        exit_exc = _drive_monitor_with_success_line(monkeypatch, tmp_path, OutputMode.PRETTY)
        assert exit_exc.code == 0
        out = capsys.readouterr().out
        assert "envelope/1" not in out

    def test_launch_failure_is_structured_error(self, monkeypatch, capsys):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(launch_mod, "check_comfy_server_running", lambda port: False)
        # A fresh (unbacked) ConfigManager: no background running.
        fake_cfg = type("C", (), {"background": None})()
        monkeypatch.setattr(launch_mod, "ConfigManager", lambda: fake_cfg)
        monkeypatch.setattr(launch_mod, "launch_and_monitor", AsyncMock(return_value=["boom\n"]))

        exits = {}
        monkeypatch.setattr(launch_mod.os, "_exit", lambda code: exits.setdefault("code", code))
        _force_renderer(OutputMode.JSON)

        launch_mod.background_launch(extra=[])

        assert exits["code"] == 1
        env = _last_envelope(capsys)
        _validator_for("envelope.json").validate(env)
        assert env["ok"] is False
        assert env["command"] == "launch"
        assert env["error"]["code"] == "launch_failed"

    def test_already_running_is_structured_error(self, monkeypatch, capsys):
        fake_cfg = type("C", (), {"background": ("127.0.0.1", 8188, 4242)})()
        monkeypatch.setattr(launch_mod, "ConfigManager", lambda: fake_cfg)
        monkeypatch.setattr(launch_mod.utils, "is_running", lambda pid: True)
        _force_renderer(OutputMode.JSON)

        with pytest.raises(typer.Exit):
            launch_mod.background_launch(extra=[])

        env = _last_envelope(capsys)
        _validator_for("envelope.json").validate(env)
        assert env["ok"] is False
        assert env["error"]["code"] == "background_already_running"

    def test_invalid_port_is_structured_error(self, monkeypatch, capsys):
        fake_cfg = type("C", (), {"background": None})()
        monkeypatch.setattr(launch_mod, "ConfigManager", lambda: fake_cfg)
        _force_renderer(OutputMode.JSON)

        with pytest.raises(typer.Exit):
            launch_mod.background_launch(extra=["--port", "notaport"])

        env = _last_envelope(capsys)
        _validator_for("envelope.json").validate(env)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_port"
        assert env["error"]["details"]["port"] == "notaport"


# --------------------------------------------------------------------------
# discovery registration
# --------------------------------------------------------------------------


def test_launch_and_stop_registered_in_discovery():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy stop"] == "stop"
    assert COMMAND_SCHEMAS["comfy launch"] == "launch"
    # Both schema files ship so `comfy discover` can advertise the shapes.
    assert (SCHEMAS_DIR / "stop.json").is_file()
    assert (SCHEMAS_DIR / "launch.json").is_file()
