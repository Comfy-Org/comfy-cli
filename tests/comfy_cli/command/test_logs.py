"""Tests for `comfy logs` — the background ComfyUI log tail reader + verb.

Covers the pure tail reader (last-N and the line/byte caps), the no-log-file
error envelope, the success envelope shape, the `--where` guard, and that the
background monitor redirects the child's own fds to a truncate-on-launch
workspace logfile and records its path.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import constants
from comfy_cli.caller import Caller
from comfy_cli.command import launch
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer() -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _envelope(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    assert out.strip(), "no envelope on stdout"
    return json.loads(out.strip().splitlines()[-1])


class _FakeConfigManager:
    """Just the surface `comfy logs` resolution reads off ConfigManager."""

    def __init__(self, recorded: str | None, background: tuple[str, int, int] | None):
        self._recorded = recorded
        self.background = background

    def get(self, key: str):
        return self._recorded if key == constants.CONFIG_KEY_BACKGROUND_LOG else None


def _fake_env(
    monkeypatch,
    *,
    workspace=None,
    recorded: str | None = None,
    background: tuple[str, int, int] | None = None,
):
    """Point log resolution at a tmp workspace + a synthetic config state."""
    monkeypatch.setattr(
        launch.workspace_manager,
        "workspace_path",
        str(workspace) if workspace is not None else None,
    )
    monkeypatch.setattr(launch, "ConfigManager", lambda: _FakeConfigManager(recorded, background))


def _write_log(workspace, name: str, text: str = "hello\n"):
    user_dir = workspace / "user"
    user_dir.mkdir(exist_ok=True)
    path = user_dir / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------- #
# read_log_tail
# --------------------------------------------------------------------------- #


def test_read_log_tail_returns_last_n(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("".join(f"line {i}\n" for i in range(100)))

    lines, truncated = launch.read_log_tail(str(p), 10)

    assert lines == [f"line {i}\n" for i in range(90, 100)]
    assert truncated is False


def test_read_log_tail_file_shorter_than_n_is_not_truncated(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("a\nb\nc\n")

    lines, truncated = launch.read_log_tail(str(p), 50)

    assert lines == ["a\n", "b\n", "c\n"]
    assert truncated is False


def test_read_log_tail_line_cap(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("".join(f"{i}\n" for i in range(50)))

    lines, truncated = launch.read_log_tail(str(p), 40, max_lines=10)

    assert len(lines) == 10
    assert lines[0] == "40\n" and lines[-1] == "49\n"
    assert truncated is True


def test_read_log_tail_byte_cap_trims_from_top(tmp_path):
    p = tmp_path / "log.txt"
    # 20 lines of ~100 bytes each = ~2KB total.
    p.write_text("".join(("x" * 99 + "\n") for _ in range(20)))

    lines, truncated = launch.read_log_tail(str(p), 20, max_bytes=500)

    assert truncated is True
    assert 0 < len(lines) < 20
    assert sum(len(line.encode("utf-8")) for line in lines) <= 500


def test_read_log_tail_handles_final_line_without_newline(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("a\nb\nlast-no-newline")

    lines, _ = launch.read_log_tail(str(p), 2)

    assert lines == ["b\n", "last-no-newline"]


def test_read_log_tail_single_huge_line_kept_byte_truncated(tmp_path):
    # A single newline-less line larger than the byte cap must not drop all
    # output; keep a byte-truncated tail of it instead.
    p = tmp_path / "log.txt"
    p.write_text("z" * 5000)

    lines, truncated = launch.read_log_tail(str(p), 10, max_bytes=500)

    assert truncated is True
    assert len(lines) == 1
    assert lines[0] == "z" * 500


# --------------------------------------------------------------------------- #
# `comfy logs` verb
# --------------------------------------------------------------------------- #


def test_logs_no_file_emits_clean_error(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    _fake_env(monkeypatch, workspace=tmp_path)  # empty workspace — no candidate exists

    with pytest.raises(typer.Exit) as exc:
        launch.logs(tail=50)

    assert exc.value.exit_code == 1
    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["command"] == "logs"
    assert env["error"]["code"] == "no_log_file"
    assert env["error"]["hint"]


def test_logs_no_workspace_emits_error(monkeypatch, capsys):
    _force_json_renderer()
    _fake_env(monkeypatch, workspace=None)

    with pytest.raises(typer.Exit):
        launch.logs(tail=50)

    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "no_log_file"
    # Nothing recorded and no workspace → nothing was checked, so no path list.
    assert "Looked for" not in env["error"]["message"]


def test_logs_success_envelope(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("".join(f"line {i}\n" for i in range(5)))
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=3)

    env = _envelope(capsys)
    assert env["ok"] is True
    assert env["command"] == "logs"
    assert env["where"] == "local"
    assert env["data"]["path"] == str(log)
    assert env["data"]["lines"] == ["line 2\n", "line 3\n", "line 4\n"]
    assert env["data"]["truncated"] is False


def test_logs_rejects_non_local_where(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    # resolve should never be reached, but guard anyway.
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(tmp_path / "x.log"), "recorded"))

    with pytest.raises(typer.Exit):
        launch.logs(tail=50, where="cloud")

    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "where_invalid"


def test_logs_where_local_is_accepted(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("hello\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=10, where="local")

    env = _envelope(capsys)
    assert env["ok"] is True
    assert env["data"]["lines"] == ["hello\n"]


def test_logs_read_error_emits_clean_error(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("hello\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    def boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(launch, "read_log_tail", boom)

    with pytest.raises(typer.Exit):
        launch.logs(tail=50)

    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "log_read_failed"


def test_logs_pretty_honors_large_tail_past_line_cap(monkeypatch, tmp_path, capsys):
    # Pretty output goes to a human terminal, so --tail beyond the JSON line cap
    # must not be silently truncated.
    log = tmp_path / "comfyui_8188.log"
    n = launch.LOGS_MAX_LINES + 50
    log.write_text("".join(f"line {i}\n" for i in range(n)))
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=n)

    out = capsys.readouterr().out
    assert "line 0\n" in out  # earliest line present → nothing was capped away
    assert f"line {n - 1}\n" in out


def test_logs_pretty_writes_raw_lines(monkeypatch, tmp_path, capsys):
    # Default renderer is pretty; log text with '[...]' must not be reinterpreted.
    log = tmp_path / "comfyui_8188.log"
    log.write_text("[INFO] hello [world]\nplain\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=10)

    out = capsys.readouterr().out
    assert "[INFO] hello [world]" in out
    assert "plain" in out


# --------------------------------------------------------------------------- #
# background monitor → logfile redirection
# --------------------------------------------------------------------------- #


@patch("comfy_cli.command.launch.os._exit", side_effect=SystemExit)
@patch("comfy_cli.command.launch.subprocess.Popen")
@patch("comfy_cli.command.launch.ConfigManager")
def test_launch_and_monitor_redirects_to_logfile_and_records_path(
    mock_cfg, mock_popen, mock_exit, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # workspace = cwd

    cfg = MagicMock()
    cfg.config = {"DEFAULT": {}}
    mock_cfg.return_value = cfg

    proc = MagicMock()
    proc.pid = 4321
    proc.poll.return_value = None

    captured: dict[str, Any] = {}

    def fake_popen(cmd, **kwargs):
        # The child writes to its OWN fd (the logfile), not a PIPE the monitor
        # owns — this is what lets post-monitor lines still land in the file.
        captured["kwargs"] = kwargs
        fh = kwargs["stdout"]
        fh.write("Launching ComfyUI from: /ws\n")
        fh.write("To see the GUI go to: http://127.0.0.1:8188\n")
        fh.flush()
        return proc

    mock_popen.side_effect = fake_popen

    with pytest.raises(SystemExit):
        asyncio.run(launch.launch_and_monitor(["comfy", "launch"], "127.0.0.1", 8188))

    # stdout points at the workspace logfile; stderr is folded into it.
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    log_path = str(tmp_path / "user" / "comfyui_8188.log")
    assert cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND_LOG] == log_path
    assert "8188" in cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND]
    # Written twice: the log path is recorded up front (so a crash log is
    # findable even if startup fails) and again with the background info on success.
    assert cfg.write_config.call_count == 2
    # The logfile the child wrote is on disk with both lines.
    assert "To see the GUI go to:" in (tmp_path / "user" / "comfyui_8188.log").read_text()


# --------------------------------------------------------------------------- #
# candidate-based resolution
# --------------------------------------------------------------------------- #


def test_candidate_paths_ordering_with_live_background(monkeypatch, tmp_path):
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded="/recorded/comfyui_9000.log",
        background=("127.0.0.1", 8188, 1234),
    )

    assert launch.candidate_log_paths() == [
        ("/recorded/comfyui_9000.log", "recorded"),
        (str(tmp_path / "user" / "comfyui_8188.log"), "derived_port"),
        (str(tmp_path / "user" / "comfyui.log"), "fallback_unsuffixed"),
        (str(tmp_path / "user" / "comfyui_*.log"), "fallback_glob"),
    ]


def test_candidate_paths_without_background_uses_default_port(monkeypatch, tmp_path):
    _fake_env(monkeypatch, workspace=tmp_path)

    sources = dict((source, path) for path, source in launch.candidate_log_paths())

    assert "recorded" not in sources
    assert sources["default_port"] == str(tmp_path / "user" / f"comfyui_{launch.DEFAULT_LOG_PORT}.log")


def test_candidate_paths_with_port_is_restricted(monkeypatch, tmp_path):
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded="/recorded/comfyui_9000.log",
        background=("127.0.0.1", 8188, 1234),
    )

    # --port ignores the recorded/derived/glob candidates entirely.
    assert launch.candidate_log_paths(8189) == [
        (str(tmp_path / "user" / "comfyui_8189.log"), "explicit_port"),
        (str(tmp_path / "user" / "comfyui.log"), "fallback_unsuffixed"),
    ]


def test_resolve_skips_missing_recorded_and_derived(monkeypatch, tmp_path):
    unsuffixed = _write_log(tmp_path, "comfyui.log")
    _fake_env(monkeypatch, workspace=tmp_path, recorded=str(tmp_path / "user" / "gone.log"))

    assert launch.resolve_background_log_path() == (str(unsuffixed), "fallback_unsuffixed")


def test_resolve_glob_picks_newest_and_skips_prev_rotations(monkeypatch, tmp_path):
    old = _write_log(tmp_path, "comfyui_9001.log", "old\n")
    newest = _write_log(tmp_path, "comfyui_9002.log", "new\n")
    rotated = _write_log(tmp_path, "comfyui_9003.prev.log", "rotated\n")
    rotated2 = _write_log(tmp_path, "comfyui_9004.prev2.log", "rotated2\n")
    os.utime(old, (1_600_000_000, 1_600_000_000))
    os.utime(newest, (1_700_000_000, 1_700_000_000))
    # The rotations are the NEWEST files on disk — they must still be skipped.
    os.utime(rotated, (1_800_000_000, 1_800_000_000))
    os.utime(rotated2, (1_900_000_000, 1_900_000_000))

    _fake_env(monkeypatch, workspace=tmp_path)

    assert launch.resolve_background_log_path() == (str(newest), "fallback_glob")


def test_logs_serves_unsuffixed_manager_log(monkeypatch, tmp_path, capsys):
    """A server started outside `comfy launch --background` logs only here."""
    _force_json_renderer()
    unsuffixed = _write_log(tmp_path, "comfyui.log", "manager line\n")
    _fake_env(monkeypatch, workspace=tmp_path)  # nothing recorded, no background

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["ok"] is True
    assert env["data"]["path"] == str(unsuffixed)
    assert env["data"]["source"] == "fallback_unsuffixed"
    assert env["data"]["lines"] == ["manager line\n"]
    assert env["data"]["port_mismatch"] is False


def test_logs_serves_recorded_crash_log_after_dead_pid(monkeypatch, tmp_path, capsys):
    """The recorded pointer survives a crash, so the crash log is still served."""
    _force_json_renderer()
    crash = _write_log(tmp_path, "comfyui_8189.log", "Traceback (most recent call last):\n")
    # Dead pid → ConfigManager cleared `background` but KEPT the log pointer.
    _fake_env(monkeypatch, workspace=tmp_path, recorded=str(crash), background=None)

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["data"]["path"] == str(crash)
    assert env["data"]["source"] == "recorded"
    assert env["data"]["port_mismatch"] is False  # no live server to mismatch against


def test_logs_reports_port_mismatch_for_stale_record(monkeypatch, tmp_path, capsys):
    """A failed launch attempt leaves an empty wrong-port log recorded."""
    _force_json_renderer()
    stale = _write_log(tmp_path, "comfyui_8189.log", "")
    _write_log(tmp_path, "comfyui_8188.log", "the real live log\n")
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded=str(stale),
        background=("127.0.0.1", 8188, 1234),
    )

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["data"]["path"] == str(stale)
    assert env["data"]["source"] == "recorded"
    assert env["data"]["port_mismatch"] is True
    assert env["data"]["size"] == 0
    assert env["data"]["mtime"]


def test_logs_metadata_reports_mtime_and_size(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = _write_log(tmp_path, "comfyui_8188.log", "abc\n")
    os.utime(log, (1_700_000_000, 1_700_000_000))
    _fake_env(monkeypatch, workspace=tmp_path, background=("127.0.0.1", 8188, 1234))

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["data"]["source"] == "derived_port"
    assert env["data"]["size"] == 4
    assert env["data"]["mtime"] == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc).isoformat()
    assert env["data"]["port_mismatch"] is False


def test_logs_no_log_file_lists_every_candidate(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded="/recorded/comfyui_9000.log",
        background=("127.0.0.1", 8188, 1234),
    )

    with pytest.raises(typer.Exit):
        launch.logs(tail=10)

    message = _envelope(capsys)["error"]["message"]
    assert "/recorded/comfyui_9000.log" in message
    assert str(tmp_path / "user" / "comfyui_8188.log") in message
    assert str(tmp_path / "user" / "comfyui.log") in message
    assert str(tmp_path / "user" / "comfyui_*.log") in message


# --------------------------------------------------------------------------- #
# `comfy logs --port N`
# --------------------------------------------------------------------------- #


def test_logs_port_serves_that_ports_log(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    wanted = _write_log(tmp_path, "comfyui_8189.log", "port 8189\n")
    _write_log(tmp_path, "comfyui_8188.log", "port 8188\n")
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded=str(tmp_path / "user" / "comfyui_8188.log"),
        background=("127.0.0.1", 8188, 1234),
    )

    launch.logs(tail=10, port=8189)

    env = _envelope(capsys)
    assert env["data"]["path"] == str(wanted)
    assert env["data"]["source"] == "explicit_port"
    assert env["data"]["lines"] == ["port 8189\n"]
    # The served file's port differs from the live background server's.
    assert env["data"]["port_mismatch"] is True


def test_logs_port_falls_back_to_unsuffixed(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    unsuffixed = _write_log(tmp_path, "comfyui.log", "no --port in argv\n")
    _fake_env(monkeypatch, workspace=tmp_path)

    launch.logs(tail=10, port=8189)

    env = _envelope(capsys)
    assert env["data"]["path"] == str(unsuffixed)
    assert env["data"]["source"] == "fallback_unsuffixed"


def test_logs_port_with_no_match_errors_listing_both_candidates(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    _write_log(tmp_path, "comfyui_8188.log", "not the requested port\n")
    _fake_env(monkeypatch, workspace=tmp_path)

    with pytest.raises(typer.Exit):
        launch.logs(tail=10, port=8189)

    env = _envelope(capsys)
    assert env["error"]["code"] == "no_log_file"
    message = env["error"]["message"]
    assert str(tmp_path / "user" / "comfyui_8189.log") in message
    assert str(tmp_path / "user" / "comfyui.log") in message
    # The restricted walk must not silently widen back to the other port's log.
    assert "comfyui_8188.log" not in message


def test_logs_pretty_warns_on_port_mismatch(monkeypatch, tmp_path, capsys):
    stale = _write_log(tmp_path, "comfyui_8189.log", "stale\n")
    _fake_env(
        monkeypatch,
        workspace=tmp_path,
        recorded=str(stale),
        background=("127.0.0.1", 8188, 1234),
    )

    launch.logs(tail=10)

    out = capsys.readouterr().out
    assert "8189" in out and "8188" in out
    assert "stale\n" in out


def test_logs_payload_matches_published_schema(monkeypatch, tmp_path, capsys):
    """The `--json` payload is a published contract (`comfy --json discover`)."""
    import jsonschema

    from comfy_cli import discovery

    _force_json_renderer()
    _write_log(tmp_path, "comfyui_8188.log", "line\n")
    _fake_env(monkeypatch, workspace=tmp_path, background=("127.0.0.1", 8188, 1234))

    launch.logs(tail=10)

    schema = discovery.load_all_schemas()["logs"]
    jsonschema.Draft202012Validator(schema).validate(_envelope(capsys)["data"])


def test_resolve_glob_handles_workspace_with_glob_metacharacters(monkeypatch, tmp_path):
    # A workspace path like `/Users/a[1]/comfy` must not be treated as a pattern.
    workspace = tmp_path / "ws[1]"
    workspace.mkdir()
    log = _write_log(workspace, "comfyui_9001.log", "found me\n")
    _fake_env(monkeypatch, workspace=workspace)

    assert launch.resolve_background_log_path() == (str(log), "fallback_glob")
