"""Tests for `comfy logs` — the background ComfyUI log tail reader + verb.

Covers the pure tail reader (last-N and the line/byte caps), the no-log-file
error envelope, the success envelope shape, the `--where` guard, and that the
background monitor redirects the child's own fds to a truncate-on-launch
workspace logfile and records its path.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer

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
    missing = tmp_path / "user" / "comfyui_8188.log"
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(missing))

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
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: None)

    with pytest.raises(typer.Exit):
        launch.logs(tail=50)

    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "no_log_file"


def test_logs_success_envelope(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("".join(f"line {i}\n" for i in range(5)))
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(log))

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
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(tmp_path / "x.log"))

    with pytest.raises(typer.Exit):
        launch.logs(tail=50, where="cloud")

    env = _envelope(capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "where_invalid"


def test_logs_where_local_is_accepted(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("hello\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(log))

    launch.logs(tail=10, where="local")

    env = _envelope(capsys)
    assert env["ok"] is True
    assert env["data"]["lines"] == ["hello\n"]


def test_logs_read_error_emits_clean_error(monkeypatch, tmp_path, capsys):
    _force_json_renderer()
    log = tmp_path / "comfyui_8188.log"
    log.write_text("hello\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(log))

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
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(log))

    launch.logs(tail=n)

    out = capsys.readouterr().out
    assert "line 0\n" in out  # earliest line present → nothing was capped away
    assert f"line {n - 1}\n" in out


def test_logs_pretty_writes_raw_lines(monkeypatch, tmp_path, capsys):
    # Default renderer is pretty; log text with '[...]' must not be reinterpreted.
    log = tmp_path / "comfyui_8188.log"
    log.write_text("[INFO] hello [world]\nplain\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda: str(log))

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
    from comfy_cli import constants

    assert cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND_LOG] == log_path
    assert "8188" in cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND]
    # Written twice: the log path is recorded up front (so a crash log is
    # findable even if startup fails) and again with the background info on success.
    assert cfg.write_config.call_count == 2
    # The logfile the child wrote is on disk with both lines.
    assert "To see the GUI go to:" in (tmp_path / "user" / "comfyui_8188.log").read_text()
