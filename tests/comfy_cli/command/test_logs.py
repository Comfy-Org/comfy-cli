"""Tests for `comfy logs` — the background ComfyUI log tail reader + verb.

Covers the pure tail reader (last-N and the line/byte caps), the no-log-file
error envelope, the success envelope shape, the `--where` guard, and that the
background monitor redirects the child's own fds to a truncate-on-launch
workspace logfile and records its path.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
from datetime import UTC, datetime
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


def _symlink_or_skip(link, target) -> None:
    """``link -> target``, skipping the test where the OS forbids symlinks.

    Windows only grants SeCreateSymbolicLinkPrivilege under Developer Mode or
    elevation, so an unprivileged runner raises OSError here. The behaviour under
    test (never follow a symlink out of the glob scan) is a shared-host POSIX
    concern, so skipping is the right degradation rather than failing the suite.
    """
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as e:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create symlinks on this platform: {e}")


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
    # Default renderer is pretty; log text with '[...]' must not be reinterpreted
    # as Rich markup. Escape bytes ARE stripped (see the sanitization tests
    # below), so this no longer pins the output byte-for-byte — but every
    # bracketed run has to survive verbatim, including an unbalanced closer that
    # would raise MarkupError from a markup-parsing sink.
    log = tmp_path / "comfyui_8188.log"
    log.write_text("[INFO] hello [world]\n[####  ] 50%\n[/red]\nplain\n")
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=10)

    out = capsys.readouterr().out
    assert "[INFO] hello [world]" in out
    assert "[####  ] 50%" in out
    assert "[/red]" in out
    assert "plain" in out


# --------------------------------------------------------------------------- #
# `comfy logs` pretty replay — terminal-control sanitization
# --------------------------------------------------------------------------- #


class _TtyStream(io.StringIO):
    """A pretty sink that claims to be a terminal, for the SGR-reset gate."""

    def isatty(self) -> bool:
        return True


def _logs_pretty_out(monkeypatch, tmp_path, capsys, contents: str, *, tty: bool = False) -> str:
    """Run pretty-mode ``logs()`` over ``contents`` and return what it printed.

    ``tty=True`` swaps in a sink that reports ``isatty()``, which is what gates
    the closing SGR resets — under pytest the captured stdout never is one.
    """
    log = tmp_path / "comfyui_8188.log"
    log.write_text(contents)
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))
    if not tty:
        launch.logs(tail=50)
        return capsys.readouterr().out

    renderer = Renderer.resolve(
        is_stdout_tty=True,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
    )
    renderer.mode = OutputMode.PRETTY
    renderer.pretty_stream = _TtyStream()
    set_renderer(renderer)
    launch.logs(tail=50)
    return renderer.pretty_stream.getvalue()


@pytest.mark.parametrize(
    ("contents", "banned"),
    [
        # OSC 0 retitles the user's terminal window; BEL-terminated form.
        ("start\x1b]0;evil\x07end\n", "\x1b]"),
        # ...and the unterminated form. A terminal would swallow forward looking
        # for the terminator; on a *replayed* log that would delete every line
        # below it, so the sanitizer bounds it at the newline instead — hence
        # the `end` on the following line, which must survive.
        ("start\x1b]0;evil\nend\n", "\x1b]"),
        ("start\x1bPdcs never terminated\nend\n", "\x1bP"),
        ("start\x9d0;evil\nend\n", "\x9d"),
        # CSI erase-display and cursor-up repaint CLI-owned lines above.
        ("start\x1b[2Jend\n", "\x1b[2J"),
        ("start\x1b[3Aend\n", "\x1b[3A"),
        # 8-bit C1 CSI — not honored by UTF-8 terminals, so it is stripped too.
        ("start\x9b31mend\n", "\x9b"),
    ],
)
def test_logs_pretty_strips_terminal_control_sequences(monkeypatch, tmp_path, capsys, contents, banned):
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, contents)

    assert banned not in out
    # Text on both sides of the stripped sequence survives: the point is to
    # remove the bytes the terminal acts on, not to truncate the log there.
    assert "start" in out
    assert "end" in out


def test_logs_pretty_keeps_lines_below_an_unterminated_introducer(monkeypatch, tmp_path, capsys):
    # The tail-swallow regression in full: a crashed process dumping one stray
    # introducer must not silently cut the replay short while the JSON path
    # still shows the rest.
    contents = "line one\n\x1b]0;truncator\nline two\nline three\n"
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, contents)

    assert "line one" in out
    assert "line two" in out
    assert "line three" in out


def test_logs_pretty_preserves_sgr_colour(monkeypatch, tmp_path, capsys):
    # The reason this is `sanitize_terminal_stream` and not `sanitize`: a
    # coloured ComfyUI log must still be coloured when replayed.
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, "\x1b[31mred\x1b[0m\n", tty=True)

    assert "\x1b[31mred\x1b[0m" in out


def test_logs_pretty_closes_styles_per_line(monkeypatch, tmp_path, capsys):
    # A log line that never resets (or conceals with \x1b[8m) would otherwise
    # style every line replayed after it — and then whatever the user types
    # next. Closing at the newline, not just once at the end, is what stops one
    # planted SGR from hiding the log lines below it.
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, "\x1b[8mconcealed\nvisible\n", tty=True)

    assert out == "\x1b[8mconcealed\x1b[0m\nvisible\n"


def test_logs_pretty_adds_no_reset_when_the_log_has_no_sgr(monkeypatch, tmp_path, capsys):
    # The overwhelmingly common case stays byte-identical to the pre-sanitize
    # output: an uncoloured log gains nothing.
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, "plain one\nplain two\n")

    assert out == "plain one\nplain two\n"


def test_logs_pretty_adds_no_reset_when_the_sink_is_not_a_tty(monkeypatch, tmp_path, capsys):
    # `comfy logs > file` / `| grep` has no terminal state to protect, so it
    # must not gain a reset the source file never contained. The log's own SGR
    # still replays — that byte *was* in the file.
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, "\x1b[8mconcealed\n")

    assert out == "\x1b[8mconcealed\n"


def test_logs_pretty_preserves_progress_lines_and_layout(monkeypatch, tmp_path, capsys):
    # Tabs and newlines are layout and survive. The '\r' a tqdm progress line is
    # recorded with survives the sanitizer too (see the unit tests) — but it
    # never reaches the sanitizer, because `read_log_tail` opens the file in the
    # default universal-newline mode, which normalizes '\r' to '\n' one layer
    # up. That is pre-existing reader behavior shared with the JSON envelope, so
    # this asserts what the pretty path actually prints rather than pinning a
    # '\r' the display layer never sees.
    out = _logs_pretty_out(monkeypatch, tmp_path, capsys, "50%|##   |\r100%|#####|\ncol1\tcol2\n")

    assert "50%|##   |" in out
    assert "100%|#####|" in out
    assert "col1\tcol2\n" in out


def test_logs_json_lines_are_byte_faithful(monkeypatch, tmp_path, capsys):
    # The machine path stays raw: stripping here would mutate the data agents
    # parse, and the envelope is not a terminal — its consumer decodes JSON
    # rather than acting on control sequences.
    #
    # Note the envelope is *not* escape-free the way the pretty path is. The
    # renderer dumps with `ensure_ascii=False`, which escapes C0 (so `\x1b`
    # leaves as a six-character `\u` escape) but emits U+0080-U+009F literally — the 8-bit C1
    # introducers `\x9b`/`\x90`/`\x9d` therefore cross the envelope raw. That is
    # deliberate for a machine sink; anything that later renders these lines to
    # a human terminal owes them a trip through `sanitize_terminal_stream`.
    _force_json_renderer()
    raw = "\x1b]0;evil\x07\x1b[2J\x1b[31mred\x1b[0m\x9b31m\n"
    log = tmp_path / "comfyui_8188.log"
    log.write_text(raw)
    monkeypatch.setattr(launch, "resolve_background_log_path", lambda port=None: (str(log), "recorded"))

    launch.logs(tail=50)

    env = _envelope(capsys)
    assert env["data"]["lines"] == [raw]


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
    assert env["data"]["mtime"] == datetime.fromtimestamp(1_700_000_000, tz=UTC).isoformat()
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
    # The served file's port differs from the live background server's, but the
    # user asked for it explicitly — flagging that as a mismatch (and advising a
    # retry on 8188) would contradict the request.
    assert env["data"]["port_mismatch"] is False


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


# --------------------------------------------------------------------------- #
# staleness / cross-workspace guards on the `recorded` pointer
# --------------------------------------------------------------------------- #


def test_stale_cross_workspace_recorded_log_does_not_shadow_local(monkeypatch, tmp_path, capsys):
    """A dead pointer into ANOTHER workspace must lose to this workspace's log.

    `clear_background_process` keeps CONFIG_KEY_BACKGROUND_LOG so a crash log
    stays reachable; without a workspace check that leftover pointer would be
    served from any other workspace, with no live server for `port_mismatch` to
    flag it.
    """
    _force_json_renderer()
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    stale = _write_log(other_ws, "comfyui_9000.log", "someone else's log\n")

    this_ws = tmp_path / "here"
    this_ws.mkdir()
    mine = _write_log(this_ws, "comfyui_8188.log", "my own log\n")

    _fake_env(monkeypatch, workspace=this_ws, recorded=str(stale), background=None)

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["data"]["path"] == str(mine)
    assert env["data"]["source"] == "default_port"
    assert env["data"]["lines"] == ["my own log\n"]


def test_cross_workspace_recorded_log_still_last_resort(monkeypatch, tmp_path):
    """Demoted, not dropped: with no local log at all it is still served."""
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    stale = _write_log(other_ws, "comfyui_9000.log", "last resort\n")
    this_ws = tmp_path / "here"
    this_ws.mkdir()

    _fake_env(monkeypatch, workspace=this_ws, recorded=str(stale), background=None)

    candidates = launch.candidate_log_paths()
    assert candidates[-1] == (str(stale), "recorded")
    assert launch.resolve_background_log_path() == (str(stale), "recorded")


def test_live_background_keeps_recorded_first_even_cross_workspace(monkeypatch, tmp_path):
    """A LIVE server's pointer is authoritative wherever it points."""
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    live_log = _write_log(other_ws, "comfyui_9000.log", "live\n")
    this_ws = tmp_path / "here"
    this_ws.mkdir()
    _write_log(this_ws, "comfyui_8188.log", "local\n")

    _fake_env(
        monkeypatch,
        workspace=this_ws,
        recorded=str(live_log),
        background=("127.0.0.1", 9000, 1234),
    )

    assert launch.candidate_log_paths()[0] == (str(live_log), "recorded")
    assert launch.resolve_background_log_path() == (str(live_log), "recorded")


def test_same_workspace_recorded_crash_log_stays_first(monkeypatch, tmp_path):
    """The crash-surviving pointer keeps priority inside its own workspace."""
    crash = _write_log(tmp_path, "comfyui_8189.log", "Traceback\n")
    _write_log(tmp_path, "comfyui_8188.log", "older default-port log\n")
    _fake_env(monkeypatch, workspace=tmp_path, recorded=str(crash), background=None)

    assert launch.resolve_background_log_path() == (str(crash), "recorded")


def test_glob_fallback_skips_symlinks(monkeypatch, tmp_path):
    """A planted symlink must not be followed, even as the newest match.

    Read-side counterpart of `_open_log_for_write`'s O_NOFOLLOW: on a shared host
    `<workspace>/user` may be writable by an attacker who links a `comfyui_*.log`
    name at a file outside the workspace.
    """
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY\n")
    real = _write_log(tmp_path, "comfyui_9001.log", "the real log\n")
    _symlink_or_skip(tmp_path / "user" / "comfyui_9999.log", secret)
    os.utime(real, (1_600_000_000, 1_600_000_000))
    # Backdate the real log and forward-date the link's TARGET: that is what the
    # pre-guard code compared (it stat'd through the link), so the link was the
    # newest match and won. It must now lose regardless.
    os.utime(secret, (1_900_000_000, 1_900_000_000))

    _fake_env(monkeypatch, workspace=tmp_path)

    assert launch.resolve_background_log_path() == (str(real), "fallback_glob")


def test_explicit_port_still_serves_a_deliberate_symlink(monkeypatch, tmp_path):
    """The symlink guard is scoped to the untrusted auto-scan, not to `--port`.

    Skipping symlinks in the newest-mtime glob must not take away a user's own
    deliberately symlinked logfile: naming it explicitly still resolves, because
    only the attacker-controllable "pick whatever is newest in this directory"
    step needs the guard.
    """
    real = tmp_path / "elsewhere.log"
    real.write_text("a deliberately symlinked log\n")
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _symlink_or_skip(user_dir / "comfyui_9001.log", real)

    _fake_env(monkeypatch, workspace=tmp_path)

    # Guarded: the auto-scan will not follow it...
    assert launch.resolve_background_log_path() is None
    # ...but asking for that port by name still does.
    assert launch.resolve_background_log_path(9001) == (
        str(user_dir / "comfyui_9001.log"),
        "explicit_port",
    )


def test_logs_metadata_survives_out_of_range_mtime(monkeypatch, tmp_path, capsys):
    """A corrupt st_mtime degrades to null metadata, it does not crash."""
    _force_json_renderer()
    log = _write_log(tmp_path, "comfyui_8188.log", "abc\n")
    _fake_env(monkeypatch, workspace=tmp_path, background=("127.0.0.1", 8188, 1234))

    real_stat = os.stat

    class _BadStat:
        # st_mode is real so the isfile() candidate check still works; only the
        # timestamp is corrupt.
        st_mode = real_stat(log).st_mode
        st_mtime = 1e30  # out of range for datetime.fromtimestamp
        st_size = 4

    monkeypatch.setattr(os, "stat", lambda p, *a, **k: _BadStat() if str(p) == str(log) else real_stat(p, *a, **k))

    launch.logs(tail=10)

    env = _envelope(capsys)
    assert env["data"]["mtime"] is None
    assert env["data"]["size"] is None
    assert env["data"]["lines"] == ["abc\n"]


def test_logs_pretty_no_mismatch_warning_for_explicit_port(monkeypatch, tmp_path, capsys):
    """`--port 8189` while 8188 runs must not advise retrying on 8188."""
    _write_log(tmp_path, "comfyui_8189.log", "asked for this\n")
    _fake_env(monkeypatch, workspace=tmp_path, background=("127.0.0.1", 8188, 1234))

    launch.logs(tail=10, port=8189)

    out = capsys.readouterr().out
    assert "asked for this\n" in out
    assert "8188" not in out
    assert "Warning" not in out


def test_logs_pretty_warns_when_explicit_port_falls_back_to_unsuffixed(monkeypatch, tmp_path, capsys):
    """The unsuffixed fallback encodes no port, so say so instead of staying silent."""
    _write_log(tmp_path, "comfyui.log", "no --port in argv\n")
    _fake_env(monkeypatch, workspace=tmp_path)

    launch.logs(tail=10, port=8189)

    out = capsys.readouterr().out
    assert "comfyui_8189.log was found" in out
    assert "does not" in out and "record which port" in out
    assert "no --port in argv\n" in out


def test_logs_pretty_warning_does_not_swallow_markup_in_path(monkeypatch, tmp_path, capsys):
    """A workspace path holding Rich-style tags must survive verbatim."""
    workspace = tmp_path / "ws[bold red]"
    workspace.mkdir()
    _write_log(workspace, "comfyui.log", "line\n")
    _fake_env(monkeypatch, workspace=workspace)

    launch.logs(tail=10, port=8189)

    # Rich soft-wraps the warning line to the console width, which can fold a
    # long tmp_path right inside the literal "[bold red]" text depending on
    # the absolute path length (environment-dependent) — strip the wrap
    # newlines rather than assert on a specific layout.
    out = capsys.readouterr().out.replace("\n", "")
    assert "ws[bold red]" in out
