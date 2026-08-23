"""Renderer mode resolution + envelope shape."""

from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest

from comfy_cli.caller import Caller
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    _json_default,
    get_renderer,
    reset_renderer_for_testing,
    set_renderer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _resolve(**kwargs):
    return Renderer.resolve(
        is_stdout_tty=kwargs.pop("is_stdout_tty", True),
        env=kwargs.pop("env", {}),
        caller=kwargs.pop("caller", Caller(kind="user", agentic=False, source_env=None)),
        **kwargs,
    )


def test_default_is_pretty_when_tty():
    r = _resolve()
    assert r.mode is OutputMode.PRETTY


def test_no_tty_defaults_to_json():
    r = _resolve(is_stdout_tty=False)
    assert r.mode is OutputMode.JSON


def test_agentic_caller_defaults_to_json_even_when_tty():
    r = _resolve(caller=Caller(kind="claude", agentic=True, source_env="CLAUDECODE"))
    assert r.mode is OutputMode.JSON


def test_explicit_no_json_beats_agentic():
    r = _resolve(no_json_flag=True, caller=Caller(kind="claude", agentic=True, source_env="X"))
    assert r.mode is OutputMode.PRETTY


def test_json_stream_beats_json_flag():
    r = _resolve(json_flag=True, json_stream_flag=True)
    assert r.mode is OutputMode.NDJSON


def test_comfy_output_env_picks_mode():
    r = _resolve(env={"COMFY_OUTPUT": "ndjson"}, is_stdout_tty=True)
    assert r.mode is OutputMode.NDJSON


def test_flag_beats_env():
    r = _resolve(no_json_flag=True, env={"COMFY_OUTPUT": "json"})
    assert r.mode is OutputMode.PRETTY


def test_envelope_shape_on_success():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON  # bypass pretty branch
    r.machine_stream = stream
    r.command = "env"
    r.version = "1.2.3"
    r.emit({"foo": "bar"})
    line = stream.getvalue().strip()
    env = json.loads(line)
    # Contract versioning: the discriminator + schema version lead the
    # envelope so NDJSON consumers can pick out the final line by `type`.
    assert list(env)[:2] == ["schema", "type"]
    assert env["schema"] == "envelope/1"
    assert env["type"] == "envelope"
    assert env["ok"] is True
    assert env["command"] == "env"
    assert env["version"] == "1.2.3"
    assert env["data"] == {"foo": "bar"}
    assert env["error"] is None
    assert env["where"] is None


def test_emit_ok_false_carries_data():
    """`validate` on an invalid workflow emits its structured payload as data
    but with ok=False, so the envelope agrees with the exit code."""
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "validate"
    r.emit({"valid": False, "error_count": 2}, ok=False)
    env = json.loads(stream.getvalue().strip())
    assert env["ok"] is False
    assert env["data"] == {"valid": False, "error_count": 2}
    assert env["error"] is None  # the verdict rides in data, not error


def test_envelope_shape_on_error():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "which"
    r.error("not_in_workspace", "no workspace", hint="run: comfy install")
    env = json.loads(stream.getvalue().strip())
    assert env["schema"] == "envelope/1"
    assert env["type"] == "envelope"
    assert env["ok"] is False
    assert env["error"]["code"] == "not_in_workspace"
    assert env["error"]["message"] == "no workspace"
    assert env["error"]["hint"] == "run: comfy install"
    assert env["error"]["details"] is None
    assert r.exit_code == 1


def test_error_where_override_lands_on_envelope():
    """`where=` on error() stamps the routed target, mirroring emit()."""
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "run"
    r.error("workflow_unknown_nodes", "bad graph", where="cloud")
    env = json.loads(stream.getvalue().strip())
    assert env["ok"] is False
    assert env["where"] == "cloud"


def test_error_falls_back_to_renderer_where():
    """With no override, error() inherits the renderer's routed target — this
    is what the run path assigns once `--where` resolves."""
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "run"
    r.where = "local"
    r.error("server_not_running", "no server")
    env = json.loads(stream.getvalue().strip())
    assert env["where"] == "local"


def test_error_explicit_where_beats_renderer_where():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "run"
    r.where = "local"
    r.error("workflow_unknown_nodes", "bad graph", where="cloud")
    env = json.loads(stream.getvalue().strip())
    assert env["where"] == "cloud"


def test_error_hint_falls_back_to_registry():
    """An error emitted without an explicit hint inherits its code's REGISTERED
    navigation hint — so every error points toward correctness, never a dead end."""
    from comfy_cli import error_codes

    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "run"
    r.error("server_not_running", "no server")  # NO hint passed
    env = json.loads(stream.getvalue().strip())
    registered = error_codes.get("server_not_running").hint
    assert registered  # the code documents navigation
    assert env["error"]["hint"] == registered


def test_empty_string_hint_also_falls_back_to_registry():
    """Call sites commonly pass `hint=e.hint or ""`; a blank hint must still
    inherit the registered navigation, not emit a dead-end empty hint."""
    from comfy_cli import error_codes

    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.error("server_not_running", "no server", hint="")  # empty, not None
    env = json.loads(stream.getvalue().strip())
    assert env["error"]["hint"] == error_codes.get("server_not_running").hint


def test_explicit_hint_overrides_registry():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.error("server_not_running", "no server", hint="custom next step")
    env = json.loads(stream.getvalue().strip())
    assert env["error"]["hint"] == "custom next step"


def test_error_exit_code_is_observable_after_call():
    """main() reads renderer.exit_code as a backstop so call sites that
    forgot to also `raise typer.Exit(1)` still produce a non-zero process
    exit. Pin the public surface so that backstop keeps working."""
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = io.StringIO()
    assert r.exit_code == 0
    r.error("boom", "something broke")
    assert r.exit_code == 1
    # Custom exit codes (e.g. 130 for SIGINT) flow through too.
    r2 = _resolve()
    r2.mode = OutputMode.JSON
    r2.machine_stream = io.StringIO()
    r2.error("cancelled", "user cancelled", exit_code=130)
    assert r2.exit_code == 130


def test_emit_is_noop_in_pretty():
    stream = io.StringIO()
    r = _resolve()
    assert r.is_pretty()
    r.machine_stream = stream
    r.emit({"foo": "bar"})
    assert stream.getvalue() == ""


def test_pretty_emit_writes_nothing_to_stdout(pretty_no_stdout):
    """Regression: in pretty mode, ``renderer.emit`` is a no-op for stdout.

    Uses the conftest fixture that pins the contract — if a future change
    makes pretty mode print the envelope to stdout, this test fails.
    """
    r = _resolve()
    assert r.is_pretty()
    r.emit({"hello": "world"}, command="test")
    # Fixture asserts no stdout after the test body finishes.


def test_emit_is_idempotent():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.emit({"a": 1})
    r.emit({"a": 2})  # second call ignored
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["data"] == {"a": 1}


def test_event_only_in_ndjson():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON  # not ndjson
    r.machine_stream = stream
    r.event("progress", node="K", completed=1, total=2)
    assert stream.getvalue() == ""

    r.mode = OutputMode.NDJSON
    r.event("progress", node="K", completed=1, total=2)
    line = json.loads(stream.getvalue().strip())
    assert line == {"schema": "event/1", "type": "progress", "node": "K", "completed": 1, "total": 2}


def test_throttled_event_drops_within_window():
    import time as _time

    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.NDJSON
    r.machine_stream = stream
    # First emits, immediate second drops, then sleep > 1/max_hz emits again.
    assert r.throttled_event("k1", "progress", max_hz=20, v=1) is True
    assert r.throttled_event("k1", "progress", max_hz=20, v=2) is False
    _time.sleep(0.08)  # 1/20 = 0.05
    assert r.throttled_event("k1", "progress", max_hz=20, v=3) is True
    emitted = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [e["v"] for e in emitted] == [1, 3]


def test_singleton_set_and_get():
    r = _resolve()
    set_renderer(r)
    assert get_renderer() is r


def test_get_renderer_default_is_pretty():
    # No prior set; default is pretty so unsuspecting callers don't crash.
    r = get_renderer()
    assert r.is_pretty()


# ---------------------------------------------------------------------------
# Control-sequence sanitizing at the pretty boundary (BE-4794)
# ---------------------------------------------------------------------------

_EVIL = "job \x1b[2Jevil"


def _pretty_output(call) -> str:
    """Run ``call(renderer)`` in pretty mode and return what hit the stream.

    ``force_terminal`` is left off, so Rich emits no SGR of its own: every ESC
    in the result would have come from the message itself.
    """
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.PRETTY
    r.pretty_stream = stream
    call(r)
    return stream.getvalue()


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.warn(_EVIL),
        lambda r: r.warn("ok", hint=_EVIL),
        lambda r: r.info(_EVIL),
        lambda r: r.info("ok", hint=_EVIL),
        lambda r: r.success(_EVIL),
        lambda r: r.error("ws_disconnected", _EVIL, hint=_EVIL, details={"prompt_id": _EVIL}),
    ],
    ids=["warn", "warn-hint", "info", "info-hint", "success", "error-panel"],
)
def test_pretty_helpers_strip_escape_sequences(call):
    out = _pretty_output(call)
    assert "\x1b" not in out
    assert "evil" in out  # the text survives; only the escape is gone


# Rich re-creates escape sequences from *markup* it finds in the text, so
# stripping the raw bytes above is only half the boundary. These need a real
# terminal: Rich only emits OSC 8 hyperlinks when the console is a tty.
_MARKUP_LINK = "job [link=https://attacker.example]click[/link] done"
_MARKUP_UNBALANCED = "server said [/] oops"
_RICH_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def force_terminal(monkeypatch):
    """Make Rich treat the StringIO stream as a color terminal.

    ``COLUMNS`` is pinned too: these assertions look for an un-wrapped
    substring, and ``rich.print`` gives no way to pass ``width`` through, so
    without it the result depends on the runner's default console width.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.warn(_MARKUP_LINK),
        lambda r: r.warn("ok", hint=_MARKUP_LINK),
        lambda r: r.info(_MARKUP_LINK),
        lambda r: r.info("ok", hint=_MARKUP_LINK),
        lambda r: r.success(_MARKUP_LINK),
        lambda r: r.error("ws_disconnected", "boom", details={"prompt_id": _MARKUP_LINK}),
    ],
    ids=["warn", "warn-hint", "info", "info-hint", "success", "error-panel-details"],
)
def test_pretty_helpers_do_not_let_markup_forge_escape_sequences(call, force_terminal):
    """``[link=...]`` must not become a live OSC 8 hyperlink.

    Only the SGR codes Rich emits for its own styling may remain; once those
    are stripped, any surviving ESC was manufactured from the payload.
    """
    out = _pretty_output(call)
    assert "\x1b]8;" not in out  # no hyperlink sequence
    assert "\x1b" not in _RICH_SGR_RE.sub("", out)
    assert "link=https://attacker.example" in _RICH_SGR_RE.sub("", out)  # shown as inert text


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.warn(_MARKUP_UNBALANCED),
        lambda r: r.warn("ok", hint=_MARKUP_UNBALANCED),
        lambda r: r.info(_MARKUP_UNBALANCED),
        lambda r: r.info("ok", hint=_MARKUP_UNBALANCED),
        lambda r: r.success(_MARKUP_UNBALANCED),
        lambda r: r.error("ws_disconnected", "boom", details={"prompt_id": _MARKUP_UNBALANCED}),
    ],
    ids=["warn", "warn-hint", "info", "info-hint", "success", "error-panel-details"],
)
def test_pretty_helpers_survive_unbalanced_markup(call):
    """An unbalanced ``[/]`` used to raise ``MarkupError`` and kill the CLI
    while it was merely printing a line."""
    out = _pretty_output(call)
    assert "oops" in out


def test_json_error_envelope_is_unchanged_by_sanitizing():
    """The machine path must stay byte-identical: ``json.dumps`` already
    neutralizes ESC as ``\\u001b``, and stripping it would mutate data agents
    parse."""
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    r.command = "run"
    r.error("ws_disconnected", _EVIL, hint=_EVIL, details={"prompt_id": "\x1b[2Jx"})
    raw = stream.getvalue()

    # Byte level: escaped, never raw — and never stripped.
    assert "\x1b" not in raw
    assert raw.count("\\u001b") == 3
    # Value level: the payload round-trips exactly as it went in.
    env = json.loads(raw.strip())
    assert env["error"]["message"] == _EVIL
    assert env["error"]["hint"] == _EVIL
    assert env["error"]["details"] == {"prompt_id": "\x1b[2Jx"}


def test_ndjson_event_payload_is_unchanged_by_sanitizing():
    stream = io.StringIO()
    r = _resolve()
    r.mode = OutputMode.NDJSON
    r.machine_stream = stream
    r.event("progress", prompt_id="\x1b[2Jx")
    raw = stream.getvalue()
    assert "\x1b" not in raw
    assert "\\u001b" in raw
    assert json.loads(raw.strip())["prompt_id"] == "\x1b[2Jx"


def test_pretty_helpers_tolerate_non_string_messages():
    """The f-string interpolation these helpers used to do coerced anything to
    ``str``; sanitizing must not turn a loose caller into a TypeError."""
    assert "42" in _pretty_output(lambda r: r.warn(42))
    assert "42" in _pretty_output(lambda r: r.info(42))
    assert "42" in _pretty_output(lambda r: r.success(42))


class TestResolveSurvivesAnUnusableStdout:
    """``Renderer.resolve`` runs from the main Typer callback (``cmdline.py``)
    before any command dispatch, and every call site there omits
    ``is_stdout_tty``, so it probes ``sys.stdout`` itself.

    Under ``pythonw``, a Windows service, or a detached parent, ``sys.stdout``
    is ``None`` or an already-closed file. A bare ``sys.stdout.isatty()`` there
    raises before argument parsing, which takes down EVERY command — including
    ``comfy --help`` and runs with tracking disabled. The probe is routed
    through ``caller.stream_is_tty``, so it degrades to "not a TTY" (→ JSON
    mode, the correct read of a process with no terminal) instead.
    """

    def test_missing_stdout_resolves_to_json_not_attribute_error(self, monkeypatch):
        import comfy_cli.output.renderer as renderer_mod

        monkeypatch.setattr(renderer_mod.sys, "stdout", None)
        r = Renderer.resolve(env={})
        assert r.mode is OutputMode.JSON

    def test_closed_stdout_resolves_to_json_not_value_error(self, monkeypatch, tmp_path):
        import comfy_cli.output.renderer as renderer_mod

        handle = open(tmp_path / "out.txt", "w")
        handle.close()
        monkeypatch.setattr(renderer_mod.sys, "stdout", handle)
        assert Renderer.resolve(env={}).mode is OutputMode.JSON

    def test_revoked_fd_stdout_resolves_to_json_not_os_error(self, monkeypatch):
        import comfy_cli.output.renderer as renderer_mod

        class Revoked:
            def isatty(self):
                raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(renderer_mod.sys, "stdout", Revoked())
        assert Renderer.resolve(env={}).mode is OutputMode.JSON

    def test_a_live_tty_still_resolves_to_pretty(self, monkeypatch):
        """The guard must not flip healthy terminals to JSON."""
        import comfy_cli.output.renderer as renderer_mod

        class Tty:
            def isatty(self):
                return True

        monkeypatch.setattr(renderer_mod.sys, "stdout", Tty())
        assert Renderer.resolve(env={}).mode is OutputMode.PRETTY


class TestWritesTolerateADeadMachineStream:
    """Resolving to JSON mode against an unusable stdout must not merely DEFER
    the crash to the first emit.

    `machine_stream` falls back to `sys.stdout`, so surviving `resolve` only to
    die inside `_write_json_line` is strictly worse than dying at startup: by
    then the command has run and its side effects have landed. A stream that
    cannot be written to cannot receive output, so the write is a no-op and the
    process still exits with the right code.
    """

    def _dead_stdout_renderer(self, monkeypatch, stream):
        import comfy_cli.output.renderer as renderer_mod

        monkeypatch.setattr(renderer_mod.sys, "stdout", stream)
        r = Renderer.resolve(env={})
        assert r.mode is OutputMode.JSON
        return r

    def test_missing_stdout_emit_is_a_noop(self, monkeypatch):
        r = self._dead_stdout_renderer(monkeypatch, None)
        r.emit({"hello": "world"})  # must not raise
        assert r.exit_code == 0

    def test_missing_stdout_error_still_sets_the_exit_code(self, monkeypatch):
        r = self._dead_stdout_renderer(monkeypatch, None)
        r.error(code="server_not_running", message="nope", exit_code=1)
        assert r.exit_code == 1

    def test_closed_stdout_emit_is_a_noop(self, monkeypatch, tmp_path):
        handle = open(tmp_path / "out.txt", "w")
        handle.close()
        r = self._dead_stdout_renderer(monkeypatch, handle)
        r.error(code="server_not_running", message="nope", exit_code=1)
        assert r.exit_code == 1

    def test_broken_pipe_still_propagates(self, monkeypatch):
        """`OSError` is deliberately excluded from the guard. A BrokenPipeError
        means the stream was real and the reader hung up, and that has to keep
        propagating: `comfy cloud login` depends on it escaping the `login_url`
        emit to fail fast instead of blocking 300s on a browser callback nobody
        will read (test_json_login_fails_fast_when_login_url_write_breaks).
        Guarding "no usable stream" must not quietly become "ignore all I/O
        errors"."""

        class BrokenPipe:
            def isatty(self):
                return False

            def write(self, _):
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self):
                pass

        r = self._dead_stdout_renderer(monkeypatch, BrokenPipe())
        with pytest.raises(BrokenPipeError):
            r.emit({"hello": "world"})

    def test_a_malformed_payload_still_surfaces(self, monkeypatch):
        """The handler is deliberately narrower than `stream_is_tty`'s: a
        TypeError from the write means our payload is wrong, which is a bug and
        must stay visible rather than being silently swallowed."""

        class Fussy:
            def isatty(self):
                return False

            def write(self, _):
                raise TypeError("not a string")

            def flush(self):
                pass

        r = self._dead_stdout_renderer(monkeypatch, Fussy())
        with pytest.raises(TypeError):
            r.emit({"hello": "world"})


class TestJsonDefault:
    def test_a_non_string_isoformat_does_not_loop(self):
        class FakeStamp:
            def isoformat(self):
                return self  # a Mock or a bad stub does exactly this

        assert json.dumps({"when": FakeStamp()}, default=_json_default)

    def test_a_real_isoformat_is_still_used(self):
        assert _json_default(datetime(2026, 8, 23, tzinfo=timezone.utc)) == "2026-08-23T00:00:00+00:00"

    def test_an_enum_wrapping_an_opaque_value_is_coerced(self):
        class Wrapping(Enum):
            OPAQUE = object()

        assert _json_default(Wrapping.OPAQUE).startswith("<object object")

    def test_the_hook_runs_once_per_value(self):
        """Single-pass is the whole guarantee: json re-enters the hook for
        anything it still cannot encode, so one re-entry per value is the same
        defect as an endless one, caught earlier."""

        class Wrapping(Enum):
            OPAQUE = object()

        class BadStamp:
            def isoformat(self):
                return self

        seen = []

        def counted(obj):
            seen.append(obj)
            return _json_default(obj)

        json.dumps({"e": Wrapping.OPAQUE, "s": BadStamp(), "p": Path("/tmp/x")}, default=counted)
        assert len(seen) == 3
