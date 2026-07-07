"""Renderer mode resolution + envelope shape."""

from __future__ import annotations

import io
import json

import pytest

from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, get_renderer, reset_renderer_for_testing, set_renderer


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
