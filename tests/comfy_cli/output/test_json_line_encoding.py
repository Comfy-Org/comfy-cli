"""A JSON envelope survives a stdout that cannot encode every character."""

from __future__ import annotations

import io
import json

from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


def _json_renderer_on(stream) -> Renderer:
    r = Renderer.resolve(is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True)
    r.mode = OutputMode.JSON
    r.machine_stream = stream
    set_renderer(r)
    return r


def test_envelope_is_written_escaped_when_stdout_cannot_encode_it():
    """Windows: a redirected stdout is cp1252, and "→" is not in it. The
    UnicodeEncodeError is a ValueError, which the write path treats as "no
    stream at all" — so the envelope vanished and `comfy --json skills list`
    printed nothing. The same envelope goes out ASCII-escaped instead."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    try:
        _json_renderer_on(stream).emit({"description": "template → fragment"}, command="skill list")
        line = raw.getvalue().decode("cp1252")
        assert line.endswith("\n")
        assert "\\u2192" in line, line
        assert json.loads(line)["data"]["description"] == "template → fragment"
    finally:
        reset_renderer_for_testing()


def test_envelope_stays_unescaped_on_a_utf8_stream():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
    try:
        _json_renderer_on(stream).emit({"description": "template → fragment"}, command="skill list")
        assert "→" in raw.getvalue().decode("utf-8")
    finally:
        reset_renderer_for_testing()
