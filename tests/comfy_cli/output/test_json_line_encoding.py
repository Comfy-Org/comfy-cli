"""A JSON envelope survives a stdout that cannot encode every character."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from comfy_cli.caller import Caller
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    _is_utf8_stream,
    reset_renderer_for_testing,
    set_renderer,
)


def _json_renderer_on(stream) -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
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


def test_envelope_is_escaped_whenever_the_stream_is_not_utf8():
    """The em-dash IS in cp1252 (0x97), so no UnicodeEncodeError fires and the
    escape path above never runs — the line goes out as legacy bytes. Every
    reader of this stream decodes it as UTF-8, where 0x97 is invalid: the
    comfy-agent stored `no output nodes � the server will reject it` for
    every validate failure on the Windows box. A non-UTF-8 stream gets the
    ASCII-escaped envelope, which any JSON reader decodes back."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    try:
        _json_renderer_on(stream).emit({"message": "no output nodes — rejected"}, command="workflow validate")
        written = raw.getvalue()
        assert written.decode("utf-8") == written.decode("ascii"), written
        assert json.loads(written.decode("utf-8"))["data"]["message"] == "no output nodes — rejected"
    finally:
        reset_renderer_for_testing()


def test_a_non_utf8_stream_receives_utf8_bytes():
    """Escaping to ASCII is not enough on a stream that re-encodes it.

    A UTF-16 wrapper turns even pure ASCII into two-byte sequences, so a
    consumer decoding stdout as UTF-8 cannot parse the envelope at all. The
    machine stream's contract is "a reader decodes this as UTF-8", so the line
    is written to the underlying binary buffer as UTF-8 when the text wrapper
    would encode it as anything else.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-16", write_through=True)
    try:
        _json_renderer_on(stream).emit({"message": "plain ascii"}, command="workflow validate")
        assert json.loads(raw.getvalue().decode("utf-8"))["data"]["message"] == "plain ascii"
    finally:
        reset_renderer_for_testing()


def test_text_written_before_the_envelope_keeps_its_order():
    """The envelope goes out through the binary buffer, so anything already
    buffered in the text wrapper has to be flushed ahead of it."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", write_through=False)
    try:
        stream.write("first\n")
        _json_renderer_on(stream).emit({"message": "second"}, command="workflow validate")
        assert raw.getvalue().decode("utf-8").splitlines()[0] == "first"
    finally:
        reset_renderer_for_testing()


@pytest.mark.parametrize("encoding", ["utf-8", "utf8", "UTF-8", "utf_8", "cp65001"])
def test_every_utf8_alias_counts_as_utf8(encoding):
    """A platform reports its encoding under whichever alias it likes —
    Windows says `cp65001` for UTF-8. Resolving through `codecs.lookup`
    instead of comparing the string keeps all of them on the unescaped path."""
    assert _is_utf8_stream(io.TextIOWrapper(io.BytesIO(), encoding=encoding))


@pytest.mark.parametrize(
    "stream",
    [io.StringIO(), SimpleNamespace(encoding=None), SimpleNamespace(encoding=""), SimpleNamespace()],
    ids=["stringio", "none", "empty", "absent"],
)
def test_a_stream_that_declares_no_encoding_counts_as_utf8(stream):
    # Nothing re-encodes it, so escaping would only hurt readability.
    assert _is_utf8_stream(stream)


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16", "latin-1", "not-a-codec", 42])
def test_anything_else_is_treated_as_not_utf8(encoding):
    # Including a name no codec claims and a non-string: the fallback only
    # ever escapes more, which is always safe.
    assert not _is_utf8_stream(SimpleNamespace(encoding=encoding))
