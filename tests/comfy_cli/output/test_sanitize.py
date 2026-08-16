"""Unit tests for the pretty-mode control-sequence sanitizer."""

from __future__ import annotations

import pytest

from comfy_cli.output.sanitize import (
    close_open_sgr,
    sanitize,
    sanitize_log_markup,
    sanitize_markup,
    sanitize_optional,
    sanitize_terminal_stream,
    sanitize_value,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # CSI: clear screen, cursor moves, SGR colors.
        ("job \x1b[2Jevil running", "job evil running"),
        ("\x1b[1;31mred\x1b[0m", "red"),
        ("up\x1b[3Aover", "upover"),
        # OSC: window-title rewrite, terminated by BEL or by ST.
        ("a\x1b]0;pwned\x07b", "ab"),
        ("a\x1b]0;pwned\x1b\\b", "ab"),
        # DCS / APC / PM / SOS.
        ("a\x1bPq#0;2\x1b\\b", "ab"),
        ("a\x1b_G payload \x1b\\b", "ab"),
        # Two-character escapes and charset designators.
        ("a\x1b(Bb", "ab"),
        ("a\x1b=b", "ab"),
        # 8-bit C1 forms of CSI and OSC.
        ("a\x9b2Jb", "ab"),
        ("a\x9d0;pwned\x9cb", "ab"),
        # 8-bit SOS, the C1 counterpart of ``\x1bX`` — consumed as a unit, so
        # its payload does not survive as visible garbage.
        ("a\x98payload\x9cb", "ab"),
        ("a\x1bXpayload\x1b\\b", "ab"),
        # Bare control characters, including a lone trailing ESC and DEL.
        ("a\x00b\x07c\x7fd", "abcd"),
        ("trailing\x1b", "trailing"),
        ("trailing\x9b", "trailing"),
        # ``\x9b c`` is a complete 8-bit CSI (Device Attributes), so the final
        # byte goes with it — over-eager by one character in the pathological
        # case where U+009B appears in otherwise-legitimate text, which it
        # never does.
        ("bare\x9bcsi", "baresi"),
        # Carriage return is stripped: it lets a message repaint a printed line.
        ("real output\rspoofed", "real outputspoofed"),
        ("windows\r\nlines", "windows\nlines"),
        # Tab and newline are layout, not control — they survive.
        ("col1\tcol2\nrow2", "col1\tcol2\nrow2"),
        # Ordinary text, including non-ASCII, is untouched.
        ("", ""),
        ("plain ascii", "plain ascii"),
        ("héllo · 世界 ✓", "héllo · 世界 ✓"),
        # Rich markup is NOT stripped here — ``sanitize`` only removes the
        # bytes a terminal acts on. Neutralizing markup is ``sanitize_markup``'s
        # job, and only for sinks that actually parse it.
        ("[bold]styled[/bold]", "[bold]styled[/bold]"),
    ],
)
def test_sanitize_strips_control_sequences(raw: str, expected: str):
    assert sanitize(raw) == expected


def test_sanitize_output_never_contains_escape():
    payload = "x\x1b[2J\x1b]0;t\x07\x1bPz\x1b\\\x9b1m\x7f\x00y"
    cleaned = sanitize(payload)
    assert "\x1b" not in cleaned
    assert "\x9b" not in cleaned
    assert cleaned == "xy"


def test_unterminated_osc_consumes_the_tail():
    # A terminal would swallow the rest of the stream looking for the
    # terminator, so dropping the tail is the faithful reading.
    assert sanitize("before\x1b]0;never terminated") == "before"


def test_sanitize_is_idempotent():
    once = sanitize("job \x1b[2Jevil\x1b]0;t\x07")
    assert sanitize(once) == once


def test_sanitize_optional_passes_none_through():
    assert sanitize_optional(None) is None
    assert sanitize_optional("a\x1b[2Jb") == "ab"


def test_sanitize_optional_stringifies_non_strings():
    # A caller that ignores the ``str | None`` annotation must not hit the
    # ``TypeError`` ``re.sub`` raises on a non-string.
    assert sanitize_optional(42) == "42"


def test_sanitize_value_stringifies_non_strings():
    assert sanitize_value(42) == "42"
    assert sanitize_value(None) == "None"
    assert sanitize_value("\x1b[2Jkeep") == "keep"


def test_sanitize_markup_escapes_rich_markup():
    # Rich would turn these back into a live OSC 8 hyperlink / a restyled line.
    assert sanitize_markup("[link=https://attacker.example]x[/link]") == (r"\[link=https://attacker.example]x\[/link]")
    assert sanitize_markup("[red]spoof[/red]") == r"\[red]spoof\[/red]"


def test_sanitize_markup_also_strips_control_sequences():
    out = sanitize_markup("job \x1b[2J[bold]evil[/bold]")
    assert "\x1b" not in out
    assert out == r"job \[bold]evil\[/bold]"


def test_sanitize_markup_leaves_markup_free_text_untouched():
    # The overwhelmingly common case must be byte-identical to sanitize_value.
    for text in ["plain ascii", "col1\tcol2\nrow2", "héllo · 世界 ✓", "8188"]:
        assert sanitize_markup(text) == sanitize_value(text) == text


def test_sanitize_markup_stringifies_non_strings():
    assert sanitize_markup(42) == "42"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # SGR survives byte-for-byte — that is the whole point of this variant.
        ("\x1b[31mred\x1b[0m", "\x1b[31mred\x1b[0m"),
        ("\x1b[mreset-shorthand", "\x1b[mreset-shorthand"),
        ("\x1b[1;38;2;255;0;0mtruecolor\x1b[0m", "\x1b[1;38;2;255;0;0mtruecolor\x1b[0m"),
        # Sub-parameter colons (ITU T.416 style) are still SGR.
        ("\x1b[38:2:255:0:0mcolon", "\x1b[38:2:255:0:0mcolon"),
        # Everything else a terminal acts on still goes: erase, cursor moves,
        # scroll region, private modes, and an 'm' final byte reached through
        # private/intermediate bytes rather than plain SGR.
        ("job \x1b[2Jevil", "job evil"),
        ("up\x1b[3Aover", "upover"),
        ("a\x1b[?25lb", "ab"),
        ("a\x1b[>1mb", "ab"),
        ("a\x1b[1 mb", "ab"),
        # 8-bit C1 CSI is stripped even when it *is* an SGR: terminals in UTF-8
        # mode do not honor it, so keeping it would only leak bytes.
        ("a\x9b31mb", "ab"),
        ("a\x9b2Jb", "ab"),
        # OSC/DCS/APC, properly terminated.
        ("a\x1b]0;pwned\x07b", "ab"),
        ("a\x1b]0;pwned\x1b\\b", "ab"),
        ("a\x1bPq#0;2\x1b\\b", "ab"),
        ("a\x1b_G payload \x1b\\b", "ab"),
        # ...and unterminated, where this variant differs from ``sanitize``:
        # the sequence ends at the newline instead of eating the rest of the
        # log. Everything below the stray introducer survives.
        ("a\x1b]0;never terminated\nkeep me\n", "a\nkeep me\n"),
        ("a\x1bPnever terminated\nkeep me\n", "a\nkeep me\n"),
        ("a\x9d0;never terminated\nkeep me\n", "a\nkeep me\n"),
        # On the last line there is no newline to stop at, so the tail does go —
        # the introducer's payload is unknowable and must not be printed.
        ("before\x1b]0;never terminated", "before"),
        # Two-character escapes / charset designators.
        ("a\x1b(Bb", "ab"),
        ("a\x1b=b", "ab"),
        # Stray control bytes, including a lone trailing ESC and DEL.
        ("a\x00b\x07c\x7fd", "abcd"),
        ("trailing\x1b", "trailing"),
        ("trailing\x9b", "trailing"),
        # Layout survives — and unlike ``sanitize``, so does CR: a tqdm line
        # recorded in the log must replay as one rewritten line.
        ("col1\tcol2\nrow2", "col1\tcol2\nrow2"),
        ("50%|#####     |\r100%|##########|\n", "50%|#####     |\r100%|##########|\n"),
        ("windows\r\nlines", "windows\r\nlines"),
        # Rich markup is text here — this sink never parses it.
        ("[INFO] hello [world]", "[INFO] hello [world]"),
        ("[/red]", "[/red]"),
        ("", ""),
        ("héllo · 世界 ✓", "héllo · 世界 ✓"),
    ],
)
def test_sanitize_terminal_stream_keeps_sgr_and_cr(raw: str, expected: str):
    assert sanitize_terminal_stream(raw) == expected


def test_sanitize_terminal_stream_leaves_only_sgr_escapes():
    payload = "x\x1b[2J\x1b]0;t\x07\x1b[32mgreen\x1b[0m\x1bPz\x1b\\\x9b1m\x7f\x00y"
    cleaned = sanitize_terminal_stream(payload)
    assert cleaned == "x\x1b[32mgreen\x1b[0my"
    assert "\x9b" not in cleaned


def test_sanitize_terminal_stream_is_idempotent():
    once = sanitize_terminal_stream("\x1b[31mred\x1b[0m\x1b[2Jgone\r\n")
    assert sanitize_terminal_stream(once) == once


def test_sanitize_terminal_stream_keeps_the_log_below_a_stray_introducer():
    # The tail-swallow regression: one unterminated introducer must cost the
    # line it is on, not every line after it.
    raw = "line one\n\x1b]0;truncator\nline two\nline three\n"
    assert sanitize_terminal_stream(raw) == "line one\n\nline two\nline three\n"


def test_sanitize_terminal_stream_does_not_split_kept_sequences():
    # The regression the single-pass design exists to prevent: a two-pass
    # implementation (escapes, then a residual control-char sweep) would strip
    # the ESC out of a preserved SGR and leave a bare literal '[31m' on screen.
    assert sanitize_terminal_stream("\x1b[31mred") == "\x1b[31mred"


# --------------------------------------------------------------------------- #
# sanitize_log_markup — captured log into a markup-parsing sink
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Brackets are neutralized rather than parsed — an unbalanced closer
        # would otherwise raise MarkupError from inside the failure handler.
        ("[/red] boom", r"\[/red] boom"),
        # Rich only escapes what it would otherwise parse, so a bracketed run
        # that is not a tag (the usual log-level prefix) passes through as-is.
        ("[INFO] tail", "[INFO] tail"),
        # Escapes go entirely, colour included: the panel does its own styling.
        ("\x1b[31mred\x1b[0m", "red"),
        ("a\x1b[2Jb", "ab"),
        ("a\x9b31mb", "ab"),
        # CR goes too, unlike the stream variant: the Panel lays out its lines.
        ("real\rspoofed", "realspoofed"),
        # Layout survives.
        ("col1\tcol2\nrow2", "col1\tcol2\nrow2"),
        ("", ""),
    ],
)
def test_sanitize_log_markup_neutralizes_markup_and_escapes(raw: str, expected: str):
    assert sanitize_log_markup(raw) == expected


def test_sanitize_log_markup_keeps_the_traceback_below_a_stray_introducer():
    # Why this exists instead of plain ``sanitize_markup``: one stray '\x1b]'
    # byte in the capture would otherwise truncate the panel there, discarding
    # exactly the traceback the panel is being printed to show.
    raw = "Traceback:\n\x1b]0;stray\n  File 'x.py', line 1\nRuntimeError: boom\n"
    out = sanitize_log_markup(raw)

    assert "RuntimeError: boom" in out
    assert "x.py" in out
    assert sanitize_markup(raw) == "Traceback:\n"  # the behavior being avoided


def test_sanitize_log_markup_stringifies_non_strings():
    assert sanitize_log_markup(42) == "42"


# --------------------------------------------------------------------------- #
# close_open_sgr — containing style bleed in a replayed stream
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A line that leaves a style open is closed at its own newline, so the
        # style cannot hide or restyle the lines below it.
        ("\x1b[8mconcealed\nvisible\n", "\x1b[8mconcealed\x1b[0m\nvisible\n"),
        ("\x1b[31mred\nplain\n", "\x1b[31mred\x1b[0m\nplain\n"),
        # A line that already resets gains nothing, in any spelling of reset.
        ("\x1b[31mred\x1b[0m\n", "\x1b[31mred\x1b[0m\n"),
        ("\x1b[31mred\x1b[m\n", "\x1b[31mred\x1b[m\n"),
        ("\x1b[31mred\x1b[0;0m\n", "\x1b[31mred\x1b[0;0m\n"),
        # Only the *last* SGR on a line decides — an early reset followed by a
        # new colour still leaves the line open.
        ("\x1b[31ma\x1b[0m b\x1b[32mc\n", "\x1b[31ma\x1b[0m b\x1b[32mc\x1b[0m\n"),
        # Text with no SGR at all is returned unchanged, byte for byte.
        ("plain one\nplain two\n", "plain one\nplain two\n"),
        ("", ""),
        # An unterminated final line is closed too — there is no newline to
        # hide behind, and the shell prompt follows it directly.
        ("\x1b[31mno trailing newline", "\x1b[31mno trailing newline\x1b[0m"),
        # '\r' is a within-line repaint (tqdm), not a line break: one reset for
        # the whole progress line, not one per rewrite.
        ("\x1b[32m50%\r\x1b[32m100%\n", "\x1b[32m50%\r\x1b[32m100%\x1b[0m\n"),
    ],
)
def test_close_open_sgr(raw: str, expected: str):
    assert close_open_sgr(raw) == expected


def test_close_open_sgr_is_idempotent():
    once = close_open_sgr("\x1b[8ma\nb\n\x1b[31mc")
    assert close_open_sgr(once) == once


def test_close_open_sgr_ignores_non_sgr_escapes():
    # It runs after ``sanitize_terminal_stream``, so nothing else should be
    # left — but it must not treat a stray erase sequence as an open style.
    assert close_open_sgr("\x1b[2Jtext\n") == "\x1b[2Jtext\n"


def test_sanitize_value_leaves_container_repr_inert():
    # ``str()`` on a container reprs its members, which already renders an ESC
    # as the four visible characters ``\x1b`` — inert, so it survives as text.
    out = sanitize_value(["\x1b[2Ja"])
    assert "\x1b" not in out
    assert out == "['\\x1b[2Ja']"
