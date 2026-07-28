"""Unit tests for the pretty-mode control-sequence sanitizer."""

from __future__ import annotations

import pytest

from comfy_cli.output.sanitize import sanitize, sanitize_optional, sanitize_value


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
        # Rich markup is NOT stripped — call sites rely on it for styling.
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


def test_sanitize_value_stringifies_non_strings():
    assert sanitize_value(42) == "42"
    assert sanitize_value(None) == "None"
    assert sanitize_value("\x1b[2Jkeep") == "keep"


def test_sanitize_value_leaves_container_repr_inert():
    # ``str()`` on a container reprs its members, which already renders an ESC
    # as the four visible characters ``\x1b`` — inert, so it survives as text.
    out = sanitize_value(["\x1b[2Ja"])
    assert "\x1b" not in out
    assert out == "['\\x1b[2Ja']"
