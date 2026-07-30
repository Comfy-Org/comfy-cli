"""Unit tests for the pretty-mode control-sequence sanitizer."""

from __future__ import annotations

import pytest

from comfy_cli.output.sanitize import sanitize, sanitize_markup, sanitize_optional, sanitize_value


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


def test_sanitize_value_leaves_container_repr_inert():
    # ``str()`` on a container reprs its members, which already renders an ESC
    # as the four visible characters ``\x1b`` — inert, so it survives as text.
    out = sanitize_value(["\x1b[2Ja"])
    assert "\x1b" not in out
    assert out == "['\\x1b[2Ja']"
