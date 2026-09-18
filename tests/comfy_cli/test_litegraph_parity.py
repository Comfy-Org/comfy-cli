"""Tests for the LiteGraph constant-parity guard.

These exercise the PARSER against fixture strings, offline. The guard's own network
fetch is not tested here on purpose: a unit test that reaches GitHub is a flake, and
the thing most likely to break silently is the regex, not urllib.

The fixtures below are verbatim excerpts of the upstream declarations as of
ComfyUI_frontend main on 2026-09-17, so the "matches" case is a real parse of real
source rather than a shape invented to satisfy the regex.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_litegraph_parity.py"
_SPEC = importlib.util.spec_from_file_location("check_litegraph_parity", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - only if the script is deleted
    raise RuntimeError(f"cannot load {_SCRIPT}")
parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(parity)


GLOBALS_FIXTURE = """
export class LiteGraphGlobal {
  NODE_TITLE_HEIGHT = 30
  NODE_TITLE_TEXT_Y = 20
  NODE_SLOT_HEIGHT = 20
  NODE_WIDGET_HEIGHT = 20
  NODE_WIDTH = 140
  NODE_MIN_WIDTH = 50
  NODE_COLLAPSED_RADIUS = 10
  NODE_COLLAPSED_WIDTH = 80
  NODE_TITLE_COLOR = '#999'
  NODE_TEXT_SIZE = 14
}
"""

WIDGET_FIXTURE = """
export abstract class BaseWidget {
  static margin = 15
  static arrowMargin = 6
  static arrowWidth = 10
  static minValueWidth = 42
}
"""

# Upstream main as of 2026-09-17, after the refactor that introduced the
# `LGraphCanvas._measureText?.()` branch and changed the length expression.
NODE_FIXTURE = """
    function compute_text_size(text: string | undefined, fontStyle: string) {
      const value = text ?? ''
      return (
        LGraphCanvas._measureText?.(value, fontStyle) ??
        font_size * value.length * 0.6
      )
    }
"""

# The shape that shipped before that refactor. Kept because the guard must read both:
# repos in the wild are pinned to older frontend versions, and `--ref` can name one.
NODE_FIXTURE_LEGACY = """
    function compute_text_size(text: string, fontStyle: string) {
      return ctx
        ? ctx.measureText(text).width
        : font_size * (text?.length ?? 0) * 0.6
    }
"""


def test_finds_each_constant():
    for symbol, expected in [
        ("NODE_TITLE_HEIGHT", 30.0),
        ("NODE_SLOT_HEIGHT", 20.0),
        ("NODE_WIDGET_HEIGHT", 20.0),
        ("NODE_WIDTH", 140.0),
        ("NODE_TEXT_SIZE", 14.0),
    ]:
        assert parity.find_assignment(GLOBALS_FIXTURE, symbol) == expected


def test_similar_names_resolve_to_their_own_values():
    """Each symbol must resolve to its own declaration, not a neighbour's.

    On today's upstream these happen to be safe even without the word anchor --
    `NODE_WIDTH` is not a substring of `NODE_WIDGET_HEIGHT`, and `margin` differs
    from `arrowMargin` by case. That is luck, not design, so pin the current
    behaviour here and prove the anchor separately in the test below.
    """
    assert parity.find_assignment(GLOBALS_FIXTURE, "NODE_WIDTH") == 140.0
    assert parity.find_assignment(GLOBALS_FIXTURE, "NODE_MIN_WIDTH") == 50.0
    assert parity.find_assignment(WIDGET_FIXTURE, "margin") == 15.0
    assert parity.find_assignment(WIDGET_FIXTURE, "arrowMargin") == 6.0


def test_word_anchor_prevents_matching_a_longer_name():
    """The anchor's actual job, shown on inputs where dropping it gives a wrong answer.

    Without the leading `(?:^|\\s)`, searching for `margin` matches `topmargin = 99`
    and searching for `WIDGET_HEIGHT` matches `NODE_WIDGET_HEIGHT = 20`. Both return
    a plausible number from the wrong declaration, which is worse than returning
    nothing: the guard would report drift that does not exist, or -- if the values
    happened to agree -- report parity it never checked. Upstream only has to add one
    such name for this to start mattering.
    """
    suffix = "  static topmargin = 99\n  static margin = 15\n"
    assert parity.find_assignment(suffix, "margin") == 15.0

    prefix = "  NODE_WIDGET_HEIGHT = 20\n  WIDGET_HEIGHT = 7\n"
    assert parity.find_assignment(prefix, "WIDGET_HEIGHT") == 7.0


def test_detects_a_changed_value():
    """The guard's whole purpose: upstream moves, we notice."""
    drifted = GLOBALS_FIXTURE.replace("NODE_TITLE_HEIGHT = 30", "NODE_TITLE_HEIGHT = 34")
    assert parity.find_assignment(drifted, "NODE_TITLE_HEIGHT") == 34.0


def test_returns_none_when_renamed():
    """A rename must be distinguishable from a match.

    Returning None (rather than a default) is what lets the guard exit 2 -- 'could
    not check' -- instead of claiming parity it never verified.
    """
    renamed = GLOBALS_FIXTURE.replace("NODE_TITLE_HEIGHT", "NODE_HEADER_HEIGHT")
    assert parity.find_assignment(renamed, "NODE_TITLE_HEIGHT") is None


def test_assignment_ignores_stale_comment_and_string_matches():
    source = """
// NODE_TITLE_HEIGHT = 99
const note = 'NODE_TITLE_HEIGHT = 88 // not a comment';
/* NODE_TITLE_HEIGHT = 77 */
NODE_TITLE_HEIGHT = 30
"""
    assert parity.find_assignment(source, "NODE_TITLE_HEIGHT") == 30.0


def test_finds_glyph_width_fallback():
    assert parity.find_char_fallback(NODE_FIXTURE) == 0.6


def test_finds_glyph_width_fallback_in_the_pre_refactor_shape():
    """Regression: the first version of this guard only matched this older shape.

    It exited 2 against upstream main because the length expression had changed from
    `(text?.length ?? 0)` to `value.length`. Both must parse -- a caller can point
    `--ref` at an older frontend, and the constant being checked did not move.
    """
    assert parity.find_char_fallback(NODE_FIXTURE_LEGACY) == 0.6


def test_glyph_fallback_ignores_stale_comment_and_template_matches():
    source = """
// return font_size * old.length * 9.9
const note = `font_size * old.length * 8.8 // literal`;
/* return font_size * old.length * 7.7 */
return font_size * value.length * 0.6
"""
    assert parity.find_char_fallback(source) == 0.6


def test_glyph_width_fallback_absent_returns_none():
    assert parity.find_char_fallback("return ctx.measureText(text).width") is None


def test_glyph_width_fallback_none_when_the_constant_is_gone():
    """If upstream stops multiplying by a constant, say so rather than guess.

    Returning None here is what makes the difference between the guard reporting
    'could not check' and quietly claiming a parity it never established.
    """
    no_constant = "return font_size * value.length"
    assert parity.find_char_fallback(no_constant) is None


@pytest.mark.parametrize(
    "decl,expected",
    [
        ("  FOO = 30", 30.0),
        ("  FOO = 30.5", 30.5),
        ("  static FOO = 30", 30.0),
        ("  FOO=30", 30.0),
        ("  FOO = -4", -4.0),
    ],
)
def test_assignment_shapes(decl, expected):
    """Upstream is TypeScript written by hand; spacing and `static` vary."""
    assert parity.find_assignment(decl, "FOO") == expected
