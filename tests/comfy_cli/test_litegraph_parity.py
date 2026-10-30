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

from comfy_cli import layout

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


# Verbatim from upstream `LGraphNode.computeSize` on main as of 2026-10-29. Both numbers
# accumulate into the same variable, which is why each finder has to exclude the other's
# line rather than just search for a number near `widgets_height`.
COMPUTE_SIZE_FIXTURE = """
    let widgets_height = 0
    if (widgets?.length) {
      for (const widget of widgets) {
        if (!this.isWidgetRowVisible(widget)) continue

        let widget_height = 0
        if (widget.computeSize) {
          widget_height += widget.computeSize(size[0])[1]
        } else {
          widget_height += LiteGraph.NODE_WIDGET_HEIGHT
        }
        widgets_height += widget_height + 4
      }
      widgets_height += 8
    }
"""

# Verbatim from upstream `createImageHost` in src/scripts/ui/imagePreview.ts.
IMAGE_HOST_FIXTURE = """
  function updateSize() {
    if (currentImgs) {
      let elH = el.clientHeight
      if (first) {
        first = false
        // On first run, if we are small then grow a bit
        if (elH < 190) {
          elH = 190
        }
"""

# Verbatim from upstream `useStringWidget.ts`.
STRING_WIDGET_FIXTURE = """
  widget.options.minNodeSize = [400, 200]
"""


def test_finds_the_widget_row_gap_and_block_pad_separately():
    """Both land in `widgets_height`; each finder must read its own line.

    This is the pair that was unguarded longest, and it is the one that matters most on
    a widget-heavy node: the row gap is charged per widget, so an upstream change to it
    scales with the widget count while the block pad does not.
    """
    assert parity.find_widget_row_gap(COMPUTE_SIZE_FIXTURE) == 4.0
    assert parity.find_widget_block_pad(COMPUTE_SIZE_FIXTURE) == 8.0


def test_row_gap_finder_does_not_read_the_block_pad():
    """Drift one and only one, and only that one must move.

    If either finder could match the other's line they would agree on a single number,
    which reports parity on a term that has actually drifted.
    """
    drifted_pad = COMPUTE_SIZE_FIXTURE.replace("widgets_height += 8", "widgets_height += 12")
    assert parity.find_widget_row_gap(drifted_pad) == 4.0
    assert parity.find_widget_block_pad(drifted_pad) == 12.0

    drifted_gap = COMPUTE_SIZE_FIXTURE.replace("widget_height + 4", "widget_height + 6")
    assert parity.find_widget_row_gap(drifted_gap) == 6.0
    assert parity.find_widget_block_pad(drifted_gap) == 8.0


def test_widget_accumulator_finders_return_none_when_reshaped():
    """A refactor must read as 'could not check', never as parity."""
    assert parity.find_widget_row_gap("widgetsHeight += widget_height + 4") is None
    assert parity.find_widget_block_pad("widgetsHeight += 8") is None


def test_finds_the_image_host_floor_not_the_comparison_or_the_clientheight():
    """`elH` is assigned twice; only the numeric assignment is the floor.

    `let elH = el.clientHeight` must not match (it would yield None and mask the real
    value), and neither must `if (elH < 190)`, which happens to carry the same number
    today -- so matching it would pass for the wrong reason and keep passing after
    upstream changed only the assignment.
    """
    assert parity.find_image_host_min_height(IMAGE_HOST_FIXTURE) == 190.0

    drifted = IMAGE_HOST_FIXTURE.replace("elH = 190", "elH = 240")
    assert parity.find_image_host_min_height(drifted) == 240.0


def test_image_host_floor_absent_returns_none():
    assert parity.find_image_host_min_height("let elH = el.clientHeight") is None


def test_finds_the_multiline_min_node_width():
    assert parity.find_multiline_min_node_width(STRING_WIDGET_FIXTURE) == 400.0
    drifted = STRING_WIDGET_FIXTURE.replace("[400, 200]", "[520, 200]")
    assert parity.find_multiline_min_node_width(drifted) == 520.0


def test_multiline_min_node_width_takes_the_width_not_the_height():
    """The tuple is [width, height] and the height is a different constant.

    Reading the second term would compare a height against a width and report drift
    that does not exist -- or agree by accident and check nothing.
    """
    assert parity.find_multiline_min_node_width("minNodeSize: [400, 200]") == 400.0
    assert parity.find_multiline_min_node_width("minNodeSize = [111, 222]") == 111.0


def test_multiline_min_node_width_absent_returns_none():
    assert parity.find_multiline_min_node_width("widget.options.minNodeSize") is None


def test_every_geometry_constant_is_classified():
    """The gate that stops a new fitted constant joining the unguarded set silently.

    Four constants were unguarded for weeks for no better reason than that nobody was
    counting: upstream states all four, they just are not `NAME = value` declarations.
    A constant added to layout.py now has to be declared checkable, derived,
    browser-fitted or placer policy, and the failure names it.
    """
    gaps = parity.coverage()["unclassified"]
    assert gaps == set(), (
        f"unclassified geometry constants in layout.py: {sorted(gaps)}. Add each to "
        "CHECKS/DERIVED_CHECKS if upstream states it, to BROWSER_FITTED with a fit date "
        "if only a browser can, or to PLACER_POLICY if upstream has no opinion on it."
    )


def test_coverage_buckets_are_disjoint():
    """A constant in two buckets is a constant whose provenance is ambiguous."""
    cov = parity.coverage()
    buckets = ["checked", "derived", "fitted", "policy"]
    for i, first in enumerate(buckets):
        for second in buckets[i + 1 :]:
            overlap = cov[first] & cov[second]
            assert overlap == set(), f"{first} and {second} both claim {sorted(overlap)}"


def test_browser_fitted_values_match_the_live_constants():
    """Editing a browser-fitted constant must require re-recording how it was fitted.

    These are the two values no static check can verify, so the registry's recorded
    value is the only thing tying them to a measurement. Letting them drift apart would
    leave the guard reporting a fit date for a number that was never fitted.
    """
    for name, fitted_at, when, how in parity.BROWSER_FITTED:
        assert getattr(layout, name) == fitted_at, (
            f"{name} is now {getattr(layout, name)} but BROWSER_FITTED still records "
            f"{fitted_at} from {when}. Re-fit it against rendered geometry and update "
            "the entry, or the guard's 'not checked here' report is stale."
        )
        assert when and how, f"{name} needs a fit date and a method"


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


# --- complete numeric literals ------------------------------------------------------
#
# Every finder used to capture `(-?[\d.]+)` with nothing required after it, so it read
# the leading digits of whatever came next and reported the old number as parity. These
# are the shapes that made that happen. The required outcome is None -- "could not
# check" -- never a partially-read number, because a wrong number that happens to equal
# the local constant is a silent pass on a value that has actually been reshaped.

PARTIAL_LITERALS = [
    pytest.param("8 * scale", id="scaled"),
    pytest.param("4 + extraGap", id="extra-term"),
    pytest.param("4e1", id="exponent"),
    pytest.param("4_0", id="numeric-separator"),
    pytest.param("4.5.6", id="malformed"),
]


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_row_gap_rejects_a_partial_literal(tail):
    source = f"widgets_height += widget_height + {tail}\n"
    assert parity.find_widget_row_gap(source) is None


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_block_pad_rejects_a_partial_literal(tail):
    source = f"widgets_height += {tail}\n"
    assert parity.find_widget_block_pad(source) is None


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_image_host_floor_rejects_a_partial_literal(tail):
    source = f"if (elH < 190) {{\n  elH = {tail}\n}}\n"
    assert parity.find_image_host_min_height(source) is None


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_multiline_min_width_rejects_a_partial_literal(tail):
    assert parity.find_multiline_min_node_width(f"minNodeSize = [{tail}, 200]\n") is None


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_assignment_rejects_a_partial_literal(tail):
    assert parity.find_assignment(f"  NODE_TITLE_HEIGHT = {tail}\n", "NODE_TITLE_HEIGHT") is None


@pytest.mark.parametrize("tail", PARTIAL_LITERALS)
def test_glyph_fallback_rejects_a_partial_literal(tail):
    assert parity.find_char_fallback(f"return font_size * value.length * {tail}\n") is None


def test_a_malformed_literal_does_not_crash_the_run():
    """`4.5.6` reaching `float()` raised, and an exception exits 1 -- which reads as drift.

    Returning None instead routes it to the 'could not check' path, where a value nobody
    can parse belongs.
    """
    assert parity.find_widget_block_pad("widgets_height += 4.5.6\n") is None


def test_block_pad_does_not_read_a_reordered_row_gap():
    """`widgets_height += 4 + widget_height` is the row gap written the other way round.

    The two finders are separated by which side of the `+=` starts with a number, so
    reordering the row gap's terms used to make the block-pad finder read 4 and report
    drift against 8 -- a failure on a file where nothing had changed.
    """
    assert parity.find_widget_block_pad("widgets_height += 4 + widget_height\n") is None


# --- identifier anchoring -----------------------------------------------------------


def test_block_pad_does_not_match_a_longer_accumulator_name():
    """`\\b` does not help here: in `total_widgets_height` there is no word boundary
    before `widgets`, and in `node.widgets_height` there is one -- and that is a
    different variable. Both must miss."""
    assert parity.find_widget_block_pad("total_widgets_height += 8\n") is None
    assert parity.find_widget_block_pad("node.widgets_height += 8\n") is None


def test_row_gap_does_not_match_a_longer_accumulator_name():
    assert parity.find_widget_row_gap("total_widgets_height += widget_height + 4\n") is None


def test_image_host_floor_ignores_other_height_assignments():
    """`re.search` takes the first match anywhere in a 120-line file.

    Unscoped, `elH\\s*=\\s*<number>` read a *substring* match as the floor: `labelH = 20`
    contains `elH = 20`, and an earlier `let elH = 0` reset is the same identifier at the
    wrong moment. Both returned a plausible number from the wrong place -- false drift, or
    false parity if it happened to agree. (`maxElH` is below as filler only: its `ElH` is
    capitalised, so it never matched -- scoping to the guarded `if (elH < N) { elH = N }`
    block ties the number to its meaning regardless of how a decoy is spelled.)
    """
    decoys = """
      let labelH = 20
      let maxElH = 300
      let elH = 0
      el.style.setProperty('--comfy-widget-min-height', elH.toString())
    """
    assert parity.find_image_host_min_height(decoys) is None
    assert parity.find_image_host_min_height(decoys + IMAGE_HOST_FIXTURE) == 190.0


def test_image_host_floor_reads_the_assignment_not_the_threshold():
    """Upstream may grow below one number and grow to another; layout.py models the
    second. Drifting only the threshold must leave the answer alone."""
    threshold_only = IMAGE_HOST_FIXTURE.replace("elH < 190", "elH < 150")
    assert parity.find_image_host_min_height(threshold_only) == 190.0


def test_image_host_floor_none_when_the_guard_is_removed():
    """A bare `elH = 190` with no surrounding guard is not the first-render floor."""
    assert parity.find_image_host_min_height("  elH = 190\n") is None


# --- minNodeSize ---------------------------------------------------------------------


def test_multiline_min_width_checks_a_symbolic_height_beside_it():
    """Only the width term is modelled, so the height must not have to be a literal.

    `[400, MIN_TEXTAREA_H]` previously returned None and exited 2, reporting that a
    perfectly readable width could not be checked.
    """
    assert parity.find_multiline_min_node_width("minNodeSize = [400, MIN_TEXTAREA_H]\n") == 400.0


def test_multiline_min_width_none_when_a_second_widget_sets_one():
    """Two `minNodeSize` declarations means the guard can no longer tell whose floor it
    read. Reporting 'could not check' beats comparing against a single-line widget's."""
    two = "widget.options.minNodeSize = [200, 80]\nwidget.options.minNodeSize = [400, 200]\n"
    assert parity.find_multiline_min_node_width(two) is None


# --- the coverage walker -------------------------------------------------------------
#
# `layout_constants` only saw single-target `Assign` nodes whose entire right-hand side
# was a bare `ast.Constant`. Everything below was invisible to it, so
# `test_every_geometry_constant_is_classified` could pass while a new unguarded constant
# sat in layout.py -- the exact failure that gate exists to prevent.


def test_walker_sees_a_tuple_valued_constant():
    """`ORIGIN` and `DEFAULT_SIZE` are numeric pairs and were both unseen."""
    found = parity.constants_in("ORIGIN = (40.0, 60.0)\nDEFAULT_SIZE = [210.0, 100.0]\n")
    assert found == {"ORIGIN": (40.0, 60.0), "DEFAULT_SIZE": (210.0, 100.0)}


def test_walker_sees_an_annotated_constant():
    assert parity.constants_in("GAP: float = 4.0\n") == {"GAP": 4.0}


def test_walker_sees_a_negative_constant():
    """`-5.0` is `UnaryOp(USub, Constant)`, not a negative constant."""
    assert parity.constants_in("OFFSET = -5.0\nRAISED = +5.0\n") == {"OFFSET": -5.0, "RAISED": 5.0}


def test_walker_sees_every_name_in_a_chained_assignment():
    assert parity.constants_in("A = B = 1.0\n") == {"A": 1.0, "B": 1.0}


def test_walker_does_not_call_a_bool_or_a_string_geometry():
    """`True` is an `int` to `isinstance`, and neither it nor a string is a constant this
    gate should demand a geometry bucket for."""
    assert parity.constants_in("FLAG = True\nNAME = 'x'\nTYPES = frozenset({'Note'})\n") == {}


def test_walker_rejects_a_tuple_that_is_not_all_numbers():
    """A mixed tuple is not geometry, and must not be reported as a number."""
    assert parity.constants_in("PAIR = (40.0, 'y')\nEMPTY = ()\n") == {}


def test_expression_valued_names_are_reported_rather_than_dropped():
    """The counterpart bucket: a derived value is covered by its terms, but losing it
    entirely is how a shape the walker cannot read becomes silence instead of noise."""
    source = "A = 1.0\nPAD = A + 2.0 * 3.0\nTYPES = frozenset({'Note'})\n"
    assert parity.expressions_in(source) == {"PAD"}


def test_the_only_derived_expression_in_layout_is_the_declared_one():
    """Pins the list, so a new expression-valued geometry name has to be looked at.

    Between this and `test_every_geometry_constant_is_classified`, every module-level
    binding in layout.py that mentions a number is accounted for.
    """
    assert parity.numeric_expressions() == set(parity.DERIVED_EXPRESSIONS), (
        "layout.py has an expression-valued numeric name that DERIVED_EXPRESSIONS does "
        "not list. If it is derived from constants that are themselves covered, add it; "
        "if it is a new geometry fact, it needs a bucket in coverage() instead."
    )


def test_the_tuple_constants_are_classified():
    """Guards the fix rather than the bug: ORIGIN and DEFAULT_SIZE became visible to the
    coverage gate in the same change, so they had to be given a bucket at the same time."""
    policy = parity.coverage()["policy"]
    assert {"ORIGIN", "DEFAULT_SIZE"} <= policy
    assert {"ORIGIN", "DEFAULT_SIZE"} <= set(parity.layout_constants())


# --- one unreadable source must not hide drift in a readable one ---------------------
#
# `imagePreview.ts` and the Vue-nodes `useStringWidget.ts` are outside `lib/litegraph`
# and move far more often than the LiteGraph globals do. A single fail-fast fetch loop
# made the stable checks hostage to them: either file being moved, renamed, or absent on
# an older `--ref` returned 2 before one comparison ran, and exit 2 means "could not
# check" -- so real drift in NODE_TITLE_HEIGHT went unreported.


def _frontend_tree(tmp_path, *, omit=()):
    """A minimal ComfyUI_frontend layout on disk, for `--source-dir`."""
    files = {
        parity.GLOBALS_PATH: GLOBALS_FIXTURE,
        parity.WIDGET_PATH: WIDGET_FIXTURE,
        parity.NODE_PATH: NODE_FIXTURE + COMPUTE_SIZE_FIXTURE,
        parity.IMAGE_PREVIEW_PATH: IMAGE_HOST_FIXTURE,
        parity.STRING_WIDGET_PATH: STRING_WIDGET_FIXTURE,
    }
    for path, text in files.items():
        if path in omit:
            continue
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def test_a_complete_source_tree_passes(tmp_path):
    """The control for the two tests below: these fixtures agree with layout.py."""
    assert parity.main(source_dir=str(_frontend_tree(tmp_path))) == 0


def test_drift_is_still_reported_when_a_derived_source_is_missing(tmp_path, capsys):
    """Exit 1 beats exit 2: a mismatch found is a mismatch reported."""
    tree = _frontend_tree(tmp_path, omit=(parity.IMAGE_PREVIEW_PATH, parity.STRING_WIDGET_PATH))
    globals_file = tree / parity.GLOBALS_PATH
    globals_file.write_text(
        GLOBALS_FIXTURE.replace("NODE_TITLE_HEIGHT = 30", "NODE_TITLE_HEIGHT = 34"),
        encoding="utf-8",
    )

    assert parity.main(source_dir=str(tree)) == 1
    out = capsys.readouterr()
    assert "NODE_TITLE_HEIGHT" in out.err
    assert "CONSTANTS HAVE DRIFTED" in out.err


def test_checks_whose_source_was_read_still_run_when_another_is_missing(tmp_path, capsys):
    """Everything readable is compared; only the dependent checks become 'missing'."""
    tree = _frontend_tree(tmp_path, omit=(parity.IMAGE_PREVIEW_PATH,))

    assert parity.main(source_dir=str(tree)) == 2
    out = capsys.readouterr()
    assert "ok  NODE_TITLE_HEIGHT" in out.out
    assert "ok  MULTILINE_MIN_WIDTH" in out.out
    assert parity.IMAGE_PREVIEW_PATH in out.err
    assert "IMAGE_PREVIEW_MIN_H" not in out.out


def test_browser_fitted_is_reported_on_a_failing_run(tmp_path, capsys):
    """The docstring promises 'every run'. Behind the early returns it printed only on
    exit 0 -- the one run where nobody needs reminding what was not checked."""
    tree = _frontend_tree(tmp_path, omit=(parity.STRING_WIDGET_PATH,))

    assert parity.main(source_dir=str(tree)) == 2
    assert "NOT CHECKED HERE" in capsys.readouterr().out
