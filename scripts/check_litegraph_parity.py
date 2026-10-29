#!/usr/bin/env python3
"""Check that layout.py's geometry constants still match LiteGraph's.

comfy_cli/layout.py places nodes from a *model* of how ComfyUI_frontend's LiteGraph
renders them. The model lives here, the renderer lives in another repository on
another release cadence in another language, and nothing currently fails when the
renderer moves. That is the n8n#38093 failure mode: their SDK sized a node 96x96
while the editor drew a 320x128 card, and nothing noticed.

This fetches the upstream sources and compares the values by NAME. It deliberately
does not compare line numbers -- the citations in layout.py's comments have already
drifted (they say LiteGraphGlobal.ts:61/64/65/71; the declarations are at 48/51/52/58),
which is the smaller version of exactly the problem this guard exists to catch.

Four more constants are not plain `NAME = value` declarations upstream -- they are
numbers buried inside an expression (`widgets_height += widget_height + 4`) or inside
a widget's option literal. They were unguarded for that reason alone, not because
upstream is silent about them, so each has a purpose-built finder below. Together
they are most of the height model: get the 4 or the 8 wrong on a widget-heavy node and
the placer leaves a gap it believes is clear, which is the overlap users report.

What it does NOT catch: a change to the *algorithm* computeSize uses, or to how the
canvas measures text. Only a rendered-geometry comparison catches
that, which needs a browser and therefore lives in the frontend's Playwright harness.

Two constants are genuinely browser-fitted and cannot be read from source at all; they
are listed in BROWSER_FITTED and reported on every run with the date they were fitted,
so "this guard passed" never reads as "the whole model is verified". `coverage()`
classifies every numeric constant in layout.py into exactly one of the four buckets and
`tests/comfy_cli/test_litegraph_parity.py` fails on an unclassified one, so a new fitted
constant cannot join the unguarded set silently -- which is how these four did.

Usage:
    python scripts/check_litegraph_parity.py            # fetch from upstream main
    python scripts/check_litegraph_parity.py --ref v1.2 # pin a ref
    python scripts/check_litegraph_parity.py --source-dir /path/to/ComfyUI_frontend

Exit 0 when every constant matches, 1 on a mismatch, 2 when the sources could not be
read (network, moved file). A fetch failure is deliberately distinct from a mismatch
so CI can treat "could not check" differently from "constants have drifted".
"""

from __future__ import annotations

import ast
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from comfy_cli import layout

RAW = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI_frontend/{ref}/{path}"

GLOBALS_PATH = "src/lib/litegraph/src/LiteGraphGlobal.ts"
WIDGET_PATH = "src/lib/litegraph/src/widgets/BaseWidget.ts"
NODE_PATH = "src/lib/litegraph/src/LGraphNode.ts"
IMAGE_PREVIEW_PATH = "src/scripts/ui/imagePreview.ts"
STRING_WIDGET_PATH = "src/renderer/extensions/vueNodes/widgets/composables/useStringWidget.ts"

LAYOUT_SOURCE = Path(layout.__file__)

console = Console(highlight=False)
error_console = Console(stderr=True, highlight=False)

# (upstream file, upstream symbol, layout.py constant, current local value)
CHECKS = [
    (GLOBALS_PATH, "NODE_TITLE_HEIGHT", "TITLE_H", layout.TITLE_H),
    (GLOBALS_PATH, "NODE_SLOT_HEIGHT", "SLOT_H", layout.SLOT_H),
    (GLOBALS_PATH, "NODE_WIDGET_HEIGHT", "WIDGET_H", layout.WIDGET_H),
    (GLOBALS_PATH, "NODE_WIDTH", "LG_NODE_WIDTH", layout.LG_NODE_WIDTH),
    (GLOBALS_PATH, "NODE_TEXT_SIZE", "NODE_TEXT_SIZE", layout.NODE_TEXT_SIZE),
    (WIDGET_PATH, "margin", "_WIDGET_MARGIN", layout._WIDGET_MARGIN),
    (WIDGET_PATH, "arrowMargin", "_WIDGET_ARROW_MARGIN", layout._WIDGET_ARROW_MARGIN),
    (WIDGET_PATH, "arrowWidth", "_WIDGET_ARROW_WIDTH", layout._WIDGET_ARROW_WIDTH),
    (
        WIDGET_PATH,
        "minValueWidth",
        "_WIDGET_MIN_VALUE_WIDTH",
        layout._WIDGET_MIN_VALUE_WIDTH,
    ),
]


def find_widget_row_gap(text: str) -> float | None:
    """The inter-row gap in `LGraphNode.computeSize`: `widgets_height += widget_height + 4`.

    Anchored on the accumulator name and on `widget_height` so it cannot match the
    block pad two lines below, which assigns a bare number to the same accumulator.
    """
    m = re.search(r"widgets_height\s*\+=\s*widget_height\s*\+\s*(-?[\d.]+)", active_typescript(text))
    return float(m.group(1)) if m else None


def find_widget_block_pad(text: str) -> float | None:
    """The once-per-block pad added after computeSize's widget loop: `widgets_height += 8`.

    Requiring a number straight after `+=` is what separates this from the row gap,
    whose right-hand side starts with an identifier.
    """
    m = re.search(r"widgets_height\s*\+=\s*(-?[\d.]+)", active_typescript(text))
    return float(m.group(1)) if m else None


def find_image_host_min_height(text: str) -> float | None:
    """`createImageHost`'s first-render floor: `if (elH < 190) { elH = 190 }`.

    Matched on the assignment rather than the comparison; `let elH = el.clientHeight`
    is skipped because the pattern demands a numeric right-hand side.
    """
    m = re.search(r"elH\s*=\s*(-?[\d.]+)", active_typescript(text))
    return float(m.group(1)) if m else None


def find_multiline_min_node_width(text: str) -> float | None:
    """A multiline string widget's `minNodeSize = [400, 200]` -- the width term.

    This is the floor a node carrying a textarea is widened to, which is why
    layout.py charges multiline nodes a different minimum width from ordinary ones.
    """
    m = re.search(r"minNodeSize\s*[:=]\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", active_typescript(text))
    return float(m.group(1)) if m else None


# Constants upstream states but not as a `NAME = value` declaration, so `find_assignment`
# cannot reach them. (upstream file, finder, upstream description, layout.py name, ours)
DERIVED_CHECKS: list[tuple[str, Callable[[str], float | None], str, str, float]] = [
    (
        NODE_PATH,
        find_widget_row_gap,
        "computeSize widget row gap",
        "_WIDGET_ROW_GAP",
        layout._WIDGET_ROW_GAP,
    ),
    (
        NODE_PATH,
        find_widget_block_pad,
        "computeSize widget block pad",
        "_WIDGET_BLOCK_PAD",
        layout._WIDGET_BLOCK_PAD,
    ),
    (
        IMAGE_PREVIEW_PATH,
        find_image_host_min_height,
        "createImageHost first-render floor",
        "IMAGE_PREVIEW_MIN_H",
        layout.IMAGE_PREVIEW_MIN_H,
    ),
    (
        STRING_WIDGET_PATH,
        find_multiline_min_node_width,
        "multiline widget minNodeSize width",
        "MULTILINE_MIN_WIDTH",
        layout.MULTILINE_MIN_WIDTH,
    ),
]

# Renderer facts with no readable source: these came out of measuring real rendered
# nodes in a browser, so only a rendered-geometry test can re-verify them. Recorded
# here with the value they were fitted at, so editing the constant without re-fitting
# it fails the test suite, and so every run of this guard says out loud which part of
# the model it did not check. (layout.py name, value when fitted, when, how)
BROWSER_FITTED: list[tuple[str, float, str, str]] = [
    (
        "MULTILINE_WIDGET_H",
        166.0,
        "2026-09-17",
        "twelve-class rendered-height fit (comfy-cli #887); a textarea is a text AREA, "
        "not a widget row, and its height is DOM/CSS-driven under Nodes 2.0",
    ),
    (
        "WIDGET_MIN_WIDTH",
        270.0,
        "2026-09-18",
        "observed floor for widget-bearing nodes (comfy-cli #882); the renderer's own "
        "MIN_NODE_WIDTH is 225, so this is a widget-driven composite, not that constant",
    ),
]

# Not renderer facts at all -- the placer's own spacing policy and loop bounds. Upstream
# has no opinion on these, so there is nothing to be in parity with.
PLACER_POLICY = frozenset(
    {
        "COL_GAP",
        "ROW_GAP",
        "NODE_W",
        "HEADER_H",
        "PAD_H",
        "MIN_H",
        "_MARGIN",
        "_GUARD",
        "_ORDER_SWEEPS",
    }
)


def layout_constants() -> dict[str, float]:
    """Every module-level numeric literal constant in layout.py, by name.

    Read from the AST rather than from `dir(layout)` so an imported number or a value
    derived from other constants (`_WIDGET_PADDING`) is not demanded to be classified:
    a derived value is already covered by whatever its terms are covered by.
    """
    tree = ast.parse(LAYOUT_SOURCE.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Constant):
            continue
        if isinstance(value.value, bool) or not isinstance(value.value, (int, float)):
            continue
        found[target.id] = float(value.value)
    return found


def coverage() -> dict[str, set[str]]:
    """Which bucket each geometry constant falls in, and what falls in none."""
    checked = {name for _, _, name, _ in CHECKS} | {"_CHAR_W"}
    derived = {name for _, _, _, name, _ in DERIVED_CHECKS}
    fitted = {name for name, _, _, _ in BROWSER_FITTED}
    classified = checked | derived | fitted | PLACER_POLICY
    return {
        "checked": checked,
        "derived": derived,
        "fitted": fitted,
        "policy": set(PLACER_POLICY),
        "unclassified": set(layout_constants()) - classified,
    }


def fetch(path: str, ref: str, source_dir: str | None) -> str:
    if source_dir:
        return (Path(source_dir) / path).read_text(encoding="utf-8")
    with urllib.request.urlopen(RAW.format(ref=ref, path=path), timeout=30) as r:
        return r.read().decode("utf-8")


def active_typescript(text: str) -> str:
    """Mask comments and literals while preserving source positions and newlines."""
    chars = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(chars):
        char = chars[i]
        following = chars[i + 1] if i + 1 < len(chars) else ""

        if state == "code":
            if char == "/" and following == "/":
                chars[i] = chars[i + 1] = " "
                state = "line_comment"
                i += 2
                continue
            if char == "/" and following == "*":
                chars[i] = chars[i + 1] = " "
                state = "block_comment"
                i += 2
                continue
            if char in {"'", '"', "`"}:
                quote = char
                chars[i] = " "
                state = "literal"
                i += 1
                continue
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[i] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                chars[i] = chars[i + 1] = " "
                state = "code"
                i += 2
                continue
            if char != "\n":
                chars[i] = " "
        else:
            if char == "\\":
                chars[i] = " "
                if i + 1 < len(chars):
                    if chars[i + 1] != "\n":
                        chars[i + 1] = " "
                    i += 2
                    continue
            if char == quote:
                state = "code"
            if char != "\n":
                chars[i] = " "
        i += 1
    return "".join(chars)


def find_assignment(text: str, symbol: str) -> float | None:
    """Find `[static] NAME = <number>` for an exact symbol name.

    Anchored on a word boundary so NODE_WIDTH does not match NODE_WIDGET_HEIGHT and
    margin does not match arrowMargin.
    """
    m = re.search(
        rf"(?:^|\s)(?:static\s+)?{re.escape(symbol)}\s*=\s*(-?[\d.]+)",
        active_typescript(text),
    )
    return float(m.group(1)) if m else None


def find_char_fallback(text: str) -> float | None:
    """LiteGraph's no-canvas glyph width in compute_text_size: font_size * <len> * 0.6.

    The middle term is whatever expression upstream currently uses for the string
    length, and it moves. It has been `(text?.length ?? 0)` and is now `value.length`
    after a 2026 refactor that also introduced the `LGraphCanvas._measureText?.()`
    branch. Matching the two multiplications and taking the trailing number survives
    that; matching a specific length expression did not -- the first version of this
    guard exited 2 on upstream main for exactly that reason.

    It is deliberately still a match rather than a wildcard: if upstream inserts a
    third factor, or stops multiplying by a constant at all, this returns None and the
    caller reports "could not check" instead of inventing parity.
    """
    m = re.search(r"font_size\s*\*\s*[^*;\n]+?\*\s*(-?[\d.]+)", active_typescript(text))
    return float(m.group(1)) if m else None


def main(ref: str = "main", source_dir: str | None = None) -> int:
    sources: dict[str, str] = {}
    needed = {GLOBALS_PATH, WIDGET_PATH, NODE_PATH} | {path for path, *_ in DERIVED_CHECKS}
    for path in needed:
        try:
            sources[path] = fetch(path, ref, source_dir)
        except Exception as exc:  # noqa: BLE001 - any read failure is the same outcome here
            error_console.print(f"could not read {path}: {exc}")
            return 2

    mismatches: list[str] = []
    missing: list[str] = []

    for path, symbol, ours_name, ours in CHECKS:
        theirs = find_assignment(sources[path], symbol)
        if theirs is None:
            missing.append(f"{symbol} not found in {path} -- renamed or moved upstream")
        elif theirs != ours:
            mismatches.append(f"{symbol}: upstream {theirs}, layout.py {ours_name} = {ours}")
        else:
            console.print(f"ok  {symbol:20} {theirs}")

    for path, finder, description, ours_name, ours in DERIVED_CHECKS:
        theirs = finder(sources[path])
        if theirs is None:
            missing.append(f"{description} not found in {path} -- reshaped upstream")
        elif theirs != ours:
            mismatches.append(f"{description}: upstream {theirs}, layout.py {ours_name} = {ours}")
        else:
            console.print(f"ok  {ours_name:20} {theirs}   ({description})")

    char_w = find_char_fallback(sources[NODE_PATH])
    if char_w is None:
        missing.append(f"compute_text_size glyph fallback not found in {NODE_PATH}")
    elif char_w != layout._CHAR_W:
        mismatches.append(f"glyph width fallback: upstream {char_w}, layout.py _CHAR_W = {layout._CHAR_W}")
    else:
        console.print(f"ok  {'_CHAR_W':20} {char_w}")

    if missing:
        error_console.print("\nSYMBOLS NOT FOUND (the check could not run, not a proven mismatch):")
        for m in missing:
            error_console.print(f"  {m}")
    if mismatches:
        error_console.print("\nCONSTANTS HAVE DRIFTED:")
        for m in mismatches:
            error_console.print(f"  {m}")
        error_console.print(
            "\ncomfy_cli/layout.py models LiteGraph's geometry. When these diverge the CLI\n"
            "places nodes against a picture the browser no longer draws, and the symptom is\n"
            "overlapping nodes on an agent-built canvas. Update layout.py to match, and\n"
            "re-baseline the quality metrics in tests/comfy_cli/test_layout.py.",
        )
    if mismatches:
        return 1
    if missing:
        return 2

    console.print("\nall upstream-readable geometry constants match")
    console.print("\nNOT CHECKED HERE -- browser-fitted, re-verify with a rendered-geometry test:")
    for name, fitted_at, when, how in BROWSER_FITTED:
        console.print(f"  {name:20} {fitted_at}   fitted {when}: {how}")
    return 0


def cli(
    ref: Annotated[
        str,
        typer.Option(help="ComfyUI_frontend git ref."),
    ] = "main",
    source_dir: Annotated[
        Path | None,
        typer.Option(help="Local ComfyUI_frontend checkout instead of fetching."),
    ] = None,
) -> None:
    """Compare local layout constants with ComfyUI_frontend's LiteGraph values."""
    raise typer.Exit(main(ref=ref, source_dir=str(source_dir) if source_dir else None))


if __name__ == "__main__":
    typer.run(cli)
