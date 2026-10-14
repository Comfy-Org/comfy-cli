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

What it does NOT catch: a change to the *algorithm* computeSize uses, or to how the
canvas measures text. Only a rendered-geometry comparison catches
that, which needs a browser and therefore lives in the frontend's Playwright harness.

Usage:
    python scripts/check_litegraph_parity.py            # fetch from upstream main
    python scripts/check_litegraph_parity.py --ref v1.2 # pin a ref
    python scripts/check_litegraph_parity.py --source-dir /path/to/ComfyUI_frontend

Exit 0 when every constant matches, 1 on a mismatch, 2 when the sources could not be
read (network, moved file). A fetch failure is deliberately distinct from a mismatch
so CI can treat "could not check" differently from "constants have drifted".
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from comfy_cli import layout

RAW = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI_frontend/{ref}/{path}"

GLOBALS_PATH = "src/lib/litegraph/src/LiteGraphGlobal.ts"
WIDGET_PATH = "src/lib/litegraph/src/widgets/BaseWidget.ts"
NODE_PATH = "src/lib/litegraph/src/LGraphNode.ts"

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
    for path in {GLOBALS_PATH, WIDGET_PATH, NODE_PATH}:
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
    console.print("\nall geometry constants match upstream")
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
