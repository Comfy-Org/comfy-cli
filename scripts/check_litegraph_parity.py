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

import argparse
import importlib.util
import re
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI_frontend/{ref}/{path}"

GLOBALS_PATH = "src/lib/litegraph/src/LiteGraphGlobal.ts"
WIDGET_PATH = "src/lib/litegraph/src/widgets/BaseWidget.ts"
NODE_PATH = "src/lib/litegraph/src/LGraphNode.ts"

# (upstream file, upstream symbol, layout.py constant)
CHECKS = [
    (GLOBALS_PATH, "NODE_TITLE_HEIGHT", "TITLE_H"),
    (GLOBALS_PATH, "NODE_SLOT_HEIGHT", "SLOT_H"),
    (GLOBALS_PATH, "NODE_WIDGET_HEIGHT", "WIDGET_H"),
    (GLOBALS_PATH, "NODE_WIDTH", "LG_NODE_WIDTH"),
    (GLOBALS_PATH, "NODE_TEXT_SIZE", "NODE_TEXT_SIZE"),
]
WIDGET_TERMS = ("margin", "arrowMargin", "arrowWidth", "minValueWidth")


def fetch(path: str, ref: str, source_dir: str | None) -> str:
    if source_dir:
        return (Path(source_dir) / path).read_text(encoding="utf-8")
    with urllib.request.urlopen(RAW.format(ref=ref, path=path), timeout=30) as r:
        return r.read().decode("utf-8")


def find_assignment(text: str, symbol: str) -> float | None:
    """Find `[static] NAME = <number>` for an exact symbol name.

    Anchored on a word boundary so NODE_WIDTH does not match NODE_WIDGET_HEIGHT and
    margin does not match arrowMargin.
    """
    m = re.search(rf"(?:^|\s)(?:static\s+)?{re.escape(symbol)}\s*=\s*(-?[\d.]+)", text)
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
    m = re.search(r"font_size\s*\*\s*[^*;\n]+?\*\s*(-?[\d.]+)", text)
    return float(m.group(1)) if m else None


def load_layout(path: Path):
    """Load the production constants without importing comfy_cli or its dependencies."""
    spec = importlib.util.spec_from_file_location("_comfy_cli_layout_parity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main", help="ComfyUI_frontend git ref (default: main)")
    ap.add_argument("--source-dir", help="local ComfyUI_frontend checkout instead of fetching")
    ap.add_argument(
        "--layout-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "comfy_cli" / "layout.py",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    sources: dict[str, str] = {}
    for path in {GLOBALS_PATH, WIDGET_PATH, NODE_PATH}:
        try:
            sources[path] = fetch(path, args.ref, args.source_dir)
        except Exception as exc:  # noqa: BLE001 - any read failure is the same outcome here
            print(f"could not read {path}: {exc}", file=sys.stderr)
            return 2

    try:
        layout = load_layout(args.layout_file)
    except Exception as exc:  # noqa: BLE001 - an unreadable model is inconclusive
        print(f"could not read production layout constants from {args.layout_file}: {exc}", file=sys.stderr)
        return 2

    mismatches: list[str] = []
    missing: list[str] = []

    for path, symbol, ours_name in CHECKS:
        theirs = find_assignment(sources[path], symbol)
        if theirs is None:
            missing.append(f"{symbol} not found in {path} -- renamed or moved upstream")
            continue
        try:
            ours = float(getattr(layout, ours_name))
        except (AttributeError, TypeError, ValueError):
            missing.append(f"{ours_name} not readable in {args.layout_file}")
            continue
        if theirs != ours:
            mismatches.append(f"{symbol}: upstream {theirs}, layout.py {ours_name} = {ours}")
        else:
            print(f"ok  {symbol:20} {theirs}")

    widget_values = {symbol: find_assignment(sources[WIDGET_PATH], symbol) for symbol in WIDGET_TERMS}
    absent_widget_terms = [symbol for symbol, value in widget_values.items() if value is None]
    if absent_widget_terms:
        missing.extend(
            f"{symbol} not found in {WIDGET_PATH} -- renamed or moved upstream" for symbol in absent_widget_terms
        )
    else:
        upstream_padding = widget_values["minValueWidth"] + 2.0 * (
            widget_values["margin"] + widget_values["arrowMargin"] + widget_values["arrowWidth"]
        )
        try:
            layout_padding = float(layout._WIDGET_PADDING)
        except (AttributeError, TypeError, ValueError):
            missing.append(f"_WIDGET_PADDING not readable in {args.layout_file}")
        else:
            if upstream_padding != layout_padding:
                mismatches.append(
                    f"widget padding: upstream composed value {upstream_padding}, "
                    f"layout.py _WIDGET_PADDING = {layout_padding}"
                )
            else:
                print(f"ok  {'_WIDGET_PADDING':20} {upstream_padding}")

    char_w = find_char_fallback(sources[NODE_PATH])
    if char_w is None:
        missing.append(f"compute_text_size glyph fallback not found in {NODE_PATH}")
    else:
        try:
            layout_char_w = float(layout._CHAR_W)
        except (AttributeError, TypeError, ValueError):
            missing.append(f"_CHAR_W not readable in {args.layout_file}")
            layout_char_w = None
        if layout_char_w is not None and char_w != layout_char_w:
            mismatches.append(f"glyph width fallback: upstream {char_w}, layout.py _CHAR_W = {layout_char_w}")
        elif layout_char_w is not None:
            print(f"ok  {'_CHAR_W':20} {char_w}")

    if missing:
        print("\nSYMBOLS NOT FOUND (the check could not run, not a proven mismatch):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    if mismatches:
        print("\nCONSTANTS HAVE DRIFTED:", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(
            "\ncomfy_cli/layout.py models LiteGraph's geometry. When these diverge the CLI\n"
            "places nodes against a picture the browser no longer draws, and the symptom is\n"
            "overlapping nodes on an agent-built canvas. Update layout.py to match, and\n"
            "re-baseline the quality metrics in tests/comfy_cli/test_layout.py.",
            file=sys.stderr,
        )
    if mismatches:
        return 1
    if missing:
        return 2
    print("\nall geometry constants match upstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
