"""Replays tests/data/selector_conformance.json through ``comfy_cli.selector``.

The corpus is language-neutral on purpose: a Go port of this grammar in a
downstream consumer replays a verbatim copy of the same file, so a change here
that the port does not make fails there too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli.selector import select

CORPUS = json.loads((Path(__file__).parent.parent / "data" / "selector_conformance.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[c["expr"] or "<empty>" for c in CORPUS["cases"]])
def test_selector_conformance(case):
    result, matched = select(CORPUS["data"], case["expr"])
    assert matched is case["matched"]
    if case["matched"]:
        assert result == case["result"]
