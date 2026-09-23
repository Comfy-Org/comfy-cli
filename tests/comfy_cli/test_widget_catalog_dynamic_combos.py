"""The widget catalog must describe EVERY dynamic-combo option, not just the first.

``widget_order`` is value-blind: it expands each ``COMFY_DYNAMICCOMBO_V3`` at
its first key (``Graph.widget_order_default``). A consumer that only has the
catalog — the doc host's applier (comfy-multi-player) — therefore cannot name,
validate or position the sub-widgets of any OTHER selection. Measured on
stg-v2/nightly comfy-agent traces (2026-09-20..21): after
``set_widget 1.mode faithful`` on ``MagnificImageSkinEnhancerNode``, the
applier refused ``1.mode.skin_detail`` with "available: sharpen, smart_grain,
mode", although ``list_slots`` had just advertised it.

The frontend (``src/core/graph/widgets/dynamicWidgets.ts``) names an option's
inputs ``<selector>.<key>`` (required, then optional), inserts the widget ones
right after the selector, and seeds them from the spec defaults when the
selection changes. ``types[c].dynamic_combos`` carries exactly that per option:
the option's DIRECT widget slots in order (a nested selector gets its own
entry) plus their defaults.

THE CONTRACT IS THE ENGINE, as for ``widget_order``: expanding the catalog for a
selection must reproduce ``Graph.widget_order_for_node`` for that selection.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.cql.widget_catalog import build_types

FIXTURES = Path(__file__).parent / "fixtures"

# Verbatim from cloud services/ingest/data/object_info.json (the class behind the
# stg-v2 unknown_widget failures), hidden inputs dropped.
MAGNIFIC = {
    "MagnificImageSkinEnhancerNode": {
        "input": {
            "required": {
                "image": ["IMAGE", {}],
                "sharpen": ["INT", {"default": 0, "min": 0, "max": 100}],
                "smart_grain": ["INT", {"default": 2, "min": 0, "max": 100}],
                "mode": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {"key": "creative", "inputs": {"required": {}}},
                            {
                                "key": "faithful",
                                "inputs": {"required": {"skin_detail": ["INT", {"default": 80, "min": 0, "max": 100}]}},
                            },
                            {
                                "key": "flexible",
                                "inputs": {
                                    "required": {
                                        "optimized_for": [
                                            "COMBO",
                                            {"options": ["enhance_skin", "improve_lighting", "enhance_everything"]},
                                        ]
                                    }
                                },
                            },
                        ]
                    },
                ],
            }
        },
        "input_order": {"required": ["image", "sharpen", "smart_grain", "mode"]},
        "output": ["IMAGE"],
        "output_node": False,
    }
}


def _object_info() -> dict:
    info = json.loads((FIXTURES / "dynamic_combo_object_info.json").read_text(encoding="utf-8"))
    info.update(MAGNIFIC)
    return info


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


def test_catalog_describes_every_option_of_a_dynamic_combo(graph):
    entry = build_types(graph)["MagnificImageSkinEnhancerNode"]
    assert entry["widget_order"] == ["sharpen", "smart_grain", "mode"]
    assert entry["dynamic_combos"] == {
        "mode": {
            "default": "creative",
            "options": {
                "creative": {"widgets": [], "defaults": {}},
                "faithful": {"widgets": ["mode.skin_detail"], "defaults": {"mode.skin_detail": 80}},
                "flexible": {
                    "widgets": ["mode.optimized_for"],
                    "defaults": {"mode.optimized_for": "enhance_skin"},
                },
            },
        }
    }


def test_classes_without_dynamic_combos_carry_no_key(graph):
    assert (
        "dynamic_combos"
        not in build_types(
            Graph.from_object_info(json.loads((FIXTURES / "sd15_object_info.json").read_text(encoding="utf-8")))
        )["KSampler"]
    )


def _expand(entry: dict, selection: dict[str, str]) -> list[str]:
    """The catalog consumer's algorithm: strip every option-owned slot from
    ``widget_order``, then re-insert, after each selector, the widgets of the
    option ``selection`` names (default otherwise), recursing into nested
    selectors."""
    combos = entry.get("dynamic_combos") or {}
    owned = {w for c in combos.values() for o in c["options"].values() for w in o["widgets"]}
    base = [n for n in entry["widget_order"] if n not in owned]

    def expand(names: list[str]) -> list[str]:
        out: list[str] = []
        for name in names:
            out.append(name)
            combo = combos.get(name)
            if combo is not None:
                key = selection.get(name, combo["default"])
                out.extend(expand(combo["options"][key]["widgets"]))
        return out

    return expand(base)


def _positional(graph: Graph, cls: str, selection: dict[str, str]) -> list:
    """A widgets_values list carrying ``selection`` at each selector's slot."""
    values: list = []
    for _ in range(8):  # re-walk until the selector slots stabilise (nested combos)
        order = graph.widget_order_for_node(cls, values)
        values = [selection.get(name) for name in order]
    return values


@pytest.mark.parametrize("cls", ["MagnificImageSkinEnhancerNode", "ClaudeNode"])
def test_expanding_the_catalog_reproduces_the_engine_for_every_selection(graph, cls):
    entry = build_types(graph)[cls]
    combos = entry["dynamic_combos"]
    names = sorted(combos)
    for keys in itertools.product(*(sorted(combos[n]["options"]) for n in names)):
        selection = dict(zip(names, keys))
        expected = graph.widget_order_for_node(cls, _positional(graph, cls, selection))
        assert _expand(entry, selection) == expected, selection
