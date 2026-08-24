"""The implicit control_after_generate rule must reach EVERY widget-order surface.

The frontend's ``useIntWidget`` composable appends a ``control_after_generate``
companion widget after seed-like INT inputs even when the schema omits the
flag — partner nodes ship ``image_seed``/``model_seed``/``Seed``/``rand_seed``
unflagged (Tripo, Rodin3D, …). The UI→API converter already models this
(``workflow_to_api._has_control_after_generate_companion``); the engine's
``_has_control_after_generate_slot`` predicate exists for the same purpose but
``widget_order`` / ``widget_order_default`` / ``widget_defaults`` were checking
the raw schema flag instead of the predicate — so the exported widget catalog
(sha256-versioned, consumed by the CRDT doc host for name<->index mapping) was
off by one for every implicitly-companioned node: a consumer writing by index
landed in the control marker slot. Silent canvas corruption.
"""

from __future__ import annotations

import pytest

from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    # Tripo shape: unflagged seed-like INT, then a combo the off-by-one would eat.
    "TripoLike": {
        "input": {
            "required": {
                "image_seed": ["INT", {"default": 0}],
                "style": [["clay", "steel"], {}],
            },
        },
        "input_order": {"required": ["image_seed", "style"]},
        "output": [],
        "output_name": [],
        "category": "test",
        "display_name": "TripoLike",
        "python_module": "nodes",
    },
    # Explicitly flagged — the path that always worked; pins no regression.
    "KSamplerLike": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
            },
        },
        "input_order": {"required": ["seed", "steps"]},
        "output": [],
        "output_name": [],
        "category": "test",
        "display_name": "KSamplerLike",
        "python_module": "nodes",
    },
    # A non-seed INT must NOT grow a companion.
    "PlainInt": {
        "input": {"required": {"steps": ["INT", {"default": 20}]}},
        "input_order": {"required": ["steps"]},
        "output": [],
        "output_name": [],
        "category": "test",
        "display_name": "PlainInt",
        "python_module": "nodes",
    },
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


EXPECTED_TRIPO = ["image_seed", "control_after_generate", "style"]


class TestImplicitSeedCompanionInEveryOrderSurface:
    def test_widget_order(self, graph: Graph):
        assert graph.widget_order("TripoLike") == EXPECTED_TRIPO

    def test_widget_order_default(self, graph: Graph):
        assert graph.widget_order_default("TripoLike") == EXPECTED_TRIPO

    def test_widget_order_for_node(self, graph: Graph):
        assert graph.widget_order_for_node("TripoLike", [42, "fixed", "clay"]) == EXPECTED_TRIPO

    def test_widget_defaults_carry_the_marker(self, graph: Graph):
        assert graph.widget_defaults("TripoLike").get("control_after_generate") == "fixed"

    def test_explicit_flag_unchanged(self, graph: Graph):
        assert graph.widget_order_default("KSamplerLike") == ["seed", "control_after_generate", "steps"]

    def test_plain_int_gets_no_companion(self, graph: Graph):
        assert graph.widget_order_default("PlainInt") == ["steps"]

    def test_all_three_order_surfaces_agree(self, graph: Graph):
        """The three order functions may disagree only about dynamic-combo
        expansion — never about control markers."""
        for cls in _OBJECT_INFO:
            assert graph.widget_order(cls) == graph.widget_order_default(cls) == graph.widget_order_for_node(cls, [])
