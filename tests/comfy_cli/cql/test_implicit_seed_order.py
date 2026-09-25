"""The implicit control_after_generate rule must reach EVERY widget-order surface.

The frontend's ``useIntWidget`` composable appends a ``control_after_generate``
companion widget after an INT named exactly ``seed``/``noise_seed`` even when
the schema omits the flag. Other seed-like names (``image_seed`` on Tripo) get
one only when flagged: saved Tripo templates carry no marker after them (see
``test_seed_companion_frontend_parity``). ``widget_order`` /
``widget_order_default`` / ``widget_defaults`` must all use the engine's
``_has_control_after_generate_slot`` predicate, or the exported widget catalog
(consumed by the CRDT doc host for name<->index mapping) disagrees with the
edit path about where the marker sits.
"""

from __future__ import annotations

import pytest

from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    # Unflagged INT named exactly ``seed``, then a combo the off-by-one would eat.
    "TripoLike": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0}],
                "style": [["clay", "steel"], {}],
            },
        },
        "input_order": {"required": ["seed", "style"]},
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


EXPECTED_TRIPO = ["seed", "control_after_generate", "style"]


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
