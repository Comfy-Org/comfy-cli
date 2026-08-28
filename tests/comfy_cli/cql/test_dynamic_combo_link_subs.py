"""Connection-only dynamic-combo sub-inputs own no ``widgets_values`` slot.

A ``COMFY_DYNAMICCOMBO_V3`` option can declare sub-inputs that are sockets, not
widgets: a nested ``COMFY_AUTOGROW_V3`` image list (``GeminiNanoBanana2V2
.model.images``, ``MinimaxHailuo03ReferenceNode.model.reference_images``), a
``GEMINI_INPUT_FILES`` link, a bare ``IMAGE`` / ``VIDEO`` / ``AUDIO``. The
frontend renders those as sockets and never writes a value for them, so the
canvas serialises Nano Banana 2 as 9 values (``seed`` at index 5) and MiniMax
H3 as 8.

``_dynamic_sub_widget_defaults`` used to emit an entry for every sub-input
regardless, and it feeds ``widget_order_default`` (the exported widget catalog
the CRDT doc host maps names to indexes with) and ``widget_defaults`` (what
``add-node`` materialises). The per-node path ``widget_order_for_node`` already
skipped links. Result: ``add-node`` wrote 11 values for Nano Banana 2 with two
``null`` phantoms in front of ``seed``; converting that node gave
``response_modalities: null, system_prompt: 42, temperature: "fixed"``; and
through the catalog a UI-built MiniMax H3 node's ``seed`` position was read as
``model.reference_images`` (PM-273 / BE-10291).

Fixtures: ``dynamic_combo_link_subs_object_info.json`` is the cloud catalog
entry for the affected classes; ``dynamic_combo_link_subs_frontend_nodes.json``
is the node as the frontend serialised it in the gallery template for each
class (``api_google_nano_banana2_image_edit``, ``api_minimax_h3_r2v``,
``api_seedance2_5_r2v``) — the ground truth the CLI must agree with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api

_FIXTURES = Path(__file__).parent.parent / "fixtures"

NESTED_LINK_CLASSES = ["GeminiNanoBanana2V2", "MinimaxHailuo03ReferenceNode", "ByteDance2ReferenceNodeV2"]


@pytest.fixture(scope="module")
def object_info() -> dict:
    return json.loads((_FIXTURES / "dynamic_combo_link_subs_object_info.json").read_text())


@pytest.fixture(scope="module")
def graph(object_info: dict) -> Graph:
    return Graph.from_object_info(object_info)


@pytest.fixture(scope="module")
def frontend_nodes() -> dict:
    return json.loads((_FIXTURES / "dynamic_combo_link_subs_frontend_nodes.json").read_text())


def _empty_workflow() -> dict:
    return {"nodes": [], "links": [], "groups": [], "version": 0.4, "last_node_id": 0, "last_link_id": 0}


class TestLinkSubInputsOwnNoSlot:
    @pytest.mark.parametrize("class_type", NESTED_LINK_CLASSES)
    def test_catalog_order_names_no_link_sub_input(self, graph: Graph, class_type: str):
        order = graph.widget_order_default(class_type)
        m = graph.node(class_type)
        assert m is not None
        # Every dotted entry must be a widget-like sub-port of the selected option.
        link_subs = {
            f"{p.name}.{sub}"
            for p in m.inputs
            if p.dynamic_options
            for opt in p.dynamic_options[:1]
            for section in ("required", "optional")
            for sub, spec in (opt.get("inputs", {}).get(section) or {}).items()
            if isinstance(spec, list)
            and isinstance(spec[0], str)
            and (spec[0].startswith("COMFY_AUTOGROW") or spec[0] in {"IMAGE", "VIDEO", "AUDIO", "GEMINI_INPUT_FILES"})
        }
        assert link_subs, f"fixture for {class_type} no longer carries a link-typed sub-input"
        assert not link_subs & set(order), (
            f"{class_type}: link sub-inputs leaked into widget_order: {link_subs & set(order)}"
        )

    @pytest.mark.parametrize("class_type", NESTED_LINK_CLASSES)
    def test_every_widget_order_surface_agrees(self, graph: Graph, class_type: str):
        """The catalog order, the add-node defaults and the value-aware order
        must name the same slots in the same positions for a fresh node."""
        order = graph.widget_order_default(class_type)
        defaults = graph.widget_defaults(class_type)
        assert list(defaults) == order
        assert graph.widget_order_for_node(class_type, [defaults[n] for n in order]) == order

    def test_all_classes_self_consistent(self, graph: Graph, object_info: dict):
        for class_type in object_info:
            order = graph.widget_order_default(class_type)
            assert list(graph.widget_defaults(class_type)) == order, class_type


class TestAgreesWithFrontendSerialisation:
    @pytest.mark.parametrize("class_type", NESTED_LINK_CLASSES)
    def test_seed_index_matches_the_canvas(self, graph: Graph, frontend_nodes: dict, class_type: str):
        """A frontend-authored node stores ``seed`` where the catalog says it is —
        the property the CRDT doc host relies on when it maps names to indexes."""
        node = frontend_nodes[class_type]
        values = node["widgets_values"]
        catalog = graph.widget_order_default(class_type)
        per_node = graph.widget_order_for_node(class_type, values)
        assert catalog.index("seed") == per_node.index("seed")
        assert isinstance(values[catalog.index("seed")], int)
        # The frontend may omit trailing optional widgets it did not know about
        # (older frontends), but it never writes MORE slots than the catalog names.
        assert len(values) <= len(catalog)
        assert per_node[: len(values)] == catalog[: len(values)]

    def test_set_widget_on_a_ui_built_node_hits_the_seed_slot(self, graph: Graph, frontend_nodes: dict):
        node = frontend_nodes["MinimaxHailuo03ReferenceNode"]
        wf = _empty_workflow()
        wf["nodes"].append(
            {
                "id": 4,
                "type": "MinimaxHailuo03ReferenceNode",
                "pos": [0, 0],
                "size": [300, 300],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [dict(i) for i in node["inputs"]],
                "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": []}],
                "properties": {},
                "widgets_values": list(node["widgets_values"]),
            }
        )
        wf["last_node_id"] = 4
        before = list(node["widgets_values"])
        wf, _op = workflow_ops.set_widget(wf, graph, 4, "seed", 4242)
        after = wf["nodes"][0]["widgets_values"]
        assert len(after) == len(before)
        assert after[5] == 4242  # the frontend's seed position
        assert after[:5] == before[:5] and after[6:] == before[6:]


class TestAddNodeMatchesTheCanvas:
    def test_add_node_writes_the_frontend_slot_count(self, graph: Graph, frontend_nodes: dict):
        wf, op = workflow_ops.add_node(_empty_workflow(), graph, "GeminiNanoBanana2V2")
        values = op["node"]["widgets_values"]
        order = graph.widget_order_default("GeminiNanoBanana2V2")
        assert len(values) == len(order)
        assert None not in values[: order.index("seed") + 1], "phantom null slots in front of seed"
        assert values[order.index("seed")] == 42

    def test_add_node_converts_with_every_widget_in_its_own_field(self, graph: Graph, object_info: dict):
        wf, op = workflow_ops.add_node(_empty_workflow(), graph, "GeminiNanoBanana2V2")
        wf, _ = workflow_ops.set_widget(wf, graph, op["node_id"], "seed", 777)
        api = convert_ui_to_api(wf, object_info)
        inputs = api[str(op["node_id"])]["inputs"]
        assert inputs["seed"] == 777
        assert inputs["response_modalities"] == "IMAGE"
        assert inputs["temperature"] == 1
        assert inputs["top_p"] == 0.95
        assert isinstance(inputs["system_prompt"], str)
        # Socket-only sub-inputs never become API keys with widget values.
        assert "model.images" not in inputs and "model.files" not in inputs

    def test_add_node_minimax_h3_converts_clean(self, graph: Graph, object_info: dict):
        wf, op = workflow_ops.add_node(_empty_workflow(), graph, "MinimaxHailuo03ReferenceNode")
        api = convert_ui_to_api(wf, object_info)
        inputs = api[str(op["node_id"])]["inputs"]
        assert inputs["seed"] == 42
        assert inputs["watermark"] is False
        assert "model.reference_images" not in inputs
