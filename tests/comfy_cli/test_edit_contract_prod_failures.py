"""Edit refusals production agents hit on writes the graph could express.

From two days of cloud agent traffic (set_widget / apply_ops / connect
``workflow_edit_invalid``):

* ``connect 24.IMAGE`` on a canvas built by ``get_template`` failed "node 24
  not found" while the refusal itself listed ``insert:…:root:node:24``.
  set_widget already resolved such a template id; connect, delete_node and
  set_node_field did not (21 calls).
* ``connect … → <heygen>.speech.audio`` after ``set_widget speech=audio``
  failed "input 'speech.audio' not found … inputs: ['image']": a dynamic
  combo's LINK sub-input was not connectable (23 calls).
* ``set_widget <math>.expression = 800`` failed "expected STRING, got int":
  the value is parsed as JSON before the write (7 calls).
* ``set_widget <switch>.boolean`` on ``Image Input Switch`` (a ``forceInput``
  socket) said only "available widgets: (none — all inputs are links)" (9).
"""

from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api


def _node(inputs: dict, outputs: list[str], **extra) -> dict:
    return {
        "input": {"required": inputs},
        "input_order": {"required": list(inputs)},
        "output": outputs,
        "output_name": outputs,
        "category": "test",
        "display_name": "x",
        "python_module": "nodes",
        **extra,
    }


OBJECT_INFO = {
    "LoadImage": _node({"image": [["a.png"], {"image_upload": True}]}, ["IMAGE", "MASK"]),
    "LoadAudio": _node({"audio": [["a.wav"], {"audio_upload": True}]}, ["AUDIO"]),
    "SaveImage": _node({"images": ["IMAGE"]}, [], output_node=True),
    "SaveVideo": _node({"video": ["VIDEO"]}, [], output_node=True),
    "TalkingPhoto": _node(
        {
            "image": ["IMAGE"],
            "speech": [
                "COMFY_DYNAMICCOMBO_V3",
                {
                    "options": [
                        {"key": "script", "inputs": {"required": {"text": ["STRING", {"default": ""}]}}},
                        {"key": "audio", "inputs": {"required": {"audio": ["AUDIO", {}]}}},
                    ]
                },
            ],
        },
        ["VIDEO"],
    ),
    "MathExpression": _node({"expression": ["STRING", {"default": "a + b", "multiline": True}]}, ["INT"]),
    "Image Input Switch": _node(
        {"image_a": ["IMAGE"], "image_b": ["IMAGE"], "boolean": ["BOOLEAN", {"forceInput": True}]}, ["IMAGE"]
    ),
    "PrimitiveBoolean": _node({"value": ["BOOLEAN", {"default": False}]}, ["BOOLEAN"]),
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(OBJECT_INFO)


def _canvas() -> dict:
    """A get_template canvas: every top-level id remapped by insert_workflow."""
    ins = "insert:0e86918138a414987b0dbc8d8d3973a7:root:node:"
    return {
        "nodes": [
            {
                "id": ins + "16",
                "type": "LoadImage",
                "inputs": [],
                "widgets_values": ["a.png", "image"],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": []},
                    {"name": "MASK", "type": "MASK", "links": []},
                ],
            },
            {
                "id": ins + "9",
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [],
        "last_node_id": 0,
        "last_link_id": 0,
        "version": 0.4,
    }


class TestTemplateIds:
    def test_connect_resolves_both_template_ids(self, graph):
        wf, op = workflow_ops.connect(_canvas(), graph, 16, "IMAGE", 9, "images")
        assert op["from_node"].endswith(":root:node:16") and op["to_node"].endswith(":root:node:9")
        assert wf["nodes"][1]["inputs"][0]["link"] == op["link_id"]

    def test_batch_connect_with_bare_id(self, graph):
        wf, ops, _ = workflow_ops.apply_specs(
            _canvas(), graph, [{"op": "connect", "from": "16.IMAGE", "to": "9.images"}]
        )
        assert ops[0]["to_node"].endswith(":root:node:9")

    def test_set_node_field_and_delete_node(self, graph):
        wf, op = workflow_ops.set_node_field(_canvas(), 16, "title", "Reference")
        assert op["node_id"].endswith(":root:node:16")
        wf, op = workflow_ops.delete_node(wf, graph, 9)
        assert op["node_id"].endswith(":root:node:9")
        assert [n["id"][-2:] for n in wf["nodes"]] == ["16"]

    def test_a_real_node_with_that_id_still_wins(self, graph):
        wf = _canvas()
        wf["nodes"].append(
            {
                "id": 9,
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [],
                "widgets_values": [],
            }
        )
        _, op = workflow_ops.connect(wf, graph, 16, "IMAGE", 9, "images")
        assert op["to_node"] == 9


class TestBindingNames:
    """print_workflow names every node (``load_image``); set_widget already
    accepts that name as the node part of an address. connect, delete_node and
    set_node_field must too, or the same address works for one op and fails
    "not found" for the next."""

    def test_connect_by_binding_names(self, graph):
        wf, op = workflow_ops.connect(_canvas(), graph, "load_image", "IMAGE", "save_image", "images")
        assert op["from_node"].endswith(":root:node:16") and op["to_node"].endswith(":root:node:9")
        assert wf["nodes"][1]["inputs"][0]["link"] == op["link_id"]

    def test_set_node_field_by_binding_name(self, graph):
        wf, op = workflow_ops.set_node_field(_canvas(), "load_image", "title", "Reference")
        assert op["node_id"].endswith(":root:node:16")
        assert wf["nodes"][0]["title"] == "Reference"

    def test_delete_node_by_binding_name(self, graph):
        wf, op = workflow_ops.delete_node(_canvas(), graph, "save_image")
        assert op["node_id"].endswith(":root:node:9")
        assert [n["id"][-2:] for n in wf["nodes"]] == ["16"]

    def test_an_unknown_name_still_lists_the_nodes(self, graph):
        with pytest.raises(ValueError, match="Nodes in this workflow"):
            workflow_ops.connect(_canvas(), graph, "no_such_node", "IMAGE", "save_image", "images")


def _talking(graph, speech: str) -> tuple[dict, int, int]:
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "LoadAudio")
    audio = op["node_id"]
    wf, op = workflow_ops.add_node(wf, graph, "TalkingPhoto")
    talk = op["node_id"]
    wf, _ = workflow_ops.set_widget(wf, graph, talk, "speech", speech)
    return wf, audio, talk


class TestDynamicComboLinkInput:
    def test_connects_once_its_option_is_selected(self, graph):
        wf, audio, talk = _talking(graph, "audio")
        wf, op = workflow_ops.connect(wf, graph, audio, "AUDIO", talk, "speech.audio")
        node = next(n for n in wf["nodes"] if n["id"] == talk)
        slot = next(i for i in node["inputs"] if i["name"] == "speech.audio")
        assert slot["type"] == "AUDIO" and slot["link"] == op["link_id"]

        api = convert_ui_to_api(wf, OBJECT_INFO)
        assert api[str(talk)]["inputs"]["speech.audio"] == [str(audio), 0]
        assert not [e for e in graph.validate_workflow(api)["errors"] if "speech" in str(e.get("field"))]

        # A second connect re-wires the same socket instead of growing another.
        wf, op2 = workflow_ops.connect(wf, graph, audio, "AUDIO", talk, "speech.audio")
        node = next(n for n in wf["nodes"] if n["id"] == talk)
        assert [i["name"] for i in node["inputs"]].count("speech.audio") == 1

    def test_unselected_option_names_the_selector_value(self, graph):
        wf, audio, talk = _talking(graph, "script")
        with pytest.raises(ValueError, match=r"set_widget speech='audio' first"):
            workflow_ops.connect(wf, graph, audio, "AUDIO", talk, "speech.audio")

    def test_type_is_checked(self, graph):
        wf, _audio, talk = _talking(graph, "audio")
        wf, op = workflow_ops.add_node(wf, graph, "LoadImage")
        with pytest.raises(ValueError, match="type mismatch"):
            workflow_ops.connect(wf, graph, op["node_id"], "IMAGE", talk, "speech.audio")


def test_concurrent_connects_into_one_dynamic_link_input_converge(graph):
    """Christian's repro on #967: two actors each connect a different AUDIO
    source into ``speech.audio`` before either replica has grown the socket.
    The sub-input is ONE register keyed by its name (like a promoted subgraph
    input, which comfy-multi-player keys the same way), so both apply orders
    must end with the same occupant and no collision-renamed phantom slot."""
    base, audio_a, talk = _talking(graph, "audio")
    base, op = workflow_ops.add_node(base, graph, "LoadAudio")
    audio_b = op["node_id"]
    _, op_a = workflow_ops.connect(copy.deepcopy(base), graph, audio_a, "AUDIO", talk, "speech.audio", actor="actor-a")
    _, op_b = workflow_ops.connect(copy.deepcopy(base), graph, audio_b, "AUDIO", talk, "speech.audio", actor="actor-b")

    ab = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op_a, graph), op_b, graph)
    ba = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op_b, graph), op_a, graph)
    assert workflow_ops.canonical(ab) == workflow_ops.canonical(ba)
    for wf in (ab, ba):
        node = next(n for n in wf["nodes"] if n["id"] == talk)
        speech = [i["name"] for i in node["inputs"] if str(i.get("name", "")).startswith("speech")]
        assert speech == ["speech.audio"], speech
        assert len(wf["links"]) == 1


def test_a_reconnect_and_a_concurrent_grow_share_one_register(graph):
    """CodeRabbit on #967: once the socket exists a reconnect must claim the
    SAME register a concurrent first connect's grow claims. Actor A grows
    ``speech.audio``; B (having seen A) re-wires it; C (not having seen A)
    grows it too. Every causally ordered interleaving (A before B) must end
    with one slot holding the same link."""
    base, audio_a, talk = _talking(graph, "audio")
    base, op = workflow_ops.add_node(base, graph, "LoadAudio")
    audio_b = op["node_id"]
    base, op = workflow_ops.add_node(base, graph, "LoadAudio")
    audio_c = op["node_id"]
    after_a, op_a = workflow_ops.connect(
        copy.deepcopy(base), graph, audio_a, "AUDIO", talk, "speech.audio", actor="actor-a"
    )
    _, op_b = workflow_ops.connect(after_a, graph, audio_b, "AUDIO", talk, "speech.audio", actor="actor-b")
    _, op_c = workflow_ops.connect(copy.deepcopy(base), graph, audio_c, "AUDIO", talk, "speech.audio", actor="actor-c")

    results = []
    for order in ([op_a, op_b, op_c], [op_a, op_c, op_b], [op_c, op_a, op_b]):
        wf = copy.deepcopy(base)
        for o in order:
            wf = workflow_ops.apply_op(wf, o, graph)
        node = next(n for n in wf["nodes"] if n["id"] == talk)
        assert [i["name"] for i in node["inputs"] if str(i.get("name", "")).startswith("speech")] == ["speech.audio"]
        results.append(workflow_ops.canonical(wf))
    assert results[0] == results[1] == results[2]


class TestStringWidgetValues:
    @pytest.mark.parametrize(
        ("value", "text"),
        [(800, "800"), (1.5, "1.5"), ({"positive": [{"x": 1}]}, '{"positive": [{"x": 1}]}'), ([1, 2], "[1, 2]")],
    )
    def test_parsed_json_is_written_as_text(self, graph, value, text):
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
        wf, op = workflow_ops.add_node(wf, graph, "MathExpression")
        wf, op = workflow_ops.set_widget(wf, graph, op["node_id"], "expression", value)
        assert op["value"] == text
        assert op["warnings"][0]["code"] == "normalized_value"

    def test_a_boolean_is_still_refused(self, graph):
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
        wf, op = workflow_ops.add_node(wf, graph, "MathExpression")
        with pytest.raises(ValueError, match="expected STRING"):
            workflow_ops.set_widget(wf, graph, op["node_id"], "expression", True)


def test_value_write_to_a_socket_says_to_connect(graph):
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "Image Input Switch")
    with pytest.raises(ValueError, match=r"'boolean' is an input socket \(BOOLEAN\).*add_node PrimitiveBoolean"):
        workflow_ops.set_widget(wf, graph, op["node_id"], "boolean", True)
