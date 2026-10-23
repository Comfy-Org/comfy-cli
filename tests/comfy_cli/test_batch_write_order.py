"""Two writes to one register inside an ``apply_specs`` batch: the later spec wins.

Every op in a batch carries the same stamp ``[base_version, actor]``, so the
``op_id`` tiebreak in ``_stamp_key`` decides between them. With uuid4 op_ids the
earlier write won about half the time and the later one became a silent no-op
(#1006). Amendment v1.10 mints increasing op_ids, so spec order decides.
"""

from __future__ import annotations

import copy

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

_SUB = {"prompt": ["STRING", {"default": ""}]}
_MAX_SUB = {**_SUB, "mode": ["COMBO", {"default": "balanced", "options": ["balanced", "quality"]}]}

GRAPH = Graph.from_object_info(
    {
        "Gen": {
            "input": {
                "required": {
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {"key": "base", "inputs": {"required": _SUB}},
                                {"key": "max", "inputs": {"required": _MAX_SUB}},
                            ]
                        },
                    ],
                    "label": ["STRING", {"default": ""}],
                }
            },
            "input_order": {"required": ["model", "label"]},
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
            "category": "test",
            "display_name": "Gen",
            "python_module": "nodes",
        }
    }
)

RUNS = 20


def _fresh() -> tuple[dict, int]:
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, GRAPH, "Gen")
    return wf, op["node_id"]


def _widget(wf: dict, nid: int, name: str):
    node = next(n for n in wf["nodes"] if n["id"] == nid)
    values = node["widgets_values"]
    return values[workflow_ops._widget_index(GRAPH, "Gen", name, values)]


def test_later_write_to_the_same_widget_wins():
    for _ in range(RUNS):
        wf, nid = _fresh()
        wf, _, _ = workflow_ops.apply_specs(
            wf,
            GRAPH,
            [
                {"op": "set_widget", "node": nid, "widget": "label", "value": "x"},
                {"op": "set_widget", "node": nid, "widget": "label", "value": "y"},
            ],
        )
        assert _widget(wf, nid, "label") == "y"


def test_later_selector_write_shapes_the_specs_after_it():
    """A structural register: the second selector write must win, or the
    sub-widget write after it fails with "not found"."""
    for _ in range(RUNS):
        wf, nid = _fresh()
        wf, _, _ = workflow_ops.apply_specs(
            wf,
            GRAPH,
            [
                {"op": "set_widget", "node": nid, "widget": "model", "value": "base"},
                {"op": "set_widget", "node": nid, "widget": "model", "value": "max"},
                {"op": "set_widget", "node": nid, "widget": "model.mode", "value": "quality"},
            ],
        )
        assert _widget(wf, nid, "model") == "max"
        assert _widget(wf, nid, "model.mode") == "quality"


def test_batch_stamps_increase_in_spec_order():
    wf, nid = _fresh()
    specs = [{"op": "set_widget", "node": nid, "widget": "label", "value": str(i)} for i in range(50)]
    _, ops, _ = workflow_ops.apply_specs(wf, GRAPH, specs)
    keys = [workflow_ops._stamp_key(op) for op in ops]
    assert keys == sorted(keys) and len(set(map(tuple, keys))) == len(keys)


def test_batch_replays_to_the_same_graph_in_any_order_and_twice():
    base, nid = _fresh()
    specs = [
        {"op": "set_widget", "node": nid, "widget": "label", "value": "x"},
        {"op": "set_widget", "node": nid, "widget": "label", "value": "y"},
    ]
    local, ops, _ = workflow_ops.apply_specs(copy.deepcopy(base), GRAPH, specs)
    expected = workflow_ops.canonical(local)
    for order in (ops, ops[::-1], ops + ops[::-1]):
        wf = copy.deepcopy(base)
        for op in order:
            workflow_ops.apply_op(wf, op, GRAPH)
        assert workflow_ops.canonical(wf) == expected
