"""validate reports broken link rows instead of lowering them away.

Prod trace f8d27ae4: three links of a SAM3 subgraph targeted input slot 6 on
nodes with inputs 0-5. The UI→API lowering reads each input's own ``link``
and never a row's slots, so the rows vanished, validate said 0 errors, and
the agent typed the prompts in by hand instead of re-wiring them.
"""

from __future__ import annotations

import copy
import json

from test_connect_interior import _workflow as _interior_workflow  # type: ignore[import-not-found]
from test_workflow_validate import _run, _write, reset_singleton  # type: ignore[import-not-found]  # noqa: F401

from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.link_integrity import broken_link_findings

OBJECT_INFO = {
    "StringSource": {
        "input": {"required": {"value": ["STRING", {"default": ""}]}},
        "input_order": {"required": ["value"]},
        "output": ["STRING"],
        "output_name": ["STRING"],
        "category": "utils",
        "display_name": "String Source",
        "output_node": False,
        "python_module": "nodes",
    },
    "TextSink": {
        "input": {"required": {"text": ["STRING", {"default": ""}]}},
        "input_order": {"required": ["text"]},
        "output": [],
        "output_name": [],
        "category": "utils",
        "display_name": "Text Sink",
        "output_node": True,
        "python_module": "nodes",
    },
}


def _canvas(link_row: list) -> dict:
    return {
        "last_node_id": 2,
        "last_link_id": 7,
        "nodes": [
            {
                "id": 1,
                "type": "StringSource",
                "inputs": [{"name": "value", "type": "STRING", "widget": {"name": "value"}, "link": None}],
                "outputs": [{"name": "STRING", "type": "STRING", "links": [7]}],
                "widgets_values": ["a red car"],
                "mode": 0,
            },
            {
                "id": 2,
                "type": "TextSink",
                "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}],
                "outputs": [],
                "widgets_values": [""],
                "mode": 0,
            },
        ],
        "links": [link_row],
        "version": 0.4,
    }


def _validate(tmp_path, capsys, wf: dict) -> dict:
    oi = _write(tmp_path, "oi.json", OBJECT_INFO)
    path = _write(tmp_path, "wf.json", wf)
    _code, env, _ = _run(workflow_cmd.app, ["validate", "--workflow", path, "--input", oi], capsys)
    return env


def test_a_link_to_a_missing_input_slot_is_an_error_with_the_rewire(tmp_path, capsys):
    env = _validate(tmp_path, capsys, _canvas([7, 1, 0, 2, 6, "STRING"]))
    data = env["data"]
    assert data["valid"] is False
    (err,) = [e for e in data["errors"] if e["code"] == "link_slot_out_of_range"]
    assert err["node_id"] == "2"
    assert err["field"] == "text"
    assert "`connect 1.0 2.text`" in err["hint"]
    assert "don't retype" in err["hint"]


def test_a_link_from_a_missing_output_slot_is_an_error(tmp_path, capsys):
    wf = _canvas([7, 1, 3, 2, 0, "STRING"])
    wf["nodes"][1]["inputs"][0]["link"] = 7
    env = _validate(tmp_path, capsys, wf)
    (err,) = [e for e in env["data"]["errors"] if e["code"] == "link_slot_out_of_range"]
    assert "output slot 3 of node 1, which has 1 output(s)" in err["message"]
    assert "`connect 1.<output> 2.text`" in err["hint"]


def test_a_healthy_link_is_clean(tmp_path, capsys):
    wf = _canvas([7, 1, 0, 2, 0, "STRING"])
    wf["nodes"][1]["inputs"][0]["link"] = 7
    env = _validate(tmp_path, capsys, wf)
    assert env["data"]["valid"] is True, env["data"]["errors"]


def test_interior_rows_are_addressed_through_the_instance():
    errors, _warnings = broken_link_findings(_interior_workflow())
    (err,) = errors
    assert err["node_id"] == "70/2011"
    assert "`connect 70/2005.0 70/2011.text`" in err["hint"]


def test_a_rewired_leftover_row_is_not_an_error():
    from test_connect_interior import _graph  # type: ignore[import-not-found]

    from comfy_cli import workflow_ops

    wf, _op = workflow_ops.connect(
        copy.deepcopy(_interior_workflow()), _graph(), "70/2005", "STRING", "70/2011", "text"
    )
    errors, warnings = broken_link_findings(wf)
    assert errors == [] and warnings == []


def test_a_row_with_no_empty_input_to_take_it_is_only_a_warning():
    wf = _interior_workflow()
    node = next(n for n in wf["definitions"]["subgraphs"][0]["nodes"] if n["id"] == 2011)
    node["inputs"] = [i for i in node["inputs"] if i["name"] != "text"]
    errors, warnings = broken_link_findings(json.loads(json.dumps(wf)))
    assert errors == [] and len(warnings) == 1
