"""``connect`` must not crash when the source output has never been wired.

A real ComfyUI-serialized never-wired output carries `"links": null` — the
key EXISTS (unlike a freshly-added output that may omit it entirely), so
`src["outputs"][slot].setdefault("links", [])` returns the existing `None`
rather than installing a fresh list. The next line, `if op["link_id"] not in
out_links`, then raises `TypeError: argument of type 'NoneType' is not
iterable`.

This hits EVERY connect from a loaded/fetched real workflow's unwired output
slot, not just one node/type. Prod argv:

    workflow connect workflow.json --actor ... --base-version 3 --where cloud \
        -- 301.audio 349.audio

5 prod failures. Fix: an explicit `None` check before appending, instead of
relying on `setdefault`'s "key missing" semantics.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    "AudioSource": {
        "input": {"required": {}},
        "input_order": {"required": []},
        "output": ["AUDIO"],
        "output_name": ["audio"],
        "category": "audio",
        "display_name": "Audio Source",
        "python_module": "nodes",
    },
    "AudioSink": {
        "input": {"required": {"audio": ["AUDIO", {}]}},
        "input_order": {"required": ["audio"]},
        "output": [],
        "output_name": [],
        "category": "audio",
        "display_name": "Audio Sink",
        "python_module": "nodes",
    },
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _workflow_with_null_output_links() -> dict:
    return {
        "nodes": [
            {
                "id": 301,
                "type": "AudioSource",
                "outputs": [{"name": "audio", "type": "AUDIO", "links": None}],
            },
            {
                "id": 349,
                "type": "AudioSink",
                "inputs": [{"name": "audio", "type": "AUDIO", "link": None}],
            },
        ],
        "links": [],
    }


def test_connect_from_a_never_wired_output_does_not_crash(graph: Graph):
    wf = _workflow_with_null_output_links()
    wf, op = W.connect(wf, graph, 301, "audio", 349, "audio")
    assert op["op"] == "connect"
    assert wf["nodes"][1]["inputs"][0]["link"] == op["link_id"]


def test_connect_from_a_never_wired_output_records_the_link(graph: Graph):
    wf = _workflow_with_null_output_links()
    wf, op = W.connect(wf, graph, 301, "audio", 349, "audio")
    out_links = wf["nodes"][0]["outputs"][0]["links"]
    assert out_links == [op["link_id"]]


def test_second_connect_from_the_same_output_appends(graph: Graph):
    """Two links off the same never-wired output both survive (not just the
    first) — proves the fix builds a real list, not just swallowing the crash."""
    wf = _workflow_with_null_output_links()
    wf["nodes"].append(
        {
            "id": 350,
            "type": "AudioSink",
            "inputs": [{"name": "audio", "type": "AUDIO", "link": None}],
        }
    )
    wf, op1 = W.connect(wf, graph, 301, "audio", 349, "audio")
    wf, op2 = W.connect(wf, graph, 301, "audio", 350, "audio")
    out_links = wf["nodes"][0]["outputs"][0]["links"]
    assert set(out_links) == {op1["link_id"], op2["link_id"]}
