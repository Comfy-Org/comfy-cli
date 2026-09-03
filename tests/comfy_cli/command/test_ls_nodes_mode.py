"""`workflow ls-nodes` must report a node's BYPASS/MUTE state.

ComfyUI lets a user disable a node without deleting it: mode 4 = bypass (passes
input through), mode 2 = mute/never (removed from execution). workflow_to_api
already understands both — it strips them at API conversion
(workflow_to_api.py:149) and has a bypassed-id helper (:550).

But ls-nodes emitted only id/type/title, so a caller inspecting the graph could
not tell a disabled node from a live one. The consequence is worse than a
cosmetic gap: the agent "repairs" a graph whose node is merely bypassed, or
reports a workflow ready to run when a required node is muted.
"""

from __future__ import annotations

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow_edit

MODE_MUTED, MODE_BYPASS = 2, 4


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _wf_with_modes() -> dict:
    wf = _base_workflow()
    wf["nodes"][0]["mode"] = MODE_BYPASS  # KSampler, id 3
    wf["nodes"][1]["mode"] = MODE_MUTED  # EmptyLatentImage, id 7
    wf["nodes"].append(
        {
            "id": 9,
            "type": "VAEDecode",
            "pos": [0, 0],
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": [],
        }
    )
    return wf


def _rows(tmp_path, capsys) -> dict:
    path = _write(tmp_path, _wf_with_modes())
    env = _run(["ls-nodes", str(path)], capsys)
    assert env["ok"] is True, env
    return {r["id"]: r for r in env["data"]["nodes"]}


def test_ls_nodes_reports_bypassed_and_muted(patched_graph, tmp_path, capsys):
    rows = _rows(tmp_path, capsys)
    assert rows[3].get("mode") == "bypass", rows[3]
    assert rows[7].get("mode") == "mute", rows[7]


def test_ls_nodes_omits_mode_for_normal_nodes(patched_graph, tmp_path, capsys):
    """A live node must stay a single clean row — no mode noise on the 99% case."""
    rows = _rows(tmp_path, capsys)
    assert "mode" not in rows[9], rows[9]


def test_ls_nodes_unchanged_when_no_modes_set(patched_graph, tmp_path, capsys):
    path = _write(tmp_path, _base_workflow())
    env = _run(["ls-nodes", str(path)], capsys)
    assert all("mode" not in r for r in env["data"]["nodes"]), env["data"]["nodes"]


# ---------------------------------------------------------------------------
# subgraph interiors — `data.subgraph_nodes[]`
#
# `workflow["nodes"]` is the TOP LEVEL only: a subgraph instance is one opaque
# node whose `type` is its definition UUID, and the nodes it actually executes
# live under `definitions.subgraphs[].nodes`. `workflow_to_api` expands those
# interiors and then DROPS a muted/bypassed one — so the graph runs without the
# node and, before this, no reader of `ls-nodes` could tell.
#
# They are emitted under a NEW sibling key, never appended to `nodes[]`: the
# cloud agent renders every `nodes[]` entry as a model-visible line and pins
# that listing.
# ---------------------------------------------------------------------------

SG_UUID = "8f1e0a2c-0000-4000-8000-000000000001"
SG_UUID_2 = "8f1e0a2c-0000-4000-8000-000000000002"


def _sg_workflow(instance_ids=(10,), interior_mode=MODE_MUTED) -> dict:
    """Top-level EmptyLatentImage 7 plus one instance per id, all pointing at a
    single definition whose interior node 9 carries ``interior_mode``."""
    wf = {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [
            {"id": 7, "type": "EmptyLatentImage", "pos": [0, 0], "widgets_values": [512, 512, 1]},
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": SG_UUID,
                    "name": "Text to Image",
                    "inputs": [],
                    "nodes": [
                        {"id": 9, "type": "CLIPTextEncode", "mode": interior_mode, "widgets_values": ["a cat"]},
                        {"id": 11, "type": "VAEDecode", "mode": 0},
                    ],
                    "links": [],
                }
            ]
        },
    }
    for nid in instance_ids:
        wf["nodes"].append({"id": nid, "type": SG_UUID, "pos": [100, 0]})
    return wf


def _env(tmp_path, wf: dict, capsys) -> dict:
    path = _write(tmp_path, wf)
    env = _run(["ls-nodes", str(path)], capsys)
    assert env["ok"] is True, env
    return env


def test_interior_node_mode_is_reported(patched_graph, tmp_path, capsys):
    """(i) a live instance 10 whose definition has interior node 9 with mode 2."""
    env = _env(tmp_path, _sg_workflow(), capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/9"]["instance"] == "10"
    assert by_path["10/9"]["id"] == 9
    assert by_path["10/9"]["mode"] == "mute"
    assert by_path["10/9"]["type"] == "CLIPTextEncode"
    # a normally-executing interior stays label-free, same as the top-level rows
    assert "mode" not in by_path["10/11"], by_path["10/11"]


def test_top_level_nodes_unchanged_by_interiors(patched_graph, tmp_path, capsys):
    """`nodes[]` keeps its exact meaning — the instance stays ONE opaque row and
    no interior is appended. The cloud agent pins this listing."""
    env = _env(tmp_path, _sg_workflow(), capsys)
    assert [r["id"] for r in env["data"]["nodes"]] == [7, 10]
    assert env["data"]["count"] == 2
    assert env["data"]["subgraph_count"] == len(env["data"]["subgraph_nodes"]) == 2


def test_nested_instance_two_levels_deep(patched_graph, tmp_path, capsys):
    """(ii) 10/3/7, two levels deep, bypassed."""
    wf = _sg_workflow()
    wf["definitions"]["subgraphs"][0]["nodes"].append({"id": 3, "type": SG_UUID_2})
    wf["definitions"]["subgraphs"].append(
        {
            "id": SG_UUID_2,
            "name": "Inner",
            "inputs": [],
            "nodes": [{"id": 7, "type": "KSampler", "mode": MODE_BYPASS}],
            "links": [],
        }
    )
    env = _env(tmp_path, wf, capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/3/7"]["mode"] == "bypass"
    assert by_path["10/3/7"]["instance"] == "10", "instance stays the TOP-LEVEL id"
    assert by_path["10/3/7"]["id"] == 7
    # the nested instance itself is a row too, so a reader can see the chain
    assert by_path["10/3"]["type"] == SG_UUID_2


def test_two_instances_of_same_definition_both_emit(patched_graph, tmp_path, capsys):
    """(iii) 10 and 11 share one definition — both must appear, addressed apart."""
    env = _env(tmp_path, _sg_workflow(instance_ids=(10, 11)), capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/9"]["mode"] == by_path["11/9"]["mode"] == "mute"
    assert by_path["10/9"]["instance"] == "10"
    assert by_path["11/9"]["instance"] == "11"


def test_no_definitions_emits_empty_list(patched_graph, tmp_path, capsys):
    """(iv) the 99% workflow: the key is always present, and empty."""
    env = _env(tmp_path, _base_workflow(), capsys)
    assert env["data"]["subgraph_nodes"] == []
    assert env["data"]["subgraph_count"] == 0
    assert env["data"]["count"] == 2


def test_self_referencing_definition_terminates(patched_graph, tmp_path, capsys):
    """(v) a definition that contains an instance of ITSELF must not recurse
    forever. ComfyUI cannot author this; a hand-written or corrupt document can."""
    wf = _sg_workflow()
    wf["definitions"]["subgraphs"][0]["nodes"].append({"id": 5, "type": SG_UUID})
    env = _env(tmp_path, wf, capsys)
    paths = [r["path"] for r in env["data"]["subgraph_nodes"]]
    assert "10/5" in paths
    assert len(paths) == len(set(paths)), "addresses must stay unique"
    assert len(paths) < 100, f"cycle was not bounded: {len(paths)} rows"


def test_depth_cap_stops_a_long_nesting_chain(patched_graph, tmp_path, capsys):
    """A chain of distinct definitions deeper than `_MAX_SUBGRAPH_DEPTH` is
    truncated at the cap rather than walked to the bottom."""
    from comfy_cli.cql.engine import _MAX_SUBGRAPH_DEPTH

    depth = _MAX_SUBGRAPH_DEPTH + 5
    uuids = [f"8f1e0a2c-0000-4000-8000-{i:012d}" for i in range(depth)]
    subgraphs = []
    for i, u in enumerate(uuids):
        inner = [{"id": 1, "type": "VAEDecode"}]
        if i + 1 < depth:
            inner.append({"id": 2, "type": uuids[i + 1]})
        subgraphs.append({"id": u, "name": f"L{i}", "inputs": [], "nodes": inner, "links": []})
    wf = {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [{"id": 10, "type": uuids[0], "pos": [0, 0]}],
        "links": [],
        "definitions": {"subgraphs": subgraphs},
    }
    env = _env(tmp_path, wf, capsys)
    levels = {r["path"].count("/") for r in env["data"]["subgraph_nodes"]}
    assert max(levels) == _MAX_SUBGRAPH_DEPTH, sorted(levels)


def test_malformed_definitions_do_not_crash(patched_graph, tmp_path, capsys):
    """Non-dict entries in `nodes`/`subgraphs`, and an instance whose `type`
    resolves to nothing, are skipped rather than raising."""
    wf = _sg_workflow()
    wf["nodes"].append("not-a-node")
    wf["nodes"].append({"id": 12, "type": ["unhashable"]})
    wf["definitions"]["subgraphs"].append("not-a-subgraph")
    wf["definitions"]["subgraphs"][0]["nodes"].append(None)
    env = _env(tmp_path, wf, capsys)
    assert [r["path"] for r in env["data"]["subgraph_nodes"]] == ["10/9", "10/11"]
