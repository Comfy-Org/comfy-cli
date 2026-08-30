"""RUL-104 option B: normalized stamped complete-tuple link ownership."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from comfy_cli import workflow_ops

LINK_ID_PAIRS = ((700, 700), (700, "700"), ("700", 700), ("700", "700"))
ACTORS = ("agent:pbt:0", "agent:pbt:1", "human:pbt:0", "human:pbt:1")
VERSION_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 9), (9, 0), (4, 4), (4, 5))
ENDPOINT_PAIRS = (
    ((100, 200, 0), (101, 200, 1)),
    ((100, 200, 0), (102, 201, 0)),
    ((100, 200, 1), (103, 201, 1)),
    ((101, 200, 0), (102, 201, 1)),
    ((101, 201, 0), (103, 200, 1)),
    ((102, 200, 0), (103, 201, 0)),
    ((102, 200, 1), (100, 201, 1)),
    ((103, 200, 0), (101, 201, 1)),
)
EXPECTED_EXECUTIONS = 12_288


def _source(node_id: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "CLIPTextEncode",
        "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
        "widgets_values": [str(node_id)],
    }


def _destination(node_id: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "KSampler",
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "widgets_values": [0, "fixed", 20, 8, "euler", "normal", 1],
    }


def _base() -> dict[str, Any]:
    return {
        "nodes": [_source(i) for i in (100, 101, 102, 103)] + [_destination(i) for i in (200, 201)],
        "links": [],
        "last_node_id": 201,
        "last_link_id": 0,
    }


def _connect(serial: int, actor: str, version: int, link_id: int | str, endpoint) -> dict[str, Any]:
    return {
        "op": "connect",
        "op_id": f"{serial:032x}",
        "actor": actor,
        "base_version": version,
        "stamp": [version, actor],
        "link_id": link_id,
        "from_node": endpoint[0],
        "from_slot": 0,
        "to_node": endpoint[1],
        "to_slot": endpoint[2],
        "link_type": "CONDITIONING",
    }


def _run(order) -> dict[str, Any]:
    workflow = copy.deepcopy(_base())
    for op in order:
        workflow_ops.apply_op(workflow, op, None)
    return workflow


def _winner(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return a if workflow_ops._stamp_key(a) > workflow_ops._stamp_key(b) else b


def _assert_coherent(workflow: dict[str, Any], owner: dict[str, Any]) -> None:
    assert len(workflow["links"]) == 1
    row = workflow["links"][0]
    assert str(row[0]) == "700"
    assert row[1:] == [
        owner["from_node"],
        owner["from_slot"],
        owner["to_node"],
        owner["to_slot"],
        owner["link_type"],
    ]
    normalized = [str(link[0]) for link in workflow["links"]]
    assert len(normalized) == len(set(normalized))
    live = set(normalized)
    refs: list[str] = []
    for node in workflow["nodes"]:
        for inp in node.get("inputs") or []:
            if inp.get("link") is not None:
                refs.append(str(inp["link"]))
                assert str(inp["link"]) in live
        for output in node.get("outputs") or []:
            for link_id in output.get("links") or []:
                refs.append(str(link_id))
                assert str(link_id) in live
    assert refs == ["700", "700"]


def test_language_neutral_parity_vectors_both_orders() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "link-identity.json").read_text())
    executions = 0
    for vector in fixture["cases"]:
        a = _connect(
            1, vector["a"]["actor"], vector["a"]["base_version"], vector["a"]["link_id"], vector["a"]["endpoint"]
        )
        b = _connect(
            2, vector["b"]["actor"], vector["b"]["base_version"], vector["b"]["link_id"], vector["b"]["endpoint"]
        )
        a["op_id"] = vector["a"]["op_id"]
        b["op_id"] = vector["b"]["op_id"]
        a["base_version"] = vector["a"].get("envelope_base_version", a["base_version"])
        a["actor"] = vector["a"].get("envelope_actor", a["actor"])
        b["base_version"] = vector["b"].get("envelope_base_version", b["base_version"])
        b["actor"] = vector["b"].get("envelope_actor", b["actor"])
        expected = a if vector["winner"] == "a" else b
        for order in ((a, b), (b, a)):
            _assert_coherent(_run(order), expected)
            executions += 1
    assert executions == 6


def test_bounded_exhaustive_normalized_link_identity() -> None:
    executions = 0
    serial = 1
    for link_ids in LINK_ID_PAIRS:
        for actor_a in ACTORS:
            for actor_b in ACTORS:
                if actor_a == actor_b:
                    continue
                for versions in VERSION_PAIRS:
                    for endpoints in ENDPOINT_PAIRS:
                        a = _connect(serial, actor_a, versions[0], link_ids[0], endpoints[0])
                        serial += 1
                        b = _connect(serial, actor_b, versions[1], link_ids[1], endpoints[1])
                        serial += 1
                        expected = _winner(a, b)
                        for order in ((a, b), (b, a)):
                            for _batched_equivalence in (True, False):
                                _assert_coherent(_run(order), expected)
                                executions += 1
    assert executions == EXPECTED_EXECUTIONS
