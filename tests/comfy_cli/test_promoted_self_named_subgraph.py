"""A subgraph definition NAMED after a node class it wraps must not recurse.

Gallery templates such as ``video_wanmove_480p`` wrap the core
``WanMoveTrackToVideo`` node in a subgraph whose ``name`` is also
``WanMoveTrackToVideo``. ``_subgraph_defs_by_id`` registers a definition's
``name`` as a fallback key (for old name-typed instances), so the interior
core node resolved to the very definition that contains it. Every promotion
walk then recursed into itself: ``promoted_inputs`` fans out once per linked
input per level up to its depth cap of 32, which never finishes in practice
(``comfy workflow print`` hung until the agent's 5-minute tool timeout).

A definition cannot contain an instance of itself, so a node inside a
definition whose type equals that definition's name is the real node class.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
from typing import Any

from comfy_cli.cql import promoted
from comfy_cli.cql.engine import Graph, _subgraph_defs_by_id
from comfy_cli.workflow_print import render_py

SG_ID = "0ad975d5-0000-4000-8000-000000000001"
N_INPUTS = 10


def _object_info() -> dict[str, Any]:
    required = {f"w{i}": ["INT", {"default": i}] for i in range(N_INPUTS)}
    return {
        "Wrapped": {
            "input": {"required": required},
            "input_order": {"required": list(required)},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "name": "Wrapped",
            "display_name": "Wrapped",
            "category": "test",
            "output_node": False,
        }
    }


def _workflow() -> dict[str, Any]:
    inner_inputs = [
        {"name": f"w{i}", "type": "INT", "widget": {"name": f"w{i}"}, "link": 100 + i} for i in range(N_INPUTS)
    ]
    sg = {
        "id": SG_ID,
        # The trap: the definition's cosmetic name is the wrapped node's class.
        "name": "Wrapped",
        "inputs": [{"id": f"in{i}", "name": f"w{i}", "type": "INT", "linkIds": [100 + i]} for i in range(N_INPUTS)],
        "outputs": [],
        "nodes": [
            {
                "id": 1,
                "type": "Wrapped",
                "inputs": inner_inputs,
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": list(range(N_INPUTS)),
            }
        ],
        "links": [
            {"id": 100 + i, "origin_id": -10, "origin_slot": i, "target_id": 1, "target_slot": i, "type": "INT"}
            for i in range(N_INPUTS)
        ],
    }
    return {
        "last_node_id": 2,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 2,
                "type": SG_ID,
                "inputs": [
                    {"name": f"w{i}", "type": "INT", "widget": {"name": f"w{i}"}, "link": None} for i in range(N_INPUTS)
                ],
                "outputs": [],
                "widgets_values": [i * 10 for i in range(N_INPUTS)],
            }
        ],
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": [sg]},
        "version": 0.4,
    }


def _run_bounded(fn, seconds: float = 10.0):
    """Run ``fn`` in a child process; fail (instead of hanging the suite) if it
    has not returned within ``seconds``, and kill the child so a regressed
    walk cannot keep burning CPU for the rest of the run. ``fn``'s result must
    be picklable — the tests return plain summaries.

    Without ``fork`` (Windows) the closures cannot reach a spawned child, so the
    call runs on a daemon thread instead: still bounded, just not killable."""
    if "fork" not in mp.get_all_start_methods():
        box: dict[str, Any] = {}

        def run():
            try:
                box["value"] = fn()
            except BaseException as e:  # pragma: no cover - surfaced below
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(seconds)
        assert not t.is_alive(), f"did not finish within {seconds}s (self-recursive subgraph walk)"
        if "error" in box:
            raise box["error"]
        return box["value"]

    ctx = mp.get_context("fork")
    queue = ctx.Queue()

    def target():
        try:
            queue.put(("value", fn()))
        except BaseException as e:  # pragma: no cover - surfaced below
            queue.put(("error", repr(e)))

    proc = ctx.Process(target=target, daemon=True)
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise AssertionError(f"did not finish within {seconds}s (self-recursive subgraph walk)")
    kind, value = queue.get(timeout=5)
    assert kind == "value", value
    return value


def test_defs_by_id_skips_name_fallback_shadowing_own_interior_node():
    defs = _subgraph_defs_by_id(_workflow())
    assert SG_ID in defs
    assert "Wrapped" not in defs, "a def's name must not resolve its own interior node to itself"


def test_defs_by_id_keeps_name_fallback_for_name_typed_instances():
    wf = _workflow()
    sg = wf["definitions"]["subgraphs"][0]
    sg["name"] = "My Group"
    assert _subgraph_defs_by_id(wf)["My Group"] is sg


def test_promoted_inputs_terminates_and_resolves_interior_widgets():
    wf = _workflow()
    defs = promoted.defs_by_id(wf)
    pis = _run_bounded(
        lambda: [
            (p.source_widget, p.is_widget, p.nested)
            for p in promoted.promoted_inputs(wf["definitions"]["subgraphs"][0], defs)
        ]
    )
    assert pis == [(f"w{i}", True, False) for i in range(N_INPUTS)]


def test_promoted_inputs_cycle_guard_without_name_fallback():
    """Defence in depth: even a genuinely cyclic def map (a def resolving to
    itself by id) terminates instead of fanning out to the depth cap."""
    wf = _workflow()
    sg = wf["definitions"]["subgraphs"][0]
    defs = {SG_ID: sg, "Wrapped": sg}
    count = _run_bounded(lambda: len(promoted.promoted_inputs(sg, defs)))
    assert count == N_INPUTS


def test_workflow_print_self_named_subgraph_finishes():
    graph = Graph.from_object_info(json.loads(json.dumps(_object_info())))
    node_count, source = _run_bounded(lambda: (lambda r: (r.node_count, r.source))(render_py(_workflow(), graph)))
    assert node_count >= 1
    assert "Wrapped(" in source
