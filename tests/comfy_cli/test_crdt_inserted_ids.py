"""Canvases whose ids were remapped by the doc host's ``insert_workflow`` op.

Since V1.5 (#863) ``get_template`` lands on the canvas through cmp's
``insert_workflow``, which deterministically remaps EVERY id the template
carries (comfy-multi-player ``src/remap.ts``)::

    node      insert:<op_id>:root:node:57
    link      insert:<op_id>:root:link:12
    interior  insert:<op_id>:root/definition:%22<uuid>%22:node:27

So a live canvas holds string link ids, top-level node ids with ``:`` in them,
and interior node ids with ``/`` in them. The fixtures under
``fixtures/crdt_insert/`` are the output of cmp's own
``remapInsertedWorkflowIds`` (comfy-multi-player 622865b) over the gallery
fixtures, with the op_id of a real failing stg-v2 turn — see the README there.

Measured on stg-v2/nightly comfy-agent traces (2026-09-20..21): every
``validate`` failure was a link-carried ``required_input_missing`` on an
``insert:`` node, and set_widget refused the exact interior address
``list_slots`` had just advertised (77 times).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph, _extract_frontend_slots
from comfy_cli.workflow_to_api import convert_ui_to_api

FIXTURES = Path(__file__).parent / "fixtures"
OP_ID = "1ed4449ae23f3bcc8b599de88f69fd6a"


def _load(rel: str) -> dict:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


def _node_ref(node_str: str):
    # Same coercion `comfy workflow set-widget` applies (workflow_edit._split_addr).
    return int(node_str) if node_str.lstrip("-").isdigit() else node_str


Z_IMAGE = {
    "original": "gallery/image_z_image_turbo.json",
    "cmp_inserted": "crdt_insert/image_z_image_turbo.cmp_inserted.json",
}
SD15 = {
    "original": "sd15_ui_workflow.json",
    "cmp_inserted": "crdt_insert/sd15_ui_workflow.cmp_inserted.json",
}


@pytest.fixture(scope="module")
def promoted_graph() -> Graph:
    return Graph.from_object_info(_load("object_info_subgraph_promoted.json"))


@pytest.mark.parametrize("variant", ["original", "cmp_inserted"])
def test_every_advertised_slot_resolves_for_set_widget(variant, promoted_graph):
    """An address `workflow slots` advertises must be one set-widget can reach.

    The value written is the slot's own current value, so the only thing under
    test is address resolution — a catalog enum refusal (the fixture object_info
    lists placeholder model files) is not a resolution failure.
    """
    wf = _load(Z_IMAGE[variant])
    slots = _extract_frontend_slots(wf, promoted_graph)
    interior = [s for s in slots if "/" in s["address"]]
    assert interior, "fixture no longer advertises any subgraph-interior slot"

    unresolved = []
    for slot in slots:
        node_str, _, widget = slot["address"].partition(".")
        try:
            workflow_ops.set_widget(
                copy.deepcopy(wf), promoted_graph, _node_ref(node_str), widget, slot["current_value"]
            )
        except ValueError as e:
            if "not found" in str(e):
                unresolved.append(f"{slot['address']}: {e}")
    assert not unresolved, "\n".join(unresolved)


def _linked_inputs(api: dict) -> list[tuple[str, str, int]]:
    """(class_type, input, source slot) for every link-carried API input, id-free."""
    out = []
    for node in api.values():
        for name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2:
                out.append((node["class_type"], name, value[1]))
    return sorted(out)


@pytest.mark.parametrize("fixture", [SD15, Z_IMAGE], ids=["sd15", "z_image_turbo"])
def test_lowering_keeps_links_whose_ids_are_strings(fixture):
    """cmp's string link ids must lower to the same wiring integer ids do."""
    original = _linked_inputs(convert_ui_to_api(_load(fixture["original"]), {}))
    assert original, "control fixture lowered with no links at all"
    inserted = _linked_inputs(convert_ui_to_api(_load(fixture["cmp_inserted"]), {}))
    assert inserted == original


def _run_validate(tmp_path: Path, workflow: dict, object_info: dict) -> dict:
    from comfy_cli.cmdline import app

    wf_path = tmp_path / "workflow.json"
    wf_path.write_text(json.dumps(workflow), encoding="utf-8")
    oi_path = tmp_path / "object_info.json"
    oi_path.write_text(json.dumps(object_info), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["validate", "--workflow", str(wf_path), "--input", str(oi_path), "--where", "local"],
        env={"COMFY_OUTPUT": "json"},
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON envelope in output: {result.stdout!r}"
    return json.loads(lines[-1])


def _errors(env: dict) -> list[dict]:
    return ((env.get("error") or {}).get("details") or {}).get("errors") or []


@pytest.mark.parametrize("variant", ["original", "cmp_inserted"])
def test_validate_accepts_a_fully_wired_inserted_template(variant, tmp_path):
    env = _run_validate(tmp_path, _load(SD15[variant]), _load("sd15_object_info.json"))
    missing = [e for e in _errors(env) if e.get("code") == "required_input_missing"]
    assert not missing, missing


def test_validate_names_an_inserted_node_by_its_real_id(tmp_path):
    """A validate error on a top-level `insert:` node must carry that node's id
    verbatim — rewriting its `:` to `/` turns it into a subgraph path that
    addresses nothing."""
    wf = _load(SD15["cmp_inserted"])
    sampler = next(n for n in wf["nodes"] if n["type"] == "KSampler")
    model_in = next(i for i in sampler["inputs"] if i["name"] == "model")
    wf["links"] = [ln for ln in wf["links"] if ln[0] != model_in["link"]]
    model_in["link"] = None

    env = _run_validate(tmp_path, wf, _load("sd15_object_info.json"))
    hits = [e for e in _errors(env) if e.get("field") == "model"]
    assert hits, _errors(env)
    assert hits[0]["node_id"] == sampler["id"]
    assert sampler["id"].startswith(f"insert:{OP_ID}:root:node:")
