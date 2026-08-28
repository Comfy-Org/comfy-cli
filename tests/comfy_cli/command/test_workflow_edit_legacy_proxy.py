"""``set-widget`` / ``set-slot`` on a legacy ``proxyWidgets`` promotion the
subgraph definition does NOT back with a linked input.

Before: the write followed ``proxyWidgets`` into the interior node — an edit
under layer 2 that the frontend's load-time migration then re-seeds the host
from (so it *appeared* to work) but that diverged from every other promoted
write and left no host value for ``comfy run`` to submit.

After: the write first performs the frontend's forward migration on that
instance (``cql.promoted.flush_proxy_migration`` — a port of
``flushProxyWidgetMigration``), turning the legacy entry into a linked
subgraph input, and then writes the HOST value like any other promoted widget.
The op carries the deterministic ids the repair minted, so replaying it on
another replica produces a byte-identical document.

Fixtures are verbatim gallery templates — see ``tests/comfy_cli/cql/test_proxy_migration.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql import promoted
from comfy_cli.cql.engine import Graph, _deterministic_fork_id
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer
from comfy_cli.workflow_to_api import convert_ui_to_api

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_GALLERY = _FIXTURES / "gallery"
OBJECT_INFO = _FIXTURES / "object_info_legacy_proxy.json"

RECOMPOSER = "templates_graphic_design_recomposer.json"
EXTENSION = "template_horizontal_vertical_extension.json"
FLUX = "flux_dev_checkpoint_example.json"
FLUX_SEED = 53943644181156


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(OBJECT_INFO.read_text()))


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


def _node(wf: dict, node_id: Any) -> dict:
    return next(n for n in wf["nodes"] if str(n["id"]) == str(node_id))


def _def_of(wf: dict, instance: dict) -> dict:
    return next(d for d in wf["definitions"]["subgraphs"] if d["id"] == instance["type"])


def _interior(wf: dict, instance_id: Any, inner_id: Any) -> dict:
    sg = _def_of(wf, _node(wf, instance_id))
    return next(n for n in sg["nodes"] if str(n["id"]) == str(inner_id))


def _stripped(wf: dict) -> str:
    w = copy.deepcopy(wf)
    w.pop("_applied_ops", None)
    w.pop("_widget_stamps", None)
    return json.dumps(w, sort_keys=True)


# --------------------------------------------------------------------------- #
# the write lands on the host, after repairing the definition
# --------------------------------------------------------------------------- #


def test_legacy_write_repairs_the_definition_and_writes_the_host(graph):
    wf = _load(FLUX)
    wf, op = workflow_ops.set_widget(wf, graph, 56, "seed", 5)
    inst = _node(wf, 56)
    sg = _def_of(wf, inst)
    assert [i["name"] for i in sg["inputs"]][-1] == "seed"
    assert inst["widgets_values"][-1] == 5
    assert _interior(wf, 56, 52)["widgets_values"][0] == FLUX_SEED  # interior is only the default
    assert "proxyWidgets" not in inst["properties"]
    assert "path" not in op
    assert op["node_id"] == 56 and op["widget"] == "seed"
    assert op["old"] == FLUX_SEED
    assert op["promoted"]["value_index"] == 7
    assert op["promoted"]["host_widgets_values"][-1] == 5
    repair = op["promoted"]["repair"]
    assert repair["entry"] == ["52", "seed"]
    assert repair["ids"] == {"52.seed": promoted.repair_ids(["56"], "52", "seed", 1)}
    assert repair["ids"]["52.seed"]["links"][0] == sg["inputs"][-1]["linkIds"][0]


def test_interior_address_of_a_legacy_promotion_is_redirected_to_the_host(graph):
    wf = _load(FLUX)
    wf, op = workflow_ops.set_widget(wf, graph, "56/52", "seed", 5)
    assert op["node_id"] == 56 and op["widget"] == "seed"
    assert op["redirected_from"] == "56/52.seed"
    assert _node(wf, 56)["widgets_values"][-1] == 5
    assert _interior(wf, 56, 52)["widgets_values"][0] == FLUX_SEED


def test_flat_and_interior_forms_share_one_write_target(graph):
    wf = _load(FLUX)
    _, flat = workflow_ops.set_widget(copy.deepcopy(wf), graph, 56, "seed", 1)
    _, nested = workflow_ops.set_widget(copy.deepcopy(wf), graph, "56/52", "seed", 2)
    assert workflow_ops._write_target(flat) == workflow_ops._write_target(nested) == ("widget", "56", "seed")
    assert workflow_ops.detect_conflict(flat, nested) is True


def test_unrepairable_legacy_entry_still_writes_the_interior_and_leaves_the_document_alone(graph):
    """``control_after_generate`` has no backing input slot: the frontend
    quarantines it and the interior widget stays the live one, so the write
    lands there and no repair is triggered."""
    wf = _load(FLUX)
    wf, op = workflow_ops.set_widget(wf, graph, 56, "control_after_generate", "fixed")
    assert op["path"] == ["56", "52"] and op["inner_widget"] == "control_after_generate"
    assert _interior(wf, 56, 52)["widgets_values"][1] == "fixed"
    inst = _node(wf, 56)
    assert len(inst["properties"]["proxyWidgets"]) == 9
    assert "proxyWidgetErrorQuarantine" not in inst["properties"]


def test_repair_quarantines_the_unrepairable_sibling_entries(graph):
    wf = _load(FLUX)
    wf, _ = workflow_ops.set_widget(wf, graph, 56, "seed", 5)
    assert _node(wf, 56)["properties"]["proxyWidgetErrorQuarantine"] == [
        {"originalEntry": ["52", "control_after_generate"], "reason": "missingSubgraphInput", "attemptedAtVersion": 1}
    ]


def test_type_mismatch_is_refused_before_any_repair(graph):
    wf = _load(FLUX)
    before = _stripped(wf)
    with pytest.raises(ValueError):
        workflow_ops.set_widget(wf, graph, 56, "seed", "notanumber")
    assert _stripped(wf) == before


def test_primitive_fanout_write_lands_on_the_host_and_previews_stay(graph):
    wf = _load(RECOMPOSER)
    wf, op = workflow_ops.set_widget(wf, graph, 143, "resolution", "1K")
    inst = _node(wf, 143)
    assert op["widget"] == "resolution" and op["old"] == "2K"
    assert op["promoted"]["repair"]["entry"] == ["156", "value"]
    assert inst["widgets_values"] == ["Nano Banana 2", "1K", "16:9"]
    assert _interior(wf, 143, 156)["widgets_values"][0] == "2K"  # the primitive is inert, untouched

    wf = _load(EXTENSION)
    wf, op = workflow_ops.set_widget(wf, graph, 252, "aspect_ratio", "1:1 (Square)")
    inst = _node(wf, 252)
    assert inst["widgets_values"] == ["1:1 (Square)"]
    assert [e[1] for e in inst["properties"]["proxyWidgets"]] == ["$$canvas-image-preview"] * 3


def test_ambiguous_legacy_alias_is_refused_with_the_minted_names(graph):
    """``143.value`` — the legacy tuple's own widget name — is an alias for
    the input the repair mints. Two primitives share it here, so guessing it
    is refused with both real addresses instead of silently picking one;
    with one entry left the alias redirects."""
    wf = _load(RECOMPOSER)
    before = _stripped(wf)
    with pytest.raises(ValueError) as e:
        workflow_ops.set_widget(wf, graph, 143, "value", "1K")
    assert "143.resolution" in str(e.value) and "143.aspect_ratio" in str(e.value)
    assert _stripped(wf) == before
    inst = _node(wf, 143)
    inst["properties"]["proxyWidgets"] = [x for x in inst["properties"]["proxyWidgets"] if x != ["142", "value"]]
    wf, op = workflow_ops.set_widget(wf, graph, 143, "value", "1K")
    assert op["widget"] == "resolution" and op["redirected_from"] == "143.value"


def test_interior_address_owned_by_a_legacy_entry_is_redirected_even_when_link_fed(graph):
    """``143/135.choice`` fed by interior node 2: with no legacy entry naming
    it the widget write is refused (the link supplies the value). With the
    legacy entry, the frontend's own migration replaces that link with the
    boundary link on load, so the host value is what will run — the write is
    redirected there and the feeder link is dropped, as the frontend drops it."""
    wf = _load(RECOMPOSER)
    sg = _def_of(wf, _node(wf, 143))
    sg["links"].append(
        {"id": 999001, "origin_id": 2, "origin_slot": 0, "target_id": 135, "target_slot": 0, "type": "STRING"}
    )
    _interior(wf, 143, 135)["inputs"][0]["link"] = 999001
    _interior(wf, 143, 2)["outputs"][0]["links"].append(999001)
    unowned = copy.deepcopy(wf)
    _node(unowned, 143)["properties"]["proxyWidgets"] = [["156", "value"], ["142", "value"]]
    with pytest.raises(ValueError, match="fed by interior node 143/2"):
        promoted.resolve_write(unowned, graph, ["143", "135"], "choice")
    target = promoted.resolve_write(wf, graph, ["143", "135"], "choice")
    assert target.kind == "host" and target.widget == "choice" and target.redirected_from == "143/135.choice"
    assert target.repair is not None and target.repair.original == ["135", "choice"]
    # (``choice`` is a COMBO the catalog lists no options for, so the write
    # itself is refused by validation; the flush that drops the feeder link is
    # pinned in tests/comfy_cli/cql/test_proxy_migration.py.)


# --------------------------------------------------------------------------- #
# replay: deterministic on another replica; shared definitions are forked
# --------------------------------------------------------------------------- #


def test_repair_op_replays_to_a_byte_identical_document(graph):
    for name, node_id, widget, value in ((FLUX, 56, "seed", 5), (RECOMPOSER, 143, "aspect_ratio", "1:1")):
        base = _load(name)
        applied, op = workflow_ops.set_widget(copy.deepcopy(base), graph, node_id, widget, value)
        replayed = workflow_ops.apply_op(copy.deepcopy(base), op, graph)
        assert _stripped(replayed) == _stripped(applied)
        # idempotent: a second delivery is a no-op
        assert _stripped(workflow_ops.apply_op(replayed, op, graph)) == _stripped(applied)


def test_concurrent_repairs_of_one_instance_converge(graph):
    base = _load(RECOMPOSER)
    _, a = workflow_ops.set_widget(copy.deepcopy(base), graph, 143, "resolution", "1K", actor="a")
    _, b = workflow_ops.set_widget(copy.deepcopy(base), graph, 143, "aspect_ratio", "1:1", actor="b")
    ab = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), a, graph), b, graph)
    ba = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), b, graph), a, graph)
    assert _stripped(ab) == _stripped(ba)
    assert _node(ab, 143)["widgets_values"] == ["Nano Banana 2", "1K", "1:1"]


def test_shared_definition_is_forked_not_mutated(graph):
    wf = _load(FLUX)
    sibling = copy.deepcopy(_node(wf, 56))
    sibling["id"] = 156
    wf["nodes"].append(sibling)
    original_def_id = sibling["type"]
    original_def = copy.deepcopy(_def_of(wf, sibling))
    wf, op = workflow_ops.set_widget(wf, graph, 56, "seed", 5)
    assert _node(wf, 56)["type"] == _deterministic_fork_id(original_def_id, 56)
    assert _node(wf, 156)["type"] == original_def_id
    assert _def_of(wf, _node(wf, 156)) == original_def
    assert _node(wf, 156)["properties"]["proxyWidgets"] == sibling["properties"]["proxyWidgets"]
    assert [i["name"] for i in _def_of(wf, _node(wf, 56))["inputs"]][-1] == "seed"
    # the sibling repairs independently: it now owns the original definition
    # alone, so that one is repaired in place and the fork stays as it was
    forked = copy.deepcopy(_def_of(wf, _node(wf, 56)))
    wf, _ = workflow_ops.set_widget(wf, graph, 156, "seed", 6)
    assert _node(wf, 156)["type"] == original_def_id
    assert [i["name"] for i in _def_of(wf, _node(wf, 156))["inputs"]][-1] == "seed"
    assert _def_of(wf, _node(wf, 56)) == forked
    assert _node(wf, 56)["widgets_values"][-1] == 5 and _node(wf, 156)["widgets_values"][-1] == 6


# --------------------------------------------------------------------------- #
# the converter runs the host value
# --------------------------------------------------------------------------- #


def _api(wf: dict) -> dict:
    return convert_ui_to_api(wf, json.loads(OBJECT_INFO.read_text()))


def test_converter_emits_the_host_value_after_repair(graph):
    wf = _load(FLUX)
    assert _api(wf)["56:52"]["inputs"]["seed"] == FLUX_SEED
    wf, _ = workflow_ops.set_widget(wf, graph, 56, "seed", 5)
    assert _api(wf)["56:52"]["inputs"]["seed"] == 5


def test_converter_applies_a_fanned_out_host_value_to_every_target(graph):
    wf = _load(RECOMPOSER)
    wf, _ = workflow_ops.set_widget(wf, graph, 143, "aspect_ratio", "1:1")
    api = _api(wf)
    assert api["143:130"]["inputs"]["aspect_ratio"] == "1:1"
    assert api["143:157"]["inputs"]["model.aspect_ratio"] == "1:1"


# --------------------------------------------------------------------------- #
# reads: slots advertise the promotion at the host, before any repair
# --------------------------------------------------------------------------- #


def test_slots_advertise_a_legacy_promotion_at_the_host_address(graph):
    wf = _load(FLUX)
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert slots["56.seed"]["current_value"] == FLUX_SEED
    assert slots["56.seed"]["type"] == "INT"
    assert "56/52.seed" not in slots  # layer 2: the surface value wins
    assert "56/52.cfg" in slots  # unpromoted interior widgets stay reachable
    assert "56.control_after_generate" not in slots  # quarantined: never a host widget


def test_slots_advertise_primitive_fanouts_under_their_planned_name(graph):
    wf = _load(RECOMPOSER)
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert slots["143.resolution"]["current_value"] == "2K"
    assert slots["143.aspect_ratio"]["current_value"] == "16:9"
    assert slots["143.choice"]["current_value"] == "Nano Banana 2"
    assert "143/130.resolution" not in slots and "143/157.model.resolution" not in slots


def test_set_slot_path_repairs_like_set_widget(graph):
    via_op, _ = workflow_ops.set_widget(_load(FLUX), graph, 56, "seed", 5)
    via_slot, warnings = graph.apply_slots(_load(FLUX), {"56.seed": 5})
    assert warnings == []
    assert _stripped(via_slot) == _stripped(via_op)


def test_effective_value_reads_an_unrepaired_legacy_promotion(graph):
    wf = _load(FLUX)
    assert promoted.effective_value(wf, _node(wf, 56), "seed", graph) == FLUX_SEED
    with pytest.raises(ValueError):
        promoted.effective_value(wf, _node(wf, 56), "control_after_generate", graph)


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def _run(args: list[str], capsys) -> dict[str, Any]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(workflow_cmd.app, args, standalone_mode=False)
    out = capsys.readouterr().out
    if not out.strip():
        out = result.stdout or ""
    for line in reversed([ln for ln in out.strip().splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={out[:600]})")


def test_cli_set_widget_repairs_then_slots_show_the_host_value(tmp_path, capsys):
    path = tmp_path / "flux.json"
    path.write_text(json.dumps(_load(FLUX)))
    env = _run(["slots", str(path), "--input", str(OBJECT_INFO)], capsys)
    before = {s["address"]: s for s in env["data"]["slots"]}
    assert before["56.seed"]["current_value"] == FLUX_SEED
    env = _run(["set-widget", str(path), "56.seed", "5", "--input", str(OBJECT_INFO)], capsys)
    assert env["ok"] is True, env
    assert env["data"]["op"]["promoted"]["repair"]["entry"] == ["52", "seed"]
    saved = json.loads(path.read_text())
    assert "proxyWidgets" not in _node(saved, 56)["properties"]
    env = _run(["slots", str(path), "--input", str(OBJECT_INFO)], capsys)
    after = {s["address"]: s for s in env["data"]["slots"]}
    assert after["56.seed"]["current_value"] == 5
