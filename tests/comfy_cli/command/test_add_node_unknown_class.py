"""`workflow add-node` must fail like `nodes show` does when a class is unknown.

When an agent named a class that does not exist, `workflow add-node` gave NO
suggestions back:

  add_node:  "unknown node type 'MarkdownNote'"
  add_node:  "unknown node type 'Note'"
  add_node:  "unknown node type '2454ad83-157c-40dd-9f19-5daaf4041ce0'"
  show_node: "Node class 'RadianceShowText' not found ..."  + close_matches

`nodes show` emits code=node_not_found with details.close_matches, so the agent
self-corrects in one retry. `workflow add-node` emitted
code=workflow_edit_invalid with hint "run `comfy nodes types`" — which lists
CONNECTION types (MODEL/LATENT/IMAGE), not class_types, and which the agent has
no tool for. It also means the agent-side annotateNodeNotFound (which keys on
code == "node_not_found") never fired on this path.
"""

from __future__ import annotations

import copy
import json

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import layout, workflow_ops
from comfy_cli.command import workflow_edit


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _add(tmp_path, capsys, class_type: str) -> dict:
    path = _write(tmp_path, _base_workflow())
    return _run(["add-node", str(path), class_type], capsys)


def test_unknown_class_emits_node_not_found_with_close_matches(patched_graph, tmp_path, capsys):
    # 'KSample' is a near-miss for the catalog's 'KSampler'.
    env = _add(tmp_path, capsys, "KSample")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found", f"must match `nodes show`'s code: {err}"
    assert "KSampler" in (err.get("details") or {}).get("close_matches", []), err
    assert "KSampler" in (err.get("hint") or ""), err
    # The misleading hint must be gone: `nodes types` lists connection types.
    assert "nodes types" not in json.dumps(err)


def test_ui_only_node_is_rejected_with_a_specific_reason(patched_graph, tmp_path, capsys):
    """Reroute/GetNode/SetNode/PrimitiveNode exist only in the UI graph and carry
    data flow, so they stay refused. difflib gives no useful match for them (and
    for GetNode returns actively misleading ones), so they need their own message.

    Note/MarkdownNote used to be in this list; they are now authorable (see
    test_add_node_virtual_notes.py) — the two traces at the top of this file were
    the agent asking for exactly that."""
    for cls in ("GetNode", "Reroute", "SetNode", "PrimitiveNode"):
        env = _add(tmp_path, capsys, cls)
        assert env["ok"] is False, cls
        err = env["error"]
        assert err["code"] == "node_not_found", f"{cls}: {err}"
        blob = json.dumps(err).lower()
        assert "ui-only" in blob or "ui only" in blob, f"{cls} must be named as UI-only: {err}"
        assert (err.get("details") or {}).get("ui_only") is True, err


def test_uuid_class_type_is_named_as_a_subgraph_instance(patched_graph, tmp_path, capsys):
    """An unknown UUID is still rejected as a missing subgraph definition."""
    env = _add(tmp_path, capsys, "2454ad83-157c-40dd-9f19-5daaf4041ce0")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found"
    blob = json.dumps(err).lower()
    assert "subgraph" in blob, f"must explain the UUID is a subgraph instance id: {err}"


def _workflow_with_subgraph_instance() -> tuple[dict, str]:
    definition_id = "2454ad83-157c-40dd-9f19-5daaf4041ce0"
    workflow = _base_workflow()
    workflow["definitions"] = {
        "subgraphs": [
            {
                "id": definition_id,
                "name": "Prompt module",
                "inputs": [],
                "outputs": [],
                "nodes": [],
                "links": [],
            }
        ]
    }
    workflow["nodes"].append(
        {
            "id": 88,
            "type": definition_id,
            "pos": [400, 300],
            "size": [240, 160],
            "flags": {"collapsed": True},
            "order": 3,
            "mode": 2,
            "inputs": [
                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 51},
                {"name": "clip", "type": "CLIP", "link": 52},
            ],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [53]}],
            "properties": {"Node name for S&R": "Prompt module"},
            "widgets_values": ["a photo of a pier", 7],
            "widgets_values_form": {"order": ["text", "seed"]},
            "title": "custom first instance",
        }
    )
    workflow["last_node_id"] = 88
    return workflow, definition_id


def test_existing_subgraph_definition_reuses_exact_host_schema_with_fresh_instance_state(
    patched_graph, tmp_path, capsys
):
    workflow, definition_id = _workflow_with_subgraph_instance()
    path = _write(tmp_path, workflow)

    env = _run(["add-node", str(path), definition_id, "--at", "700,300"], capsys)

    assert env["ok"] is True, env
    op = env["data"]["op"]
    saved = json.loads(path.read_text())
    instances = [node for node in saved["nodes"] if node["type"] == definition_id]
    assert len(instances) == 2
    original, reused = instances
    assert reused == op["node"]
    assert op["subgraph_reuse"] == {
        "definition_id": definition_id,
        "existing_instance_paths": [["88"]],
    }
    assert reused["id"] != original["id"]
    assert reused["pos"] == [700.0, 300.0]
    assert reused["size"] == original["size"]
    assert reused["widgets_values"] == original["widgets_values"]
    assert reused["widgets_values_form"] == original["widgets_values_form"]
    assert reused["properties"] == original["properties"]
    assert [item["link"] for item in reused["inputs"]] == [None, None]
    assert reused["outputs"][0]["links"] == []
    assert reused["flags"] == {}
    assert reused["order"] == 0
    assert reused["mode"] == 0
    assert "title" not in reused

    reused["widgets_values"][0] = "second instance"
    assert original["widgets_values"][0] == "a photo of a pier"


def test_legacy_name_typed_host_is_resolved_by_definition_id(patched_graph, tmp_path, capsys):
    workflow, definition_id = _workflow_with_subgraph_instance()
    workflow["nodes"][-1]["type"] = "Prompt module"
    path = _write(tmp_path, workflow)

    env = _run(["add-node", str(path), definition_id, "--at", "700,300"], capsys)

    assert env["ok"] is True, env
    assert env["data"]["op"]["node"]["type"] == definition_id
    assert env["data"]["op"]["subgraph_reuse"]["existing_instance_paths"] == [["88"]]


def test_definition_without_top_level_host_stays_node_not_found(patched_graph, tmp_path, capsys):
    workflow, definition_id = _workflow_with_subgraph_instance()
    workflow["nodes"].pop()
    path = _write(tmp_path, workflow)

    env = _run(["add-node", str(path), definition_id], capsys)

    assert env["ok"] is False
    assert env["error"]["code"] == "node_not_found"
    assert "matching definition and reusable instance" in json.dumps(env["error"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inputs", 1),
        ("outputs", {}),
        ("properties", []),
        ("widgets_values", "not-a-container"),
        ("widgets_values_form", []),
    ],
)
def test_malformed_reusable_host_schema_fails_closed(field, value):
    workflow, definition_id = _workflow_with_subgraph_instance()
    workflow["nodes"][-1][field] = value

    with pytest.raises(ValueError, match=f"host field {field!r}"):
        workflow_ops.add_node(workflow, _graph(), definition_id)


def test_reused_host_preserves_object_form_size_as_a_list():
    workflow, definition_id = _workflow_with_subgraph_instance()
    workflow["nodes"][-1]["size"] = {"0": 640, "1": 360}

    _saved, op = workflow_ops.add_node(workflow, _graph(), definition_id, pos=[900, 300])

    assert op["node"]["size"] == [640.0, 360.0]


def test_batch_layout_uses_reused_host_size_for_collision_avoidance():
    workflow, definition_id = _workflow_with_subgraph_instance()
    workflow["nodes"][-1]["size"] = {"0": 640, "1": 360}

    saved, ops, _aliases = workflow_ops.apply_specs(
        workflow,
        _graph(),
        [
            {"op": "add_node", "class_type": definition_id, "as": "copy"},
            {"op": "add_node", "class_type": "VAEDecode", "as": "decode"},
        ],
    )

    copied, decode = (next(node for node in saved["nodes"] if node["id"] == op["node_id"]) for op in ops)
    copied_rect = layout.occupied(copied["pos"], copied["size"])
    decode_rect = layout.occupied(decode["pos"], decode["size"])
    assert not layout._overlaps(copied_rect, decode_rect)


def test_reuse_and_concurrent_interior_write_are_reported_as_a_conflict():
    base, definition_id = _workflow_with_subgraph_instance()
    inner = copy.deepcopy(next(node for node in _base_workflow()["nodes"] if node["type"] == "KSampler"))
    inner["id"] = 27
    base["definitions"]["subgraphs"][0]["nodes"] = [inner]

    _with_copy, reuse = workflow_ops.add_node(copy.deepcopy(base), _graph(), definition_id, actor="copy")
    _with_edit, interior = workflow_ops.set_widget(copy.deepcopy(base), _graph(), "88/27", "steps", 9, actor="edit")
    copy_then_edit = workflow_ops.apply_op(
        workflow_ops.apply_op(copy.deepcopy(base), reuse, _graph()), interior, _graph()
    )
    edit_then_copy = workflow_ops.apply_op(
        workflow_ops.apply_op(copy.deepcopy(base), interior, _graph()), reuse, _graph()
    )

    assert workflow_ops.canonical(copy_then_edit) != workflow_ops.canonical(edit_then_copy)
    assert workflow_ops.detect_conflict(reuse, interior) is True


def test_known_class_still_adds(patched_graph, tmp_path, capsys):
    env = _add(tmp_path, capsys, "VAEDecode")
    assert env["ok"] is True, env


def test_non_note_ui_only_nodes_keep_their_hint(patched_graph, tmp_path, capsys):
    """Reroute/GetNode are wiring helpers, not annotations — no note route."""
    env = _add(tmp_path, capsys, "Reroute")
    assert "insert-workflow" not in (env["error"].get("hint") or "")


def test_set_widget_on_a_note_names_the_replace_route(patched_graph, tmp_path, capsys):
    """An agent calling set_widget
    {"address": "insert:cdcd…:root:node:101.text"} on a MarkdownNote got
    "widget 'text' not found on MarkdownNote; available widgets: (none — all
    inputs are links)" — false (a note has no inputs) and
    no way forward. A note's text is not name-addressable (the catalog has no
    schema for it; the doc host stores it opaquely), so say that and name the
    route that works: delete the note and insert a replacement."""
    note_id = "insert:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd:root:node:101"
    wf = _base_workflow()
    wf["nodes"].append(
        {"id": note_id, "type": "MarkdownNote", "pos": [0, 400], "inputs": [], "outputs": [], "widgets_values": [""]}
    )
    env = _run(["set-widget", str(_write(tmp_path, wf)), f"{note_id}.text", "## Seedance 2.5"], capsys)
    assert env["ok"] is False
    err = env["error"]
    blob = json.dumps(err)
    assert "all inputs are links" not in blob, err
    assert "delete" in (err.get("hint") or "") and "insert-workflow" in (err.get("hint") or ""), err


def _note(node_id, cls="MarkdownNote") -> dict:
    return {"id": node_id, "type": cls, "pos": [0, 400], "inputs": [], "outputs": [], "widgets_values": [""]}


_SG = "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f"


def _note_workflows() -> list[tuple[str, dict, object]]:
    """Every address form that resolves to a note: a string spelling of an int
    id, an interior note inside a subgraph instance, and a bare template id
    that only resolves through set_widget's retry onto the inserted node."""
    by_str = _base_workflow()
    by_str["nodes"].append(_note(42))

    interior = _base_workflow()
    interior["nodes"].append({"id": 5, "type": _SG, "pos": [0, 0], "inputs": [], "outputs": [], "widgets_values": []})
    interior["definitions"] = {
        "subgraphs": [{"id": _SG, "name": "sg", "nodes": [_note(9, "Note")], "links": [], "inputs": [], "outputs": []}]
    }

    retried = _base_workflow()
    retried["nodes"].append(_note("insert:abababababababababababababababab:root:node:101"))
    return [("string id", by_str, "42"), ("subgraph interior", interior, "5/9"), ("retry", retried, 101)]


@pytest.mark.parametrize("case", [c[0] for c in _note_workflows()])
def test_every_address_that_resolves_to_a_note_is_refused_as_a_note(case):
    """The note guard must look at the RESOLVED target, not the raw id: a
    string id, a subgraph interior path and the retry onto an inserted node
    all reach a note, and each must raise NoteTextNotWritable instead of
    the generic widget error."""
    from comfy_cli import workflow_ops

    _, wf, addr = next(c for c in _note_workflows() if c[0] == case)
    with pytest.raises(workflow_ops.NoteTextNotWritable):
        workflow_ops.set_widget(wf, _graph(), addr, "text", "hello")


@pytest.mark.parametrize("case", [c[0] for c in _note_workflows()])
def test_the_note_replacement_route_names_an_address_delete_node_accepts(case):
    """The hint's delete step must name the note's RESOLVED top-level id — the
    exact id delete_node requires — not the address the caller typed ("42"
    for node 42, a bare template id for an inserted node). An interior note
    cannot be removed by delete_node at all, so its hint must not promise it."""
    from comfy_cli import workflow_ops

    _, wf, addr = next(c for c in _note_workflows() if c[0] == case)
    with pytest.raises(workflow_ops.NoteTextNotWritable) as info:
        workflow_ops.set_widget(wf, _graph(), addr, "text", "hello")
    hint = info.value.hint
    if case == "subgraph interior":
        assert "delete node" not in hint, hint
        assert "subgraph" in hint, hint
        return
    token = hint.split("delete node ", 1)[1].split(" ", 1)[0]
    node_id = int(token) if token.lstrip("-").isdigit() else token
    workflow_ops.delete_node(wf, _graph(), node_id)  # must not raise
    assert info.value.node_id == node_id
