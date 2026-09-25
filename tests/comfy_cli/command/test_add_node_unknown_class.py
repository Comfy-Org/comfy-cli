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

import json

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

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
    """Note/MarkdownNote/Reroute/GetNode/SetNode/PrimitiveNode exist only in the
    UI graph. difflib gives no useful match for them (and for GetNode returns
    actively misleading ones), so they need their own message."""
    for cls in ("Note", "MarkdownNote", "GetNode", "Reroute"):
        env = _add(tmp_path, capsys, cls)
        assert env["ok"] is False, cls
        err = env["error"]
        assert err["code"] == "node_not_found", f"{cls}: {err}"
        blob = json.dumps(err).lower()
        assert "ui-only" in blob or "ui only" in blob, f"{cls} must be named as UI-only: {err}"
        assert (err.get("details") or {}).get("ui_only") is True, err


def test_uuid_class_type_is_named_as_a_subgraph_instance(patched_graph, tmp_path, capsys):
    """A subgraph INSTANCE's `type` is its definition UUID, and ls-nodes passes
    it through verbatim — so the agent sees a UUID that looks like a class name.
    There is no instantiate command, so this can never succeed; say so."""
    env = _add(tmp_path, capsys, "2454ad83-157c-40dd-9f19-5daaf4041ce0")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found"
    blob = json.dumps(err).lower()
    assert "subgraph" in blob, f"must explain the UUID is a subgraph instance id: {err}"


def test_known_class_still_adds(patched_graph, tmp_path, capsys):
    env = _add(tmp_path, capsys, "VAEDecode")
    assert env["ok"] is True, env


@pytest.mark.parametrize("cls", ["Note", "MarkdownNote"])
def test_note_refusal_hints_the_insert_workflow_route(patched_graph, tmp_path, capsys, cls):
    """Asked to add a Note, an agent could call add_node {"class_type": "Note"},
    read the hint "use a real node class; to annotate the graph, set a
    title/widget on an existing node instead", and tell the user notes cannot
    be added. They can: an insert_workflow carrying the note node (with an `id`
    and its text in widgets_values) lands it. The hint must name that route,
    and the example it gives must itself be accepted by
    `workflow insert-workflow`."""
    env = _add(tmp_path, capsys, cls)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found"
    hint = err.get("hint") or ""
    assert "insert-workflow" in hint, f"must name the route that adds a note: {err}"
    assert "set a title/widget on an existing node instead" not in hint, err

    example = json.loads(hint[hint.index("{") : hint.rindex("}") + 1])
    node = example["nodes"][0]
    assert node["type"] == cls and "id" in node and node["widgets_values"], example

    tpl = _write(tmp_path, example, name="note.json")
    ins = _run(["insert-workflow", str(_write(tmp_path, _base_workflow())), str(tpl)], capsys)
    assert ins["ok"] is True, ins
    assert ins["data"]["op"]["workflow"]["nodes"][0]["type"] == cls


@pytest.mark.parametrize("cls", ["Note", "MarkdownNote"])
def test_note_hint_is_a_runnable_invocation_whose_op_must_be_applied(patched_graph, tmp_path, capsys, cls):
    """`insert-workflow` takes the template as a FILE PATH or `-` (stdin), never
    inline JSON, and it only EMITS an op. A hint reading "insert {json}" is not
    something a caller can run, and following it without applying the op
    leaves no note on the canvas. The hint must spell a stdin invocation and
    say the emitted op has to be applied to the source workflow."""
    hint = _add(tmp_path, capsys, cls)["error"].get("hint") or ""
    assert "comfy workflow insert-workflow <workflow.json> -" in hint, hint
    assert "apply" in hint.lower(), hint

    payload = hint[hint.index("echo '") + len("echo '") : hint.index("' | comfy workflow insert-workflow")]
    from test_workflow_edit import _force_json_renderer  # type: ignore[import-not-found]
    from typer.testing import CliRunner

    from comfy_cli.command import workflow as workflow_cmd

    _force_json_renderer()
    src = _write(tmp_path, _base_workflow())
    res = CliRunner().invoke(workflow_cmd.app, ["insert-workflow", str(src), "-"], input=payload, standalone_mode=False)
    out = capsys.readouterr().out or res.stdout
    env = json.loads([ln for ln in out.splitlines() if ln.strip()][-1])
    assert env["ok"] is True, env
    assert env["data"]["op"]["workflow"]["nodes"][0]["type"] == cls


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
