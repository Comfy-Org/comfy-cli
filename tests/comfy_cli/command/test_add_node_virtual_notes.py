"""`add_node` authors the two documentation-only UI nodes, `Note` and `MarkdownNote`.

Measured on prod comfy-agent traces (see test_add_node_unknown_class.py): the
agent asked for `Note` / `MarkdownNote` and was refused as "UI-only". Every
downstream replica already round-trips those nodes — the multi-player applier
stores the uncatalogued positional `widgets_values` opaquely and the frontend
follower materializes a Note with its text box — so the deny-list here was the
only thing standing between "agent adds a note" and a note on the canvas.

`Note` / `MarkdownNote` have no backend class (no `/object_info` entry), no
sockets, and exactly one positional widget: the text. They still never reach
the API prompt; `workflow_to_api` owns that exclusion and a test below pins the
two sides together on the exact node shape `add_node` mints.

The other UI-only types (`Reroute`, `PrimitiveNode`, `GetNode`, `SetNode`)
stay refused: they carry data flow the API converter resolves differently, and
authoring them here would need link semantics this surface does not model.
"""

from __future__ import annotations

import copy
import json
import math

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _object_info,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import workflow_ops
from comfy_cli.command import workflow_edit
from comfy_cli.workflow_to_api import convert_ui_to_api

NOTE_CLASSES = ("Note", "MarkdownNote")
STILL_REFUSED = ("Reroute", "PrimitiveNode", "GetNode", "SetNode")


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _is_finite_pair(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


# ---------------------------------------------------------------------------
# T4 — workflow_ops.add_node mints a well-formed frontend node for a note class
# ---------------------------------------------------------------------------


class TestAddNoteNode:
    @pytest.mark.parametrize("cls", NOTE_CLASSES)
    def test_mints_note_with_empty_text_by_default(self, cls):
        wf = _base_workflow()
        before = len(wf["nodes"])
        wf, op = workflow_ops.add_node(wf, _graph(), cls)

        assert op["op"] == "add_node"
        assert op["class_type"] == cls
        node = op["node"]
        # The applier replicas require node.id == node_id and node.type == class_type.
        assert node["id"] == op["node_id"]
        assert node["type"] == cls
        # One positional widget — the text — and nothing else. A note has no sockets.
        assert node["widgets_values"] == [""]
        assert node["inputs"] == []
        assert node["outputs"] == []
        assert _is_finite_pair(node["pos"]) and node["pos"] == op["pos"]
        assert _is_finite_pair(node["size"]) and all(x > 0 for x in node["size"])
        assert node["mode"] == 0 and node["flags"] == {} and node["properties"] == {}
        # Landed in the workflow under a fresh id, and bumped last_node_id.
        assert len(wf["nodes"]) == before + 1
        assert wf["nodes"][-1] == node
        assert wf["last_node_id"] == node["id"] > 7

    @pytest.mark.parametrize("cls", NOTE_CLASSES)
    def test_text_lands_in_widgets_values_verbatim(self, cls):
        text = "注 line1\nline2\ttab  trailing "
        _, op = workflow_ops.add_node(_base_workflow(), _graph(), cls, text=text)
        assert op["node"]["widgets_values"] == [text]

    def test_non_string_text_is_refused(self):
        with pytest.raises(ValueError, match="text"):
            workflow_ops.add_node(_base_workflow(), _graph(), "Note", text=123)  # type: ignore[arg-type]

    def test_explicit_pos_and_mode_are_honoured(self):
        _, op = workflow_ops.add_node(_base_workflow(), _graph(), "Note", pos=[10, 20], mode=2, text="muted")
        assert op["node"]["pos"] == [10, 20]
        assert op["pos"] == [10, 20]
        assert op["node"]["mode"] == 2
        assert op["mode"] == 2

    def test_text_is_not_accepted_for_a_catalog_class(self):
        """`text` is the note's positional widget; a catalog class has named
        widgets addressed through set_widget, so a stray `text` is an error, not
        a silently ignored field."""
        with pytest.raises(ValueError, match="text"):
            workflow_ops.add_node(_base_workflow(), _graph(), "VAEDecode", text="nope")

    def test_replay_of_minted_op_reproduces_the_node(self):
        """op.node is authoritative for replay (§8.5): applying the op onto an
        empty doc — with a catalog that has never heard of Note — yields the
        identical node."""
        _, op = workflow_ops.add_node(_base_workflow(), _graph(), "MarkdownNote", text="# doc")
        empty = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        replayed = workflow_ops.apply_op(copy.deepcopy(empty), copy.deepcopy(op), _graph())
        assert replayed["nodes"] == [op["node"]]
        assert replayed["last_node_id"] == op["node_id"]


# ---------------------------------------------------------------------------
# T5 — apply_specs: the batch surface every agent path goes through
# ---------------------------------------------------------------------------


class TestApplySpecsNotes:
    def test_two_notes_each_keep_their_own_text(self):
        specs = [
            {"op": "add_node", "class_type": "Note", "text": "注 line1\nline2", "as": "n"},
            {"op": "add_node", "class_type": "MarkdownNote", "text": "# md\n- α", "as": "m"},
        ]
        wf, ops, aliases = workflow_ops.apply_specs(_base_workflow(), _graph(), specs)
        assert [o["class_type"] for o in ops] == ["Note", "MarkdownNote"]
        assert ops[0]["node"]["widgets_values"] == ["注 line1\nline2"]
        assert ops[1]["node"]["widgets_values"] == ["# md\n- α"]
        assert aliases == {"n": ops[0]["node_id"], "m": ops[1]["node_id"]}
        assert ops[0]["node_id"] != ops[1]["node_id"]
        # Layout: the two auto-placed notes do not land on top of each other.
        assert ops[0]["node"]["pos"] != ops[1]["node"]["pos"]
        by_id = {n["id"]: n for n in wf["nodes"]}
        assert by_id[ops[0]["node_id"]]["type"] == "Note"
        assert by_id[ops[1]["node_id"]]["type"] == "MarkdownNote"

    def test_note_without_text_is_empty_and_a_real_node_still_adds(self):
        specs = [
            {"op": "add_node", "class_type": "Note"},
            {"op": "add_node", "class_type": "VAEDecode"},
        ]
        _, ops, _ = workflow_ops.apply_specs(_base_workflow(), _graph(), specs)
        assert ops[0]["node"]["widgets_values"] == [""]
        assert ops[1]["class_type"] == "VAEDecode"

    def test_non_string_text_aborts_the_batch_atomically(self):
        wf = _base_workflow()
        snapshot = copy.deepcopy(wf)
        specs = [
            {"op": "add_node", "class_type": "VAEDecode"},
            {"op": "add_node", "class_type": "Note", "text": ["not", "a", "string"]},
        ]
        with pytest.raises(ValueError, match="text") as ei:
            workflow_ops.apply_specs(wf, _graph(), specs)
        assert getattr(ei.value, "spec_index", None) == 1
        assert getattr(ei.value, "applied_count", None) == 0
        # `apply_specs` mutates in place on success; the caller discards on failure,
        # but the failing spec itself must not have written anything.
        assert [n["type"] for n in wf["nodes"] if n["type"] == "Note"] == []
        assert snapshot["last_node_id"] == 7


# ---------------------------------------------------------------------------
# T6 — the data-flow UI-only types stay refused, with the same specific reason
# ---------------------------------------------------------------------------


class TestOtherUiOnlyStillRefused:
    @pytest.mark.parametrize("cls", STILL_REFUSED)
    def test_ops_layer_refuses(self, cls):
        with pytest.raises(workflow_ops.UnknownNodeType) as ei:
            workflow_ops.add_node(_base_workflow(), _graph(), cls)
        assert ei.value.ui_only is True

    @pytest.mark.parametrize("cls", STILL_REFUSED)
    def test_cli_refuses_with_ui_only_envelope(self, patched_graph, tmp_path, capsys, cls):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), cls], capsys)
        assert env["ok"] is False, cls
        err = env["error"]
        assert err["code"] == "node_not_found"
        assert (err.get("details") or {}).get("ui_only") is True
        # The file was not touched.
        assert json.loads(path.read_text())["last_node_id"] == 7

    def test_cli_text_on_a_refused_class_is_still_refused(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "Reroute", "--text", "x"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "node_not_found"

    def test_deny_list_constant_still_names_every_ui_only_type(self):
        """The converter's exclusion set and this module's must keep agreeing on
        what is UI-only; authoring a subset does not shrink the set."""
        assert set(NOTE_CLASSES) | set(STILL_REFUSED) == set(workflow_ops.UI_ONLY_NODE_TYPES)


# ---------------------------------------------------------------------------
# T10 — the minted node never reaches the API prompt
# ---------------------------------------------------------------------------


class TestNotesExcludedFromApi:
    @pytest.mark.parametrize("cls", NOTE_CLASSES)
    def test_exact_minted_shape_is_dropped_by_the_converter(self, cls):
        wf, op = workflow_ops.add_node(_base_workflow(), _graph(), cls, text="doc")
        # Strip the op bookkeeping the converter has no business seeing.
        wf = {k: v for k, v in wf.items() if not k.startswith("_")}
        api = convert_ui_to_api(wf, _object_info())
        assert str(op["node_id"]) not in api
        assert cls not in {v["class_type"] for v in api.values()}
        # The real graph survived untouched: EmptyLatentImage -> KSampler.
        assert {v["class_type"] for v in api.values()} == {"EmptyLatentImage", "KSampler"}


# ---------------------------------------------------------------------------
# T11 — the CLI surface: `add-node <file> Note --text`, and a recipe with `text`
# ---------------------------------------------------------------------------


class TestCliAddNote:
    @pytest.mark.parametrize("cls", NOTE_CLASSES)
    def test_add_node_with_text_writes_the_note(self, patched_graph, tmp_path, capsys, cls):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), cls, "--text", "hello\nworld"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "add_node" and op["class_type"] == cls
        assert op["node"]["widgets_values"] == ["hello\nworld"]
        saved = json.loads(path.read_text())
        note = [n for n in saved["nodes"] if n["type"] == cls]
        assert len(note) == 1
        assert note[0]["id"] == op["node_id"]
        assert note[0]["widgets_values"] == ["hello\nworld"]

    def test_add_node_without_text_writes_an_empty_note(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "Note"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["op"]["node"]["widgets_values"] == [""]

    def test_text_on_a_catalog_class_is_an_edit_error(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "VAEDecode", "--text", "x"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert json.loads(path.read_text())["last_node_id"] == 7

    def test_apply_recipe_with_text_matches_the_direct_command(self, patched_graph, tmp_path, capsys):
        recipe = [{"op": "add_node", "class_type": "MarkdownNote", "text": "# via recipe", "as": "doc"}]
        rp = tmp_path / "recipe.json"
        rp.write_text(json.dumps(recipe), encoding="utf-8")
        path = _write(tmp_path, _base_workflow())
        env = _run(["apply", str(path), "--ops", str(rp)], capsys)
        assert env["ok"] is True, env
        (op,) = env["data"]["ops"]
        assert op["class_type"] == "MarkdownNote"
        assert op["node"]["widgets_values"] == ["# via recipe"]
        assert env["data"]["aliases"] == {"doc": op["node_id"]}


# ---------------------------------------------------------------------------
# T12 — the read-back surfaces show the note the agent just wrote
# ---------------------------------------------------------------------------


class TestNoteReadBack:
    def test_ls_nodes_and_notes_show_the_added_note(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "Note", "--text", "trigger word: zxc"], capsys)
        assert env["ok"] is True, env
        node_id = env["data"]["op"]["node_id"]

        ls = _run(["ls-nodes", str(path)], capsys)
        assert ls["ok"] is True
        assert {"id": node_id, "type": "Note", "title": None} in ls["data"]["nodes"]

        notes = _run(["notes", str(path)], capsys)
        assert notes["ok"] is True
        assert notes["data"]["count"] == 1
        (note,) = notes["data"]["notes"]
        assert note["id"] == node_id
        assert note["type"] == "Note"
        assert note["text"] == "trigger word: zxc"
