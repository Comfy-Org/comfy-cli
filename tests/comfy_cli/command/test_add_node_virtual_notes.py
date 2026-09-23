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

from comfy_cli import layout, workflow_ops
from comfy_cli.command import workflow_edit
from comfy_cli.cql.engine import Graph
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
        # `apply_specs` mutates the in-memory dict as it goes (the caller discards
        # the whole batch on failure by never writing it). So the only difference
        # from the snapshot must be spec #0's VAEDecode: the failing note spec
        # itself wrote nothing and minted no id.
        assert [n["type"] for n in wf["nodes"] if n["type"] == "Note"] == []
        added = [n for n in wf["nodes"] if n["id"] not in {m["id"] for m in snapshot["nodes"]}]
        assert [n["type"] for n in added] == ["VAEDecode"]
        assert wf["last_node_id"] == added[0]["id"]
        assert wf["last_node_id"] != snapshot["last_node_id"]

    def test_explicit_null_text_is_rejected_not_coerced_to_empty(self):
        # `"text": null` in a recipe is a malformed spec, not shorthand for an empty
        # note. Omitting the key is the documented way to get "". Silently coercing
        # null would contradict "non-string text raises ValueError".
        wf = _base_workflow()
        specs = [{"op": "add_node", "class_type": "Note", "text": None}]
        with pytest.raises(ValueError, match="text.*null") as ei:
            workflow_ops.apply_specs(wf, _graph(), specs)
        assert getattr(ei.value, "spec_index", None) == 0
        assert getattr(ei.value, "applied_count", None) == 0
        assert [n["type"] for n in wf["nodes"] if n["type"] == "Note"] == []

    def test_explicit_null_text_hint_depends_on_the_class(self):
        # Same malformed value, two different remedies: on a note the fix is
        # "omit the key"; on a catalog class `text` was never valid, so the fix
        # is set_widget — pointing a VAEDecode user at "omit the key" would be
        # wrong advice.
        with pytest.raises(ValueError, match="omit the key") as note_err:
            workflow_ops.apply_specs(
                _base_workflow(), _graph(), [{"op": "add_node", "class_type": "Note", "text": None}]
            )
        assert "only valid for annotation nodes" not in str(note_err.value)

        with pytest.raises(ValueError, match="only valid for annotation nodes") as cat_err:
            workflow_ops.apply_specs(
                _base_workflow(), _graph(), [{"op": "add_node", "class_type": "VAEDecode", "text": None}]
            )
        assert "set_widget" in str(cat_err.value)
        assert "omit the key" not in str(cat_err.value)

    def test_recipe_text_is_prose_not_a_param_hole(self):
        # `${seed}` inside a note is documentation for a human, not an op value:
        # substitute_params must leave it verbatim (and must not raise "undeclared
        # param"), while a real `${...}` elsewhere on the same entry still resolves.
        ops = [
            {"op": "add_node", "class_type": "Note", "text": "seed is ${seed}", "as": "${alias}"},
            {"op": "add_node", "class_type": "VAEDecode", "as": "${alias}"},
        ]
        out = workflow_ops.substitute_params(ops, {"alias": "n"})
        assert out[0]["text"] == "seed is ${seed}"
        assert out[0]["as"] == "n"
        assert out[1]["as"] == "n"
        # And the verbatim text lands in the node body unchanged.
        wf, (op,), _ = workflow_ops.apply_specs(_base_workflow(), _graph(), [out[0]])
        assert op["node"]["widgets_values"] == ["seed is ${seed}"]


# ---------------------------------------------------------------------------
# T5b — name decides before the catalog; a note has no named widgets to set
# ---------------------------------------------------------------------------


def _graph_with_catalog_note():
    """A catalog that happens to publish a backend class called ``Note`` (a
    custom-node pack could). The annotation path must still win by name, or the
    same spec would mint two different shapes depending on which server the
    agent is pointed at."""
    info = copy.deepcopy(_object_info())
    info["Note"] = {
        "input": {"required": {"images": ["IMAGE"], "label": ["STRING", {"default": "x"}]}},
        "input_order": {"required": ["images", "label"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "custom",
        "display_name": "Note (custom pack)",
        "python_module": "custom_nodes.notes",
    }
    return Graph.from_object_info(info)


class TestNameDecidesBeforeCatalog:
    def test_add_node_mints_the_annotation_shape_even_with_a_catalog_note(self):
        graph = _graph_with_catalog_note()
        assert graph.node("Note") is not None, "precondition: the catalog really does publish Note"
        wf, op = workflow_ops.add_node(_base_workflow(), graph, "Note", text="doc")
        node = op["node"]
        assert node["widgets_values"] == ["doc"]
        assert node["inputs"] == [] and node["outputs"] == []
        # Not the catalog's width-from-slots estimate: the same footprint the
        # planner uses for a note.
        assert node["size"] == layout.note_size("Note")

    def test_assign_positions_sizes_by_name_even_with_a_catalog_note(self):
        graph = _graph_with_catalog_note()
        specs = [{"op": "add_node", "class_type": "Note", "as": "n"}]
        (out,) = layout.assign_positions(_base_workflow(), graph, specs)
        assert _is_finite_pair(out["at"])
        # The planner and the minted node must agree on footprint (the invariant
        # `note_size` exists for), so mint through the same graph and compare.
        _, op = workflow_ops.add_node(_base_workflow(), graph, "Note")
        assert op["node"]["size"] == layout.note_size("Note")


class TestSetWidgetOnANote:
    @pytest.mark.parametrize("cls", NOTE_CLASSES)
    def test_set_widget_refuses_with_a_note_specific_reason(self, cls):
        wf, op = workflow_ops.add_node(_base_workflow(), _graph(), cls, text="before")
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, _graph(), op["node_id"], "text", "after")
        msg = str(ei.value)
        assert "no named widgets" in msg and "text" in msg and cls in msg
        # The generic catalog wording would be wrong here and must not leak through.
        assert "all inputs are links" not in msg
        # Nothing changed.
        note = next(n for n in wf["nodes"] if n["id"] == op["node_id"])
        assert note["widgets_values"] == ["before"]


# ---------------------------------------------------------------------------
# T5c — replace_ops carries a note's body so a replayed template keeps it
# ---------------------------------------------------------------------------


class TestReplaceOpsCarriesNoteText:
    def test_add_node_spec_half_has_text_and_replays_it(self):
        src = _base_workflow()
        src, _ = workflow_ops.add_node(src, _graph(), "MarkdownNote", text="# read me")
        src = {k: v for k, v in src.items() if not k.startswith("_")}

        ops = workflow_ops.replace_ops({"nodes": [], "links": []}, src)
        note_ops = [o for o in ops if o["op"] == "add_node" and o["class_type"] == "MarkdownNote"]
        assert len(note_ops) == 1
        assert note_ops[0]["text"] == "# read me"
        # Catalog nodes do not grow a `text` key — it is the annotation body only.
        assert all("text" not in o for o in ops if o["op"] == "add_node" and o["class_type"] != "MarkdownNote")

        # Replaying the SPEC half (what `apply` runs) reproduces the body, not "".
        wf, applied, _ = workflow_ops.apply_specs({"nodes": [], "links": [], "last_node_id": 0}, _graph(), ops)
        replayed = [n for n in wf["nodes"] if n["type"] == "MarkdownNote"]
        assert len(replayed) == 1
        assert replayed[0]["widgets_values"] == ["# read me"]

    def test_foreign_note_without_a_string_body_replays_as_empty(self):
        # A hand-edited document may carry a note with a missing or non-string
        # slot; the spec contract is "text is a string", so it degrades to "".
        src = _base_workflow()
        src["nodes"].append({"id": 99, "type": "Note", "pos": [0, 0], "size": [200, 100], "widgets_values": [None]})
        src["last_node_id"] = 99
        ops = workflow_ops.replace_ops({"nodes": [], "links": []}, src)
        (note_op,) = [o for o in ops if o["op"] == "add_node" and o["class_type"] == "Note"]
        assert note_op["text"] == ""

    @pytest.mark.parametrize("bad_type", [["Note"], {"type": "Note"}, 7])
    def test_non_string_node_type_is_refused_not_a_type_error(self, bad_type):
        # A malformed `type` is truthy, so a bare falsiness check lets it reach
        # the authorable-set membership test, where an unhashable value raises
        # TypeError instead of the NotExpressibleError contract callers rely on.
        src = _base_workflow()
        src["nodes"].append({"id": 99, "type": bad_type, "pos": [0, 0], "size": [200, 100]})
        src["last_node_id"] = 99
        with pytest.raises(workflow_ops.NotExpressibleError, match="no id or no type"):
            workflow_ops.replace_ops({"nodes": [], "links": []}, src)


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

    def test_authorable_set_is_a_subset_of_ui_only(self):
        """`add_node` treats "authorable" as a carve-out FROM the UI-only set; a
        type listed as authorable but not UI-only would be minted as an
        annotation AND lowered to the API prompt — the converter only drops
        UI_ONLY_NODE_TYPES."""
        assert set(workflow_ops.AUTHORABLE_VIRTUAL_NODE_TYPES) <= set(workflow_ops.UI_ONLY_NODE_TYPES)
        assert set(workflow_ops.AUTHORABLE_VIRTUAL_NODE_TYPES) == set(NOTE_CLASSES)


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
