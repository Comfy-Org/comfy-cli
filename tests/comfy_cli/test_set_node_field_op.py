"""``set_node_field`` — a per-field write to a node's durable scalar state
(op-vocabulary-v1 proposal, superseding the withdrawn ``set_title`` proposal).

The vocabulary can already express "this node changed" only as an ``add_node``
upsert, which replaces the WHOLE node: it rewrites the node's widget values
and clears its widget stamps, so a title/mode/flag change concurrent with a
``set_widget`` write on that node discards the write. Nothing wrote
``title``, ``mode`` or ``flags.collapsed``/``flags.pinned`` after a node
already existed — by hand, by a script, or by the in-app agent — so a merge
consumer had nothing to receive and a concurrent edit from two sources
resolved by accident (arrival order) rather than by any of this document's
convergence guarantees. This is the same clobber class comfy-multi-player#235
(merged, superseding that repo's earlier title-only #232/ADR-032 prototype)
fixed for the CRDT multiplayer doc; this suite proves (and then closes) the
matching gap in comfy-cli's own local op vocabulary.

Before this change: ``workflow_ops`` has no ``set_node_field``, no CLI command
exists to write one of these fields without hand-editing the JSON, and
``apply_op`` rejects a ``set_node_field`` op outright (``unknown op``). The
tests below fail against that baseline; the accompanying implementation makes
them pass.

See docs/op-vocabulary-v1.md §1.8 / Amendment v1.6 for the normative shape.
This is a PROPOSED amendment (not yet ratified) — see the PR description.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _object_info() -> dict[str, Any]:
    return {
        "KSampler": {
            "input": {"required": {"seed": ["INT", {"default": 0}]}},
            "input_order": {"required": ["seed"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "python_module": "nodes",
        }
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _sampler(node_id: int = 3, **extra: Any) -> dict[str, Any]:
    node = {
        "id": node_id,
        "type": "KSampler",
        "pos": [0, 0],
        "inputs": [],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "widgets_values": [0],
    }
    node.update(extra)
    return node


def _base_workflow(**extra: Any) -> dict[str, Any]:
    return {"last_node_id": 3, "last_link_id": 0, "nodes": [_sampler(3, **extra)], "links": []}


def _write(tmp_path: Path, data: dict, name: str = "wf.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    lines = [ln for ln in captured.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


def _op(tag: str, actor: str, base_version: int, node_id: int, field: str, value: Any) -> dict[str, Any]:
    op_id = (tag + "0" * 32)[:32]
    return {
        "op": "set_node_field",
        "op_id": op_id,
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "node_id": node_id,
        "field": field,
        "value": value,
    }


# ---------------------------------------------------------------------------
# workflow_ops core: minting, applying, converging
# ---------------------------------------------------------------------------


class TestSetNodeFieldOp:
    def test_kind_is_in_the_frozen_vocabulary(self):
        assert "set_node_field" in workflow_ops.FROZEN_OPS
        assert "set_node_field" in workflow_ops.BATCHABLE_OPS

    def test_writable_fields_is_the_closed_set(self):
        assert workflow_ops.WRITABLE_NODE_FIELDS == ("title", "mode", "flags.collapsed", "flags.pinned")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("title", "My Sampler"),
            ("mode", 4),
            ("flags.collapsed", True),
            ("flags.pinned", True),
        ],
    )
    def test_set_node_field_emits_op_and_writes_the_field(self, field, value):
        wf = _base_workflow()
        wf, op = workflow_ops.set_node_field(wf, 3, field, value)
        assert op["op"] == "set_node_field"
        assert op["node_id"] == 3
        assert op["field"] == field
        assert op["value"] == value
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        head, _, leaf = field.partition(".")
        assert (node[head][leaf] if leaf else node[head]) == value

    @pytest.mark.parametrize("field", ["title", "mode", "flags.collapsed", "flags.pinned"])
    def test_null_value_clears_the_field(self, field):
        head, _, leaf = field.partition(".")
        extra = {head: {leaf: True}} if leaf else {head: "was set"}
        wf = _base_workflow(**extra)
        wf, op = workflow_ops.set_node_field(wf, 3, field, None)
        assert op["value"] is None
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        if leaf:
            assert leaf not in node.get(head, {})
        else:
            assert head not in node

    def test_unknown_node_is_rejected_at_mint_time(self):
        wf = _base_workflow()
        with pytest.raises(ValueError):
            workflow_ops.set_node_field(wf, 999, "title", "Nope")

    def test_field_outside_the_allowlist_is_rejected(self):
        """``widgets_values`` is ``set_widget``'s register and ``type`` is
        node identity; neither may be moved by a field write."""
        wf = _base_workflow()
        with pytest.raises(ValueError):
            workflow_ops.set_node_field(wf, 3, "widgets_values", [])
        with pytest.raises(ValueError):
            workflow_ops.set_node_field(wf, 3, "type", "Other")

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("title", 42),
            ("mode", "not-a-number"),
            ("mode", True),  # bool is an int subclass -- must NOT pass as mode
            ("flags.collapsed", "true"),
            ("flags.pinned", 1),
        ],
    )
    def test_value_of_the_wrong_type_is_rejected(self, field, bad_value):
        wf = _base_workflow()
        with pytest.raises(ValueError):
            workflow_ops.set_node_field(wf, 3, field, bad_value)

    def test_op_id_is_frozen_shape(self):
        wf = _base_workflow()
        _, op = workflow_ops.set_node_field(wf, 3, "title", "Renamed")
        assert len(op["op_id"]) == 32
        assert all(c in "0123456789abcdef" for c in op["op_id"])
        assert op["stamp"] == [op["base_version"], op["actor"]]

    def test_idempotent_replay(self):
        wf = _base_workflow()
        op = _op("a", "human:u1:tab_1", 0, 3, "title", "Renamed once")
        wf = workflow_ops.apply_op(wf, op, None)
        wf = workflow_ops.apply_op(wf, op, None)  # re-delivered, same op_id
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Renamed once"
        assert wf["_applied_ops"].count(op["op_id"]) == 1

    def test_delete_wins_over_a_racing_write(self):
        wf = _base_workflow()
        write = _op("a", "human:u1:tab_1", 0, 3, "title", "Too late")
        delete = {
            "op": "delete_node",
            "op_id": "b" * 32,
            "actor": "human:u2:tab_1",
            "base_version": 0,
            "stamp": [0, "human:u2:tab_1"],
            "node_id": 3,
            "removed_links": [],
        }
        wf = workflow_ops.apply_op(wf, delete, None)
        wf = workflow_ops.apply_op(wf, write, None)  # target already gone
        assert wf["nodes"] == []

    @pytest.mark.parametrize("order", ["low_then_high", "high_then_low"])
    def test_lww_converges_regardless_of_apply_order(self, order):
        """Two concurrent writes of the same field converge on the
        higher-stamped value in EITHER apply order (mirrors set_widget's LWW
        register, docs/op-vocabulary-v1.md §3)."""
        low = _op("a", "human:u1:tab_1", 0, 3, "title", "Alice's title")
        high = _op("b", "human:u2:tab_1", 1, 3, "title", "Bob's title")
        ops = [low, high] if order == "low_then_high" else [high, low]
        wf = _base_workflow()
        for op in ops:
            wf = workflow_ops.apply_op(wf, op, None)
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Bob's title"

    def test_two_fields_of_one_node_do_not_contend(self):
        """One register per (node, field): a title write and a flag write on
        the same node claim different LWW targets, so neither drops the other
        whatever order they replay in."""
        title_op = _op("a", "human:u1:tab_1", 1, 3, "title", "T")
        flag_op = _op("b", "human:u2:tab_1", 1, 3, "flags.collapsed", True)
        assert workflow_ops._write_target(title_op) != workflow_ops._write_target(flag_op)
        assert not workflow_ops.detect_conflict(title_op, flag_op)
        for order in ([title_op, flag_op], [flag_op, title_op]):
            wf = _base_workflow()
            for op in order:
                wf = workflow_ops.apply_op(wf, op, None)
            node = next(n for n in wf["nodes"] if n["id"] == 3)
            assert node["title"] == "T"
            assert node["flags"]["collapsed"] is True

    def test_write_target_is_its_own_namespace_not_the_widget_one(self):
        """A field must not alias a same-named widget's LWW register."""
        op = _op("a", "cli", 0, 3, "title", "X")
        target = workflow_ops._write_target(op)
        assert target[0] == "node_field"
        widget_op = {"op": "set_widget", "node_id": 3, "widget": "title", "value": "X"}
        assert target != workflow_ops._write_target(widget_op)

    def test_apply_specs_batches_set_node_field(self):
        wf = _base_workflow()
        g = _graph()
        wf, ops, _aliases = workflow_ops.apply_specs(
            wf, g, [{"op": "set_node_field", "node": 3, "field": "title", "value": "Batched"}]
        )
        assert ops[0]["op"] == "set_node_field"
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Batched"


# ---------------------------------------------------------------------------
# CLI surface: `comfy workflow set-node-field` — the emitting affordance a
# human or the in-app agent uses; wiring this is what actually closes the
# sync gap (workflow_ops alone is unreachable without it).
# ---------------------------------------------------------------------------


class TestSetNodeFieldCommand:
    @pytest.mark.parametrize(
        ("field", "raw", "expected"),
        [
            ("title", "New Name", "New Name"),
            ("mode", "4", 4),
            ("flags.collapsed", "true", True),
            ("flags.pinned", "true", True),
        ],
    )
    def test_writes_the_field_and_emits_op(self, tmp_path, capsys, field, raw, expected):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-node-field", str(path), "3", field, raw], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_node_field"
        assert op["node_id"] == 3
        assert op["field"] == field
        assert op["value"] == expected
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == 3)
        head, _, leaf = field.partition(".")
        assert (node[head][leaf] if leaf else node[head]) == expected

    def test_clear_flag_resets_the_field_to_absent(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow(title="Custom"))
        env = _run(["set-node-field", str(path), "3", "title", "--clear"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["op"]["value"] is None
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == 3)
        assert "title" not in node

    def test_value_and_clear_together_is_rejected(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-node-field", str(path), "3", "title", "New Name", "--clear"], capsys)
        assert env["ok"] is False

    def test_unknown_node_errors(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-node-field", str(path), "999", "title", "New Name"], capsys)
        assert env["ok"] is False

    def test_field_outside_the_allowlist_errors(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-node-field", str(path), "3", "widgets_values", "[]"], capsys)
        assert env["ok"] is False
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == 3)
        assert "widgets_values" in node
        assert node["widgets_values"] == [0]

    def test_wrong_type_for_field_errors(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-node-field", str(path), "3", "mode", "not-a-number"], capsys)
        assert env["ok"] is False
