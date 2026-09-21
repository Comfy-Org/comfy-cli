"""``set_title`` — a node-rename op (op-vocabulary-v1 proposal).

A node's ``title`` enters a frontend-format workflow only as a passthrough
field on ``add_node``'s initial ``node`` snapshot (docs/op-vocabulary-v1.md
§1.1). Nothing wrote it afterward: renaming a node already in the document —
by hand, by a script, or by the in-app agent — produced no op at all, so
there was nothing for a merge consumer (or a second collaborator) to receive.
This is the same title-stomp bug class comfy-multi-player#232 / ADR-032 fixed
for the CRDT multiplayer doc; this suite proves (and then closes) the
matching gap in comfy-cli's own local op vocabulary.

Before this change: `workflow_ops` has no `set_title`, no CLI command exists
to rename a node without hand-editing the JSON, and `apply_op` rejects a
`set_title` op outright (`unknown op`). The tests below fail against that
baseline; the accompanying implementation makes them pass.

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
from comfy_cli.command import workflow_edit
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


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _sampler(node_id: int = 3, title: str | None = None) -> dict[str, Any]:
    node = {
        "id": node_id,
        "type": "KSampler",
        "pos": [0, 0],
        "inputs": [],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "widgets_values": [0],
    }
    if title is not None:
        node["title"] = title
    return node


def _base_workflow(title: str | None = None) -> dict[str, Any]:
    return {"last_node_id": 3, "last_link_id": 0, "nodes": [_sampler(3, title)], "links": []}


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


def _op(tag: str, actor: str, base_version: int, node_id: int, title: str | None) -> dict[str, Any]:
    op_id = (tag + "0" * 32)[:32]
    return {
        "op": "set_title",
        "op_id": op_id,
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        "node_id": node_id,
        "title": title,
    }


# ---------------------------------------------------------------------------
# workflow_ops core: minting, applying, converging
# ---------------------------------------------------------------------------


class TestSetTitleOp:
    def test_kind_is_in_the_frozen_vocabulary(self):
        assert "set_title" in workflow_ops.FROZEN_OPS
        assert "set_title" in workflow_ops.BATCHABLE_OPS

    def test_set_title_emits_op_and_renames_node(self):
        wf = _base_workflow()
        wf, op = workflow_ops.set_title(wf, 3, "My Sampler")
        assert op["op"] == "set_title"
        assert op["node_id"] == 3
        assert op["title"] == "My Sampler"
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "My Sampler"

    def test_null_title_clears_a_custom_title(self):
        wf = _base_workflow(title="Old Custom Title")
        wf, op = workflow_ops.set_title(wf, 3, None)
        assert op["title"] is None
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert "title" not in node

    def test_unknown_node_is_rejected_at_mint_time(self):
        wf = _base_workflow()
        with pytest.raises(ValueError):
            workflow_ops.set_title(wf, 999, "Nope")

    def test_non_string_non_null_title_is_rejected(self):
        wf = _base_workflow()
        with pytest.raises(ValueError):
            workflow_ops.set_title(wf, 3, 42)

    def test_op_id_is_frozen_shape(self):
        wf = _base_workflow()
        _, op = workflow_ops.set_title(wf, 3, "Renamed")
        assert len(op["op_id"]) == 32
        assert all(c in "0123456789abcdef" for c in op["op_id"])
        assert op["stamp"] == [op["base_version"], op["actor"]]

    def test_idempotent_replay(self):
        wf = _base_workflow()
        op = _op("a", "human:u1:tab_1", 0, 3, "Renamed once")
        wf = workflow_ops.apply_op(wf, op, None)
        wf = workflow_ops.apply_op(wf, op, None)  # re-delivered, same op_id
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Renamed once"
        assert wf["_applied_ops"].count(op["op_id"]) == 1

    def test_delete_wins_over_a_racing_rename(self):
        wf = _base_workflow()
        rename = _op("a", "human:u1:tab_1", 0, 3, "Too late")
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
        wf = workflow_ops.apply_op(wf, rename, None)  # target already gone
        assert wf["nodes"] == []

    @pytest.mark.parametrize("order", ["low_then_high", "high_then_low"])
    def test_lww_converges_regardless_of_apply_order(self, order):
        """Two concurrent renames of the same node converge on the
        higher-stamped title in EITHER apply order (mirrors set_widget's LWW
        register, docs/op-vocabulary-v1.md §3)."""
        low = _op("a", "human:u1:tab_1", 0, 3, "Alice's title")
        high = _op("b", "human:u2:tab_1", 1, 3, "Bob's title")
        ops = [low, high] if order == "low_then_high" else [high, low]
        wf = _base_workflow()
        for op in ops:
            wf = workflow_ops.apply_op(wf, op, None)
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Bob's title"

    def test_write_target_is_its_own_namespace_not_the_widget_one(self):
        """`title` must not alias a same-named widget's LWW register (ADR-032:
        rejected alternative 'folding title into set_widget')."""
        op = _op("a", "cli", 0, 3, "X")
        target = workflow_ops._write_target(op)
        assert target[0] == "title"
        widget_op = {"op": "set_widget", "node_id": 3, "widget": "title", "value": "X"}
        assert target != workflow_ops._write_target(widget_op)

    def test_apply_specs_batches_set_title(self):
        wf = _base_workflow()
        g = _graph()
        wf, ops, _aliases = workflow_ops.apply_specs(wf, g, [{"op": "set_title", "node": 3, "title": "Batched"}])
        assert ops[0]["op"] == "set_title"
        node = next(n for n in wf["nodes"] if n["id"] == 3)
        assert node["title"] == "Batched"


# ---------------------------------------------------------------------------
# CLI surface: `comfy workflow set-title` — the emitting affordance a human
# or the in-app agent uses; wiring this is what actually closes the sync gap
# (workflow_ops alone is unreachable without it).
# ---------------------------------------------------------------------------


class TestSetTitleCommand:
    def test_renames_node_and_emits_op(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-title", str(path), "3", "New Name"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_title"
        assert op["node_id"] == 3
        assert op["title"] == "New Name"
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == 3)
        assert node["title"] == "New Name"

    def test_clear_flag_resets_to_class_default(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow(title="Custom"))
        env = _run(["set-title", str(path), "3", "--clear"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["op"]["title"] is None
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == 3)
        assert "title" not in node

    def test_title_and_clear_together_is_rejected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-title", str(path), "3", "New Name", "--clear"], capsys)
        assert env["ok"] is False

    def test_unknown_node_errors(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-title", str(path), "999", "New Name"], capsys)
        assert env["ok"] is False
