"""``set_node_field`` — a per-field write to a node's durable scalar state.

The vocabulary can already express "this node changed" only as an ``add_node``
upsert, which replaces the WHOLE node: it rewrites the node's widget values and
clears its widget stamps, so a title or flag change concurrent with a widget
write on that node discards the write. ``set_node_field`` claims one LWW
register per ``(node, field)`` instead.

These tests pin the CLI half of that: the ``comfy workflow set-node-field``
command mints a replayable op, ``apply_op`` replays it, the field allowlist is
closed, and it rides inside a batch.
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
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


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
        "TinyLoader": {
            "input": {"required": {"ckpt_name": [["a.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL"],
            "output_name": ["MODEL"],
            "category": "loaders",
            "display_name": "Tiny Loader",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _populated() -> dict[str, Any]:
    return {
        "id": "wf-1",
        "revision": 0,
        "nodes": [
            {
                "id": 1,
                "type": "TinyLoader",
                "pos": [0, 0],
                "title": "Tiny Loader",
                "mode": 0,
                "flags": {},
                "inputs": [],
                "outputs": [],
                "widgets_values": ["a.safetensors"],
            },
        ],
        "links": [],
        "last_node_id": 1,
        "last_link_id": 0,
    }


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


class TestSetNodeFieldCommand:
    @pytest.mark.parametrize(
        ("field", "raw", "expected"),
        [
            ("title", "Renamed", "Renamed"),
            ("mode", "4", 4),
            ("flags.collapsed", "true", True),
            ("flags.pinned", "true", True),
        ],
    )
    def test_writes_the_field_and_emits_a_replayable_op(self, tmp_path: Path, capsys, field, raw, expected):
        wf = tmp_path / "wf_set_node_field.json"
        wf.write_text(json.dumps(_populated()), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "1", field, raw], capsys)

        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_node_field"
        assert str(op["node_id"]) == "1"
        assert op["field"] == field
        assert op["value"] == expected
        assert op["stamp"] == [op["base_version"], op["actor"]]

        node = json.loads(wf.read_text(encoding="utf-8"))["nodes"][0]
        head, _, leaf = field.partition(".")
        assert (node[head][leaf] if leaf else node[head]) == expected

    def test_rejects_a_field_outside_the_allowlist(self, tmp_path: Path, capsys):
        """`widgets_values` is `set_widget`'s register and `type` is node
        identity; neither may be moved by a field write."""
        wf = tmp_path / "wf_bad_field.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "1", "widgets_values", "[]"], capsys)

        assert env["ok"] is False
        assert json.loads(wf.read_text(encoding="utf-8")) == before

    def test_rejects_a_missing_node(self, tmp_path: Path, capsys):
        wf = tmp_path / "wf_missing_node.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "999", "title", "ghost"], capsys)

        assert env["ok"] is False
        assert json.loads(wf.read_text(encoding="utf-8")) == before


class TestSetNodeFieldReplay:
    def test_apply_op_replays_the_op_idempotently(self):
        workflow = _populated()
        _, op = workflow_ops.set_node_field(_populated(), "1", "title", "Renamed", actor="cli", base_version=1)

        workflow_ops.apply_op(workflow, op, _graph())
        assert workflow["nodes"][0]["title"] == "Renamed"

        workflow_ops.apply_op(workflow, op, _graph())
        assert workflow["_applied_ops"].count(op["op_id"]) == 1

    def test_two_fields_of_one_node_do_not_contend(self):
        """One register per (node, field): a title write and a flag write on
        the same node claim different LWW targets, so neither drops the other
        whatever order they replay in."""
        title_op = workflow_ops.set_node_field(_populated(), "1", "title", "T", actor="a", base_version=1)[1]
        flag_op = workflow_ops.set_node_field(_populated(), "1", "flags.collapsed", True, actor="b", base_version=1)[1]

        assert workflow_ops._write_target(title_op) != workflow_ops._write_target(flag_op)
        assert not workflow_ops.detect_conflict(title_op, flag_op)

        for order in ([title_op, flag_op], [flag_op, title_op]):
            workflow = _populated()
            for op in order:
                workflow_ops.apply_op(workflow, op, _graph())
            node = workflow["nodes"][0]
            assert node["title"] == "T"
            assert node["flags"]["collapsed"] is True

    def test_two_writers_of_one_field_resolve_by_stamp(self):
        early = workflow_ops.set_node_field(_populated(), "1", "title", "early", actor="a", base_version=1)[1]
        late = workflow_ops.set_node_field(_populated(), "1", "title", "late", actor="b", base_version=2)[1]

        for order in ([early, late], [late, early]):
            workflow = _populated()
            for op in order:
                workflow_ops.apply_op(workflow, op, _graph())
            assert workflow["nodes"][0]["title"] == "late"


class TestSetNodeFieldIsBatchable:
    def test_rides_inside_a_batch(self):
        workflow, ops, _aliases = workflow_ops.apply_specs(
            _populated(),
            _graph(),
            [{"op": "set_node_field", "node_id": "1", "field": "title", "value": "batched"}],
        )

        assert [op["op"] for op in ops] == ["set_node_field"]
        assert workflow["nodes"][0]["title"] == "batched"
