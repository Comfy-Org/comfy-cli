"""Tests for ``comfy workflow delete-nodes <id...>`` (V1-020).

The measured agent loop runs ``delete-node`` once per doomed node (×22 runs) —
N file loads, N writes, N catalog loads. ``delete-nodes`` is the batch verb:
N ids, ONE atomic write, one frozen ``delete_node`` op per id (via
``workflow_ops.delete_node`` — no new op kind).

Contract under test:
  * batch removes every named node in one write; ``data.ops`` carries one
    stamped ``delete_node`` op per id (``--actor``/``--base-version`` honored).
  * any invalid id fails the WHOLE batch atomically — file unchanged — with
    the node-inventory hint rendered from the PRE-batch graph.
  * dangling links are cleaned exactly as the single-delete verb cleans them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

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
        "TinyLatent": {
            "input": {"required": {}},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "latent",
            "display_name": "Tiny Latent",
            "python_module": "nodes",
        },
        "TinyKS": {
            "input": {"required": {"latent_image": ["LATENT"]}},
            "input_order": {"required": ["latent_image"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "Tiny KSampler",
            "python_module": "nodes",
        },
        "TinyNote": {
            "input": {"required": {}},
            "output": [],
            "category": "util",
            "display_name": "Tiny Note",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _workflow() -> dict:
    """TinyLatent(7) --link 1--> TinyKS(3); TinyNote(5) free-standing."""
    return {
        "last_node_id": 7,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 3,
                "type": "TinyKS",
                "pos": [200, 100],
                "inputs": [{"name": "latent_image", "type": "LATENT", "link": 1}],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                "widgets_values": [],
            },
            {
                "id": 5,
                "type": "TinyNote",
                "pos": [0, 300],
                "inputs": [],
                "outputs": [],
                "widgets_values": [],
            },
            {
                "id": 7,
                "type": "TinyLatent",
                "pos": [0, 100],
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}],
                "widgets_values": [],
            },
        ],
        "links": [[1, 7, 0, 3, 0, "LATENT"]],
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "delete_nodes_wf.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


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


class TestDeleteNodesBatch:
    def test_batch_removes_all_and_emits_stamped_ops(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _workflow())
        env = _run(
            ["delete-nodes", str(path), "7", "5", "--actor", "agent-x", "--base-version", "4"],
            capsys,
        )
        assert env["ok"] is True, env
        data = env["data"]
        assert data["count"] == 2
        ops = data["ops"]
        # One frozen delete_node op per id — NOT a new op kind.
        assert [op["op"] for op in ops] == ["delete_node", "delete_node"]
        assert [op["node_id"] for op in ops] == [7, 5]
        for op in ops:
            assert op["actor"] == "agent-x"
            assert op["base_version"] == 4
            assert op["stamp"] == [4, "agent-x"]
            assert isinstance(op["op_id"], str) and op["op_id"]
        assert data["base_version"] == 4
        assert data["version"] == 4 + len(ops)
        # ONE write, both nodes gone.
        on_disk = json.loads(path.read_text())
        assert {n["id"] for n in on_disk["nodes"]} == {3}
        # Bookkeeping never serialized.
        assert "_applied_ops" not in on_disk

    def test_dangling_links_cleaned_exactly_like_single_delete(self, patched_graph, tmp_path, capsys):
        batch_path = _write(tmp_path, _workflow())
        env = _run(["delete-nodes", str(batch_path), "7"], capsys)
        assert env["ok"] is True
        batch_disk = json.loads(batch_path.read_text())

        single_path = tmp_path / "delete_nodes_single_wf.json"
        single_path.write_text(json.dumps(_workflow(), indent=2), encoding="utf-8")
        env2 = _run(["delete-node", str(single_path), "7"], capsys)
        assert env2["ok"] is True
        single_disk = json.loads(single_path.read_text())

        assert batch_disk["links"] == single_disk["links"] == []
        for disk in (batch_disk, single_disk):
            ks = next(n for n in disk["nodes"] if n["id"] == 3)
            assert ks["inputs"][0]["link"] is None

    def test_invalid_id_fails_whole_batch_atomically(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _workflow())
        before = path.read_text()
        env = _run(["delete-nodes", str(path), "7", "999"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        # File untouched even though id 7 was valid and listed first.
        assert path.read_text() == before
        # The node-inventory hint reflects the PRE-batch graph: node 7 still
        # exists on disk, so it must still be advertised.
        msg = env["error"]["message"]
        assert "999" in msg
        assert "7 (TinyLatent)" in msg
        assert "No changes were applied" in msg

    def test_stdout_mode_writes_nothing(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _workflow())
        before = path.read_text()
        env = _run(["delete-nodes", str(path), "7", "--stdout"], capsys)
        assert env["ok"] is True
        assert env["data"]["wrote"] is None
        assert path.read_text() == before
