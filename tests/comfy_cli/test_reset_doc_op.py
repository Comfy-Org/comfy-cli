"""``reset_doc`` — the guarded, standalone-only document reset (V1-038 / BE-7171).

``reset_doc`` was frozen in ``docs/op-vocabulary-v1.md`` §1.6 but left deferred:
``apply_op`` rejected it and ``DEFERRED_OPS`` pinned that rejection. This ticket
un-defers it, so the guarantees that make it safe move from prose into tests:

* it is **guarded** — ``comfy workflow reset-doc <file>`` fails closed without an
  explicit ``--confirm`` and writes nothing;
* it is **standalone-only** — a batch containing it is rejected atomically with a
  registered error code, exactly like ``clear``;
* it is a **history barrier** — unlike ``clear`` it drops the id high-water marks
  and the applied-op bookkeeping, so it is not merely "delete every node".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import error_codes, workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.command import workflow_edit
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
            {"id": 1, "type": "TinyLoader", "pos": [0, 0], "inputs": [], "outputs": [], "widgets_values": []},
            {"id": 2, "type": "TinyLoader", "pos": [10, 0], "inputs": [], "outputs": [], "widgets_values": []},
        ],
        "links": [],
        "groups": [{"title": "g"}],
        "last_node_id": 2,
        "last_link_id": 0,
        "_applied_ops": ["deadbeef" * 4],
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


class TestResetDocGuard:
    def test_reset_doc_requires_confirm(self, tmp_path: Path, capsys):
        """Without --confirm the command fails closed: nothing is written.

        The guard is the whole reason reset_doc is safe to expose — it erases
        replay history, so an accidental invocation is unrecoverable by replay.
        """
        wf = tmp_path / "wf_reset_guard.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["reset-doc", str(wf)], capsys)

        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_reset_doc_unconfirmed"
        assert "--confirm" in (env["error"].get("hint") or "")
        # The file is byte-for-byte untouched — a guard that writes anything is
        # not a guard.
        assert json.loads(wf.read_text(encoding="utf-8")) == before

    def test_reset_doc_with_confirm_empties_the_document(self, tmp_path: Path, capsys):
        wf = tmp_path / "wf_reset_confirm.json"
        wf.write_text(json.dumps(_populated()), encoding="utf-8")

        env = _run(["reset-doc", str(wf), "--confirm"], capsys)

        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "reset_doc"
        assert op["removed_nodes"] == [1, 2]
        assert op["stamp"] == [op["base_version"], op["actor"]]

        after = json.loads(wf.read_text(encoding="utf-8"))
        assert after["nodes"] == []
        assert after["links"] == []
        assert after["groups"] == []
        # History barrier, not a clear: the high-water marks go back to the
        # empty baseline (clear preserves them, §1.5 vs §1.6).
        assert after["last_node_id"] == 0
        assert after["last_link_id"] == 0


class TestResetDocIsNotBatchable:
    def test_reset_doc_rejected_in_batch(self):
        """A batch containing reset_doc is rejected atomically, with its own
        registered code naming the standalone command."""
        with pytest.raises(workflow_ops.NotBatchableError) as ei:
            workflow_ops.apply_specs(
                {"nodes": [], "links": []},
                _graph(),
                [{"op": "add_node", "class_type": "TinyLoader"}, {"op": "reset_doc"}],
            )
        err = ei.value
        assert err.code == "workflow_reset_doc_not_batchable"
        assert error_codes.is_registered(err.code)
        registered = error_codes.get(err.code)
        assert registered is not None and "comfy workflow reset-doc" in (registered.hint or "")
        assert "comfy workflow reset-doc" in err.hint
        assert "no changes were applied" in str(err).lower()

    def test_reset_doc_rejected_through_the_apply_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())
        wf = tmp_path / "wf_reset_batch.json"
        wf.write_text(json.dumps(_populated()), encoding="utf-8")
        ops = tmp_path / "reset_batch_ops.json"
        ops.write_text(json.dumps([{"op": "reset_doc"}]), encoding="utf-8")

        env = _run(["apply", str(wf), "--ops", str(ops)], capsys)

        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_reset_doc_not_batchable"
        # Atomic: the graph the batch was rejected against is untouched.
        assert len(json.loads(wf.read_text(encoding="utf-8"))["nodes"]) == 2


class TestResetDocReplay:
    def test_apply_op_replays_reset_doc_and_records_it(self):
        """Un-deferred: apply_op dispatches reset_doc. Its own op_id survives
        the wipe, so a re-delivered reset is a no-op rather than a second wipe."""
        wf, op = workflow_ops.reset_doc(_populated(), actor="agent:t:1", base_version=3)
        assert wf["nodes"] == [] and wf["links"] == []
        assert wf["_applied_ops"] == [op["op_id"]]

        # Idempotent re-delivery: put a node back, replay the same op — the
        # op_id gate drops it, so the node survives.
        wf["nodes"].append({"id": 9, "type": "TinyLoader"})
        wf = workflow_ops.apply_op(wf, op, None)
        assert [n["id"] for n in wf["nodes"]] == [9]

    def test_reset_doc_is_no_longer_deferred(self):
        assert "reset_doc" in workflow_ops.FROZEN_OPS
        assert "reset_doc" not in workflow_ops.DEFERRED_OPS
        assert "reset_doc" not in workflow_ops.BATCHABLE_OPS
