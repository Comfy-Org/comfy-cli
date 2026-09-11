from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import typer

from comfy_cli import workflow_ops
from comfy_cli.command import workflow as workflow_cmd  # noqa: F401 -- initializes the edit command cycle
from comfy_cli.command import workflow_edit

SUBGRAPH_ID = "12345678-1234-4123-8123-123456789abc"


def _definition() -> dict:
    return {
        "id": SUBGRAPH_ID,
        "name": "One",
        "inputs": [],
        "outputs": [],
        "nodes": [{"id": 10, "type": "Inner"}],
        "links": [],
    }


def test_define_subgraph_emits_cmp_payload_without_mutating_workflow():
    workflow = {"nodes": [{"id": 1, "type": "Existing"}], "links": []}
    before = copy.deepcopy(workflow)
    definition = _definition()

    result, op = workflow_ops.define_subgraph(workflow, definition, actor="agent", base_version=4)

    assert result is workflow
    assert workflow == before
    assert op == {
        "op": "define_subgraph",
        "op_id": op["op_id"],
        "actor": "agent",
        "base_version": 4,
        "stamp": [4, "agent"],
        "subgraph_id": SUBGRAPH_ID,
        "subgraph_definition": definition,
    }


def test_define_subgraph_emits_semantically_odd_definition_for_cmp_validation():
    definition = {"id": SUBGRAPH_ID, "nodes": "future-cmp-shape", "links": {"also": "future"}}

    _, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition)

    assert op["subgraph_definition"] == definition


def test_define_subgraph_rejects_local_replay():
    _, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, _definition())

    with pytest.raises(ValueError, match="unknown op 'define_subgraph'"):
        workflow_ops.apply_op({"nodes": [], "links": []}, op, None)


def test_define_subgraph_can_assign_an_explicit_new_id():
    definition = _definition()
    definition.pop("id")

    _, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition, subgraph_id=SUBGRAPH_ID)

    assert op["subgraph_id"] == SUBGRAPH_ID
    assert op["subgraph_definition"]["id"] == SUBGRAPH_ID


def test_define_subgraph_command_emits_without_writing_workflow(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow.json"
    original = {"nodes": [], "links": []}
    workflow.write_text(json.dumps(original))
    definition_file = tmp_path / "definition.json"
    definition_file.write_text(json.dumps(_definition()))
    emitted = []

    class Renderer:
        command = ""

        def is_pretty(self):
            return False

        def emit(self, payload, **kwargs):
            emitted.append((payload, kwargs))

    monkeypatch.setattr(workflow_edit, "get_renderer", Renderer)

    workflow_edit.define_subgraph_cmd(str(workflow), str(definition_file))

    assert json.loads(workflow.read_text()) == original
    assert emitted[0][0]["op"]["op"] == "define_subgraph"
    assert emitted[0][0]["wrote"] is None
    assert emitted[0][1]["changed"] is False


def test_define_subgraph_command_rejects_non_regular_definition_file(tmp_path, monkeypatch):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"nodes": [], "links": []}))
    definition_dir = tmp_path / "definition"
    definition_dir.mkdir()
    errors = []

    class Renderer:
        command = ""

        def error(self, **kwargs):
            errors.append(kwargs)

    monkeypatch.setattr(workflow_edit, "get_renderer", Renderer)
    with pytest.raises(typer.Exit):
        workflow_edit.define_subgraph_cmd(str(workflow), str(definition_dir))

    assert "regular file" in errors[0]["message"]


def test_definition_reader_rejects_oversized_files(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (workflow_edit._MAX_DEFINITION_BYTES + 1))

    with pytest.raises(ValueError, match="too large"):
        workflow_edit._read_subgraph_definition(Path(path))
