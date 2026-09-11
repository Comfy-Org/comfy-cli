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
NESTED_ID = "abcdefab-cdef-4abc-8def-abcdefabcdef"
OTHER_NESTED_ID = "fedcbafe-dcba-4fed-8cba-fedcbafedcba"


def _definition(value: int = 1) -> dict:
    return {
        "id": SUBGRAPH_ID,
        "name": "One",
        "inputs": [],
        "outputs": [],
        "nodes": [{"id": 10, "type": "Inner", "widgets_values": [value]}],
        "links": [],
    }


def test_define_subgraph_emits_cmp_payload_and_inserts_definition():
    workflow = {"nodes": [], "links": []}
    definition = _definition()
    snapshot = copy.deepcopy(definition)

    result, op = workflow_ops.define_subgraph(workflow, definition, actor="agent", base_version=4)

    assert definition == snapshot
    assert op == {
        "op": "define_subgraph",
        "op_id": op["op_id"],
        "actor": "agent",
        "base_version": 4,
        "stamp": [4, "agent"],
        "subgraph_id": SUBGRAPH_ID,
        "subgraph_definition": definition,
    }
    assert result["definitions"]["subgraphs"] == [definition]


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        ([], "JSON object"),
        ({"id": 7, "nodes": [], "links": []}, "non-empty string id"),
        ({"id": "not-a-uuid", "nodes": [], "links": []}, "valid UUID"),
        ({"id": SUBGRAPH_ID, "nodes": {}, "links": []}, "nodes and links must be arrays"),
        (
            {
                "id": SUBGRAPH_ID,
                "nodes": [],
                "links": [],
                "definitions": {"subgraphs": [{"id": "not-a-uuid", "nodes": [], "links": []}]},
            },
            "definitions.subgraphs\\[0\\].*valid UUID",
        ),
    ],
)
def test_define_subgraph_rejects_malformed_input_before_mutation(definition, match):
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)

    with pytest.raises(ValueError, match=match):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_define_subgraph_can_assign_an_explicit_new_id():
    definition = _definition()
    definition.pop("id")

    _, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition, subgraph_id=SUBGRAPH_ID)

    assert op["subgraph_id"] == SUBGRAPH_ID
    assert op["subgraph_definition"]["id"] == SUBGRAPH_ID


def test_define_subgraph_rejects_existing_id_and_different_definition():
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [_definition()]}}

    with pytest.raises(ValueError, match="already exists"):
        workflow_ops.define_subgraph(workflow, _definition(2))


def test_define_subgraph_preserves_nested_definitions_inside_single_parent_op():
    nested = {"id": NESTED_ID, "nodes": [], "links": []}
    definition = {**_definition(), "definitions": {"subgraphs": [nested]}}

    result, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition)

    assert op["subgraph_definition"]["definitions"] == {"subgraphs": [nested]}
    assert result["definitions"]["subgraphs"] == [definition]
    assert op["op"] == "define_subgraph"


@pytest.mark.parametrize(
    "definition",
    [
        {
            **_definition(),
            "definitions": {"subgraphs": [{"id": SUBGRAPH_ID, "nodes": [], "links": []}]},
        },
        {
            **_definition(),
            "definitions": {
                "subgraphs": [
                    {
                        "id": NESTED_ID,
                        "nodes": [],
                        "links": [],
                        "definitions": {"subgraphs": [{"id": OTHER_NESTED_ID, "nodes": [], "links": []}]},
                    },
                    {"id": OTHER_NESTED_ID, "nodes": [], "links": []},
                ]
            },
        },
    ],
    ids=["ancestor", "across-branches"],
)
def test_define_subgraph_rejects_duplicate_ids_across_entire_definition_tree(definition):
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)

    with pytest.raises(ValueError, match="duplicates subgraph definition id"):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_define_subgraph_rejects_id_already_nested_in_workflow():
    existing = {
        "id": OTHER_NESTED_ID,
        "nodes": [],
        "links": [],
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": [], "links": []}]},
    }
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [existing]}}
    before = copy.deepcopy(workflow)
    definition = {
        **_definition(),
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": [], "links": []}]},
    }

    with pytest.raises(ValueError, match="duplicates subgraph definition id"):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_apply_define_subgraph_rejects_malformed_nested_definition_atomically():
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)
    definition = {
        **_definition(),
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": {}, "links": []}]},
    }
    op = {
        "op": "define_subgraph",
        "op_id": "a" * 32,
        "actor": "peer",
        "base_version": 0,
        "stamp": [0, "peer"],
        "subgraph_id": SUBGRAPH_ID,
        "subgraph_definition": definition,
    }

    with pytest.raises(ValueError, match="malformed_op:.*definitions.subgraphs\\[0\\].*nodes and links"):
        workflow_ops.apply_op(workflow, op, None)

    assert workflow == before


def test_apply_define_subgraph_replays_are_idempotent():
    workflow = {"nodes": [], "links": []}
    _, op = workflow_ops.define_subgraph(workflow, _definition())
    before = copy.deepcopy(workflow)

    workflow_ops.apply_op(workflow, op, None)
    assert workflow == before

    conflicting = {**op, "op_id": "f" * 32, "subgraph_definition": _definition(2)}
    workflow_ops.apply_op(workflow, conflicting, None)
    assert workflow["definitions"] == before["definitions"]


def test_define_subgraph_write_targets_are_scoped_by_definition_id():
    first = {"op": "define_subgraph", "subgraph_id": SUBGRAPH_ID}
    other = {"op": "define_subgraph", "subgraph_id": NESTED_ID}

    assert workflow_ops.detect_conflict(first, other) is False
    assert workflow_ops.detect_conflict(first, dict(first)) is True


def test_apply_define_subgraph_accepts_null_definitions_container():
    workflow = {"nodes": [], "links": [], "definitions": None}
    _, op = workflow_ops.define_subgraph(workflow, _definition())

    assert workflow["definitions"]["subgraphs"] == [op["subgraph_definition"]]


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        ({**_definition(), "nodes": [{"id": 1, "type": "Inner", "inputs": 1}]}, "inputs must be an array"),
        ({**_definition(), "links": [[1, 2]]}, "link.*tuple"),
        ({**_definition(), "nodes": [{"id": 1, "type": SUBGRAPH_ID}]}, "cyclic subgraph reference"),
        (
            {
                **_definition(),
                "nodes": [{"id": 1, "type": NESTED_ID}],
                "definitions": {
                    "subgraphs": [{"id": NESTED_ID, "nodes": [{"id": 2, "type": SUBGRAPH_ID}], "links": []}]
                },
            },
            "cyclic subgraph reference",
        ),
    ],
)
def test_define_subgraph_rejects_malformed_or_recursive_interior(definition, match):
    with pytest.raises(ValueError, match=match):
        workflow_ops.define_subgraph({"nodes": [], "links": []}, definition)


def test_define_subgraph_preserves_explicit_falsy_ids_for_validation():
    with pytest.raises(ValueError, match="non-empty string id"):
        workflow_ops.define_subgraph({"nodes": [], "links": []}, _definition(), subgraph_id="")
    with pytest.raises(ValueError, match="non-empty string id"):
        workflow_ops.define_subgraph({"nodes": [], "links": []}, {"id": 0, "nodes": [], "links": []})


def test_apply_define_subgraph_different_redelivery_is_a_noop():
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [_definition(2)]}}
    before = copy.deepcopy(workflow)
    _, original_op = workflow_ops.define_subgraph({"nodes": [], "links": []}, _definition())

    workflow_ops.apply_op(workflow, original_op, None)

    assert workflow["definitions"] == before["definitions"]
    assert original_op["op_id"] in workflow["_applied_ops"]


def test_apply_define_subgraph_fresh_op_id_exercises_definition_idempotency():
    workflow = {"nodes": [], "links": []}
    _, op = workflow_ops.define_subgraph(workflow, _definition())
    workflow_ops.strip_internal(workflow)

    workflow_ops.apply_op(workflow, {**op, "op_id": "e" * 32}, None)

    assert workflow["definitions"]["subgraphs"] == [_definition()]


def test_replace_ops_emits_definition_before_nodes():
    new = {"nodes": [], "links": [], "definitions": {"subgraphs": [_definition()]}}

    ops = workflow_ops.replace_ops({"nodes": [], "links": []}, new)

    assert [op["op"] for op in ops] == ["define_subgraph"]


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
