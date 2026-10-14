"""DynamicGroup refusals carry their registered ``error.code`` in the envelope,
so a caller can count them without matching message text."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "dynamic_group.json").read_text())


def _unsupported(info, workflow):
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][0] = "COMFY_DYNAMICGROUP_V4"


def _mismatched(info, workflow):
    workflow["nodes"][0]["widgets_values"][1] = 3
    del workflow["nodes"][0]["widgets_values_named"]


def _linked(info, workflow):
    workflow["nodes"][0]["inputs"][-1]["link"] = 10


REFUSALS = {
    "dynamic_group_unsupported": (_unsupported, "after", '"edited"'),
    "dynamic_group_binding_mismatch": (_mismatched, "after", '"edited"'),
    "dynamic_group_count_out_of_range": (lambda info, workflow: None, "loras", "4"),
    "dynamic_group_rows_linked": (_linked, "loras", "1"),
}


def _run(args: list[str]) -> dict:
    from comfy_cli.cmdline import app

    result = CliRunner().invoke(app, args, env={"COMFY_OUTPUT": "json"})
    lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON envelope: {result.stdout!r}"
    return json.loads(lines[-1])


def _files(tmp_path: Path, code: str) -> tuple[str, str]:
    info = copy.deepcopy(FIXTURE["object_info"])
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    REFUSALS[code][0](info, workflow)
    info_path = tmp_path / "object_info.json"
    info_path.write_text(json.dumps(info))
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(workflow))
    return str(workflow_path), str(info_path)


@pytest.mark.parametrize("code", list(REFUSALS))
def test_set_widget_refusal_carries_its_code(tmp_path, code):
    workflow, info = _files(tmp_path, code)
    _, widget, value = REFUSALS[code]

    env = _run(["workflow", "set-widget", workflow, f"1.{widget}", value, "--input", info])

    assert env["error"]["code"] == code


@pytest.mark.parametrize("code", list(REFUSALS))
def test_batch_refusal_carries_its_code(tmp_path, code):
    workflow, info = _files(tmp_path, code)
    _, widget, value = REFUSALS[code]
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "set_widget", "node": 1, "widget": widget, "value": json.loads(value)}]))

    env = _run(["workflow", "apply", workflow, "--ops", str(ops), "--ack", "summary", "--input", info])

    assert env["error"]["code"] == code
    assert env["error"]["details"]["failed"]["code"] == code


@pytest.mark.parametrize("code", ["dynamic_group_unsupported", "dynamic_group_binding_mismatch"])
@pytest.mark.parametrize("command", ["validate", "slots", "decompose"])
def test_read_refusal_carries_its_code(tmp_path, code, command):
    workflow, info = _files(tmp_path, code)
    args = ["--workflow", workflow] if command == "validate" else [workflow]
    if command == "decompose":
        args += ["--lib", str(tmp_path / "fragments")]

    env = _run(["workflow", command, *args, "--input", info])

    assert env["error"]["code"] == code


@pytest.mark.parametrize("command", ["add-node", "foreach"])
def test_adding_an_unsupported_group_class_carries_its_code(tmp_path, command):
    workflow, info = _files(tmp_path, "dynamic_group_unsupported")
    if command == "add-node":
        args = ["add-node", workflow, "DevToolsNodeWithDynamicGroup"]
    else:
        recipe = tmp_path / "recipe.json"
        recipe.write_text(json.dumps({"ops": [{"op": "add_node", "class_type": "DevToolsNodeWithDynamicGroup"}]}))
        params = tmp_path / "params.json"
        params.write_text("[{}]")
        args = ["foreach", str(recipe), "--params", str(params), "--out-dir", str(tmp_path / "out")]

    env = _run(["workflow", *args, "--input", info])

    assert env["error"]["code"] == "dynamic_group_unsupported"
