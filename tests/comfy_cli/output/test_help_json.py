"""Help JSON coverage: every visible Typer command appears."""

import json

import pytest

from comfy_cli.cmdline import app
from comfy_cli.help_json import build_help_json, iter_command_paths


def test_help_json_includes_top_level_commands():
    doc = build_help_json(app)
    cmds = doc["commands"]
    # A representative subset that must always be there.
    for expected in ("env", "which", "install", "run", "launch", "model", "node"):
        assert expected in cmds, f"missing top-level command: {expected}"


def test_help_json_has_examples_for_well_known_commands():
    doc = build_help_json(app)
    assert doc["commands"]["env"]["examples"], "expected examples for `comfy env`"
    assert doc["commands"]["run"]["examples"], "expected examples for `comfy run`"


def test_help_json_run_has_workflow_param():
    doc = build_help_json(app)
    params = doc["commands"]["run"]["params"]
    assert any(p["name"] == "workflow" for p in params), "comfy run should expose --workflow"


def test_help_json_is_json_serializable():
    doc = build_help_json(app)
    s = json.dumps(doc, default=str)
    # Round-trip
    again = json.loads(s)
    assert sorted(again["commands"]) == sorted(doc["commands"])


def test_iter_command_paths_includes_subcommands():
    paths = iter_command_paths(app)
    # `comfy model download` is a known nested command via add_typer.
    assert any(p.endswith("model download") for p in paths), paths


def test_global_options_present_on_root():
    doc = build_help_json(app)
    root_params = {p["name"]: p for p in doc["root"]["params"]}
    for expected in ("workspace", "json_output", "json_stream", "no_json", "help_json"):
        assert expected in root_params, f"missing global option: {expected}"


@pytest.mark.parametrize(
    "cmd_path",
    [
        "env",
        "which",
        "install",
        "run",
        "launch",
    ],
)
def test_visible_command_has_help_text(cmd_path):
    doc = build_help_json(app)
    cmd = doc["commands"][cmd_path]
    assert cmd["help"], f"`comfy {cmd_path}` should have a help string"
