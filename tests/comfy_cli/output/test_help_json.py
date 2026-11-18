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


def _find_param(doc, command_path, param_name):
    """Look up a single param dict by space-separated command path + name."""
    node = {"subcommands": doc["commands"], "params": doc["root"]["params"]}
    for part in command_path.split():
        node = node["subcommands"][part]
    for p in node["params"]:
        if p["name"] == param_name:
            return p
    raise AssertionError(f"no param {param_name!r} on {command_path!r}")


# (command path, param name, expected ``type``). These pin the help JSON ``type``
# contract across typer versions: typer >= 0.29 dropped ``param.type`` and we now
# rebuild the value from ``type_descriptor``, so a drift in that mapping must fail
# here rather than silently change the contract agents consume. The newest-typer
# CI lane runs this against an unpinned typer so the next break surfaces early.
@pytest.mark.parametrize(
    ("command_path", "param_name", "expected_type"),
    [
        ("", "workspace", "str"),  # root option, plain string
        ("", "recent", "boolean"),  # root flag
        ("jobs ls", "limit", "int"),  # plain int
        ("stop", "port", "int range"),  # bounded int
        ("jobs wait", "poll_interval", "float"),  # plain float
        ("deploy run", "timeout", "float range"),  # bounded float
        ("preview", "file", "path"),  # pathlib.Path
        ("deploy run", "asset_root", "path"),  # list[Path] -> element type
        ("run", "output_node_ids", "str"),  # list[str] -> element type
        ("install", "cuda_version", "choice"),  # enum choice
    ],
)
def test_param_type_contract(command_path, param_name, expected_type):
    doc = build_help_json(app)
    param = _find_param(doc, command_path, param_name)
    assert param["type"] == expected_type


def test_choice_param_emits_string_choices():
    doc = build_help_json(app)
    cuda = _find_param(doc, "install", "cuda_version")
    assert cuda["choices"], "enum option should list its choices"
    assert all(isinstance(c, str) for c in cuda["choices"]), cuda["choices"]
    # Enum *values* (not member reprs) are the contract agents consume.
    assert "12.8" in cuda["choices"]


def test_every_param_type_is_a_nonempty_string():
    """`schemas/help.json` requires every param to carry a string ``type``."""
    doc = build_help_json(app)

    def walk(node):
        for p in node.get("params", []):
            assert isinstance(p.get("type"), str) and p["type"], p
        for sub in node.get("subcommands", {}).values():
            walk(sub)

    walk({"params": doc["root"]["params"], "subcommands": doc["commands"]})
