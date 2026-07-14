"""`comfy workflow validate` (canonical) + the deprecated `comfy validate` alias.

Both entry points share one implementation (`workflow.validate_api_workflow`).
These tests drive them offline via `--input <object_info.json>` so no ComfyUI
server is needed, and assert the alias stays behavior-compatible while warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app as root_app
from comfy_cli.command import workflow as workflow_cmd
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


def _object_info() -> dict:
    """Minimal object_info: one output node with a single INT widget input."""
    return {
        "TrivialNode": {
            "input": {"required": {"value": ["INT", {"default": 0}]}},
            "input_order": {"required": ["value"]},
            "output": [],
            "output_name": [],
            "category": "testing",
            "display_name": "Trivial",
            "description": "A trivial node for tests.",
            "output_node": True,
            "python_module": "nodes",
        }
    }


def _valid_workflow() -> dict:
    return {"1": {"class_type": "TrivialNode", "inputs": {"value": 5}}}


def _invalid_workflow() -> dict:
    # `value` is declared INT; a string is a shape mismatch -> a validation error.
    return {"1": {"class_type": "TrivialNode", "inputs": {"value": "not-an-int"}}}


def _write(tmp_path: Path, name: str, data: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def _run(app, args: list[str], capsys):
    """Invoke a command and return (exit_code, envelope, combined_output)."""
    _force_json_renderer()
    runner = CliRunner()
    # Default standalone_mode so `typer.Exit(code=1)` maps to result.exit_code.
    result = runner.invoke(app, args)
    captured = capsys.readouterr()
    combined = (captured.out or "") + (captured.err or "")
    if not combined.strip():
        combined = result.stdout or ""
    envelope = None
    for line in reversed(combined.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return result.exit_code, envelope, combined


def test_workflow_validate_valid(tmp_path, capsys):
    oi = _write(tmp_path, "oi.json", _object_info())
    wf = _write(tmp_path, "wf.json", _valid_workflow())

    code, env, _ = _run(workflow_cmd.app, ["validate", "--workflow", wf, "--input", oi], capsys)

    assert code == 0
    assert env is not None
    assert env["ok"] is True
    assert env["command"] == "workflow validate"
    assert env["data"]["valid"] is True
    assert env["data"]["error_count"] == 0


def test_workflow_validate_invalid_exits_nonzero(tmp_path, capsys):
    oi = _write(tmp_path, "oi.json", _object_info())
    wf = _write(tmp_path, "wf.json", _invalid_workflow())

    code, env, _ = _run(workflow_cmd.app, ["validate", "--workflow", wf, "--input", oi], capsys)

    assert code == 1
    assert env is not None
    assert env["ok"] is False
    assert env["data"]["valid"] is False
    assert env["data"]["error_count"] >= 1


def test_deprecated_alias_still_works_and_warns(tmp_path, capsys):
    oi = _write(tmp_path, "oi.json", _object_info())
    wf = _write(tmp_path, "wf.json", _valid_workflow())

    code, env, combined = _run(root_app, ["validate", "--workflow", wf, "--input", oi], capsys)

    # Still functional...
    assert code == 0
    assert env is not None
    assert env["ok"] is True
    assert env["data"]["valid"] is True
    # ...but the alias labels itself and warns to point at the canonical home.
    assert env["command"] == "validate"
    assert "deprecated" in combined.lower()
    assert "comfy workflow validate" in combined


def test_alias_matches_canonical_validation_payload(tmp_path, capsys):
    """The deprecated alias must produce the same validation verdict as the
    canonical command for the same inputs (only the `command` label differs)."""
    oi = _write(tmp_path, "oi.json", _object_info())
    wf = _write(tmp_path, "wf.json", _valid_workflow())

    _, canonical, _ = _run(workflow_cmd.app, ["validate", "--workflow", wf, "--input", oi], capsys)
    _, alias, _ = _run(root_app, ["validate", "--workflow", wf, "--input", oi], capsys)

    for key in ("valid", "error_count", "warning_count", "errors", "warnings"):
        assert canonical["data"][key] == alias["data"][key]
