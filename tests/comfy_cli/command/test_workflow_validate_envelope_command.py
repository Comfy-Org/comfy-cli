"""The envelope's `command` must name the command that failed.

`workflow validate` stamped the Typer GROUP ("workflow") on every early-exit
error path — a missing file, unreadable JSON, an object_info that could not be
loaded — while the verdict path stamped "workflow validate". A caller routing
on `command` files a third of this command's own failures under the group.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner


def _run(args: list[str]) -> dict:
    from comfy_cli.cmdline import app

    result = CliRunner().invoke(app, args, env={"COMFY_OUTPUT": "json"})
    lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON envelope: {result.stdout!r}"
    return json.loads(lines[-1])


def _object_info(tmp_path: Path) -> Path:
    path = tmp_path / "object_info.json"
    path.write_text(
        json.dumps(
            {
                "PreviewImage": {
                    "input": {"required": {"images": "IMAGE"}},
                    "output": [],
                    "output_name": [],
                    "output_node": True,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_missing_workflow_file_names_the_subcommand(tmp_path: Path) -> None:
    env = _run(
        ["workflow", "validate", "--workflow", str(tmp_path / "nope.json"), "--input", str(_object_info(tmp_path))]
    )
    assert env["error"]["code"] == "workflow_not_found"
    assert env["command"] == "workflow validate"


def test_unloadable_catalog_names_the_subcommand(tmp_path: Path) -> None:
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}), encoding="utf-8")
    env = _run(["workflow", "validate", "--workflow", str(wf), "--input", str(tmp_path / "missing_catalog.json")])
    assert env["error"]["code"] == "cql_no_graph"
    assert env["command"] == "workflow validate"


def test_unparseable_workflow_names_the_subcommand(tmp_path: Path) -> None:
    wf = tmp_path / "wf.json"
    wf.write_text("{not json", encoding="utf-8")
    env = _run(["workflow", "validate", "--workflow", str(wf), "--input", str(_object_info(tmp_path))])
    assert env["error"]["code"] == "workflow_invalid_json"
    assert env["command"] == "workflow validate"


def test_non_object_workflow_names_the_subcommand(tmp_path: Path) -> None:
    wf = tmp_path / "wf.json"
    wf.write_text("[]", encoding="utf-8")
    env = _run(["workflow", "validate", "--workflow", str(wf), "--input", str(_object_info(tmp_path))])
    assert env["error"]["code"] == "workflow_not_api_format"
    assert env["command"] == "workflow validate"
