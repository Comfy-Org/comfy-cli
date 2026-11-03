"""`comfy workflow validate` folds many pruned-node warnings into one.

A canvas that keeps finished branches around produced one
`node_not_reachable_from_output` warning per pruned node (up to 66 per call on a
76-node workflow), and an agent re-read every one on each later round. More than
three now fold into one entry carrying the count and every id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _object_info() -> dict:
    return {
        "Src": {
            "input": {"required": {}},
            "input_order": {"required": []},
            "output": ["INT"],
            "output_name": ["INT"],
            "category": "testing",
            "display_name": "Src",
            "description": "",
            "output_node": False,
            "python_module": "nodes",
        },
        "Sink": {
            "input": {"required": {"value": ["INT", {}]}},
            "input_order": {"required": ["value"]},
            "output": [],
            "output_name": [],
            "category": "testing",
            "display_name": "Sink",
            "description": "",
            "output_node": True,
            "python_module": "nodes",
        },
    }


def _validate(tmp_path: Path, capsys, dangling: int) -> dict:
    wf = {"1": {"class_type": "Src", "inputs": {}}, "99": {"class_type": "Sink", "inputs": {"value": ["1", 0]}}}
    for i in range(2, 2 + dangling):
        wf[str(i)] = {"class_type": "Src", "inputs": {}}
    oi_path = tmp_path / "oi.json"
    oi_path.write_text(json.dumps(_object_info()), encoding="utf-8")
    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps(wf), encoding="utf-8")

    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(workflow_cmd.app, ["validate", "--workflow", str(wf_path), "--input", str(oi_path)])
    out = (capsys.readouterr().out or "") or (result.stdout or "")
    envelope = json.loads([ln for ln in out.strip().splitlines() if ln.strip()][-1])
    assert envelope["ok"] is True, envelope
    return envelope["data"]


def _unreachable(data: dict) -> list[dict]:
    return [w for w in data["warnings"] if w["code"] == "node_not_reachable_from_output"]


def test_many_pruned_nodes_fold_into_one_warning(tmp_path, capsys):
    data = _validate(tmp_path, capsys, dangling=6)
    warns = _unreachable(data)
    assert len(warns) == 1, warns
    w = warns[0]
    assert w["count"] == 6
    assert sorted(w["node_ids"]) == ["2", "3", "4", "5", "6", "7"]
    assert w["node_id"] in w["node_ids"]
    for nid in w["node_ids"]:
        assert nid in w["message"]
    assert "Src x6" in w["message"]
    assert data["valid"] is True


def test_long_id_lists_are_truncated_in_the_message_only(tmp_path, capsys):
    data = _validate(tmp_path, capsys, dangling=14)
    w = _unreachable(data)[0]
    assert w["count"] == 14
    assert len(w["node_ids"]) == 14
    assert "and 4 more" in w["message"]


def test_a_few_pruned_nodes_keep_one_warning_each(tmp_path, capsys):
    data = _validate(tmp_path, capsys, dangling=3)
    warns = _unreachable(data)
    assert sorted(w["node_id"] for w in warns) == ["2", "3", "4"]
    assert all("count" not in w for w in warns)
