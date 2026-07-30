"""`workflow slots` must advertise dynamic-combo SUB-widgets.

A COMFY_DYNAMICCOMBO_V3 input is one port (`model`) whose selected option
contributes extra widgets addressed as `model.<sub>`. `widget_order_for_node`
knows them; `_node_widget_slots` iterated `m.inputs` instead, so the dotted
addresses never appeared in `workflow slots` — the only place they surfaced was
the set-widget ERROR:

  widget 'prompt' not found on ByteDance2ReferenceNode;
  available: model, model.prompt, model.resolution

i.e. the CLI knew the answer and would not advertise it. 102 catalog node types
carry a dynamic combo; 4 prod set_widget failures in 6 days were this shape.
"""

from __future__ import annotations

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow as workflow_cmd


@pytest.fixture
def patched_graph(monkeypatch):
    # `slots` resolves its catalog through command/workflow.py, not workflow_edit.
    monkeypatch.setattr(workflow_cmd, "_get_graph", lambda *a, **kw: _graph())


def _wf() -> dict:
    return {
        "last_node_id": 5,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 5,
                "type": "KlingFLFTest",
                "pos": [0, 0],
                "inputs": [
                    {"name": "first_frame", "type": "IMAGE", "link": None},
                    {"name": "last_frame", "type": "IMAGE", "link": None},
                ],
                "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": []}],
                "widgets_values": ["a prompt", "kling-v3", "1080p"],
            }
        ],
        "links": [],
    }


def test_slots_include_dynamic_combo_subwidgets(patched_graph, tmp_path, capsys):
    env = _run(["slots", str(_write(tmp_path, _wf()))], capsys)
    assert env["ok"] is True, env
    addrs = {s["address"] for s in env["data"]["slots"]}
    assert "5.prompt" in addrs, addrs
    assert "5.model" in addrs, addrs
    assert "5.model.resolution" in addrs, f"the dynamic-combo sub-widget must be advertised: {addrs}"


def test_subwidget_carries_its_current_value(patched_graph, tmp_path, capsys):
    env = _run(["slots", str(_write(tmp_path, _wf()))], capsys)
    by = {s["address"]: s for s in env["data"]["slots"]}
    assert by["5.model"]["current_value"] == "kling-v3"
    assert by["5.model.resolution"]["current_value"] == "1080p", by["5.model.resolution"]
