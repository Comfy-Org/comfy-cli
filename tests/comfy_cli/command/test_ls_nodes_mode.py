"""`workflow ls-nodes` must report a node's BYPASS/MUTE state.

ComfyUI lets a user disable a node without deleting it: mode 4 = bypass (passes
input through), mode 2 = mute/never (removed from execution). workflow_to_api
already understands both — it strips them at API conversion
(workflow_to_api.py:149) and has a bypassed-id helper (:550).

But ls-nodes emitted only id/type/title, so a caller inspecting the graph could
not tell a disabled node from a live one. The consequence is worse than a
cosmetic gap: the agent "repairs" a graph whose node is merely bypassed, or
reports a workflow ready to run when a required node is muted.
"""

from __future__ import annotations

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow_edit

MODE_MUTED, MODE_BYPASS = 2, 4


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _wf_with_modes() -> dict:
    wf = _base_workflow()
    wf["nodes"][0]["mode"] = MODE_BYPASS   # KSampler, id 3
    wf["nodes"][1]["mode"] = MODE_MUTED    # EmptyLatentImage, id 7
    wf["nodes"].append({
        "id": 9, "type": "VAEDecode", "pos": [0, 0], "mode": 0,
        "inputs": [], "outputs": [], "widgets_values": [],
    })
    return wf


def _rows(tmp_path, capsys) -> dict:
    path = _write(tmp_path, _wf_with_modes())
    env = _run(["ls-nodes", str(path)], capsys)
    assert env["ok"] is True, env
    return {r["id"]: r for r in env["data"]["nodes"]}


def test_ls_nodes_reports_bypassed_and_muted(patched_graph, tmp_path, capsys):
    rows = _rows(tmp_path, capsys)
    assert rows[3].get("mode") == "bypass", rows[3]
    assert rows[7].get("mode") == "mute", rows[7]


def test_ls_nodes_omits_mode_for_normal_nodes(patched_graph, tmp_path, capsys):
    """A live node must stay a single clean row — no mode noise on the 99% case."""
    rows = _rows(tmp_path, capsys)
    assert "mode" not in rows[9], rows[9]


def test_ls_nodes_unchanged_when_no_modes_set(patched_graph, tmp_path, capsys):
    path = _write(tmp_path, _base_workflow())
    env = _run(["ls-nodes", str(path)], capsys)
    assert all("mode" not in r for r in env["data"]["nodes"]), env["data"]["nodes"]
