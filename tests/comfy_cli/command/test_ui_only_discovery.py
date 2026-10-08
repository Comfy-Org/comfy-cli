"""UI-only frontend nodes and subgraph instances, as the discovery surfaces show them.

A caller that knows ComfyUI from the canvas can ask `workflow add-node` for
`Reroute`, or pass a subgraph INSTANCE uuid as if it were a class. add-node
already refuses both with node_not_found ("'Reroute' is a UI-only node ...").

`nodes search`/`nodes ls` read object_info, which never carries Reroute/Note/
PrimitiveNode, so the catalog did not advertise them. Two surfaces still let a
caller walk into the wall:

* `nodes show Reroute` answered the generic "not found in the loaded
  environment" (plus difflib noise), not the UI-only explanation add-node gives
  — so a caller that checks before adding learns nothing.
* `workflow ls-nodes` printed a canvas Reroute or a subgraph instance as a
  plain `{id, type}` row, indistinguishable from an addable class.
"""

from __future__ import annotations

import json

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _force_json_renderer,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)
from typer.testing import CliRunner

from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.command import workflow_edit

_SG_UUID = "84e2cf3f-de93-40ef-ab22-b9375296917b"


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph())
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _show(capsys, name: str) -> dict:
    _force_json_renderer()
    result = CliRunner().invoke(nodes_cmd.app, ["show", name], standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout or ""
    for line in reversed([ln for ln in captured.strip().splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


@pytest.mark.parametrize("name", ["Reroute", "PrimitiveNode", "GetNode", "SetNode", "Note", "MarkdownNote"])
def test_nodes_show_names_a_ui_only_node(patched_graph, capsys, name):
    env = _show(capsys, name)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found", err
    assert "UI-only" in err["message"], err
    details = err.get("details") or {}
    assert details.get("ui_only") is True, err
    # difflib against a frontend-only name is noise, same as for a uuid.
    assert not details.get("close_matches"), err


def _ls_rows(tmp_path, capsys, wf: dict) -> dict:
    env = _run(["ls-nodes", str(_write(tmp_path, wf))], capsys)
    assert env["ok"] is True, env
    return {r["id"]: r for r in env["data"]["nodes"]}


def _wf_with_reroute_and_subgraph() -> dict:
    wf = _base_workflow()
    wf["nodes"].append({"id": 20, "type": "Reroute", "pos": [0, 0], "inputs": [], "outputs": []})
    wf["nodes"].append({"id": 21, "type": _SG_UUID, "pos": [0, 0], "inputs": [], "outputs": []})
    return wf


def test_ls_nodes_marks_ui_only_and_subgraph_rows(patched_graph, tmp_path, capsys):
    rows = _ls_rows(tmp_path, capsys, _wf_with_reroute_and_subgraph())
    assert rows[20]["type"] == "Reroute"  # still readable: the agent must understand the graph
    assert rows[20].get("ui_only") is True, rows[20]
    assert rows[21]["type"] == _SG_UUID
    assert rows[21].get("subgraph") is True, rows[21]


def test_ls_nodes_real_classes_stay_clean(patched_graph, tmp_path, capsys):
    rows = _ls_rows(tmp_path, capsys, _wf_with_reroute_and_subgraph())
    for nid in (3, 7):
        assert "ui_only" not in rows[nid] and "subgraph" not in rows[nid], rows[nid]
