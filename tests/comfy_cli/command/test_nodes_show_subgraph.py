"""`nodes show <uuid>` must name the UUID as a subgraph type, not "not found".

Measured on prod comfy-agent traces (2026-08-05, session 0cc9d03b): the agent
read a subgraph instance's `type` — its definition UUID — out of `ls-nodes` and
asked `show_node` about it, seven times:

  show_node "84e2cf3f-de93-40ef-ab22-b9375296917b"
  => "Node class '84e2cf3f-…' not found in the loaded environment."

`workflow add-node` already explains this shape (see
test_add_node_unknown_class.py::test_uuid_class_type_is_named_as_a_subgraph_instance);
`nodes show` was left behind with the generic catalog miss.
"""

from __future__ import annotations

import json

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _force_json_renderer,
    _graph,
    reset_singleton,  # noqa: F401  (autouse fixture)
)
from typer.testing import CliRunner

from comfy_cli.command import nodes as nodes_cmd

_SG_UUID = "84e2cf3f-de93-40ef-ab22-b9375296917b"


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph())


def _show(capsys, name: str) -> dict:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(nodes_cmd.app, ["show", name], standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed([ln for ln in captured.strip().splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


def test_uuid_is_named_as_a_subgraph_type(patched_graph, capsys):
    env = _show(capsys, _SG_UUID)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found", err
    blob = json.dumps(err).lower()
    assert "subgraph" in blob, f"must explain the UUID is a subgraph type id: {err}"
    # Point at the surface that CAN inspect it.
    assert "slots" in blob or "ls-nodes" in blob, err
    assert (err.get("details") or {}).get("subgraph_id") is True, err
    # difflib matches against a UUID are noise; don't emit any.
    assert not (err.get("details") or {}).get("close_matches"), err


def test_plain_unknown_class_keeps_close_matches(patched_graph, capsys):
    env = _show(capsys, "KSample")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found"
    assert "KSampler" in (err.get("details") or {}).get("close_matches", []), err
