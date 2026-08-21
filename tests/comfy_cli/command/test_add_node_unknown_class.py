"""`workflow add-node` must fail like `nodes show` does when a class is unknown.

Measured on prod comfy-agent traces (2026-07-23 → 07-28): 12 failures where the
agent named a class that does not exist, and it got NO suggestions back:

  add_node:  "unknown node type 'MarkdownNote'"
  add_node:  "unknown node type 'Note'"
  add_node:  "unknown node type '2454ad83-157c-40dd-9f19-5daaf4041ce0'"
  show_node: "Node class 'RadianceShowText' not found ..."  + close_matches

`nodes show` emits code=node_not_found with details.close_matches, so the agent
self-corrects in one retry. `workflow add-node` emitted
code=workflow_edit_invalid with hint "run `comfy nodes types`" — which lists
CONNECTION types (MODEL/LATENT/IMAGE), not class_types, and which the agent has
no tool for. It also means the agent-side annotateNodeNotFound (which keys on
code == "node_not_found") never fired on this path.
"""

from __future__ import annotations

import json

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow_edit


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _add(tmp_path, capsys, class_type: str) -> dict:
    path = _write(tmp_path, _base_workflow())
    return _run(["add-node", str(path), class_type], capsys)


def test_unknown_class_emits_node_not_found_with_close_matches(patched_graph, tmp_path, capsys):
    # 'KSample' is a near-miss for the catalog's 'KSampler'.
    env = _add(tmp_path, capsys, "KSample")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found", f"must match `nodes show`'s code: {err}"
    assert "KSampler" in (err.get("details") or {}).get("close_matches", []), err
    assert "KSampler" in (err.get("hint") or ""), err
    # The misleading hint must be gone: `nodes types` lists connection types.
    assert "nodes types" not in json.dumps(err)


def test_ui_only_node_is_rejected_with_a_specific_reason(patched_graph, tmp_path, capsys):
    """Note/MarkdownNote/Reroute/GetNode/SetNode/PrimitiveNode exist only in the
    UI graph. difflib gives no useful match for them (and for GetNode returns
    actively misleading ones), so they need their own message."""
    for cls in ("Note", "MarkdownNote", "GetNode", "Reroute"):
        env = _add(tmp_path, capsys, cls)
        assert env["ok"] is False, cls
        err = env["error"]
        assert err["code"] == "node_not_found", f"{cls}: {err}"
        blob = json.dumps(err).lower()
        assert "ui-only" in blob or "ui only" in blob, f"{cls} must be named as UI-only: {err}"
        assert (err.get("details") or {}).get("ui_only") is True, err


def test_uuid_class_type_is_named_as_a_subgraph_instance(patched_graph, tmp_path, capsys):
    """A subgraph INSTANCE's `type` is its definition UUID, and ls-nodes passes
    it through verbatim — so the agent sees a UUID that looks like a class name.
    There is no instantiate command, so this can never succeed; say so."""
    env = _add(tmp_path, capsys, "2454ad83-157c-40dd-9f19-5daaf4041ce0")
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "node_not_found"
    blob = json.dumps(err).lower()
    assert "subgraph" in blob, f"must explain the UUID is a subgraph instance id: {err}"


def test_known_class_still_adds(patched_graph, tmp_path, capsys):
    env = _add(tmp_path, capsys, "VAEDecode")
    assert env["ok"] is True, env
