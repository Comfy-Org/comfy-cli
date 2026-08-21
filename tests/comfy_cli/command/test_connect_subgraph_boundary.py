"""`workflow connect` must explain a subgraph boundary instead of "not found".

Measured on prod comfy-agent traces (2026-08-05, session 0cc9d03b): the agent
read `129/93.text` out of `comfy workflow slots` (which advertises interior
subgraph addresses precisely so widgets can be slot-edited), then tried to wire
a link into it:

  connect 1263680240999073.STRING -> 129/93.text
  => "node 129/93 not found in workflow. Nodes in this workflow: ...
      Use an id from `comfy workflow slots` / `ls-nodes` — never rebuild it."

The hint told it to consult the exact tool that advertised the address, so it
retried the SAME call seven times over eleven minutes and the turn died. The
truth: a link cannot cross a subgraph boundary — interior addresses are
writable (set-widget) but not wirable, and `connect` must say that instead of
"not found" + an inventory that fuels the retry loop.
"""

from __future__ import annotations

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _graph,
    _run,
    _subgraph_workflow,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow_edit

LOOP_FUEL = (
    "not found in workflow",
    "Nodes in this workflow",
)


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _connect(tmp_path, capsys, source: str, target: str) -> dict:
    path = _write(tmp_path, _subgraph_workflow())
    return _run(["connect", str(path), source, target], capsys)


def _assert_boundary_error(env: dict) -> dict:
    assert env["ok"] is False, env
    err = env["error"]
    assert err["code"] == "workflow_edit_invalid", err
    for fuel in LOOP_FUEL:
        assert fuel not in err["message"], f"retry-loop fuel {fuel!r} in: {err['message']}"
    return err


class TestConnectInteriorAddress:
    def test_slash_interior_target_names_the_boundary(self, patched_graph, tmp_path, capsys):
        """`57/27.text` — the exact shape of the prod failure (129/93.text)."""
        env = _connect(tmp_path, capsys, "9.LATENT", "57/27.text")
        err = _assert_boundary_error(env)
        msg = err["message"]
        assert "inside subgraph 57" in msg, msg
        # The actionable alternative: the address IS writable, just not wirable.
        assert "set-widget" in msg, msg
        assert "57/27" in msg, msg

    def test_colon_interior_target_is_the_same_boundary(self, patched_graph, tmp_path, capsys):
        """`57:27.text` — the flattened namespace validate/node_errors report."""
        env = _connect(tmp_path, capsys, "9.LATENT", "57:27.text")
        err = _assert_boundary_error(env)
        assert "inside subgraph 57" in err["message"], err

    def test_interior_source_is_also_guarded(self, patched_graph, tmp_path, capsys):
        """Wiring FROM an interior node out is the same boundary violation."""
        env = _connect(tmp_path, capsys, "57/3.LATENT", "9.width")
        err = _assert_boundary_error(env)
        assert "inside subgraph 57" in err["message"], err

    def test_unknown_interior_lists_what_the_subgraph_contains(self, patched_graph, tmp_path, capsys):
        env = _connect(tmp_path, capsys, "9.LATENT", "57/99.text")
        err = _assert_boundary_error(env)
        msg = err["message"]
        assert "no node 99 inside subgraph 57" in msg, msg
        # Inventory of the DEFINITION, not the top-level graph.
        assert "27 (CLIPTextEncode)" in msg, msg
        assert "3 (KSampler)" in msg, msg

    def test_path_under_a_non_subgraph_node_says_so(self, patched_graph, tmp_path, capsys):
        """`9/27.text` — node 9 exists but is a plain node, not a subgraph."""
        env = _connect(tmp_path, capsys, "9.LATENT", "9/27.text")
        err = _assert_boundary_error(env)
        assert "not a subgraph" in err["message"], err

    def test_unknown_head_still_gets_the_standard_not_found(self, patched_graph, tmp_path, capsys):
        """`999/27.text` — nothing to explain; the classic enriched error stays."""
        env = _connect(tmp_path, capsys, "9.LATENT", "999/27.text")
        assert env["ok"] is False
        assert "not found in workflow" in env["error"]["message"], env


class TestConnectPromotedWidget:
    def test_flat_promoted_widget_target_points_at_set_widget(self, patched_graph, tmp_path, capsys):
        """`57.text` is a promoted WIDGET (proxyWidgets), not a link input.

        The old error — "input 'text' not found on node 57; inputs: []" plus the
        top-level node inventory — reads as "wrong id, try another", when the
        truth is "right id, wrong verb"."""
        env = _connect(tmp_path, capsys, "9.LATENT", "57.text")
        err = _assert_boundary_error(env)
        msg = err["message"]
        assert "promoted widget" in msg, msg
        assert "set-widget" in msg, msg
        assert "57.text" in msg, msg

    def test_unpromoted_missing_input_keeps_the_standard_error(self, patched_graph, tmp_path, capsys):
        """A name that is neither a link input nor a promoted widget stays a
        plain input-not-found so genuinely wrong names are still called wrong."""
        env = _connect(tmp_path, capsys, "9.LATENT", "57.nonsense")
        assert env["ok"] is False
        assert "promoted widget" not in env["error"]["message"], env
