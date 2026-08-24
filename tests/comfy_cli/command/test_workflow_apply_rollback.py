"""A FAILED atomic `workflow apply` batch must not advertise node ids that the
rollback discards.

Regression guard for a bug measured in prod comfy-agent Langfuse traces
(2026-07-27). Sequence in trace ff11b86931f3fbaa:

  03:34:58 apply_ops  ok=false  "batch failed: input 'image' not found on node
                                868300940744052; inputs: ['images','files'].
                                Nodes in this workflow: 3686911078754972
                                (LoadImage), 868300940744052 (GeminiNanoBanana2),
                                4045462123562940 (KlingStartEndFrameNode), …"
  03:35:13 connect    ok=false  "node 3686911078754972 not found in workflow"
  … six more connects, all "not found", every one using an id from that hint

Across the 78-trace prod sample, 16/16 (100%) of "node <id> not found in
workflow" edit failures used an id that an earlier FAILED batch had advertised.

Cause: `apply_specs` threads an accumulating `workflow` dict through the batch,
so when spec #N fails that dict already holds the nodes added by specs #0..N-1,
and `_enrich_resolution_error` renders its inventory from it. The batch is
atomic, so those ids never reach disk — but the hint is phrased as an
instruction ("Use an id from ... never rebuild it"), so a model treats them as
real and addresses them next.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _run,
    _write,
    patched_graph,  # noqa: F401  (pytest fixture)
    reset_singleton,  # noqa: F401  (autouse fixture)
)


def _failing_batch(tmp_path) -> Path:
    """add VAEDecode + BatchImagesNode, then connect into 'image' — which does
    not exist on BatchImagesNode (its autogrow input is 'images'). Same shape as
    the prod failure: two adds succeed in memory, the third spec fails."""
    ops = tmp_path / "ops.json"
    ops.write_text(
        json.dumps(
            [
                {"op": "add_node", "class_type": "VAEDecode", "as": "dec"},
                {"op": "add_node", "class_type": "BatchImagesNode", "as": "batch"},
                {"op": "connect", "from": "dec.IMAGE", "to": "batch.image"},
            ]
        )
    )
    return ops


def test_failed_batch_does_not_advertise_discarded_node_ids(patched_graph, tmp_path, capsys):  # noqa: F811
    path = _write(tmp_path, _base_workflow())
    ids_before = {n["id"] for n in json.loads(Path(path).read_text())["nodes"]}

    env = _run(["apply", str(path), "--ops", str(_failing_batch(tmp_path))], capsys)
    assert env["ok"] is False, env
    msg = env["error"]["message"]
    assert "batch failed" in msg, msg

    # The batch is atomic — the file is untouched.
    after = {n["id"] for n in json.loads(Path(path).read_text())["nodes"]}
    assert after == ids_before, "batch must be atomic"

    # Every id the message names as present must actually exist.
    advertised = {int(m) for m in re.findall(r"(\d+) \(", msg)}
    assert not (advertised - ids_before), (
        f"failed batch advertised discarded node ids: {sorted(advertised - ids_before)}\n{msg}"
    )


def test_failed_batch_states_nothing_was_applied(patched_graph, tmp_path, capsys):  # noqa: F811
    """The caller must be told the graph is unchanged, and be pointed at the
    real inventory — otherwise it re-addresses ids from the failed batch."""
    path = _write(tmp_path, _base_workflow())
    env = _run(["apply", str(path), "--ops", str(_failing_batch(tmp_path))], capsys)
    msg = env["error"]["message"]

    assert "No changes were applied" in msg, msg
    assert "The workflow still contains:" in msg, msg
    # The pre-batch nodes ARE named, so the hint stays actionable.
    assert "3 (KSampler)" in msg and "7 (EmptyLatentImage)" in msg, msg
    # And the stale mid-batch inventory is gone.
    assert "Nodes in this workflow:" not in msg, msg
    # Punctuation survives the clause strip.
    assert "] No changes" not in msg, f"missing separator before the suffix: {msg}"


def test_successful_batch_is_unaffected(patched_graph, tmp_path, capsys):  # noqa: F811
    """The re-hint must only fire on failure — a good batch still applies."""
    path = _write(tmp_path, _base_workflow())
    ops = tmp_path / "ok.json"
    ops.write_text(
        json.dumps(
            [
                {"op": "add_node", "class_type": "VAEDecode", "as": "dec"},
                {"op": "add_node", "class_type": "BatchImagesNode", "as": "batch"},
                {"op": "connect", "from": "dec.IMAGE", "to": "batch.images"},
            ]
        )
    )
    env = _run(["apply", str(path), "--ops", str(ops)], capsys)
    assert env["ok"] is True, env
    after = json.loads(Path(path).read_text())
    types = {n["type"] for n in after["nodes"]}
    assert {"VAEDecode", "BatchImagesNode"} <= types, types
