"""An output slot may be addressed with a case/separator variant of its name.

Prod comfy-agent traces (2026-07-23 → 07-28) show the agent addressing outputs
by their TYPE because no discovery surface showed it the NAME (fixed separately
in cloud#5828). Where the name and type differ only in case or separators, the
intent is unambiguous and refusing it is pure friction:

  output 'IMAGE'          on a node whose outputs are ['image','alpha']
  output 'MODEL_TASK_ID'  on Tripo nodes whose output is named 'model task_id'

Accepted ONLY when exactly one output matches after normalizing. Measured across
the 3573-node catalog: just 2 node types gain a normalized ambiguity
(Flux2KleinOutputExtractor_EditUtils has both 'mask' and 'MASK'), and with the
exactly-one guard both keep failing exactly as before — zero regression.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph


def _node(outputs):
    return {"id": 1, "type": "X", "outputs": [{"name": n, "type": t} for n, t in outputs]}


@pytest.fixture
def g():
    return Graph.from_object_info({})


@pytest.mark.parametrize(
    "asked,outputs,want_idx",
    [
        ("IMAGE", [("image", "IMAGE"), ("alpha", "MASK")], 0),  # prod: BeebleSwitchXImageEdit
        ("MODEL_TASK_ID", [("model task_id", "MODEL_TASK_ID")], 0),  # prod: Tripo* (space -> _)
        ("Florence2_Model", [("florence2_model", "FL2MODEL")], 0),  # case only
        ("alpha", [("image", "IMAGE"), ("alpha", "MASK")], 1),  # exact still wins
    ],
)
def test_normalized_output_name_resolves(g, asked, outputs, want_idx):
    idx, _ = W._resolve_output_slot(_node(outputs), g, asked)
    assert idx == want_idx


def test_exact_match_wins_over_a_normalized_rival(g):
    """A node with both 'mask' and 'MASK' must resolve 'mask' exactly, not guess."""
    idx, _ = W._resolve_output_slot(_node([("mask", "MASK"), ("MASK", "MASK")]), g, "mask")
    assert idx == 0


def test_ambiguous_normalized_match_still_fails(g):
    """The real catalog collision (mask + MASK) must keep failing when the ask
    matches neither exactly — never silently pick one."""
    with pytest.raises(ValueError, match="not found"):
        W._resolve_output_slot(_node([("mask", "MASK"), ("MASK", "MASK")]), g, "Mask")


def test_unrelated_name_still_fails_with_the_name_list(g):
    # Two outputs, so the ask stays genuinely ambiguous (a single-output node
    # auto-resolves any name — see test_single_output_slot_alias.py — so this
    # regression guard needs a node where "unrelated" really is unrelated).
    with pytest.raises(ValueError) as ei:
        W._resolve_output_slot(_node([("image", "IMAGE"), ("alpha", "MASK")]), g, "LATENT")
    assert "image" in str(ei.value), "the error must still list the real names"
