"""A single-output node should accept ANY requested output name.

Agents address an output by its TYPE when the node has exactly ONE output
(they were never shown the real name). Real prod cases (5 failures):

    LUMA_RAY32_KEYFRAME -> actual name 'keyframes'
    CAMERA_CONTROL      -> actual name 'camera_control'    (x2)
    ELEVENLABS_VOICE    -> actual name 'voice'
    IMAGE               -> actual name 'images'

The case/separator tolerance already on this branch
(test_output_slot_normalized.py) only accepts a variant of the SAME name
(case/underscore-insensitive) — it does not cover an outright rename, so all
five kept failing with "output 'X' not found ... outputs: [...]".

Fix in `_resolve_output_slot`: when the node has exactly one output, an
unmatched name resolves to that one output (there is no ambiguity — it's the
only thing the caller could mean). Multi-output nodes are unaffected and keep
today's behavior: an unmatched name still errors with the full name list.
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
    "asked,name,out_type",
    [
        ("LUMA_RAY32_KEYFRAME", "keyframes", "LUMA_RAY32_KEYFRAME"),  # prod: LUMA_RAY32_KEYFRAME
        ("CAMERA_CONTROL", "camera_control", "CAMERA_CONTROL"),  # prod: CAMERA_CONTROL (x2)
        ("ELEVENLABS_VOICE", "voice", "ELEVENLABS_VOICE"),  # prod: ELEVENLABS_VOICE
        ("IMAGE", "images", "IMAGE"),  # prod: IMAGE
        ("anything_at_all", "output", "SOMETYPE"),  # any alias at all — not just a type name
    ],
)
def test_single_output_node_accepts_any_alias(g, asked, name, out_type):
    idx, resolved_type = W._resolve_output_slot(_node([(name, out_type)]), g, asked)
    assert idx == 0
    assert resolved_type == out_type


def test_single_output_node_exact_name_still_wins(g):
    idx, _ = W._resolve_output_slot(_node([("keyframes", "LUMA_RAY32_KEYFRAME")]), g, "keyframes")
    assert idx == 0


def test_multi_output_node_still_errors_with_the_name_list(g):
    """Regression guard: a node with MORE THAN ONE output must NOT gain the
    single-output auto-resolve — an unmatched name stays ambiguous and must
    keep failing with the real name list, exactly as before."""
    outs = [("image", "IMAGE"), ("mask", "MASK")]
    with pytest.raises(ValueError) as ei:
        W._resolve_output_slot(_node(outs), g, "LATENT")
    msg = str(ei.value)
    assert "not found" in msg
    assert "image" in msg and "mask" in msg


def test_multi_output_node_exact_and_normalized_matches_unaffected(g):
    """Existing behavior for multi-output nodes (exact + case/separator
    normalization) must be untouched by this fix."""
    outs = [("image", "IMAGE"), ("alpha", "MASK")]
    idx, _ = W._resolve_output_slot(_node(outs), g, "IMAGE")
    assert idx == 0
