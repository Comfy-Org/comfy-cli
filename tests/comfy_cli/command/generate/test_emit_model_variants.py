"""`--emit-workflow` flags that the agent's generate_workflow tripped on.

Measured on stg-v2/nightly comfy-agent traces (2026-09-20..21):

* ``nano-banana --model gemini-3-pro-image-preview`` — the adapter offers that
  model, but emit always built ``GeminiImageNode``, whose ``model`` combo only
  knows the 2.5 flash models, so the emitted graph failed ``validate``
  ("'gemini-3-pro-image-preview' not in known options"). The node that runs it
  is ``GeminiImage2Node`` (Nano Banana Pro).
* ``flux-2 --output_format png`` — refused as an unmapped flag, although the
  emitted graph's ``SaveImage`` writes PNG anyway.
"""

import json
from pathlib import Path

import pytest

from comfy_cli.command.generate import emit
from comfy_cli.cql.engine import Graph

PARTNER_OBJECT_INFO = json.loads(
    (Path(__file__).parent / "fixtures" / "partner_nodes_object_info.json").read_text(encoding="utf-8")
)


def test_nano_banana_pro_model_emits_the_node_that_offers_it():
    wf = emit.build_workflow("nano-banana", {"prompt": "a fox", "model": "gemini-3-pro-image-preview"})
    partner = wf["1"]
    assert partner["class_type"] == "GeminiImage2Node"
    assert partner["inputs"]["model"] == "gemini-3-pro-image-preview"

    graph = Graph.from_object_info(PARTNER_OBJECT_INFO)
    result = graph.validate_workflow({"1": partner})
    enum_errors = [e for e in result["errors"] if e.get("code") == "unknown_enum_value"]
    assert not enum_errors, enum_errors


def test_nano_banana_default_model_still_emits_gemini_image_node():
    wf = emit.build_workflow("nano-banana", {"prompt": "a fox"})
    assert wf["1"]["class_type"] == "GeminiImageNode"


def test_flux_2_accepts_png_output_format_it_already_produces():
    wf = emit.build_workflow("flux-2", {"prompt": "a fox", "output_format": "png"})
    assert wf["1"]["class_type"] == "Flux2ImageNode"


def test_flux_2_still_refuses_an_output_format_it_cannot_produce():
    with pytest.raises(emit.EmitError, match="output_format"):
        emit.build_workflow("flux-2", {"prompt": "a fox", "output_format": "jpeg"})


def test_flux_2_nightly_input_with_png_output_format_emits():
    """Pin for an agent failure seen in telemetry: these exact args were
    refused with "does not map --output_format onto Flux2ImageNode" on a build
    that predated #921."""
    wf = emit.build_workflow(
        "bfl/flux-2-pro/generate",
        {"prompt": "a rubber duck", "width": 1024, "height": 1024, "output_format": "png"},
    )
    assert wf["1"]["class_type"] == "Flux2ImageNode"
    assert wf["1"]["inputs"]["model.width"] == 1024


# ─── models with no node: point at the nearest one that has ───────────────
#
# Nightly 2026-09-22/23: generate_workflow asked for `flux-pro` (x2) and
# `ideogram`. Neither has a partner node in the recorded catalog (the only
# flux-pro-1.1 node is the Ultra variant; there is no Ideogram node), so they
# stay unsupported. What the agent needs is the ONE alias to retry with: an
# emittable model of the same output kind, the same partner's first.


def test_unsupported_flux_pro_suggests_image_models_same_partner_first():
    with pytest.raises(emit.UnsupportedModelError) as ei:
        emit.build_workflow("bfl/flux-pro-1.1/generate", {"prompt": "x"})
    assert ei.value.model == "bfl/flux-pro-1.1/generate"
    assert ei.value.suggested == ["flux-2", "flux-ultra", "nano-banana"]
    assert "flux-2" in str(ei.value).split("Supported models")[0]


def test_unsupported_ideogram_suggests_only_image_models():
    with pytest.raises(emit.UnsupportedModelError) as ei:
        emit.build_workflow("ideogram", {"prompt": "x"})
    assert set(ei.value.suggested) == {"flux-2", "flux-ultra", "nano-banana"}


def test_unsupported_video_model_suggests_video_models():
    with pytest.raises(emit.UnsupportedModelError) as ei:
        emit.build_workflow("kling/v1/videos/text2video", {"prompt": "x"})
    assert ei.value.suggested == ["kling-i2v", "seedance"]


def test_unknown_model_suggests_nothing():
    with pytest.raises(emit.UnsupportedModelError) as ei:
        emit.build_workflow("no-such-model", {"prompt": "x"})
    assert ei.value.suggested == []
