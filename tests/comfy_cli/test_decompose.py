"""Tests for projecting an API-format workflow back into fragment source.

`decompose_workflow` is the inverse of `compose_blueprint`: it takes a flat
API-format workflow and returns a fragment JSON dict whose loaders are typed
inputs (bound to the *consumer*, the loader node stripped), scalar widgets are
named params, and a terminal save's producer is an output (the save stripped) —
every port derived from the graph, nothing hardcoded. The result round-trips
through `parse_fragment`, and composing it reproduces the original wiring while
letting a blueprint override any widget by name instead of editing the compiled
graph by hand.
"""

from __future__ import annotations

import json

from comfy_cli.fragments import compose_blueprint, decompose_workflow, parse_fragment

# A minimal but realistic restyle graph: load an image, edit it, save it.
_RESTYLE_WF = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "base.png"}, "_meta": {"title": "Base"}},
    "2": {
        "class_type": "FluxKontextProImageNode",
        "inputs": {"prompt": "as an oil painting", "input_image": ["1", 0], "seed": 7},
        "_meta": {"title": "Kontext"},
    },
    "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "out"}},
}


def test_decompose_embeds_provenance_and_edit_guidance():
    frag_json = decompose_workflow(_RESTYLE_WF, name="restyle", source="workflows/10_restyle_oil.json")
    meta = frag_json["_fragment"]

    # Self-documenting: the fragment records where it came from and how to edit it.
    assert meta["source"] == "workflows/10_restyle_oil.json"
    assert "10_restyle_oil.json" in meta["description"]
    assert "compose" in meta["description"].lower(), "description should point the reader at the compile loop"
    # Still a valid, round-trippable fragment.
    parse_fragment(frag_json)


def test_scalar_widget_becomes_named_param_with_default():
    frag = parse_fragment(decompose_workflow(_RESTYLE_WF, name="restyle"))

    prompt_params = [p for p in frag.params.values() if p.binds == "2.prompt"]
    assert len(prompt_params) == 1, f"expected the prompt widget surfaced once, got {frag.params}"
    assert prompt_params[0].type == "STRING"
    assert prompt_params[0].has_default
    assert prompt_params[0].default == "as an oil painting"


def test_loader_becomes_input_bound_to_consumer_and_is_stripped():
    frag = parse_fragment(decompose_workflow(_RESTYLE_WF, name="restyle"))

    # The loader is the input boundary — it must not survive as an interior node,
    # else compose would inject a *second* loader and double-load.
    assert "1" not in frag.nodes
    assert len(frag.inputs) == 1, f"expected one IMAGE input, got {frag.inputs}"
    inp = next(iter(frag.inputs.values()))
    assert inp.type == "IMAGE"
    assert inp.binds == "2.input_image", "input binds to the CONSUMER, not the stripped loader"


def test_save_becomes_output_bound_to_producer_and_is_stripped():
    frag = parse_fragment(decompose_workflow(_RESTYLE_WF, name="restyle"))

    # The save is the output boundary — stripped so the fragment is composable as
    # a middle step (compose re-adds a save only for the final stage).
    assert "3" not in frag.nodes
    assert frag.terminal is False
    assert len(frag.outputs) == 1, f"expected one output, got {frag.outputs}"
    out = next(iter(frag.outputs.values()))
    assert out.type == "IMAGE"
    assert out.from_node == "2"
    assert out.port == 0


def test_non_numeric_ids_are_renumbered_into_a_valid_composable_fragment():
    # Subgraph flattening yields composite ids like "139:99"; fragments require
    # numeric ids (the composer does int(old)+offset), so projection must
    # renumber and rewire every reference.
    wf = {
        "139:138": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "139:2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "hi", "clip": ["139:9", 0]},  # internal edge to another composite id
        },
        "139:9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        "139:68": {"class_type": "SaveImage", "inputs": {"images": ["139:2", 0], "filename_prefix": "o"}},
    }
    frag = parse_fragment(decompose_workflow(wf, name="t"))  # must not raise

    assert all(k.isdigit() for k in frag.nodes), f"interior ids must be numeric, got {list(frag.nodes)}"
    # The surviving internal edge (CLIPTextEncode.clip -> checkpoint) must point
    # at a real renumbered interior node, not a stale composite id.
    encode = next(n for n in frag.nodes.values() if n["class_type"] == "CLIPTextEncode")
    clip_ref = encode["inputs"]["clip"]
    assert isinstance(clip_ref, list) and clip_ref[0] in frag.nodes
    assert frag.nodes[clip_ref[0]]["class_type"] == "CheckpointLoaderSimple"
    # Boundaries still surfaced as ports.
    assert len(frag.outputs) == 1 and next(iter(frag.outputs.values())).type == "IMAGE"


def test_decompose_then_compose_overrides_param_by_name(tmp_path):
    """The whole point: project -> override a widget BY NAME -> compose. No jq."""
    frag_json = decompose_workflow(_RESTYLE_WF, name="restyle")
    frag = parse_fragment(frag_json)
    prompt_param = next(n for n, p in frag.params.items() if p.binds == "2.prompt")
    img_input = next(n for n, p in frag.inputs.items() if p.binds == "2.input_image")

    lib = tmp_path / "fragments"
    lib.mkdir()
    (lib / "restyle.json").write_text(json.dumps(frag_json))

    blueprint = {
        "output_prefix": "outputs/recompose",
        "pipeline": [
            {
                "fragment": "restyle",
                "alias": "r",
                "inputs": {img_input: "inputs/base.png"},
                "params": {prompt_param: "as a watercolor"},
            }
        ],
    }
    workflow, _summary = compose_blueprint(blueprint, lib_dir=lib)

    kontext = [n for n in workflow.values() if n["class_type"] == "FluxKontextProImageNode"]
    assert len(kontext) == 1
    assert kontext[0]["inputs"]["prompt"] == "as a watercolor", "the by-name override reached the graph"
    assert kontext[0]["inputs"]["seed"] == 7, "an untouched param kept its projected default"
    # the input image is wired through a freshly injected loader (not the stale id)
    assert isinstance(kontext[0]["inputs"]["input_image"], list)
    loader_id = kontext[0]["inputs"]["input_image"][0]
    assert workflow[loader_id]["class_type"] == "LoadImage"
    assert workflow[loader_id]["inputs"]["image"] == "inputs/base.png"
