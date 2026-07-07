"""Emit a runnable API-format workflow that calls a partner *node* instead of
the proxy endpoint.

``comfy generate <model> …`` calls the partner through Comfy's HTTP proxy. That
is convenient but leaves no reusable artifact: no workflow JSON you can re-run,
edit, or drop into a fragment pipeline. The same partner models also exist as
ComfyUI **API NODES**. ``--emit-workflow <path>`` takes the exact same
``--param`` values the proxy path would consume and writes an API-format
workflow that drives the partner node, plus a ``SaveImage``/``SaveVideo`` so the
result lands on disk when run with ``comfy run``.

The proxy-model → node-class mapping is intentionally small and explicit: it
covers the common partner models and fails loudly (listing what *is* supported)
for everything else, rather than guessing a node class that may not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_cli.command.generate import spec


class EmitError(RuntimeError):
    """``--emit-workflow`` cannot map this model to a partner node."""


@dataclass(frozen=True)
class NodeSpec:
    """How to render one partner model as a ComfyUI node.

    ``param_map`` maps a ``comfy generate`` flag name → the partner node's input
    key. ``image_params`` lists flag names whose value is a local image path
    that must be materialized with a ``LoadImage`` node and wired into the
    partner node's matching input. ``fixed`` are node inputs always set to a
    constant (defaults the node requires but that ``generate`` doesn't surface).
    ``output`` selects the save node (IMAGE → SaveImage, VIDEO → SaveVideo) and
    the partner node's output port that carries the media.
    """

    node_class: str
    param_map: dict[str, str]
    output: str  # "IMAGE" | "VIDEO"
    fixed: dict[str, Any] = field(default_factory=dict)
    image_params: dict[str, str] = field(default_factory=dict)  # flag -> node input key
    media_port: int = 0


# proxy model alias → partner node spec. Param keys are the *generate* flag
# names (the openapi property names a user already types today); values are the
# real node input keys, taken from `comfy nodes show <ClassName>`.
MODEL_NODE_MAP: dict[str, NodeSpec] = {
    # Google Gemini Flash Image (nano-banana). Node: GeminiImageNode.
    "nano-banana": NodeSpec(
        node_class="GeminiImageNode",
        param_map={
            "prompt": "prompt",
            "model": "model",
            "seed": "seed",
            "aspect_ratio": "aspect_ratio",
        },
        image_params={"image": "images", "images": "images"},
        fixed={"model": "gemini-2.5-flash-image", "seed": 42},
        output="IMAGE",
    ),
    # ByteDance Seedance image-to-video. Node: ByteDanceImageToVideoNode.
    "seedance": NodeSpec(
        node_class="ByteDanceImageToVideoNode",
        param_map={
            "prompt": "prompt",
            "model": "model",
            "resolution": "resolution",
            "aspect_ratio": "aspect_ratio",
            "duration": "duration",
            "seed": "seed",
        },
        image_params={"image": "image"},
        fixed={
            "model": "seedance-1-0-pro-fast-251015",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "duration": 5,
        },
        output="VIDEO",
    ),
    # BFL Flux (text-to-image). Node: Flux2ProImageNode.
    "flux-pro": NodeSpec(
        node_class="Flux2ProImageNode",
        param_map={
            "prompt": "prompt",
            "width": "width",
            "height": "height",
            "seed": "seed",
            "prompt_upsampling": "prompt_upsampling",
        },
        fixed={"width": 1024, "height": 768, "seed": 0, "prompt_upsampling": True},
        output="IMAGE",
    ),
    "flux-2": NodeSpec(
        node_class="Flux2ProImageNode",
        param_map={
            "prompt": "prompt",
            "width": "width",
            "height": "height",
            "seed": "seed",
            "prompt_upsampling": "prompt_upsampling",
        },
        fixed={"width": 1024, "height": 768, "seed": 0, "prompt_upsampling": True},
        output="IMAGE",
    ),
    # Kling image-to-video. Node: KlingImage2VideoNode.
    "kling-i2v": NodeSpec(
        node_class="KlingImage2VideoNode",
        param_map={
            "prompt": "prompt",
            "negative_prompt": "negative_prompt",
            "model_name": "model_name",
            "cfg_scale": "cfg_scale",
            "mode": "mode",
            "aspect_ratio": "aspect_ratio",
            "duration": "duration",
        },
        image_params={"image": "start_frame", "start_frame": "start_frame"},
        fixed={
            "negative_prompt": "",
            "model_name": "kling-v2-master",
            "cfg_scale": 0.8,
            "mode": "std",
            "aspect_ratio": "16:9",
            "duration": "5",
        },
        output="VIDEO",
    ),
}


def supported_models() -> list[str]:
    """Aliases that ``--emit-workflow`` knows how to render as a node."""
    return sorted(MODEL_NODE_MAP)


def _resolve_model(model: str) -> tuple[str, NodeSpec]:
    """Map a user-typed model to a (alias, NodeSpec). Accepts an alias or the
    canonical endpoint id that an alias resolves to."""
    if model in MODEL_NODE_MAP:
        return model, MODEL_NODE_MAP[model]
    canonical = spec.resolve_alias(model)
    pref = spec.preferred_alias(canonical)
    if pref and pref in MODEL_NODE_MAP:
        return pref, MODEL_NODE_MAP[pref]
    raise EmitError(
        f"--emit-workflow does not support model {model!r}.\n"
        f"Supported models: {', '.join(supported_models())}.\n"
        "These map to ComfyUI API nodes; other proxy models have no node mapping yet."
    )


def build_workflow(model: str, values: dict[str, Any], *, output_prefix: str = "generate") -> dict[str, Any]:
    """Build an API-format workflow that drives the partner node for ``model``.

    ``values`` are the parsed ``--param`` values (same dict the proxy client
    receives). Local-file image params are materialized as ``LoadImage`` nodes
    and wired into the partner node; scalar params override the node's fixed
    defaults. A ``SaveImage``/``SaveVideo`` is appended so ``comfy run`` writes
    the result to disk.
    """
    _alias, ns = _resolve_model(model)

    node_inputs: dict[str, Any] = dict(ns.fixed)

    next_id = 2  # the partner node is "1"; loaders/save get 2, 3, …

    # Image-path params → LoadImage nodes wired into the partner node.
    workflow: dict[str, Any] = {}
    for flag, node_key in ns.image_params.items():
        raw = values.get(flag)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            raise EmitError(
                f"--{flag} received multiple files, but emit-workflow currently "
                "maps this input to a single LoadImage node."
            )
        path = str(Path(raw).expanduser())
        loader_id = str(next_id)
        next_id += 1
        workflow[loader_id] = {
            "class_type": "LoadImage",
            "_meta": {"title": f"load {Path(path).name}"},
            "inputs": {"image": path},
        }
        node_inputs[node_key] = [loader_id, 0]

    # Scalar params → node inputs, honoring the explicit param_map.
    for flag, node_key in ns.param_map.items():
        if flag in values and values[flag] is not None:
            node_inputs[node_key] = values[flag]

    partner = {
        "class_type": ns.node_class,
        "_meta": {"title": f"{ns.node_class} ({model})"},
        "inputs": node_inputs,
    }
    workflow["1"] = partner

    save_id = str(next_id)
    if ns.output == "VIDEO":
        workflow[save_id] = {
            "class_type": "SaveVideo",
            "_meta": {"title": "save generated video"},
            "inputs": {
                "video": ["1", ns.media_port],
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        }
    else:
        workflow[save_id] = {
            "class_type": "SaveImage",
            "_meta": {"title": "save generated image"},
            "inputs": {"images": ["1", ns.media_port], "filename_prefix": output_prefix},
        }
    return workflow


def write_workflow(
    model: str, values: dict[str, Any], path: Path, *, output_prefix: str = "generate"
) -> dict[str, Any]:
    """Build the workflow for ``model`` and write it to ``path`` as JSON.
    Returns the workflow dict. Raises ``EmitError`` on an unsupported model."""
    workflow = build_workflow(model, values, output_prefix=output_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    return workflow
