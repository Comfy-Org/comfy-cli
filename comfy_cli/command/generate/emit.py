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
from importlib import resources
from pathlib import Path
from typing import Any

from comfy_cli import workflow_ops
from comfy_cli.command.generate import spec
from comfy_cli.cql.engine import Graph


class EmitError(RuntimeError):
    """``--emit-workflow`` cannot build a workflow for this request."""


class UnsupportedModelError(EmitError):
    """``--emit-workflow`` has no partner-node mapping for ``model``.

    Distinct from the other :class:`EmitError` causes (bad parameters) because
    the remedy is different — pick a model from ``supported`` — and an agent
    needs to branch on that without parsing prose. ``generate list`` exposes
    the same answer up front as ``emit_supported`` per row.
    """

    def __init__(self, model: str, supported: list[str]):
        self.model = model
        self.supported = supported
        super().__init__(
            f"--emit-workflow does not support model {model!r}.\n"
            f"Supported models: {', '.join(supported)}.\n"
            "These map to ComfyUI API nodes; other proxy models have no node mapping yet."
        )


@dataclass(frozen=True)
class NodeSpec:
    """How to render one partner model as a ComfyUI node.

    ``endpoint`` is the canonical ``/proxy/`` endpoint id the node's ``execute()``
    posts to (read off the ComfyUI source). It must equal the endpoint the alias
    itself proxies to — otherwise ``--emit-workflow`` would silently swap the
    model out from under the user; ``test_emit.py`` asserts that invariant.

    ``param_map`` maps a ``comfy generate`` flag name → the partner node's input
    key. ``image_params`` lists flag names whose value is a local image path
    that must be materialized with a ``LoadImage`` node and wired into the
    partner node's matching input. ``fixed`` are node inputs always set to a
    constant (defaults the node requires but that ``generate`` doesn't surface).
    ``output`` selects the save node (IMAGE → SaveImage, VIDEO → SaveVideo) and
    the partner node's output port that carries the media.

    COMPLETENESS CONTRACT for ``fixed``: it must supply a default for EVERY
    widget (non-link) input of the node — the *optional* schema section
    included — unless the value always arrives via ``param_map``/``image_params``.
    A schema-``optional`` input is not necessarily optional at execution time:
    V3 nodes may declare an input ``optional=True`` while their ``execute()``
    signature has no Python default (ByteDanceImageToVideoNode's
    ``seed``/``camera_fixed``/``watermark`` do exactly this), so a workflow
    that omits it validates cleanly but crashes at run time with
    "missing N required positional arguments". The ComfyUI frontend always
    serializes every widget value, which is why UI-exported workflows never
    hit this — the emitter must match that behavior.
    ``tests/comfy_cli/command/generate/test_emit.py`` enforces the contract
    against a recorded object_info snapshot
    (``fixtures/partner_nodes_object_info.json``).
    """

    node_class: str
    endpoint: str  # canonical /proxy/ endpoint id the node's execute() posts to
    param_map: dict[str, str]
    output: str  # "IMAGE" | "VIDEO"
    fixed: dict[str, Any] = field(default_factory=dict)
    image_params: dict[str, str] = field(default_factory=dict)  # flag -> node input key
    media_port: int = 0
    # Node input to set to "{width}:{height}" when the user passes both flags —
    # for nodes that take an aspect ratio where the proxy schema takes w/h.
    aspect_from_wh: str | None = None


# proxy model alias → partner node spec. Param keys are the *generate* flag
# names (the openapi property names a user already types today); values are the
# real node input keys, taken from `comfy nodes show <ClassName>`.
MODEL_NODE_MAP: dict[str, NodeSpec] = {
    # Google Gemini Flash Image (nano-banana). Node: GeminiImageNode.
    "nano-banana": NodeSpec(
        node_class="GeminiImageNode",
        endpoint="vertexai/gemini/{model}",
        param_map={
            "prompt": "prompt",
            "model": "model",
            "seed": "seed",
            "aspect_ratio": "aspect_ratio",
        },
        image_params={"image": "images", "images": "images"},
        fixed={
            "model": "gemini-2.5-flash-image",
            "seed": 42,
            "aspect_ratio": "auto",
            "response_modalities": "IMAGE+TEXT",
            # Schema default (snapshot 2026-07); the node's execute() falls back
            # to "" but the UI serializes this steering prompt, so match it.
            "system_prompt": (
                "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
                "Interpret all user input—regardless of format, intent, or abstraction—as literal "
                "visual directives for image composition.\n"
                "If a prompt is conversational or lacks specific visual details, you must creatively "
                "invent a concrete visual scenario that depicts the concept.\n"
                "Prioritize generating the visual representation above any text, formatting, or "
                "conversational requests."
            ),
        },
        output="IMAGE",
    ),
    # ByteDance Seedance image-to-video. Node: ByteDanceImageToVideoNode.
    "seedance": NodeSpec(
        node_class="ByteDanceImageToVideoNode",
        endpoint="byteplus/api/v3/contents/generations/tasks",
        param_map={
            "prompt": "prompt",
            "model": "model",
            "resolution": "resolution",
            # The proxy flag is --ratio; the node input is aspect_ratio.
            "ratio": "aspect_ratio",
            "aspect_ratio": "aspect_ratio",
            "duration": "duration",
            "seed": "seed",
            # Proxy flag --camerafixed; node input camera_fixed.
            "camerafixed": "camera_fixed",
            "watermark": "watermark",
            "generate_audio": "generate_audio",
        },
        image_params={"image": "image"},
        fixed={
            "model": "seedance-1-0-pro-fast-251015",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "duration": 5,
            # Schema-optional but positionally REQUIRED by execute() — omitting
            # any of these fails the run with "missing required positional
            # arguments" (observed live on cloud). See the NodeSpec docstring.
            "seed": 0,
            "camera_fixed": False,
            "watermark": False,
            "generate_audio": False,
        },
        output="VIDEO",
    ),
    # BFL Flux 2 [pro] (text-to-image). Node: Flux2ProImageNode.
    #
    # There is deliberately NO "flux-pro" entry: that alias means BFL Flux Pro
    # 1.1 (`bfl/flux-pro-1.1/generate`), and ComfyUI has no node for it — the
    # only `flux-pro-1.1` node is the Ultra variant below. Mapping it to
    # Flux2ProImageNode would emit a workflow for a *different* model, so
    # `flux-pro` falls through to the EmitError instead.
    "flux-2": NodeSpec(
        node_class="Flux2ProImageNode",
        endpoint="bfl/flux-2-pro/generate",
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
    # BFL Flux 1.1 [pro] Ultra (text-to-image). Node: FluxProUltraImageNode.
    # The node takes an `aspect_ratio` string where the proxy schema takes
    # width/height, so w/h are folded into it via `aspect_from_wh`.
    "flux-ultra": NodeSpec(
        node_class="FluxProUltraImageNode",
        endpoint="bfl/flux-pro-1.1-ultra/generate",
        param_map={
            "prompt": "prompt",
            "seed": "seed",
        },
        # `image_prompt_strength` is schema-optional but positionally required by
        # execute(); the emitted node must carry every widget input. Default from
        # the cloud catalog (FluxProUltraImageNode).
        fixed={
            "aspect_ratio": "16:9",
            "raw": False,
            "prompt_upsampling": False,
            "seed": 0,
            "image_prompt_strength": 0.1,
        },
        aspect_from_wh="aspect_ratio",
        output="IMAGE",
    ),
    # Kling image-to-video. Node: KlingImage2VideoNode.
    "kling-i2v": NodeSpec(
        node_class="KlingImage2VideoNode",
        endpoint="kling/v1/videos/image2video",
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


# Bundled, offline object_info snapshot covering exactly the classes this
# module ever mints: every ``node_class`` in MODEL_NODE_MAP, plus the fixed
# support cast (LoadImage/ImageBatch/SaveImage/SaveVideo). ``--emit-workflow``
# has no server/API-key dependency (see module docstring), so this cannot be a
# live ``object_info`` fetch — it is the same recorded-snapshot approach
# ``tests/comfy_cli/command/generate/test_emit.py`` already used for the
# completeness-contract test, promoted from test-only to the module that needs
# it so the two never drift. Refresh both together from `comfy nodes show
# <ClassName> --json` against a current cloud catalog when a mapped node's
# schema changes.
_SUPPORT_NODE_CLASSES = ("LoadImage", "ImageBatch", "SaveImage", "SaveVideo")


def _load_catalog() -> Graph:
    data = resources.files("comfy_cli.command.generate.data").joinpath("emit_object_info.json").read_bytes()
    return Graph.from_object_info(json.loads(data))


_CATALOG: Graph | None = None


def _catalog() -> Graph:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _load_catalog()
    return _CATALOG


def supported_models() -> list[str]:
    """Aliases that ``--emit-workflow`` knows how to render as a node."""
    return sorted(MODEL_NODE_MAP)


def _lookup_model(model: str) -> tuple[str, NodeSpec] | None:
    """(alias, NodeSpec) for a user-typed model — an alias or the canonical
    endpoint id an alias resolves to — or None when nothing maps to a node."""
    if model in MODEL_NODE_MAP:
        return model, MODEL_NODE_MAP[model]
    canonical = spec.resolve_alias(model)
    pref = spec.preferred_alias(canonical)
    if pref and pref in MODEL_NODE_MAP:
        return pref, MODEL_NODE_MAP[pref]
    return None


def is_supported(model: str) -> bool:
    """Whether ``--emit-workflow`` can render ``model`` (alias or canonical id)
    as a partner node. The per-row ``emit_supported`` flag of ``generate list``."""
    return _lookup_model(model) is not None


def _resolve_model(model: str) -> tuple[str, NodeSpec]:
    found = _lookup_model(model)
    if found is None:
        raise UnsupportedModelError(model, supported_models())
    return found


def build_workflow(
    model: str, values: dict[str, Any], *, output_prefix: str = "generate"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a frontend-format workflow that drives the partner node for ``model``.

    ``values`` are the parsed ``--param`` values (same dict the proxy client
    receives). Local-file image params are materialized as ``LoadImage`` nodes
    and wired into the partner node; scalar params override the node's fixed
    defaults. A ``SaveImage``/``SaveVideo`` is appended so ``comfy run`` writes
    the result to disk.

    Returns ``(workflow, ops)``: ``workflow`` is frontend-format (``nodes``/
    ``links``, minted via ``workflow_ops.add_node``/``connect``/``set_widget``
    — the same CRDT-ready primitives ``comfy workflow`` slot-editing commands
    require), and ``ops`` is the replayable op stream those primitives emitted,
    in apply order. This is what makes the result editable downstream instead
    of a raw API-format dict `list_slots`/`set_widget`/`connect` reject as
    ``workflow_not_frontend_format``.

    The partner node classes in ``MODEL_NODE_MAP`` are addressed by name
    against the bundled snapshot regardless of that snapshot's own
    ``deprecated`` flag (some partner nodes are marked deprecated upstream in
    favor of a newer node this emitter does not yet target) — ``--emit-workflow``
    always emits the class ``NodeSpec`` names, so every ``add_node`` call in
    this module passes ``allow_deprecated=True``.
    """
    _alias, ns = _resolve_model(model)
    graph = _catalog()

    workflow: dict[str, Any] = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
    ops: list[dict[str, Any]] = []

    def _add(class_type: str) -> Any:
        nonlocal workflow
        workflow, op = workflow_ops.add_node(workflow, graph, class_type, allow_deprecated=True)
        ops.append(op)
        return op["node_id"]

    def _set(node_id: Any, widget: str, value: Any) -> None:
        nonlocal workflow
        workflow, op = workflow_ops.set_widget(workflow, graph, node_id, widget, value)
        ops.append(op)

    def _link(from_node: Any, from_slot: int, to_node: Any, to_slot: int) -> None:
        nonlocal workflow
        workflow, op = workflow_ops.connect(workflow, graph, from_node, from_slot, to_node, to_slot)
        ops.append(op)

    node_inputs: dict[str, Any] = dict(ns.fixed)
    image_links: dict[str, tuple[Any, int]] = {}

    # Image-path params → LoadImage nodes wired into the partner node.
    for flag, node_key in ns.image_params.items():
        raw = values.get(flag)
        if raw is None:
            continue
        paths = [str(Path(p).expanduser()) for p in (raw if isinstance(raw, list | tuple) else [raw])]
        loader_ids: list[Any] = []
        for path in paths:
            loader_id = _add("LoadImage")
            _set(loader_id, "image", path)
            loader_ids.append(loader_id)
        # One file wires straight in; several fold through chained core
        # ImageBatch nodes (2-input, always present) so the partner still
        # receives a single IMAGE stream.
        upstream, upstream_out = loader_ids[0], 0
        for lid in loader_ids[1:]:
            batch_id = _add("ImageBatch")
            _link(upstream, upstream_out, batch_id, "image1")
            _link(lid, 0, batch_id, "image2")
            upstream, upstream_out = batch_id, 0
        image_links[node_key] = (upstream, upstream_out)

    # Scalar params → node inputs, honoring the explicit param_map.
    for flag, node_key in ns.param_map.items():
        if flag in values and values[flag] is not None:
            node_inputs[node_key] = values[flag]

    # Nodes that take an aspect ratio instead of width/height: fold the two
    # proxy flags into the node's single "W:H" input when both are present.
    # A lone --width/--height has no fixed default to fall back on here (unlike
    # flux-2's independent param_map entries), so silently keeping the default
    # aspect ratio would drop the user's flag with no error - fail loudly instead.
    if ns.aspect_from_wh:
        width_given = values.get("width") is not None
        height_given = values.get("height") is not None
        if width_given != height_given:
            raise EmitError(
                f"--emit-workflow for {model!r} needs --width and --height together "
                "to set the aspect ratio; only one was provided."
            )
        if width_given and height_given:
            node_inputs[ns.aspect_from_wh] = f"{values['width']}:{values['height']}"

    partner_id = _add(ns.node_class)
    for node_key, value in node_inputs.items():
        _set(partner_id, node_key, value)
    for node_key, (src_node, src_slot) in image_links.items():
        _link(src_node, src_slot, partner_id, node_key)

    if ns.output == "VIDEO":
        save_id = _add("SaveVideo")
        _set(save_id, "filename_prefix", output_prefix)
        _link(partner_id, ns.media_port, save_id, "video")
    else:
        save_id = _add("SaveImage")
        _set(save_id, "filename_prefix", output_prefix)
        _link(partner_id, ns.media_port, save_id, "images")

    return workflow, ops


def write_workflow(
    model: str, values: dict[str, Any], path: Path, *, output_prefix: str = "generate"
) -> dict[str, Any]:
    """Build the workflow for ``model`` and write it to ``path`` as JSON.
    Returns the workflow dict. Raises ``EmitError`` on an unsupported model."""
    workflow, _ops = build_workflow(model, values, output_prefix=output_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    return workflow
