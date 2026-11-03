"""What one ComfyUI workflow needs, and how much of it this machine and Comfy Cloud already have.

``comfy build init --from-workflow`` asks the builder which packs a workflow uses, but
it drops what a build needs to be complete: model subfolders, the pack versions the
workflow recorded, ``embedding:`` references, and weights a loader picks without a
widget naming them. A build made from an install instead carries everything installed.
This module reads the workflow itself so the scope of a build is exactly what the
workflow loads, and nothing else on the machine.

``inventory`` is pure: it reads the JSON (UI/editing format or an API export) and
reports. The ``join_*`` functions add what is known elsewhere:

- ``join_local``  — every local file a model value could be, across model roots;
- ``join_server`` — the running ComfyUI's version and which pack directory provides
  each class;
- ``join_cloud``  — whether Comfy Cloud's standard environment runs the workflow as-is, with dropdown
  remaps, or needs a build.

The caller fetches; these functions only compare, so every one is testable offline.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")

# Loaders whose first widget names a run-time input file (an image, a video, a clip).
INPUT_LOADERS = frozenset(
    {
        "LoadImage",
        "LoadImageMask",
        "LoadImageOutput",
        "LoadVideo",
        "LoadAudio",
        "VHS_LoadVideo",
        "VHS_LoadVideoPath",
        "VHS_LoadAudioUpload",
    }
)
INPUT_WIDGETS = {
    "LoadImage": "image",
    "LoadImageMask": "image",
    "LoadImageOutput": "image",
    "LoadVideo": "file",
    "LoadAudio": "audio",
    "VHS_LoadVideo": "video",
    "VHS_LoadVideoPath": "video",
    "VHS_LoadAudioUpload": "audio",
}

# Node modes the executor skips: these nodes load nothing, so nothing they name travels.
SKIPPED_MODES = {2: "muted", 4: "bypassed"}

# Canvas-only nodes. No object_info carries them and they load nothing.
UI_ONLY_NODES = frozenset({"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "GetNode", "SetNode"})

# Prune only with evidence about execution roots. Unknown/custom classes may be
# OUTPUT_NODEs even when they also return values; retain them conservatively.
KNOWN_OUTPUT_NODES = frozenset({"SaveImage", "PreviewImage"})
KNOWN_NON_OUTPUT_NODES = frozenset(
    {
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "LoraLoader",
        "LoraLoaderModelOnly",
        "VAELoader",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
        "UpscaleModelLoader",
        "CLIPTextEncode",
        "KSampler",
        "KSamplerAdvanced",
        "VAEDecode",
        "VAEEncode",
        "EmptyLatentImage",
        "LoadImage",
        "LoadImageMask",
    }
)

# Loaders known to load weights that no widget names. A build that misses these
# fails at run time, not at build time. Extend as packs are learned.
IMPLICIT_LOADERS = {
    "IPAdapterUnifiedLoader": "picks ipadapter/* and clip_vision/* weights by preset name",
    "IPAdapterUnifiedLoaderFaceID": "as IPAdapterUnifiedLoader, plus insightface models and a FaceID LoRA",
    "InstantIDFaceAnalysis": "loads insightface/models/antelopev2/*.onnx",
    "IPAdapterInsightFaceLoader": "loads insightface/models/<provider>/*.onnx",
    "DownloadAndLoadDepthAnythingV3Model": "downloads its model on first run unless it is declared under depthanything3/",
    "UltralyticsDetectorProvider": "needs the Impact Subpack companion pack",
}

# Nodes whose only effect is memory or speed. Cloud lacking one is a lossless fix: remove it
# and pass its model input straight to its consumer. Add a class only when the pack documents
# it as memory- or speed-only; anything that can change pixels does not belong here.
PERF_ONLY_NODES = {
    "WanVideoBlockSwap": "block-swap args for the model loader (VRAM only)",
    "WanVideoSetBlockSwap": "passes the model through with block swap set (VRAM only)",
    "WanVideoVRAMManagement": "VRAM management args (memory only)",
    "WanVideoTorchCompileSettings": "torch.compile settings (speed only)",
}

_EMBEDDING_RE = re.compile(r"embedding:([^\s,:()\[\]]+)", re.IGNORECASE)

# Where Comfy Desktop keeps its settings, including `modelsDirs`. Only the macOS
# location is confirmed; on other platforms callers pass `--models-root` instead.
DESKTOP_SETTINGS_MACOS = "~/Library/Application Support/Comfy Desktop/settings.json"


def is_model_value(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.lower().endswith(MODEL_EXTENSIONS)
        and not _EMBEDDING_RE.search(value)
        and "\n" not in value
        and "\r" not in value
    )


def _posix(value: str) -> str:
    """Windows separators become ``/``: a saved ``sdxl\\model.safetensors`` names a subfolder."""
    return value.replace("\\", "/")


def _widget_strings(node: Mapping[str, Any]) -> list[str]:
    values = node.get("widgets_values")
    if isinstance(values, dict):
        values = list(values.values())
    strings: list[str] = []
    for value in values or []:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):  # LoRA stacks store rows such as {"lora": "...", "on": true}
            strings.extend(v for v in value.values() if isinstance(v, str))
    return strings


def _model_strings(node: Mapping[str, Any]) -> list[str]:
    """Exclude known text widgets without rejecting filenames containing spaces."""
    if str(node.get("type", "")).startswith("CLIPTextEncode"):
        return []
    return [value for value in _widget_strings(node) if is_model_value(value)]


def is_api_format(workflow: Any) -> bool:
    return (
        isinstance(workflow, dict)
        and "nodes" not in workflow
        and bool(workflow)
        and all(isinstance(v, dict) and "class_type" in v for v in workflow.values())
    )


def _disconnected_nodes(nodes: list[Mapping[str, Any]], edges: Iterable[tuple[Any, Any]]) -> set[str]:
    """Return provably unused nodes; an incomplete graph keeps its full scope."""
    by_id = {str(node.get("id")): node for node in nodes}
    parents: dict[str, set[str]] = {}
    for source, target in edges:
        if str(source) in by_id and str(target) in by_id:
            parents.setdefault(str(target), set()).add(str(source))
        else:
            return set()
    if not parents:
        return set()
    roots: set[str] = set()
    has_output = False
    for identifier, node in by_id.items():
        if node.get("mode") == 2:
            continue
        kind = node.get("type")
        explicit = node.get("output_node", (node.get("properties") or {}).get("output_node"))
        # Workflow metadata is caller-controlled. A false marker cannot veto a
        # known output or prove an unknown custom class has no execution effect.
        output = explicit is True or kind in KNOWN_OUTPUT_NODES
        has_output |= output
        if output or kind not in KNOWN_NON_OUTPUT_NODES:
            roots.add(identifier)
    if not has_output:
        return set()
    pending = list(roots)
    reachable: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier not in reachable:
            reachable.add(identifier)
            pending.extend(parents.get(identifier, ()))
    return set(by_id) - reachable


def _ui_nodes(workflow: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    definitions = {sg["id"]: sg for sg in (workflow.get("definitions") or {}).get("subgraphs") or [] if sg.get("id")}

    def visit(nodes, links=(), prefix="", ancestors=()):
        edges = []
        for link in links:
            if isinstance(link, dict):
                edges.append((link.get("origin_id"), link.get("target_id")))
            elif isinstance(link, (list, tuple)) and len(link) >= 4:
                edges.append((link[1], link[3]))
            else:
                edges.append((None, None))  # Incomplete metadata cannot justify pruning.
        disconnected = _disconnected_nodes(nodes, edges)
        for node in nodes:
            node_id = f"{prefix}/{node.get('id')}" if prefix else node.get("id")
            scoped = {**node, "id": node_id}
            if str(node.get("id")) in disconnected:
                yield {**scoped, "inventory_disconnected": True}
                continue
            definition = definitions.get(node.get("type"))
            if definition is None or node.get("mode") in SKIPPED_MODES:
                yield scoped
            elif node.get("type") not in ancestors:
                yield scoped
                yield from visit(
                    definition.get("nodes") or [],
                    definition.get("links") or [],
                    str(node_id),
                    (*ancestors, node.get("type")),
                )
            else:
                raise ValueError(f"cyclic subgraph definition at node {node_id}")

    yield from visit(workflow.get("nodes") or [], workflow.get("links") or [])


def _api_nodes(workflow: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    nodes = [{**node, "id": node_id, "type": node.get("class_type")} for node_id, node in workflow.items()]
    disconnected = _disconnected_nodes(
        nodes,
        (
            (value[0], node_id)
            for node_id, node in workflow.items()
            for value in (node.get("inputs") or {}).values()
            if isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        ),
    )
    for node_id, node in workflow.items():
        inputs = node.get("inputs") or {}
        yield {
            "type": node.get("class_type"),
            "id": node_id,
            "mode": 0,
            "inventory_disconnected": str(node_id) in disconnected,
            "named_inputs": inputs,
            # A list value is a link to another node's output, not a widget.
            "widgets_values": [v for v in inputs.values() if not isinstance(v, list)],
        }


def inventory(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Report every class, model, embedding and input file the workflow's active nodes use.

    Muted and bypassed nodes are listed under ``skipped`` with the models they would
    have loaded, so a caller can say what was left out and why.
    """
    if workflow.get("schema") == "envelope/1":
        raise ValueError(
            "This file is a CLI response envelope. Extract data.prompt as the API graph "
            "(jq '.data.prompt' preview.json > api.json), then inventory api.json."
        )
    api = is_api_format(workflow)
    if api:
        if any(not isinstance(n.get("inputs", {}), dict) for n in workflow.values()):
            raise ValueError("API workflow node inputs must be objects")
    else:
        if not isinstance(workflow.get("nodes"), list):
            raise ValueError("UI workflow must contain a nodes list, or pass an API workflow export")
        definitions = workflow.get("definitions") or {}
        if not isinstance(definitions, dict) or not isinstance(definitions.get("subgraphs", []), list):
            raise ValueError("workflow definitions.subgraphs must be a list")
        groups = [workflow]
        for definition in definitions.get("subgraphs", []):
            if not isinstance(definition, dict) or not isinstance(definition.get("nodes", []), list):
                raise ValueError("each subgraph definition must be an object with a nodes list")
            groups.append(definition)
        for group in groups:
            for node in group.get("nodes", []):
                if not isinstance(node, dict) or not isinstance(node.get("type"), str):
                    raise ValueError("each UI workflow node must be an object with a string type")
    report: dict[str, Any] = {
        "format": "api" if api else "ui",
        "classes": {},
        "packs": {},
        "models": [],
        "model_hints": [],
        "embeddings": [],
        "inputs": [],
        "skipped": [],
        "implicit": [],
        "warnings": [],
        "unverified": [],
        # The pack versions the frontend recorded when the workflow was saved.
        "node_versions": {} if api else (workflow.get("extra") or {}).get("node_versions") or {},
    }
    subgraph_ids = (
        set()
        if api
        else {sg.get("id") for sg in (workflow.get("definitions") or {}).get("subgraphs") or [] if sg.get("id")}
    )
    classes: dict[str, dict[str, Any]] = {}
    packs: dict[str, set[str]] = {}
    seen_paths: set[tuple[Any, str, str]] = set()

    def add_model(entry: dict[str, Any]) -> None:
        # Catalog choices are per loader. The same filename used by two loaders
        # must be checked twice, and every remap needs its own widget address.
        key = (entry["node_id"], entry["node"], entry["path"])
        if entry["path"] and key not in seen_paths:
            seen_paths.add(key)
            report["models"].append(entry)

    for node in _api_nodes(workflow) if api else _ui_nodes(workflow):
        class_type = node.get("type")
        if node.get("inventory_disconnected") and node.get("mode", 0) not in SKIPPED_MODES:
            report["skipped"].append(
                {
                    "id": node.get("id"),
                    "class": class_type,
                    "why": "disconnected",
                    "models": [_posix(value) for value in _model_strings(node)],
                }
            )
            continue
        if node.get("mode", 0) not in SKIPPED_MODES:
            if class_type in {"PrimitiveNode", "GetNode", "SetNode"}:
                report["unverified"].append(
                    f"{class_type} #{node.get('id')}: linked widget values need a local API export before inventory"
                )
            elif class_type in subgraph_ids and node.get("widgets_values"):
                report["unverified"].append(
                    f"subgraph #{node.get('id')}: host widget overrides need a local API export before inventory"
                )
        node_id = node.get("id")
        strings = _widget_strings(node)
        model_strings = _model_strings(node)
        mode = node.get("mode", 0)
        if mode in SKIPPED_MODES:
            report["skipped"].append(
                {
                    "id": node_id,
                    "class": class_type,
                    "why": SKIPPED_MODES[mode],
                    "models": [_posix(s) for s in model_strings],
                }
            )
            continue
        # A subgraph instance's type is its definition id; its interior nodes are walked separately.
        if not class_type or class_type in UI_ONLY_NODES or class_type in subgraph_ids:
            continue

        properties = node.get("properties") or {}
        pack = properties.get("cnr_id") or properties.get("aux_id") or "unknown"
        row = classes.setdefault(class_type, {"count": 0, "pack": pack, "ver": set()})
        row["count"] += 1
        if properties.get("ver"):
            row["ver"].add(str(properties["ver"]))
        packs.setdefault(pack, set()).add(class_type)

        for value in strings:
            if value in model_strings:
                path = _posix(value)
                if "\\" in value:
                    report["warnings"].append(
                        f"{class_type} #{node_id}: Windows path {value!r} will not match on Linux; keep the "
                        "subfolder in the model's type and expect the user to re-pick it"
                    )
                add_model(
                    {
                        "value": value,
                        "path": path,
                        "basename": os.path.basename(path),
                        "subfolder": os.path.dirname(path),
                        "node": class_type,
                        "node_id": node_id,
                    }
                )
            for name in _EMBEDDING_RE.findall(value):
                report["embeddings"].append({"name": name, "node": class_type, "node_id": node_id})

        # Download hints the frontend embeds next to a node (name, directory, url).
        for hint in properties.get("models") or []:
            if not isinstance(hint, dict):
                continue
            path = _posix(str(hint.get("name") or ""))
            hint_entry = {
                "value": hint.get("name"),
                "path": path,
                "basename": os.path.basename(path),
                "subfolder": hint.get("directory"),
                "url_hint": hint.get("url"),
                "node": class_type,
                "node_id": node_id,
            }
            selected = next(
                (model for model in report["models"] if model["node_id"] == node_id and model["path"] == path), None
            )
            if selected is not None:
                selected["url_hint"] = hint.get("url")
            elif path:
                # Template download metadata can survive a model dropdown
                # change. Keep it as evidence, never as a required upload.
                report["model_hints"].append(hint_entry)

        if class_type in INPUT_LOADERS:
            named = node.get("named_inputs")
            saved_widgets = node.get("widgets_values")
            if named is not None:
                input_file = named.get(INPUT_WIDGETS[class_type])
            elif isinstance(saved_widgets, dict):
                input_file = saved_widgets.get(INPUT_WIDGETS[class_type])
            else:
                input_file = strings[0] if strings else None
            if isinstance(input_file, str) and input_file and not input_file.startswith(("http://", "https://")):
                report["inputs"].append({"file": input_file, "node": class_type, "node_id": node_id})

        if class_type in IMPLICIT_LOADERS:
            report["implicit"].append({"class": class_type, "note": IMPLICIT_LOADERS[class_type]})
        elif "loader" in class_type.lower() and not any(is_model_value(s) for s in strings):
            report["implicit"].append(
                {
                    "class": class_type,
                    "note": "loader with no model widget: read the pack source for weights it fetches or picks",
                }
            )

    report["classes"] = {name: {**row, "ver": sorted(row["ver"])} for name, row in classes.items()}
    report["packs"] = {name: sorted(members) for name, members in packs.items()}
    return report


def sha256_file(path: str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def join_local(report: dict[str, Any], roots: Iterable[str], *, hash_files: bool = False) -> None:
    """Attach every local file each model value could be, by basename, across ``roots``.

    Basename matching is deliberate: a workflow saved on another machine names its
    own subfolders, and the user's copy may sit anywhere under a root. ``type_dir`` is
    the first folder under the root (``checkpoints``, ``loras``…), the build's model type.
    """
    index: dict[str, list[tuple[str, str, str]]] = {}
    for root in roots:
        root = os.path.expanduser(root)
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for filename in files:
                if filename.lower().endswith(MODEL_EXTENSIONS) and not filename.startswith("."):
                    full = os.path.join(dirpath, filename)
                    index.setdefault(filename, []).append((root, os.path.relpath(full, root), full))
    for model in report["models"]:
        model["local"] = []
        for root, rel, full in index.get(model["basename"], []):
            try:
                size = os.stat(full).st_size
            except OSError:
                continue
            parts = rel.split(os.sep)
            match: dict[str, Any] = {
                "path": full,
                "root": root,
                "rel": rel.replace(os.sep, "/"),
                "bytes": size,
                "type_dir": parts[0] if len(parts) > 1 else None,
            }
            if hash_files:
                match["sha256"] = sha256_file(full)
            model["local"].append(match)
    report["models_not_on_disk"] = [m["path"] for m in report["models"] if not m.get("local")]


def join_server(
    report: Mapping[str, Any], object_info: Mapping[str, Any], system_stats: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Which classes the running ComfyUI lacks, and which pack directory provides each one it has."""
    missing: list[str] = []
    by_pack_dir: dict[str, list[str]] = {}
    for class_type in report["classes"]:
        info = object_info.get(class_type)
        if not isinstance(info, dict):
            missing.append(class_type)
            continue
        module = str(info.get("python_module") or "")
        pack_dir = module.split(".", 2)[1] if module.startswith("custom_nodes.") else "core"
        by_pack_dir.setdefault(pack_dir, []).append(class_type)
    return {
        "comfyui_version": ((system_stats or {}).get("system") or {}).get("comfyui_version"),
        "classes_missing_locally": missing,
        "classes_by_local_pack_dir": by_pack_dir,
    }


def join_inputs(report: dict[str, Any], roots: Iterable[str], hash_file: Callable[[str], str]) -> None:
    """Find each run-time input file under ``roots`` and the name Cloud would store it under.

    Cloud names an uploaded input by the blake3 of its bytes, so ``cloud_name`` is the name an
    existing upload of the same file already has. ``join_cloud`` then says whether it is listed.
    """
    index: dict[str, set[str]] = {}
    exact: dict[str, set[str]] = {}
    for root in roots:
        root = os.path.expanduser(root)
        for dirpath, _dirs, files in os.walk(root):
            for filename in files:
                full = os.path.join(dirpath, filename)
                index.setdefault(filename, set()).add(full)
                exact.setdefault(_posix(os.path.relpath(full, root)), set()).add(full)
    for item in report["inputs"]:
        requested = _posix(item["file"])
        matches = exact.get(requested) or index.get(os.path.basename(requested), set())
        path = next(iter(matches)) if len(matches) == 1 else None
        item["local_path"] = path
        if len(matches) > 1:
            report["warnings"].append(f"{item['file']}: ambiguous input file; choose the local path before uploading")
        if path:
            digest = hash_file(path).split(":", 1)[-1]
            item["cloud_name"] = digest + os.path.splitext(path)[1].lower()


def _weights(model: Mapping[str, Any], cloud_name: str, size_of: Callable[[str], int | None] | None) -> str:
    """Same name is not same weights: compare the local copy's size with Cloud's file."""
    local = model.get("local")
    if local is None:
        return "not_checked"
    if not local:
        return "no_local_copy"
    if len(local) > 1:
        digests = {match.get("sha256") for match in local}
        if None in digests or len(digests) != 1:
            return "ambiguous_local_copies"
    size = size_of(cloud_name) if size_of else None
    if size is None:
        return "unknown"
    return "same_size" if local[0]["bytes"] == size else "different_size"


def _combo_choices(node: Mapping[str, Any]) -> dict[str, str]:
    """Every string choice a node's inputs accept, mapped to the input that accepts it."""
    choices: dict[str, str] = {}
    for item in node.get("inputs") or []:
        for choice in item.get("choices") or []:
            if isinstance(choice, str):
                choices.setdefault(choice, item.get("name"))
    return choices


def join_cloud(
    report: Mapping[str, Any],
    lookup: Callable[[str], Mapping[str, Any] | None],
    size_of: Callable[[str], int | None] | None = None,
) -> dict[str, Any]:
    """Can Comfy Cloud's standard environment run this workflow as-is, after lossless fixes, or not at all?

    ``lookup(class_type)`` returns the Cloud catalog's ``nodes show`` payload for a class
    (``inputs[].choices``, ``cloud_disabled``), or ``None`` when Cloud lacks the class.
    ``size_of(name)`` returns the byte size of Cloud's model file with that exact name.

    A model is ``present`` when Cloud's dropdown lists the saved value, ``remap`` when
    exactly one Cloud choice has the same file name under a different folder (the
    saved value must be rewritten with ``set_widget``), and ``absent`` otherwise. Its
    ``weights`` compares sizes with the local copy: ``different_size`` means Cloud's file
    is a different model under the same name, so using it is a lossy change.
    A missing class in ``PERF_ONLY_NODES`` is ``lossless_removable`` rather than blocking.
    Embeddings and implicit loaders stay ``unverified``: no dropdown lists them.
    """
    result: dict[str, Any] = {
        "missing_classes": [],
        "cloud_disabled": [],
        "lossless_removable": [],
        "models": [],
        "unverified": list(report.get("unverified", [])),
    }
    choices_by_class: dict[str, dict[str, str]] = {}
    for class_type in report["classes"]:
        node = lookup(class_type)
        if node is None:
            if class_type in PERF_ONLY_NODES:
                result["lossless_removable"].append({"class": class_type, "why": PERF_ONLY_NODES[class_type]})
            else:
                result["missing_classes"].append(class_type)
            continue
        if node.get("cloud_disabled"):
            result["cloud_disabled"].append(class_type)
        for item in node.get("inputs") or []:
            if item.get("choices_truncated"):
                result["unverified"].append(f"{class_type}.{item.get('name')}: Cloud's choice list was truncated")
        choices_by_class[class_type] = _combo_choices(node)

    for model in report["models"]:
        choices = choices_by_class.get(model["node"])
        if choices is None:
            result["models"].append({"path": model["path"], "status": "node_missing"})
            continue
        if model["value"] in choices:
            weights = _weights(model, model["value"], size_of)
            result["models"].append({"path": model["path"], "status": "present", "weights": weights})
            continue
        same_name = sorted(c for c in choices if os.path.basename(_posix(c)) == model["basename"])
        if len(same_name) == 1:
            widget = choices[same_name[0]]
            result["models"].append(
                {
                    "path": model["path"],
                    "status": "remap",
                    "node_id": model["node_id"],
                    "widget": widget,
                    "from": model["value"],
                    "to": same_name[0],
                    "set_widget": f"{model['node_id']}.{widget}",
                    "weights": _weights(model, same_name[0], size_of),
                }
            )
        else:
            result["models"].append({"path": model["path"], "status": "absent", "candidates": same_name})

    for embedding in report["embeddings"]:
        result["unverified"].append(
            f"embedding:{embedding['name']} (no loader lists embeddings; check the Cloud model library)"
        )
    for implicit in report["implicit"]:
        result["unverified"].append(f"{implicit['class']}: {implicit['note']}")

    unresolved_inputs = []
    for item in report["inputs"]:
        if item.get("cloud_name"):
            item["on_cloud"] = item["cloud_name"] in choices_by_class.get(item["node"], {})
        if not item.get("local_path") and not item.get("on_cloud"):
            unresolved_inputs.append(
                {
                    "node_id": item.get("node_id"),
                    "file": item["file"],
                    "reason": "local file not found and target availability unverified",
                }
            )
    # Dependency compatibility alone never establishes a successful run. Keep
    # missing runtime inputs separate so callers cannot mistake `covered` for it.
    result["readiness"] = {
        "status": "needs_inputs" if unresolved_inputs else "not_run",
        "unresolved_inputs": unresolved_inputs,
    }

    for model in result["models"]:
        if model.get("weights") in ("same_size", "unknown", "not_checked", "no_local_copy", "ambiguous_local_copies"):
            result["unverified"].append(
                f"{model['path']}: model identity is unverified; matching names and sizes do not prove identical bytes"
            )

    blocking = (
        result["missing_classes"]
        + result["cloud_disabled"]
        + [m["path"] for m in result["models"] if m["status"] in ("absent", "node_missing")]
        + [
            f"{m['path']} (Cloud's file differs in size from yours)"
            for m in result["models"]
            if m.get("weights") == "different_size"
        ]
    )
    if blocking:
        verdict = "needs_build_or_lossy_conversion"
    elif result["unverified"]:
        verdict = "covered_needs_check"
    elif result["lossless_removable"] or any(m["status"] == "remap" for m in result["models"]):
        verdict = "covered_after_lossless_fixes"
    else:
        verdict = "covered"
    result["verdict"] = verdict
    result["blocking"] = blocking
    return result
