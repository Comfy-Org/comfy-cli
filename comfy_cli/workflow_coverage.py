"""Can Comfy Cloud run one workflow as it is, and what is missing if not?

``comfy workflow check --where cloud`` asks this. The command fetches; this module
only reads and compares, so every rule here is testable offline:

- :func:`find_uses` reads a workflow (the editor format with ``nodes[]``, subgraphs
  included, or an API export) and lists the node types and model files its active
  nodes use. Muted and bypassed nodes load nothing, so they are listed as skipped.
- :func:`resolve_models` places each model on Cloud: in Cloud's own dropdown, among
  the person's uploaded models, as a remap to the one Cloud file of the same name in
  another folder, or missing.
- :func:`identity` says whether a local copy is proven to be the same file as
  Cloud's: the same name is not the same bytes.
- :func:`verdict` and :func:`apply_remaps` turn that into the answer and the copy.

Lifted from the Cloud inventory check in the run-on-cloud draft, cut to what
this one question needs.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".sft")

# Node modes the executor skips: these nodes load nothing.
SKIPPED_MODES = {2: "muted", 4: "bypassed"}

# Canvas-only nodes: no object_info carries them and they load nothing.
UI_ONLY_NODES = frozenset({"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "GetNode", "SetNode"})

_HASH_RE = re.compile(r"(?:blake3:)?([0-9a-f]{64})(?:\.[A-Za-z0-9]+)?")

VERDICTS = ("covered", "covered_with_remaps", "covered_needs_check", "needs_build")


def workflow_format(data: Any) -> str | None:
    """``"ui"`` for the editor format, ``"api"`` for an API export, ``None`` for neither."""
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return "ui"
    if (
        isinstance(data, dict)
        and data
        and all(isinstance(node, dict) and isinstance(node.get("class_type"), str) for node in data.values())
    ):
        return "api"
    return None


def is_model_value(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.lower().endswith(MODEL_EXTENSIONS)
        and "\n" not in value
        and "\r" not in value
        and not value.startswith(("http://", "https://"))
    )


def _basename(value: str) -> str:
    # A workflow saved on Windows names its subfolder with "\".
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _ui_model_strings(node: Mapping[str, Any]) -> list[str]:
    if str(node.get("type", "")).startswith("CLIPTextEncode"):
        return []  # a prompt can end in ".safetensors" and still be a prompt
    values = node.get("widgets_values")
    if isinstance(values, dict):
        values = list(values.values())
    found: list[str] = []
    for value in values if isinstance(values, list) else []:
        if is_model_value(value):
            found.append(value)
        elif isinstance(value, dict):  # LoRA stacks keep rows such as {"lora": "...", "on": true}
            found.extend(v for v in value.values() if is_model_value(v))
    return found


def _subgraphs(workflow: Mapping[str, Any]) -> list[Any]:
    definitions = workflow.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    return subgraphs if isinstance(subgraphs, list) else []


def _ui_nodes(workflow: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any], tuple]]:
    """Every node an instance reaches, with its scoped id and where it lives in the file.

    A subgraph instance's interior nodes get ids ``<instance>/<inner>``. ``loc`` is
    ``("nodes", i)`` or ``("subgraph", j, i)``: two instances of one definition share
    one ``loc``, which is what a remap rewrites.
    """
    by_id = {
        sg["id"]: (j, sg)
        for j, sg in enumerate(_subgraphs(workflow))
        if isinstance(sg, dict) and isinstance(sg.get("id"), str)
    }

    def visit(nodes, prefix: str, base: tuple, ancestors: tuple):
        for i, node in enumerate(nodes if isinstance(nodes, list) else []):
            if not isinstance(node, dict):
                continue
            node_id = f"{prefix}/{node.get('id')}" if prefix else str(node.get("id"))
            yield node_id, node, (*base, i)
            definition = by_id.get(node["type"]) if isinstance(node.get("type"), str) else None
            if definition is None or node.get("mode") in SKIPPED_MODES or node.get("type") in ancestors:
                continue
            j, sg = definition
            yield from visit(sg.get("nodes"), node_id, ("subgraph", j), (*ancestors, node.get("type")))

    yield from visit(workflow.get("nodes"), "", ("nodes",), ())


def find_uses(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """The node types and model files a workflow's active nodes use.

    Returns ``{"format", "nodes": {class_type: [node_id, ...]}, "models": [...],
    "skipped": [...]}``. Each model row carries ``node_id``, ``class_type``,
    ``value``, ``widget`` (the input name in an API export, else ``None``) and an
    internal ``loc`` that :func:`apply_remaps` uses and the command drops.
    """
    fmt = workflow_format(workflow)
    if fmt is None:
        raise ValueError("not a ComfyUI workflow: expected an editor save with nodes[] or an API export")
    nodes: dict[str, list[str]] = {}
    models: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    if fmt == "api":
        for node_id, node in workflow.items():
            class_type = node["class_type"]
            nodes.setdefault(class_type, []).append(str(node_id))
            inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
            for name, value in inputs.items():
                if is_model_value(value):
                    models.append(
                        {
                            "node_id": str(node_id),
                            "class_type": class_type,
                            "value": value,
                            "widget": name,
                            "loc": ("api", str(node_id)),
                        }
                    )
        return {"format": fmt, "nodes": nodes, "models": models, "skipped": skipped}

    subgraph_ids = {sg["id"] for sg in _subgraphs(workflow) if isinstance(sg, dict) and isinstance(sg.get("id"), str)}
    for node_id, node, loc in _ui_nodes(workflow):
        class_type = node.get("type")
        if node.get("mode") in SKIPPED_MODES:  # a muted subgraph's interior is never walked
            skipped.append({"node_id": node_id, "class_type": class_type, "why": SKIPPED_MODES[node["mode"]]})
            continue
        if not isinstance(class_type, str) or class_type in UI_ONLY_NODES or class_type in subgraph_ids:
            continue
        nodes.setdefault(class_type, []).append(node_id)
        for value in _ui_model_strings(node):
            if (loc, value) in seen:  # a second instance of the same subgraph
                continue
            seen.add((loc, value))
            models.append({"node_id": node_id, "class_type": class_type, "value": value, "widget": None, "loc": loc})
    return {"format": fmt, "nodes": nodes, "models": models, "skipped": skipped}


def combo_choices(class_info: Any) -> dict[str, str]:
    """Every string a node type's dropdowns accept, mapped to the input that accepts it.

    Reads both object_info forms: ``[["a", "b"], {...}]`` and ``["COMBO", {"options": [...]}]``.
    """
    choices: dict[str, str] = {}
    inputs = class_info.get("input") if isinstance(class_info, dict) else None
    for section in ("required", "optional"):
        group = inputs.get(section) if isinstance(inputs, dict) else None
        for name, spec in (group or {}).items() if isinstance(group, dict) else ():
            if not isinstance(spec, list) or not spec:
                continue
            options = spec[0]
            if not isinstance(options, list) and len(spec) > 1 and isinstance(spec[1], dict):
                options = spec[1].get("options")
            for option in options if isinstance(options, list) else []:
                if isinstance(option, str):
                    choices.setdefault(option, name)
    return choices


def content_hash(value: Any) -> str | None:
    """Cloud reports a hash as ``blake3:<hex>`` or as the stored ``<hex>.<ext>``; normalize to the first."""
    match = _HASH_RE.fullmatch(value) if isinstance(value, str) else None
    return f"blake3:{match[1]}" if match else None


def resolve_models(
    uses: Mapping[str, Any],
    object_info: Mapping[str, Any],
    own_models: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Place each model on Cloud.

    ``own_models(basename)`` returns the person's uploaded model assets whose name
    contains ``basename``. It is asked only about models Cloud's dropdowns do not
    list, once per name.

    ``status`` is ``present`` (Cloud runs the saved value), ``remap`` (exactly one
    Cloud file has the same name in another folder: the saved value must change to
    ``cloud_value``), ``missing``, or ``not_checked`` when the node type itself is
    not on Cloud. ``source`` is ``cloud`` for Cloud's own models and ``yours`` for
    the person's uploads.
    """
    asked: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for use in uses["models"]:
        value, class_type = use["value"], use["class_type"]
        row: dict[str, Any] = {
            "node_id": use["node_id"],
            "class_type": class_type,
            "widget": use["widget"],
            "value": value,
            "status": "missing",
            "source": None,
            "cloud_value": None,
            "loc": use["loc"],
        }
        rows.append(row)
        if class_type not in object_info:
            row["status"] = "not_checked"
            continue
        choices = combo_choices(object_info[class_type])
        name = _basename(value)
        same_name = sorted(c for c in choices if _basename(c) == name)
        if value in choices:
            row.update(status="present", source="cloud", cloud_value=value, widget=row["widget"] or choices[value])
            continue
        if len(same_name) == 1:
            to = same_name[0]
            row.update(status="remap", source="cloud", cloud_value=to, widget=row["widget"] or choices[to])
            continue
        if same_name:
            row["candidates"] = same_name  # several Cloud files share the name: no lossless pick
            continue
        if name not in asked:
            asked[name] = [a for a in own_models(name) if isinstance(a, dict)]
        mine = [a for a in asked[name] if a.get("name") in (value, name)]
        exact = [a for a in mine if a.get("name") == value] or mine
        if not exact:
            continue
        asset = exact[0]
        row.update(
            status="present" if asset["name"] == value else "remap",
            source="yours",
            cloud_value=asset["name"],
            cloud_hash=content_hash(asset.get("hash")),
            cloud_size=asset.get("size") if isinstance(asset.get("size"), int) else None,
        )
        if len({content_hash(a.get("hash")) for a in exact}) > 1:
            row["cloud_hash"] = None  # two uploads share the name: which one runs is not ours to say
    return rows


def identity(
    local: list[tuple[str, int]] | None,
    cloud_size: int | None,
    cloud_hash: str | None,
    hash_file: Callable[[str], str],
) -> tuple[str, str | None]:
    """``same``, ``different`` or ``unverified``, and the local path compared.

    ``local`` is every local file with the model's name, as ``(path, size)``;
    ``None`` when no models folder was given. Only a blake3 match proves ``same``;
    a size mismatch alone proves ``different``.
    """
    if not local or len({size for _, size in local}) > 1:
        return "unverified", None
    path, size = local[0]
    if cloud_size is not None and cloud_size != size:
        return "different", path
    if cloud_hash is None:
        return "unverified", path
    digests = {hash_file(p) for p, _ in local}
    if len(digests) > 1:
        return "unverified", None
    return ("same" if digests == {cloud_hash} else "different"), path


def local_index(roots: list[str]) -> dict[str, list[tuple[str, int]]]:
    """Every model file under ``roots``, by file name."""
    index: dict[str, list[tuple[str, int]]] = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(os.path.expanduser(root), followlinks=True):
            for filename in files:
                if filename.lower().endswith(MODEL_EXTENSIONS) and not filename.startswith("."):
                    full = os.path.join(dirpath, filename)
                    try:
                        index.setdefault(filename, []).append((full, os.stat(full).st_size))
                    except OSError:
                        continue
    return index


# Where ComfyUI keeps a model, by the loader input that names it and, for an editor
# save (whose widgets carry no names), by the loader type. Only the stock loaders:
# anything else reports ``folder: null`` unless the workflow's own download notes say.
_FOLDER_BY_INPUT = {
    "ckpt_name": "checkpoints",
    "lora_name": "loras",
    "vae_name": "vae",
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "control_net_name": "controlnet",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
    "hypernetwork_name": "hypernetworks",
}
_FOLDER_BY_CLASS = {
    "CheckpointLoaderSimple": "checkpoints",
    "ImageOnlyCheckpointLoader": "checkpoints",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "VAELoader": "vae",
    "UNETLoader": "diffusion_models",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "TripleCLIPLoader": "text_encoders",
    "ControlNetLoader": "controlnet",
    "UpscaleModelLoader": "upscale_models",
    "CLIPVisionLoader": "clip_vision",
    "StyleModelLoader": "style_models",
}


def download_notes(workflow: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    """``{filename: {"folder", "url"}}`` from the download notes an editor save may carry
    (top-level ``models[]`` and each node's ``properties.models[]``, as the ComfyUI
    templates write them)."""
    entries: list[Any] = []
    if isinstance(workflow.get("models"), list):
        entries.extend(workflow["models"])
    if workflow_format(workflow) == "ui":
        for _, node, _ in _ui_nodes(workflow):
            props = node.get("properties")
            if isinstance(props, dict) and isinstance(props.get("models"), list):
                entries.extend(props["models"])
    notes: dict[str, dict[str, str | None]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            folder, url = entry.get("directory"), entry.get("url")
            notes.setdefault(
                _basename(entry["name"]),
                {"folder": folder if isinstance(folder, str) else None, "url": url if isinstance(url, str) else None},
            )
    return notes


def add_sources(
    models: list[dict[str, Any]],
    workflow: Mapping[str, Any],
    index: Mapping[str, list[tuple[str, int]]] | None,
) -> None:
    """Give each model row what a person needs to find or upload the file: ``filename``,
    ``folder`` (the ComfyUI models folder, or ``None`` when unknown), ``url`` (from the
    workflow's download notes, else ``None``) and ``local_paths`` (copies under
    ``--models-dir``; empty without one)."""
    notes = download_notes(workflow)
    for model in models:
        name = _basename(model["value"])
        note = notes.get(name, {})
        model["filename"] = name
        model["folder"] = (
            note.get("folder")
            or _FOLDER_BY_INPUT.get(model.get("widget") or "")
            or _FOLDER_BY_CLASS.get(model["class_type"])
        )
        model["url"] = note.get("url")
        model["local_paths"] = [path for path, _ in (index or {}).get(name, [])]


def blocking(uses: Mapping[str, Any], object_info: Mapping[str, Any], models: list[dict[str, Any]]) -> list[dict]:
    """What stops the workflow from running on Comfy Cloud: each missing node type, then each model."""
    found: list[dict[str, Any]] = [
        {"kind": "node", "class_type": class_type, "node_ids": node_ids, "reason": "not_on_cloud"}
        for class_type, node_ids in uses["nodes"].items()
        if class_type not in object_info
    ]
    for model in models:
        reason = None
        if model["status"] == "missing":
            reason = "ambiguous" if model.get("candidates") else "not_on_cloud"
        elif model.get("identity") == "different":
            reason = "different_file"
        if reason:
            found.append(
                {
                    "kind": "model",
                    "value": model["value"],
                    "node_id": model["node_id"],
                    "class_type": model["class_type"],
                    "reason": reason,
                    **{key: model[key] for key in ("filename", "folder", "url", "local_paths") if key in model},
                }
            )
    return found


def verdict(blocked: list[dict], models: list[dict[str, Any]]) -> str:
    """A remap outranks an unproven model: the unchanged original does not run while a remap is due."""
    if blocked:
        return "needs_build"
    if any(m["status"] == "remap" for m in models):
        return "covered_with_remaps"
    if any(m.get("identity") != "same" for m in models):
        return "covered_needs_check"
    return "covered"


def apply_remaps(workflow: Mapping[str, Any], models: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """A copy of ``workflow`` with each remap's value replaced; the original is not touched."""
    out = copy.deepcopy(workflow)
    changed = 0
    for model in models:
        if model["status"] != "remap":
            continue
        loc, old, new = model["loc"], model["value"], model["cloud_value"]
        if loc[0] == "api":
            out[loc[1]]["inputs"][model["widget"]] = new
            changed += 1
            continue
        node = out["nodes"][loc[1]] if loc[0] == "nodes" else out["definitions"]["subgraphs"][loc[1]]["nodes"][loc[2]]
        values = node.get("widgets_values")
        items = list(values.items()) if isinstance(values, dict) else list(enumerate(values or []))
        for key, value in items:
            if value == old:
                values[key] = new
                changed += 1
            elif isinstance(value, dict):
                for inner, inner_value in list(value.items()):
                    if inner_value == old:
                        value[inner] = new
                        changed += 1
    return out, changed
