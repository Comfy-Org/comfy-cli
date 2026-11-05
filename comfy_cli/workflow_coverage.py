"""Can Comfy Cloud run one workflow as it is, and what is missing if not?

``comfy workflow check --where cloud`` asks this. The command fetches; this module
only reads and compares, so every rule here is testable offline:

- :func:`find_uses` reads a workflow (the editor format with ``nodes[]``, subgraphs
  included, or an API export) and lists the node types and model files its active
  nodes use. Muted and bypassed nodes load nothing, so they are listed as skipped.
  An editor save is read through the save-to-API converter, so a subgraph
  instance's own value for a promoted widget is the one checked, as the frontend
  submits it. Whether an input names a file is read from the node type's input
  spec in Cloud's node list, never from a list of node types.
- :func:`resolve_models` places each model on Cloud: in Cloud's own dropdown, among
  the person's uploaded models, as a remap to the one Cloud file of the same name in
  another folder, or missing.
- :func:`identity` says whether a local copy is proven to be the same file as
  Cloud's: the same name is not the same bytes.
- :func:`verdict` and :func:`apply_remaps` turn that into the answer and the copy.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

# ComfyUI's own list (folder_paths.supported_pt_extensions), plus the formats
# custom loaders read.
MODEL_EXTENSIONS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pt2",
    ".pth",
    ".bin",
    ".pkl",
    ".sft",
    ".gguf",
    ".onnx",
    ".torchscript",
)

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


def _model_strings(value: Any) -> list[str]:
    """The file names in one widget value: the value itself, or the values of a
    row such as a LoRA stack's ``{"lora": "...", "on": true}``."""
    if is_model_value(value):
        return [value]
    if isinstance(value, dict):
        return [v for v in value.values() if is_model_value(v)]
    return []


def _ui_model_strings(node: Mapping[str, Any]) -> list[str]:
    values = node.get("widgets_values")
    if isinstance(values, dict):
        values = list(values.values())
    return [found for value in (values if isinstance(values, list) else []) for found in _model_strings(value)]


def _quarantined_host_values(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A subgraph instance's legacy promoted values: the frontend reads these before ``widgets_values``."""
    props = node.get("properties")
    raw = props.get("proxyWidgetErrorQuarantine") if isinstance(props, dict) else None
    return [e for e in raw if isinstance(e, dict) and "hostValue" in e] if isinstance(raw, list) else []


def _declared_spec(class_info: Any, name: str) -> Any:
    inputs = class_info.get("input") if isinstance(class_info, dict) else None
    for section in ("required", "optional"):
        group = inputs.get(section) if isinstance(inputs, dict) else None
        if isinstance(group, dict) and name in group:
            return group[name]
    return None


def names_a_file(class_info: Any, name: str) -> bool:
    """Whether input ``name`` of a node type can hold a model file name, from its input spec.

    A dropdown (a choice list, ``COMBO`` or a dynamic combo) can. A typed input
    such as ``STRING`` cannot: a prompt that ends in ``.safetensors`` is still a
    prompt. An input the spec does not declare (the node type is not on Cloud, or
    the input is a dynamic row such as a LoRA stack's) can, and is judged by its
    value alone.
    """
    spec = _declared_spec(class_info, name)
    if not isinstance(spec, (list, tuple)) or not spec:
        return True
    kind = spec[0]
    return isinstance(kind, (list, tuple)) or (isinstance(kind, str) and "COMBO" in kind)


def _subgraphs(workflow: Mapping[str, Any]) -> list[Any]:
    definitions = workflow.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    return subgraphs if isinstance(subgraphs, list) else []


def _places_in(path: tuple, value: Any) -> Iterator[tuple[tuple, str]]:
    if is_model_value(value):
        yield path, value
    elif isinstance(value, dict):
        for key, inner in value.items():
            if is_model_value(inner):
                yield (*path, key), inner


def _saved_places(workflow: Mapping[str, Any]) -> Iterator[tuple[tuple, str, int, bool]]:
    """Every file name an editor save stores, as ``(path from the root, value, id of
    its node, whether it is a host's quarantined value)``.

    The places are each node's ``widgets_values`` (top level and in every subgraph
    definition), a row's fields one level down, and a subgraph instance's legacy
    quarantined host values.
    """
    owners: list[tuple[tuple, Any]] = [(("nodes",), workflow.get("nodes"))]
    for index, sg in enumerate(_subgraphs(workflow)):
        if isinstance(sg, dict):
            owners.append((("definitions", "subgraphs", index, "nodes"), sg.get("nodes")))
    for prefix, nodes in owners:
        for i, node in enumerate(nodes if isinstance(nodes, list) else []):
            if not isinstance(node, dict):
                continue
            values = node.get("widgets_values")
            if not isinstance(values, (dict, list)):
                values = []
            for key, value in values.items() if isinstance(values, dict) else enumerate(values):
                for path, found in _places_in((*prefix, i, "widgets_values", key), value):
                    yield path, found, id(node), False
            props = node.get("properties")
            raw = props.get("proxyWidgetErrorQuarantine") if isinstance(props, dict) else None
            for q, entry in enumerate(raw if isinstance(raw, list) else []):
                if isinstance(entry, dict) and "hostValue" in entry:
                    path = (*prefix, i, "properties", "proxyWidgetErrorQuarantine", q, "hostValue")
                    for at, found in _places_in(path, entry["hostValue"]):
                        yield at, found, id(node), True


def _at(root: Any, path: tuple) -> Any:
    for key in path:
        root = root[key]
    return root


def _ui_nodes(workflow: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every node an instance reaches, with its scoped id.

    A subgraph instance's interior nodes get ids ``<instance>:<inner>``, the ids the
    save-to-API converter and the frontend give them.
    """
    by_id = {sg["id"]: sg for sg in _subgraphs(workflow) if isinstance(sg, dict) and isinstance(sg.get("id"), str)}

    def visit(nodes, prefix: str, ancestors: tuple):
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            node_id = f"{prefix}:{node.get('id')}" if prefix else str(node.get("id"))
            yield node_id, node
            definition = by_id.get(node["type"]) if isinstance(node.get("type"), str) else None
            if definition is None or node.get("mode") in SKIPPED_MODES or node.get("type") in ancestors:
                continue
            yield from visit(definition.get("nodes"), node_id, (*ancestors, node.get("type")))

    yield from visit(workflow.get("nodes"), "", ())


def find_uses(workflow: Mapping[str, Any], object_info: Mapping[str, Any]) -> dict[str, Any]:
    """The node types and model files a workflow's active nodes use.

    Returns ``{"format", "nodes": {class_type: [node_id, ...]}, "models": [...],
    "skipped": [...]}``. Each model row carries ``node_id``, ``class_type``,
    ``value`` and ``widget`` (the input that holds it, or ``None`` when unknown).
    A row from an editor save also carries ``loc``, the path from the workflow's
    root to the one place its value is saved (``None`` when that cannot be told);
    :func:`apply_remaps` writes there.

    An editor save is converted the way it is submitted (subgraphs expanded, each
    instance's promoted values applied), so every instance of a subgraph gets its
    own rows. A node the converter cannot read, or whose type is not on Cloud, is
    read from its saved widget values instead. Raises ``ValueError`` for a file
    that is neither format and :class:`~comfy_cli.workflow_to_api.WorkflowConversionError`
    for an editor save the converter refuses.
    """
    fmt = workflow_format(workflow)
    if fmt is None:
        raise ValueError("not a ComfyUI workflow: expected an editor save with nodes[] or an API export")
    nodes: dict[str, list[str]] = {}
    models: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    markers: dict[str, tuple[tuple, str]] = {}

    def add(
        node_id: str,
        class_type: str,
        inputs: Mapping[str, Any],
        saved: set[str] | None = None,
        probed: Mapping[str, Any] | None = None,
    ) -> None:
        class_info = object_info.get(class_type)
        for name, value in inputs.items():
            if not names_a_file(class_info, name):
                continue
            marked = probed.get(name) if probed is not None else None
            if is_model_value(value):
                pairs = [(value, marked)]
            elif isinstance(value, dict):
                pairs = [(v, marked.get(k) if isinstance(marked, dict) else None) for k, v in value.items()]
            else:
                pairs = []
            for found, marker in pairs:
                # The converter fills an input the save left out with Cloud's first
                # choice; only a file the workflow itself names is reported.
                if not is_model_value(found) or (saved is not None and found.lower() not in saved):
                    continue
                row: dict[str, Any] = {"node_id": node_id, "class_type": class_type, "value": found, "widget": name}
                if saved is not None:
                    place = markers.get(marker) if isinstance(marker, str) else None
                    row["loc"] = place[0] if place is not None and place[1] == found else None
                models.append(row)

    if fmt == "api":
        for node_id, node in workflow.items():
            class_type = node["class_type"]
            nodes.setdefault(class_type, []).append(str(node_id))
            add(str(node_id), class_type, node.get("inputs") if isinstance(node.get("inputs"), dict) else {})
        return {"format": fmt, "nodes": nodes, "models": models, "skipped": skipped}

    from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api

    links = workflow.get("links")
    catalog = object_info if isinstance(object_info, dict) else dict(object_info)
    prompt = convert_ui_to_api({**workflow, "links": links if isinstance(links, list) else []}, catalog)
    # Where each value is saved: the same conversion again, with every saved file
    # name swapped for a marker. The marker an input ends up with names the one
    # place its value came from (a node's own widget, the instance slot holding a
    # promoted value, a quarantined host value), whatever precedence picked it, so
    # a remap writes there and nowhere else.
    places = list(_saved_places(workflow))
    probe = copy.deepcopy({**workflow, "links": links if isinstance(links, list) else []})
    for n, (path, value, _owner, _host) in enumerate(places):
        marker = f"\x00saved-at-{n}\x00"
        markers[marker] = (path, value)
        _at(probe, path[:-1])[path[-1]] = marker
    try:
        probed = convert_ui_to_api(probe, catalog)
    except WorkflowConversionError:
        probed = {}
    walked = list(_ui_nodes(workflow))
    saved = {
        found.lower()
        for _, node in walked
        for found in [
            *_ui_model_strings(node),
            *(v for e in _quarantined_host_values(node) for v in _model_strings(e["hostValue"])),
        ]
    }
    subgraph_ids = {sg["id"] for sg in _subgraphs(workflow) if isinstance(sg, dict) and isinstance(sg.get("id"), str)}
    for node_id, node in walked:
        class_type = node.get("type")
        if node.get("mode") in SKIPPED_MODES:  # a muted subgraph's interior is never walked
            skipped.append({"node_id": node_id, "class_type": class_type, "why": SKIPPED_MODES[node["mode"]]})
            continue
        if not isinstance(class_type, str) or class_type in UI_ONLY_NODES or class_type in subgraph_ids:
            continue
        nodes.setdefault(class_type, []).append(node_id)
        entry = prompt.get(node_id) if class_type in object_info else None
        if isinstance(entry, dict) and isinstance(entry.get("inputs"), dict):
            probed_entry = probed.get(node_id)
            probed_inputs = probed_entry.get("inputs") if isinstance(probed_entry, dict) else None
            add(node_id, class_type, entry["inputs"], saved, probed_inputs if isinstance(probed_inputs, dict) else {})
        else:
            # Read from the node's own widget values, in the order they are saved.
            own = [(path, value) for path, value, owner, host in places if owner == id(node) and not host]
            for found, (path, _value) in zip(_ui_model_strings(node), own):
                models.append(
                    {"node_id": node_id, "class_type": class_type, "value": found, "widget": None, "loc": path}
                )
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
    """Cloud reports a hash as ``blake3:<hex>`` or as the stored ``<hex>.<ext>``; normalize to the first.

    Anything else is no hash: for its own library models Cloud puts the folder and
    file name there (``checkpoints/sd_xl_turbo_1.0_fp16.safetensors``).
    """
    match = _HASH_RE.fullmatch(value) if isinstance(value, str) else None
    return f"blake3:{match[1]}" if match else None


def _pick_upload(value: str, uploads: list[dict[str, Any]]) -> tuple[str, str | None, list[str]]:
    """Where Cloud finds ``value`` among the person's uploads: ``(status, name, candidates)``.

    Cloud's own rule when it runs the workflow: an upload named exactly the saved
    value, else the one upload whose name ends in ``/<saved value>`` (an upload
    kept under a folder, such as ``Wan22_FunReward/model.safetensors``). Both run
    the saved value unchanged, so both are ``present``. Several names ending that
    way are ambiguous and Cloud refuses to guess. Failing those, the one upload
    with the same file name under another path is a ``remap``.
    """
    names = sorted({a["name"] for a in uploads if isinstance(a.get("name"), str)})
    if value in names:
        return "present", value, []
    under_a_folder = [n for n in names if n.endswith("/" + value)]
    if len(under_a_folder) == 1:
        return "present", under_a_folder[0], []
    if under_a_folder:
        return "missing", None, under_a_folder
    same_name = [n for n in names if _basename(n) == _basename(value)]
    if len(same_name) == 1:
        return "remap", same_name[0], []
    return "missing", None, same_name


def resolve_models(
    uses: Mapping[str, Any],
    object_info: Mapping[str, Any],
    own_models: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Place each model on Cloud.

    ``own_models(basename)`` returns every one of the person's uploaded model
    assets whose name contains ``basename``. It is asked only about models Cloud's
    dropdowns do not list, once per name.

    ``status`` is ``present`` (Cloud runs the saved value), ``remap`` (exactly one
    Cloud file has the same name in another folder: the saved value must change to
    ``cloud_value``), ``missing``, or ``not_checked`` when the node type itself is
    not on Cloud. ``source`` is ``cloud`` for Cloud's own models and ``yours`` for
    the person's uploads. ``candidates`` lists the names when several fit. A use's
    ``loc`` is kept for :func:`apply_remaps`.
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
        }
        if "loc" in use:
            row["loc"] = use["loc"]
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
        status, picked, candidates = _pick_upload(value, asked[name])
        if picked is None:
            if candidates:
                row["candidates"] = candidates
            continue
        assets = [a for a in asked[name] if a.get("name") == picked]
        sizes = {a.get("size") for a in assets}
        # Two uploads under one name: which one runs is not ours to say, so neither is proof.
        hashes = {content_hash(a.get("hash")) for a in assets}
        size = sizes.pop() if len(sizes) == 1 else None
        row.update(
            status=status,
            source="yours",
            cloud_value=picked,
            cloud_hash=hashes.pop() if len(hashes) == 1 else None,
            cloud_size=size if isinstance(size, int) else None,
        )
    return rows


def identity(
    local: list[tuple[str, int]] | None,
    cloud_size: int | None,
    cloud_hash: str | None,
    hash_file: Callable[[str], str | None],
) -> tuple[str, str | None]:
    """``same``, ``different`` or ``unverified``, and the local path compared.

    ``local`` is every local file with the model's name, as ``(path, size)``;
    ``None`` when no models folder was given. Only a blake3 match proves ``same``;
    a size mismatch alone proves ``different``. ``hash_file`` returns ``None`` for
    a file it could not read, which leaves the model ``unverified``.
    """
    if not local or len({size for _, size in local}) > 1:
        return "unverified", None
    path, size = local[0]
    if cloud_size is not None and cloud_size != size:
        return "different", path
    if cloud_hash is None:  # Cloud's library models report a path as `hash`: nothing to compare, so no hashing
        return "unverified", path
    digests = {hash_file(p) for p, _ in local}
    if None in digests or len(digests) > 1:
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
        for _, node in _ui_nodes(workflow):
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


def _write(root: Any, path: tuple, old: str, new: str) -> int:
    """Set the value saved at ``path`` from ``old`` to ``new``: 0 when it does not hold
    ``old``. A row there (an API input such as a LoRA stack's) changes each field
    that holds ``old``."""
    try:
        parent = _at(root, path[:-1])
        current = parent[path[-1]]
    except (KeyError, IndexError, TypeError):
        return 0
    if current == old:
        parent[path[-1]] = new
        return 1
    changed = 0
    if isinstance(current, dict):
        for key, value in list(current.items()):
            if value == old:
                current[key] = new
                changed += 1
    return changed


def apply_remaps(
    workflow: Mapping[str, Any], models: list[dict[str, Any]], not_applied: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], int]:
    """A copy of ``workflow`` with each remap written where its value is saved; the original is not touched.

    An API export changes the remapped input of that node. An editor save changes
    the one place :func:`find_uses` read the value from (the row's ``loc``): the
    node's own widget, the subgraph instance slot that holds a promoted value, or
    its legacy quarantined host value. The same file name saved anywhere else
    (another loader, a promoted caption) is left as it is.

    Rows that read one place (a subgraph default several instances run) write it
    once. A row with no known place, or whose place another row needs set to a
    different value, is not written; it is appended to ``not_applied`` when given.
    Returns the copy and the number of values changed.
    """
    out = copy.deepcopy(workflow)
    remaps = [m for m in models if m["status"] == "remap"]
    api = workflow_format(out) == "api"

    def place(model: Mapping[str, Any]) -> tuple | None:
        if api:
            return (model["node_id"], "inputs", model["widget"]) if model.get("widget") else None
        loc = model.get("loc")
        return tuple(loc) if loc else None

    wanted: dict[tuple, set[str]] = {}
    for model in remaps:
        at = place(model)
        if at is not None:
            wanted.setdefault(at, set()).add(model["cloud_value"])
    written: dict[tuple, int] = {}
    changed = 0
    for model in remaps:
        at = place(model)
        # One place two rows need remapped two ways has no single right value: left as saved.
        if at is not None and len(wanted[at]) == 1 and at not in written:
            written[at] = _write(out, at, model["value"], model["cloud_value"])
            changed += written[at]
        if not written.get(at) and not_applied is not None:
            not_applied.append(model)
    return out, changed
