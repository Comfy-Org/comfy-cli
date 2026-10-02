"""Same model, other precision: match a missing model file to the one file a
server has that differs from it ONLY in its precision or quantization tag.

Gallery templates name a specific build of a model
(``minimax_h3_video_vae_int8_convrot.safetensors``) and a server often carries
the same weights in another precision (``minimax_h3_video_vae_fp16.safetensors``).
Without help, every load of such a template fails ``validate`` and an agent
spends a round finding the variant by hand.

The match is deliberately strict: only the tokens in :data:`_PRECISION_TOKEN`
are ignored, they must stand alone between separators, the extension must
agree, and exactly ONE option may match. Anything else (a different model, a
different size, two candidate precisions) is not a match.
"""

from __future__ import annotations

import re
from typing import Any

#: Precision / quantization tags. Each must be a whole ``_``/``-``/``.``
#: separated token: ``foo_int8_convrot`` drops both, ``fooINT8CONVROT`` drops
#: nothing.
_PRECISION_TOKEN = re.compile(
    r"(?<![a-z0-9])(?:int4|int8|fp4|fp8|fp16|fp32|bf16|nvfp4|mxfp4|e4m3fn|e4m3fnuz|e5m2|scaled|convrot)(?![a-z0-9])",
    re.IGNORECASE,
)

#: File extensions a model loader option carries.
MODEL_FILE = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$", re.IGNORECASE)


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def precision_key(name: str) -> tuple[str, str] | None:
    """``(stem without precision tags, extension)`` for a model filename, or
    ``None`` when ``name`` is not a model file or nothing is left of its stem."""
    base = _basename(name)
    m = MODEL_FILE.search(base)
    if m is None:
        return None
    stem = base[: m.start()].lower()
    stripped = re.sub(r"[_\-.]+", "_", _PRECISION_TOKEN.sub("", stem)).strip("_")
    if not stripped:
        return None
    return stripped, m.group(1).lower()


def precision_sibling(value: Any, options: list[Any]) -> str | None:
    """The ONE option that is ``value`` in another precision, else ``None``.

    ``None`` too when ``value`` is already an option, is not a model filename,
    or two or more options would match (which precision to pick is then a
    choice, not a correction).
    """
    if not isinstance(value, str):
        return None
    strs = [o for o in options if isinstance(o, str)]
    if value in strs:
        return None
    key = precision_key(value)
    if key is None:
        return None
    hits = {o for o in strs if precision_key(o) == key}
    return hits.pop() if len(hits) == 1 else None


def _workflow_nodes(workflow: dict) -> list[tuple[dict, str | None]]:
    """Every node of a frontend-format workflow with the id of the subgraph
    definition it sits in (``None`` at the top level)."""
    out: list[tuple[dict, str | None]] = [(n, None) for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    defs = workflow.get("definitions")
    for sg in (defs.get("subgraphs") if isinstance(defs, dict) else None) or []:
        if isinstance(sg, dict):
            out += [(n, str(sg.get("id"))) for n in sg.get("nodes") or [] if isinstance(n, dict)]
    return out


def _model_widgets(node: dict, graph) -> list[tuple[Any, Any, str, Any]]:
    """``(key, port, field, value)`` for each model-file COMBO value on ``node``,
    ``key`` indexing its ``widgets_values`` (a position, or a name for the
    dict form some custom nodes save)."""
    m = graph.node(str(node.get("type", "")))
    wv = node.get("widgets_values")
    if m is None or not isinstance(wv, list | dict):
        return []
    ports = {p.name: p for p in m.inputs}
    if isinstance(wv, list):
        names = graph.widget_order_for_node(node["type"], wv)
        pairs = [(i, names[i]) for i in range(min(len(wv), len(names)))]
    else:
        pairs = [(k, k) for k in wv]
    out = []
    for key, field in pairs:
        value, port = wv[key], ports.get(field)
        if not (isinstance(value, str) and MODEL_FILE.search(value)):
            continue
        if port is None or port.type != "COMBO" or not port.enum_values or port.is_upload_backed:
            continue
        out.append((key, port, field, value))
    return out


def resolve_workflow_models(workflow: dict, graph) -> tuple[list[dict], list[dict]]:
    """Point every model file the server lacks at its precision sibling.

    Rewrites ``workflow`` in place and returns ``(substitutions, unavailable)``:
    one ``normalized_value`` warning per rewritten widget, and one
    ``model_unavailable`` warning (with the closest options) per model file
    that has no unique sibling and is left as it was. A subgraph instance that
    carries the same filename as a promoted value, and the ``properties.models``
    download list, follow the rewrite.
    """
    subs: list[dict] = []
    unavailable: list[dict] = []
    # Per subgraph definition: (interior node id, widget) -> (old, new). An
    # instance of that definition may carry the value as a promoted widget.
    renamed: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
    for node, sg_id in _workflow_nodes(workflow):
        for key, port, field, value in _model_widgets(node, graph):
            if value in {str(o) for o in port.enum_values}:
                continue
            where = {"node_id": node.get("id"), "class_type": node.get("type"), "field": field}
            if sg_id is not None:
                where["subgraph"] = sg_id
            sibling = precision_sibling(value, list(port.enum_values))
            if sibling is None:
                closest = port.suggest_combo(value)
                unavailable.append(
                    {
                        "code": "model_unavailable",
                        **where,
                        "value": value,
                        "message": f"{value!r} ({field}) is not installed on this server"
                        + (f" — closest: {', '.join(closest)}" if closest else ""),
                        "did_you_mean": closest,
                    }
                )
                continue
            node["widgets_values"][key] = sibling
            _rename_download(node, value, sibling)
            if sg_id is not None:
                renamed.setdefault(sg_id, {})[(str(node.get("id")), field)] = (value, sibling)
            subs.append(
                {
                    "code": "normalized_value",
                    **where,
                    "from": value,
                    "to": sibling,
                    "message": f"{value!r} is not installed on this server; using {sibling!r}, "
                    "the same model in another precision",
                }
            )
    if renamed:
        _rename_promoted(workflow, renamed)
    return subs, unavailable


def _rename_download(node: dict, old: str, new: str) -> None:
    """Point the node's ``properties.models`` entry for ``old`` at ``new``.

    The download URL and hash describe the replaced file, so they go."""
    props = node.get("properties")
    for entry in (props.get("models") if isinstance(props, dict) else None) or []:
        if isinstance(entry, dict) and entry.get("name") == old:
            entry["name"] = new
            entry.pop("url", None)
            entry.pop("hash", None)


def _rename_promoted(workflow: dict, renamed: dict[str, dict[tuple[str, str], tuple[str, str]]]) -> None:
    """Follow a swap onto each instance's promoted copy of that exact widget.

    Only the host slot the frontend binds to the swapped interior widget
    (``PromotedInput.value_index``) changes, and only while it still holds the
    old filename; another promoted widget holding the same text is left alone."""
    from comfy_cli.cql.promoted import defs_by_id, promoted_inputs

    defs = defs_by_id(workflow)
    for node, _sg in _workflow_nodes(workflow):
        sg_id = str(node.get("type", ""))
        swaps, wv = renamed.get(sg_id), node.get("widgets_values")
        if not swaps or not isinstance(wv, list) or sg_id not in defs:
            continue
        for pi in promoted_inputs(defs[sg_id], defs):
            swap = swaps.get((str(pi.source_node), str(pi.source_widget))) if pi.is_widget and not pi.nested else None
            if swap and pi.value_index < len(wv) and wv[pi.value_index] == swap[0]:
                wv[pi.value_index] = swap[1]
