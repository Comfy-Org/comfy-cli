"""Same model, other precision: match a missing model file to the one file a
server has that differs from it ONLY in its precision or quantization tag.

Gallery templates name a specific build of a model
(``minimax_h3_video_vae_int8_convrot.safetensors``) and a server often carries
the same weights in another precision (``minimax_h3_video_vae_fp16.safetensors``).
Without help, every load of such a template fails ``validate`` and the caller
has to find the variant by hand.

The match is deliberately strict. Only the precision run that ENDS the stem is
ignored: it must start with a real precision tag (:data:`_PRECISION_CORE`, or a
GGUF quantization on a ``.gguf``), may continue with qualifiers such as
``scaled``, and must follow the model's name. Everything else must agree
exactly: the directory, the rest of the stem (case and separators included),
and the extension. The two precision tags must differ, and exactly ONE option
may match. Anything else (a different model, a different size, another
folder, two candidate precisions) is not a match.
"""

from __future__ import annotations

import copy
import re
from typing import Any

#: Precision tags: a token that, on its own, names a numeric format.
_PRECISION_CORE = frozenset({"int4", "int8", "fp4", "fp8", "fp16", "fp32", "bf16", "nvfp4", "mxfp4"})

#: Words that only QUALIFY a precision (``fp8_e4m3fn_scaled``, ``int8_convrot``).
#: They are dropped only after a precision tag in the stem's final run; alone
#: (or before the tag) they are part of the model's name (``realesrgan_x4_scaled``).
_PRECISION_MODIFIER = frozenset({"e4m3fn", "e4m3fnuz", "e5m2", "scaled", "convrot"})

#: A GGUF quantization tag (``Q4_K_M``, ``Q8_0``, ``IQ4_XS``, ``F16``) at the
#: end of a lowercased stem. It spans separators, so it is matched whole, and
#: only on a ``.gguf`` file: elsewhere the same letters are not a quantization.
_GGUF_TAIL = re.compile(
    r"(?:^|[_\-.])(?P<tok>iq[1-4]_(?:xxs|xs|s|m|nl)|iq[1-4]|q[2-8]_k(?:_[sml])?|q[4-8]_[01]|f16|f32|bf16)$"
)
#: The last separator-delimited word of a stem.
_WORD_TAIL = re.compile(r"(?:^|[_\-.])(?P<tok>[^_\-.]+)$")

#: File extensions a model loader option carries.
MODEL_FILE = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$", re.IGNORECASE)


class ModelVariantResolutionError(RuntimeError):
    """A malformed promoted-widget graph prevented safe model substitution."""


def _parse(name: str) -> tuple[tuple[str, str, str], tuple[str, ...]] | None:
    """``((directory, name, extension), precision tag)`` for a model filename.

    The tag is the stem's trailing precision run (lowercased, ``()`` when there
    is none); ``name`` is the stem before it, verbatim. ``None`` when ``name``
    is not a model file."""
    path = name.replace("\\", "/")
    directory, _, base = path.rpartition("/")
    m = MODEL_FILE.search(base)
    if m is None or m.start() == 0:
        return None
    stem, ext = base[: m.start()], m.group(1).lower()
    low = stem.lower()
    # Peel words off the end while they are precision parts, newest first.
    run: list[tuple[int, str, bool]] = []  # (start incl. separator, token, is a precision tag)
    end = len(low)
    while end > 0:
        q = _GGUF_TAIL.search(low, 0, end) if ext == "gguf" else None
        if q is not None:
            run.append((q.start(), q.group("tok").replace("-", "_").replace(".", "_"), True))
            end = q.start()
            continue
        w = _WORD_TAIL.search(low, 0, end)
        if w is None or w.group("tok") not in _PRECISION_CORE | _PRECISION_MODIFIER:
            break
        run.append((w.start(), w.group("tok"), w.group("tok") in _PRECISION_CORE))
        end = w.start()
    run.reverse()
    # The tag starts at the run's first real precision; a qualifier before it
    # belongs to the name. A run that is the whole stem IS the name (``fp16``).
    first = next((i for i, (_, _, core) in enumerate(run) if core), None)
    if first is None or run[first][0] == 0:
        return (directory, stem, ext), ()
    return (directory, stem[: run[first][0]], ext), tuple(t for _, t, _ in run[first:])


def precision_key(name: str) -> tuple[str, str, str] | None:
    """``(directory, stem without its trailing precision tag, extension)`` for
    a model filename, or ``None`` when ``name`` is not a model file. Two files
    with the same key and different tags are the same model in two precisions."""
    parsed = _parse(name)
    return parsed[0] if parsed else None


def precision_sibling(value: Any, options: list[Any]) -> str | None:
    """The ONE option that is ``value`` in another precision, else ``None``.

    ``None`` too when ``value`` is already an option, is not a model filename,
    or two or more options would match (which precision to pick is then a
    choice, not a correction). An option must sit in the same directory and
    carry a DIFFERENT precision tag: a name that differs only in case or
    separators is another file, not another precision.
    """
    if not isinstance(value, str):
        return None
    strs = [o for o in options if isinstance(o, str)]
    if value in strs:
        return None
    parsed = _parse(value)
    if parsed is None:
        return None
    key, tag = parsed
    hits = set()
    for o in strs:
        p = _parse(o)
        if p is not None and p[0] == key and p[1] != tag:
            hits.add(o)
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
    download list, follow the rewrite. Top-level-only replacements are applied
    directly; a replacement that must propagate through promoted subgraph
    values is staged on a copy so a traversal failure leaves the input intact.
    """
    from comfy_cli.cql.promoted import PromotionTraversalLimitError

    try:
        candidate = copy.deepcopy(workflow)
        result = _resolve_workflow_models(candidate, graph)
    except (PromotionTraversalLimitError, RecursionError) as exc:
        raise ModelVariantResolutionError(str(exc)) from exc
    workflow.clear()
    workflow.update(candidate)
    return result


def _resolve_workflow_models(workflow: dict, graph) -> tuple[list[dict], list[dict]]:
    """Transactional implementation for :func:`resolve_workflow_models`."""

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
    raw_models = props.get("models") if isinstance(props, dict) else None
    for entry in raw_models if isinstance(raw_models, list) else []:
        if isinstance(entry, dict) and entry.get("name") == old:
            entry["name"] = new
            entry.pop("url", None)
            entry.pop("hash", None)


def _rename_promoted(workflow: dict, renamed: dict[str, dict[tuple[str, str], tuple[str, str]]]) -> None:
    """Follow a swap onto each instance's promoted copy of that exact widget.

    Only the host slot the frontend binds to the swapped interior widget
    (``PromotedInput.value_index``) changes, and only while it still holds the
    old filename; another promoted widget holding the same text is left alone.
    A rewritten slot on an instance that itself sits inside a definition is a
    swap of that definition too, so a widget promoted through nested subgraphs
    is followed out to the outermost host."""
    from comfy_cli.cql.promoted import _MAX_NESTED_PROMOTION_DEPTH, defs_by_id, promoted_inputs

    defs = defs_by_id(workflow)
    for _ in range(_MAX_NESTED_PROMOTION_DEPTH + 1):
        grown = False
        for node, sg_loc in _workflow_nodes(workflow):
            sg_id = str(node.get("type", ""))
            swaps, wv = renamed.get(sg_id), node.get("widgets_values")
            if not swaps or not isinstance(wv, list) or sg_id not in defs:
                continue
            for pi in promoted_inputs(defs[sg_id], defs):
                if not pi.is_widget:
                    continue
                key = (str(pi.source_node), f"promoted:{pi.source_input}" if pi.nested else str(pi.source_widget))
                swap = swaps.get(key)
                if swap and pi.value_index < len(wv) and wv[pi.value_index] == swap[0]:
                    wv[pi.value_index] = swap[1]
                    if sg_loc is not None:
                        renamed.setdefault(sg_loc, {})[(str(node.get("id")), f"promoted:{pi.name}")] = swap
                        grown = True
        if not grown:
            return
    raise ModelVariantResolutionError("promoted model propagation exceeded its nesting safe limit")
