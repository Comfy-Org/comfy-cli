"""CRDT-ready structured edit operations over frontend-format ComfyUI graphs.

This is the op-model the agent (and a human via the CLI) uses to mutate a
workflow. Every primitive returns ``(workflow, op)`` where ``op`` is a
self-describing, replayable operation. The same op stream feeds both a
single-writer file edit (locally) and a merge consumer (cloud) — the CLI never
merges; it emits ops that *converge under replay* and, for the residual cases a
leaderless writer cannot decide alone, flags a conflict rather than silently
diverging.

Design (settled by the identity spike):

* **Identity is leaderless & collision-free.** New node/link ids are random
  53-bit integers (``mint_id``): no shared counter, no coordination, and still
  ``int``-typed so the API converter (which gates link ids on ``isinstance(int)``)
  and an int-keyed frontend keep working. ``last_node_id``/``last_link_id`` are
  kept only as advisory high-water marks, never as allocators.
* **Widgets are name-addressed, never index-addressed.** ``set_widget`` carries
  the widget *name*; ``apply_op`` resolves name → ``widgets_values`` index against
  the live schema at apply time, so an op survives widget-layout drift.

Convergence guarantees ``apply_op`` upholds, so any replay order reaches the same
``canonical`` graph (proved by the P8..P11 order-independence tests):

* **Idempotent** — an op whose ``op_id`` was already applied is a no-op.
* **Total** — a write (``set_widget``/``connect``) to a node that was
  concurrently deleted is a no-op: delete wins. Apply never raises on a
  since-removed target, so a merge consumer can replay a delete and an edge/edit
  in either order.
* **Last-writer-wins on widgets** — two concurrent writes to the same widget
  converge on the value with the higher causal ``stamp`` ``[base_version, actor]``
  (``op_id`` breaks exact ties into a total order), independent of apply order.
  The winning stamp per target is tracked in ``_widget_stamps`` (apply-only
  bookkeeping, stripped before serialization).
* **Deterministic structure** — forking a shared subgraph definition mints an id
  derived from ``(definition, instance)`` (not a random UUID), so two replicas
  replaying the same ops produce byte-identical graphs.
* **Non-clobbering autogrow** — a ``COMFY_AUTOGROW_V3`` connect never overwrites a
  slot already wired to another link; it grows a fresh slot keyed by ``grow_id``
  (the link id). Two concurrent autogrow connects both survive and ``canonical``
  compares grown slots by ``grow_id``, not by list position.

The one thing a leaderless writer genuinely *cannot* converge is a *sequence
decision*: the human-visible ordering/numbering of concurrently-grown autogrow
slots (a batch's element order) and of concurrent interior writes to the same
shared subgraph definition. Those are surfaced by :func:`detect_conflict` for the
merge consumer / ask-to-merge to resolve — the ops still never lose data, and
``canonical`` treats the order as immaterial, so the semantic graph converges even
while the display order does not.
"""

from __future__ import annotations

import copy
import json
import random
import re
import uuid
from typing import Any

from comfy_cli import layout

# New ids live in [2**40, 2**53): always large (never collides with small
# frontend counter ids), always inside JS Number.MAX_SAFE_INTEGER.
_ID_FLOOR = 1 << 40


def mint_id() -> int:
    """A leaderless, collision-free, int-typed identity for a node or link."""
    return _ID_FLOOR | random.getrandbits(52)


def _new_op(kind: str, actor: str, base_version: int, **fields: Any) -> dict[str, Any]:
    return {
        "op": kind,
        "op_id": uuid.uuid4().hex,
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        **fields,
    }


# ---------------------------------------------------------------------------
# The frozen op vocabulary — the normative contract is docs/op-vocabulary-v1.md;
# these constants are its machine-readable projection, and
# tests/comfy_cli/test_op_vocabulary_contract.py pins doc == constants == the
# dispatch tables in apply_op / apply_specs. Amend the doc (versioned amendment
# section) before touching any of the three.
# ---------------------------------------------------------------------------

#: Every op kind in the v1 vocabulary, including defined-but-deferred kinds.
FROZEN_OPS: tuple[str, ...] = ("add_node", "connect", "set_widget", "delete_node", "clear", "reset_doc")

#: Kinds frozen in the contract whose replay is not implemented yet.
#: ``apply_op`` must keep rejecting these. Empty since amendment v1.1:
#: ``reset_doc`` was un-deferred by the bulk-writers ticket (V1-038).
DEFERRED_OPS: tuple[str, ...] = ()

#: Kinds a batch (``apply_specs``) dispatches. ``clear`` and ``reset_doc`` are
#: standalone-only: they rewrite the whole document, so they never ride inside
#: an atomic batch.
BATCHABLE_OPS: tuple[str, ...] = ("add_node", "connect", "set_widget", "delete_node")

#: Per-kind rendering for :class:`NotBatchableError` — the registered error code
#: and the standalone command that DOES do the job. One entry per frozen kind
#: outside ``BATCHABLE_OPS``; the contract test pins that correspondence.
_NOT_BATCHABLE: dict[str, dict[str, str]] = {
    "clear": {
        "code": "workflow_clear_not_batchable",
        "command": "comfy workflow clear <file>",
        "does": "wipes the whole graph",
    },
    "reset_doc": {
        "code": "workflow_reset_doc_not_batchable",
        "command": "comfy workflow reset-doc <file> --confirm",
        "does": "resets the whole document to the empty baseline and erases its replay history",
    },
}


class NotBatchableError(ValueError):
    """A frozen op kind that is standalone-only was submitted inside a batch.

    The command layer renders this with the registered ``code``/``hint`` below
    (see ``comfy_cli/error_codes.py``) instead of the generic
    ``workflow_edit_invalid``, so a caller learns the exact standalone command
    to run rather than re-trying the batch.

    ``code``/``hint`` are per-kind INSTANCE attributes; the class attributes are
    the ``clear`` values, kept so existing callers that read
    ``NotBatchableError.code`` off the class still resolve.
    """

    code = "workflow_clear_not_batchable"
    hint = "run the standalone `comfy workflow clear <file>` first, then apply the remaining ops as a batch"

    def __init__(self, index: int, kind: str = "clear"):
        entry = _NOT_BATCHABLE.get(kind, _NOT_BATCHABLE["clear"])
        command = entry["command"]
        self.code = entry["code"]
        self.kind = kind
        self.hint = f"run the standalone `{command}` first, then apply the remaining ops as a batch"
        super().__init__(
            f"spec #{index}: `{kind}` {entry['does']} and is standalone-only (op-vocabulary-v1: "
            "batchable = no) — it never rides inside a batch. No changes were applied — the batch was "
            f"discarded. Run `{command}` as its own command, then apply the remaining ops."
        )


# Node types that live only in the UI graph and never reach the API — the
# frontend's isVirtualNode set. Mirrors workflow_to_api._UI_ONLY_NODE_TYPES;
# duplicated rather than imported to keep workflow_ops import-free of the
# converter. Keep the two in sync.
UI_ONLY_NODE_TYPES = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "GetNode", "SetNode", "Reroute"})

# A subgraph INSTANCE's node `type` is the UUID id of its definition, and
# `ls-nodes` prints that verbatim — so a caller reading ls-nodes output can
# mistake it for a class name. There is no instantiate-a-subgraph command, so
# such an add can never succeed; say why instead of "unknown node type".
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


class UnknownNodeType(ValueError):
    """add_node was given a class_type the catalog does not have.

    Carries the machine-readable detail the command layer needs to emit the same
    envelope `nodes show` does (code=node_not_found + details.close_matches), so
    a caller can self-correct in one retry. Before this, add-node emitted a bare
    workflow_edit_invalid with a hint pointing at `comfy nodes types` — which
    lists CONNECTION types, not class_types.
    """

    def __init__(
        self,
        class_type: str,
        *,
        close_matches: list[str] | None = None,
        ui_only: bool = False,
        subgraph_id: bool = False,
    ):
        self.class_type = class_type
        self.close_matches = close_matches or []
        self.ui_only = ui_only
        self.subgraph_id = subgraph_id
        if ui_only:
            msg = (
                f"{class_type!r} is a UI-only node (it exists in the editor graph but never reaches the API), "
                "so it cannot be added through this surface"
            )
        elif subgraph_id:
            msg = (
                f"{class_type!r} is a subgraph INSTANCE id, not a node class. `ls-nodes` prints a subgraph "
                "instance's definition uuid as its type; there is no command to instantiate a subgraph"
            )
        else:
            msg = f"unknown node type {class_type!r}"
        super().__init__(msg)


class DeprecatedNodeType(ValueError):
    """add_node was given a class the catalog marks ``deprecated``.

    ``replacement`` is the live class with the same display name, when one
    exists (ComfyUI retires a node by suffixing its display name with
    "(DEPRECATED)" or "(Legacy)" and registering the successor under the
    bare name).
    """

    code = "node_deprecated"

    def __init__(self, class_type: str, *, replacement: str | None):
        self.class_type = class_type
        self.replacement = replacement
        use = f"use {replacement!r} instead" if replacement else "run `comfy nodes search <text>` for a live class"
        self.hint = (
            f'{use}; to add {class_type!r} anyway set "allow_deprecated": true on the add_node op '
            "(or pass --allow-deprecated to `comfy workflow add-node`)"
        )
        super().__init__(f"{class_type!r} is deprecated")


_DEPRECATED_DISPLAY_SUFFIX = re.compile(r"\s*\((?:deprecated|legacy)\)\s*$", re.I)


def _deprecated_replacement(graph, m) -> str | None:
    want = _DEPRECATED_DISPLAY_SUFFIX.sub("", m.display_name).strip().lower()
    # A free node and a paid partner node can share a display name; never
    # answer a free class with one that bills credits (or vice versa).
    for n in graph.all_nodes():
        if not n.deprecated and n.is_api_node == m.is_api_node and n.display_name.strip().lower() == want:
            return n.id
    return None


def _find(workflow: dict, node_id: Any) -> dict | None:
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") == node_id:
            return n
    return None


def _require(workflow: dict, node_id: Any) -> dict:
    n = _find(workflow, node_id)
    if n is None:
        raise ValueError(f"node {node_id} not found in workflow")
    return n


def _available_nodes_hint(workflow: dict, *, limit: int = 12) -> str:
    """Compact ``id (type)`` list of nodes that DO exist — to correct a
    mistargeted node id."""
    out: list[str] = []
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") is not None:
            out.append(f"{n.get('id')} ({n.get('type', '?')})")
            if len(out) >= limit:
                out.append("…")
                break
    return ", ".join(out)


def _slot_types(t: Any) -> set[str]:
    """Split a slot type into the set of types it accepts.

    ComfyUI expresses a multi-type input as a COMMA-SEPARATED UNION, e.g.
    ``MESH,FILE_3D_GLB,FILE_3D_GLTF,...`` on a 3D importer's ``mesh`` input.
    """
    return {p.strip() for p in str(t or "").split(",") if p.strip()}


def _types_compatible(link_type: Any, dst_type: Any) -> bool:
    """Whether an output of ``link_type`` may drive an input of ``dst_type``.

    Compatible when the two type sets INTERSECT, not when their raw strings are
    equal: comparing whole strings refused a ``FILE_3D_GLB`` output from an input
    declaring ``MESH,FILE_3D_GLB,...`` even though it explicitly accepts it.
    Measured on prod agent traces (2026-07-23 → 07-28): ~23 connect/apply_ops
    failures of that shape, each one a correct edit being refused — and the error
    names the source type inside the accepted list, so the agent saw its own type
    listed and retried the identical call.

    Intersection (not substring) matters: ``FILE_3D_GL`` must NOT satisfy an input
    accepting only ``FILE_3D_GLTF``/``FILE_3D_GLB``. An unknown type on either
    side, or a ``*`` wildcard, stays permissive as before.
    """
    src, dst = _slot_types(link_type), _slot_types(dst_type)
    if not src or not dst:
        return True
    if "*" in src or "*" in dst:
        return True
    return bool(src & dst)


def _enrich_resolution_error(e: ValueError, workflow: dict, graph, *, widget: Any = None) -> ValueError:
    """Turn a *not-found* edit error into an actionable one.

    An LLM editing a graph tends to rebuild an identifier from memory instead of
    copying it from ``comfy workflow slots`` — and a wrong id often lands on a
    real *sibling* (e.g. ``285/288.vae_name`` hits a CLIPLoader when the VAELoader
    is ``285/29``), so the edit fails or, worse, silently mis-targets. When the
    widget name is known we scan the workflow for the address that actually
    carries it; otherwise we list the node ids that exist. Shape/enum/type
    errors (the target resolved fine) pass through unchanged.
    """
    msg = str(e)
    if "not found" not in msg:
        return e
    if widget:
        from comfy_cli.cql.engine import _suggest_slots_for_input

        addrs = _suggest_slots_for_input(workflow, str(widget), graph)
        if addrs:
            return ValueError(
                f"{msg}. Did you mean: {'; '.join(addrs)}? "
                "Copy the address verbatim from `comfy workflow slots` — never rebuild it."
            )
    nodes_hint = _available_nodes_hint(workflow)
    if nodes_hint:
        return ValueError(
            f"{msg}. Nodes in this workflow: {nodes_hint}. "
            "Use an id from `comfy workflow slots` / `ls-nodes` — never rebuild it."
        )
    return e


# Both enrichment forms above append their hint at the END of the message, so
# truncating from the first marker strips the whole (now-stale) clause.
_MIDBATCH_HINT_RE = re.compile(r"\.\s*(?:Nodes in this workflow:|Did you mean:).*\Z", re.S)


def _rehint_discarded_batch(e: Exception, pre_batch_hint: str) -> ValueError:
    """Re-render a batch failure's identifier hint against the PRE-batch graph.

    ``apply_specs`` threads an accumulating ``workflow`` through the batch, so by
    the time spec #N fails that dict already holds the nodes added by specs
    #0..N-1, and :func:`_enrich_resolution_error` renders its "Nodes in this
    workflow" / "Did you mean" inventory from it. Every caller then discards that
    graph — ``apply`` is atomic, ``foreach`` drops the failing param-set — so the
    ids in the hint are fictional the moment the command returns.

    The hint is phrased as an instruction ("Use an id from ... never rebuild
    it"), so a model reasonably treats those ids as authoritative and addresses
    them next. Measured on prod comfy-agent traces (2026-07-27): 16/16 of every
    "node <id> not found in workflow" edit failure used an id an earlier FAILED
    batch had advertised this way; one trace burned seven consecutive connects
    on ids that never existed.

    So: strip the mid-batch inventory, restate it from the graph as it actually
    stands, and say plainly that nothing was applied.
    """
    # The regex eats the separator before the stripped clause, so re-punctuate
    # rather than emit "inputs: ['images'] No changes were applied".
    msg = _MIDBATCH_HINT_RE.sub("", str(e)).rstrip().rstrip(".")
    suffix = ". No changes were applied — the batch was discarded."
    if pre_batch_hint:
        suffix += (
            f" The workflow still contains: {pre_batch_hint}. "
            "Any node id minted by this batch is gone; re-read ids with "
            "`comfy workflow slots` / `ls-nodes` before addressing nodes."
        )
    return ValueError(msg + suffix)


def _find_by_str(workflow: dict, node_id: Any) -> dict | None:
    """Locate a node comparing ids as strings — subgraph op paths carry string
    ids while top-level node ids are ints."""
    s = str(node_id)
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and str(n.get("id", "")) == s:
            return n
    return None


def _stamp_key(op: dict) -> list:
    """A total causal order for last-writer-wins: higher ``base_version`` wins,
    ties broken by ``actor`` then the unique ``op_id`` (so no two distinct ops
    ever compare equal)."""
    stamp = op.get("stamp") or [op.get("base_version", 0), op.get("actor", "")]
    return [stamp[0], stamp[1], op["op_id"]]


def _lww_gate(workflow: dict, op: dict) -> bool:
    """True iff this op's write should apply under last-writer-wins. A write to a
    target already claimed by a higher-or-equal stamp is dropped, making the
    surviving value independent of apply order.

    Gated targets (``_write_target``): ``set_widget``'s ``("widget", …)``, the
    connect-embedded ``inputcount`` bump that shares a connect's stamp (§8.4),
    and — since amendment v1.2 — a concrete connect's ``("input", to_node,
    to_slot)``. The register store is still spelled ``_widget_stamps`` for
    on-the-wire compatibility with documents written before v1.2; it holds every
    gated target, not just widgets."""
    prior = workflow.get("_widget_stamps", {}).get(json.dumps(_write_target(op), default=str))
    return prior is None or _stamp_key(op) > list(prior)


def _lww_commit(workflow: dict, op: dict) -> None:
    """Record this op's stamp as the winner for its target."""
    workflow.setdefault("_widget_stamps", {})[json.dumps(_write_target(op), default=str)] = _stamp_key(op)


def _autogrow_template(graph, node_type: str, base: str) -> dict | None:
    """The schema-declared element-naming template for the ``base`` autogrow
    input on ``node_type`` — looked up from the same object_info-derived
    ``graph`` the connect path already resolves node schemas from (see
    ``cql.engine.Port.autogrow_template``). None when ``graph`` is unavailable,
    ``node_type`` isn't in the catalog (offline edit), or the catalog entry
    carries no template — callers then fall back to the historical
    pluralization heuristic in :func:`_autogrow_elem_name`."""
    if graph is None:
        return None
    m = graph.node(node_type)
    if m is None:
        return None
    for p in m.inputs:
        if p.name == base and p.is_autogrow:
            return p.autogrow_template
    return None


def _autogrow_elem_name(base: str, n: int, template: dict | None) -> str:
    """The 0-based Nth autogrow element name for ``base``. Comes from the node
    schema when known — ``template["names"][N]`` verbatim (overflow past the
    list keeps growing as ``f"{names[-1]}{n}"``), else ``f"{prefix}{n}"`` —
    falling back to the historical ``{base[:-1]}{n}`` pluralization guess only
    when ``template`` is None (schema unavailable: offline edit, catalog miss)."""
    if template:
        names = template.get("names")
        if names:
            return names[n] if n < len(names) else f"{names[-1]}{n}"
        prefix = template.get("prefix")
        if prefix:
            return f"{prefix}{n}"
    stem = base[:-1] if base.endswith("s") else base
    return f"{stem}{n}"


def _first_free_autogrow_index(taken: set, base: str, template: dict | None) -> int:
    """The lowest N whose ``{base}.{elem(N)}`` name is not already present.

    Both autogrow namers used to seed N from ``len(inputs starting with base.)``,
    which is only correct when the existing slots are a complete, gapless,
    schema-conforming run. Legacy workflows routinely aren't:

    * a gap (``images.image0`` + ``images.image2``) counts 2 and mints a SECOND
      ``images.image2``, clobbering a wired slot;
    * a non-conforming sibling (``images.foo``) counts 1 and skips
      ``images.image0`` entirely.

    Counting names we'd actually mint — rather than inputs that merely share the
    prefix — is immune to both, and keeps the server's sequential convention by
    filling the lowest free slot.
    """
    n = 0
    while f"{base}.{_autogrow_elem_name(base, n, template)}" in taken:
        n += 1
    return n


def _next_autogrow_name(ins: list, requested: str, template: dict | None = None) -> str:
    """A free autogrow slot name. Prefer the op's requested name; if a concurrent
    connect already took it, grow the next sequential schema-derived slot (see
    :func:`_autogrow_elem_name`) so no slot is ever clobbered (the server
    convention stays sequential)."""
    taken = {i.get("name") for i in ins}
    if requested not in taken:
        return requested
    base = requested.split(".", 1)[0]
    n = _first_free_autogrow_index(taken, base, template)
    return f"{base}.{_autogrow_elem_name(base, n, template)}"


def _next_inputcount_name(ins: list, requested: str) -> str:
    """A free ``inputcount``-family slot name (bare ``{elem}_N``, NOT the
    dotted ``base.elemN`` autogrow shape). Prefers the op's requested
    (mint-time-planned) name; if a concurrent connect already claimed it,
    grows the next free bare key instead. Bare keys are this family's actual
    wire address (see :func:`_inputcount_family_match`), so collision
    resolution must stay bare too — reusing :func:`_next_autogrow_name` here
    would mint a dotted name (``image_3.image_30``) the server can't map."""
    taken = {i.get("name") for i in ins}
    if requested not in taken:
        return requested
    elem, _, n_str = requested.rpartition("_")
    n = int(n_str) if n_str.isdigit() else 1
    name = f"{elem}_{n}"
    while name in taken:
        n += 1
        name = f"{elem}_{n}"
    return name


# ---------------------------------------------------------------------------
# primitives — each returns (workflow, op); the op is applied via apply_op so
# apply(base, op) == primitive(base) holds by construction (P1 fidelity).
# ---------------------------------------------------------------------------


# Litegraph node modes: 0 always, 1 on-event, 2 never (mute), 3 on-trigger,
# 4 bypass. Mirrors workflow_to_api._MODE_MUTED/_MODE_BYPASS and the
# _MODE_LABELS table in workflow_edit's ls-nodes.
_VALID_NODE_MODES = frozenset({0, 1, 2, 3, 4})


def add_node(
    workflow: dict,
    graph,
    class_type: str,
    *,
    pos: list | None = None,
    mode: int = 0,
    actor: str = "cli",
    base_version: int = 0,
    allow_deprecated: bool = False,
) -> tuple[dict, dict]:
    m = graph.node(class_type)
    if m is None:
        if class_type in UI_ONLY_NODE_TYPES:
            raise UnknownNodeType(class_type, ui_only=True)
        if _UUID_RE.match(class_type.strip()):
            raise UnknownNodeType(class_type, subgraph_id=True)
        import difflib

        names = [n.id for n in graph.all_nodes()]
        raise UnknownNodeType(class_type, close_matches=difflib.get_close_matches(class_type, names, n=5, cutoff=0.6))
    if m.deprecated and not allow_deprecated:
        raise DeprecatedNodeType(class_type, replacement=_deprecated_replacement(graph, m))
    size = layout.estimate_size(
        len([p for p in m.inputs if p.is_link]),
        len(m.outputs),
        len(graph.widget_order_default(class_type)),
    )
    if pos is None:
        # Layout-aware default: right of the current graph, collision-free.
        # Decided at mint time so the position freezes into the op and replay
        # stays convergent (P1). Existing nodes are never moved.
        pos = layout.cascade_pos(workflow, size)
    node = _build_node(mint_id(), class_type, m, graph, pos, size)
    if mode:
        # Node mode (mute/bypass) is graph-semantic state — a bypassed node
        # executes differently — so it must survive capture→apply. op.node is
        # authoritative for replay (§8.5), so stamping the node covers it; the
        # explicit op field keeps the receipt inspectable.
        if not isinstance(mode, int) or isinstance(mode, bool) or mode not in _VALID_NODE_MODES:
            raise ValueError(
                f"invalid node mode {mode!r}; valid: 0 (always), 1 (on-event), 2 (mute), 3 (on-trigger), 4 (bypass)"
            )
        node["mode"] = mode
    op = _new_op(
        "add_node",
        actor,
        base_version,
        node_id=node["id"],
        class_type=class_type,
        pos=node["pos"],
        node=node,
        **({"mode": mode} if mode else {}),
    )
    return apply_op(workflow, op, graph), op


def set_widget(
    workflow: dict,
    graph,
    node_id: Any,
    widget: str,
    value: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    """Set a widget, enriching a not-found node/widget error with the real
    address that carries ``widget`` so a mistargeted edit self-corrects in one
    step (see :func:`_enrich_resolution_error`)."""
    try:
        return _set_widget_impl(workflow, graph, node_id, widget, value, actor=actor, base_version=base_version)
    except ValueError as e:
        raise _enrich_resolution_error(e, workflow, graph, widget=widget) from e


def _normalize_combo(graph, class_type: str, widget: str, value: Any) -> tuple[Any, dict | None]:
    """Rewrite a mangled model/COMBO value to the real option it means so the
    model actually loads (e.g. ``checkpoints/wai-illustrious-sdxl.safetensors`` →
    ``wai-illustrious-sdxl.safetensors``). Returns ``(value, note)`` — ``note`` is
    an informational warning when a rewrite happened, else ``None``. Only an
    UNAMBIGUOUS match is rewritten; anything else is left untouched so validate's
    ``unknown_enum_value`` (with ``did_you_mean``) still fires.
    """
    m = graph.node(class_type)
    if m is None:
        return value, None
    port = next((p for p in m.inputs if p.name == widget), None)
    if port is None:
        return value, None
    canon = port.canonical_combo(value)
    if canon is None or canon == value:
        return value, None
    return canon, {
        "code": "normalized_value",
        "field": widget,
        "message": f"{value!r} is not an exact option; using the matching model {canon!r}",
        "from": str(value),
        "to": canon,
    }


def _set_widget_impl(
    workflow: dict,
    graph,
    node_id: Any,
    widget: str,
    value: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    from comfy_cli.cql import engine as _engine

    # Subgraph-aware: a subgraph instance's *promoted* input (flat ``57.text`` —
    # exactly what ``comfy workflow slots`` advertises) or an interior node
    # (nested ``57/27.text``) resolves INTO the subgraph definition. Both forms
    # reuse the CQL engine's slot resolver so set-widget and slots agree. The op
    # carries the resolved interior ``path`` + ``inner_widget`` so apply/replay is
    # deterministic and writes back into the definition (the change persists).
    sub = _subgraph_write_target(workflow, node_id, widget)
    if sub is not None:
        segments, inner_widget = sub
        target = _navigate_subgraph_path(workflow, segments)  # read-only: current value + schema
        inner_type = target.get("type", "")
        value, norm_note = _normalize_combo(graph, inner_type, inner_widget, value)
        cur = _engine._widgets_as_positional(target.get("widgets_values"), graph, inner_type)
        order = graph.widget_order_for_node(inner_type, cur)
        old = None
        if inner_widget in order:
            i = order.index(inner_widget)
            old = cur[i] if i < len(cur) else None
        warnings = _validate_widget(graph, inner_type, inner_widget, value)  # raises on shape mismatch
        if norm_note:
            warnings = [norm_note, *warnings]
        op = _new_op(
            "set_widget",
            actor,
            base_version,
            node_id=node_id,
            widget=widget,
            value=value,
            old=old,
            path=[str(s) for s in segments],
            inner_widget=inner_widget,
        )
        if warnings:
            op["warnings"] = warnings
        return apply_op(workflow, op, graph), op

    node = _require(workflow, node_id)
    class_type = node.get("type", "")
    widgets = _engine._widgets_as_positional(node.get("widgets_values"), graph, class_type)
    idx = _widget_index(graph, class_type, widget, widgets)  # raises on unknown widget name
    value, norm_note = _normalize_combo(graph, class_type, widget, value)
    old = widgets[idx] if idx < len(widgets) else None
    warnings = _validate_widget(graph, class_type, widget, value)  # raises on shape mismatch
    if norm_note:
        warnings = [norm_note, *warnings]
    op = _new_op(
        "set_widget",
        actor,
        base_version,
        node_id=node_id,
        widget=widget,
        value=value,
        old=old,
    )
    if warnings:
        op["warnings"] = warnings
    return apply_op(workflow, op, graph), op


def _subgraph_write_target(workflow: dict, node_id: Any, widget: str) -> tuple[list[str], str] | None:
    """Resolve a subgraph promoted/interior widget address to ``(node_path, inner_widget)``.

    Returns ``None`` when ``node_id`` is an ordinary top-level node (the caller
    uses the direct widget path). Two address forms resolve here — the SAME ones
    ``comfy workflow slots`` advertises for a subgraph instance:

      * FLAT promoted input (``57.text``): ``node_id`` is a subgraph *instance*
        and ``widget`` names one of its promoted proxy inputs; we follow the
        instance's ``proxyWidgets`` to the interior node that backs it.
      * NESTED interior (``57/27.text``): ``node_id`` already carries the
        ``<instance>/<inner>`` path; its segments pass straight through.

    Raises ``ValueError`` (with a slots-consistent hint) when the node is a
    subgraph instance but ``widget`` is not one of its promoted inputs.
    """
    from comfy_cli.cql import engine as _engine

    node_str = str(node_id)
    # Nested interior form: the interior path is explicit.
    if _engine._SUBGRAPH_PATH_SEP in node_str:
        return node_str.split(_engine._SUBGRAPH_PATH_SEP), widget
    # Flattened composite form ("57:27"): the id namespace UI→API lowering mints
    # (workflow_to_api composes inner ids as `<outer>:<inner>`), which is what
    # `validate` output and server node_errors carry. Callers copy those ids
    # back into edit commands, so accept them as an alias for the editable
    # `<outer>/<inner>` path — unless a literal node really has that id.
    if ":" in node_str and _find_by_str(workflow, node_str) is None:
        return node_str.split(":"), widget

    defs_by_id = _engine._subgraph_defs_by_id(workflow)
    if not defs_by_id:
        return None
    instance = _find(workflow, node_id)
    if instance is None:
        return None  # let the direct path raise the canonical "node not found"
    if defs_by_id.get(instance.get("type", "")) is None:
        return None  # ordinary top-level node
    # Subgraph instance: map the promoted input name → interior node via proxyWidgets.
    proxy = (instance.get("properties") or {}).get("proxyWidgets") or []
    proxied: list[str] = []
    for entry in proxy:
        if not (isinstance(entry, list) and len(entry) >= 2):
            continue
        name = entry[1] if isinstance(entry[1], str) else str(entry[1])
        proxied.append(name)
        if name == widget:
            return [node_str, str(entry[0])], widget
    raise ValueError(
        f"promoted input {widget!r} not found on subgraph node {node_id}; "
        f"available: {', '.join(proxied) if proxied else '(none)'} "
        f"(or address an interior widget directly, e.g. {node_str}/<innerId>.<input>)"
    )


def _navigate_subgraph_path(workflow: dict, segments: list[str]) -> dict:
    """Read-only walk of a ``/``-separated node path into subgraph definitions.

    Unlike the engine's apply-time resolver this does NOT fork shared definitions
    (a read must not mutate); the forking happens at apply time. Raises
    ``ValueError`` describing the first hop that couldn't be found.
    """
    from comfy_cli.cql import engine as _engine

    defs_by_id = _engine._subgraph_defs_by_id(workflow)
    node = next(
        (n for n in workflow.get("nodes") or [] if isinstance(n, dict) and str(n.get("id", "")) == str(segments[0])),
        None,
    )
    if node is None:
        raise ValueError(f"node {segments[0]} not found in workflow")
    for seg in segments[1:]:
        sg = defs_by_id.get(node.get("type", ""))
        if sg is None:
            raise ValueError(f"node {node.get('id')} is not a subgraph; cannot descend to {seg!r}")
        node = next(
            (n for n in (sg.get("nodes") or []) if isinstance(n, dict) and str(n.get("id", "")) == str(seg)),
            None,
        )
        if node is None:
            raise ValueError(f"interior node {seg} not found in subgraph {sg.get('id')}")
    return node


def _subgraph_boundary_error(workflow: dict, node_id: Any) -> ValueError | None:
    """Explain a connect endpoint that addresses a subgraph interior.

    ``comfy workflow slots`` deliberately advertises interior addresses
    (``57/27.text``) so agents can slot-edit inside opaque template subgraphs,
    and set-widget accepts them — but a LINK cannot cross a subgraph boundary,
    so connect never can. Before this guard, such an endpoint fell through to
    the generic "node 57/27 not found in workflow" + the top-level node
    inventory + "use an id from `comfy workflow slots`" — an instruction to
    consult the exact tool that advertised the address. Measured on prod
    comfy-agent traces (2026-08-05): one session burned SEVEN identical
    connects on ``129/93.text`` and the turn died. Say what the boundary means
    and which verb works instead.

    Returns ``None`` when the endpoint is not an interior address (including
    when its head segment doesn't exist — the canonical not-found error is
    right for that).
    """
    from comfy_cli.cql import engine as _engine

    node_str = str(node_id)
    if _find_by_str(workflow, node_str) is not None:
        return None  # a literal node really has this id
    if _engine._SUBGRAPH_PATH_SEP in node_str:
        segments = node_str.split(_engine._SUBGRAPH_PATH_SEP)
    elif ":" in node_str:
        # The flattened namespace UI→API lowering mints (`<outer>:<inner>`),
        # accepted everywhere set-widget accepts the `/` form.
        segments = node_str.split(":")
    else:
        return None
    head = _find_by_str(workflow, segments[0])
    if head is None:
        return None
    sg = _engine._subgraph_defs_by_id(workflow).get(str(head.get("type", "")))
    if sg is None:
        return ValueError(
            f"node {segments[0]} is not a subgraph, so {node_str} does not address a node — "
            f"connect to node {segments[0]}'s own slots instead (see `comfy workflow slots`)"
        )
    canonical = "/".join(segments)
    try:
        _navigate_subgraph_path(workflow, segments)
    except ValueError:
        interior = ", ".join(
            f"{n.get('id')} ({n.get('type', '?')})" for n in sg.get("nodes") or [] if isinstance(n, dict)
        )
        return ValueError(
            f"no node {segments[-1]} inside subgraph {segments[0]} — its interior nodes: {interior or '(none)'}"
        )
    return ValueError(
        f"node {canonical} is inside subgraph {segments[0]} ({str(sg.get('name') or '?')!r}) — a link cannot cross "
        f"the subgraph boundary, so connect cannot reach it. Interior widgets ARE settable: "
        f"`comfy workflow set-widget <file> {canonical}.<widget> <value>`. To wire a live link, connect to one of "
        f"the instance's own slots (see `comfy workflow slots`), or promote the input in the ComfyUI editor first."
    )


def _promoted_widget_error(workflow: dict, node: dict, slot: Any) -> ValueError | None:
    """Explain a connect target that names a subgraph instance's promoted widget.

    ``slots`` advertises a curated instance's promoted inputs flat
    (``57.text``), and set-widget accepts exactly that address — but a promoted
    WIDGET is a value routed through ``proxyWidgets``, not a link input on the
    instance. The old error ("input 'text' not found on node 57; inputs: []"
    plus the node inventory) reads as *wrong id, try another*, when the truth
    is *right id, wrong verb*. Returns ``None`` unless the node is a subgraph
    instance and ``slot`` is one of its promoted widgets.
    """
    from comfy_cli.cql import engine as _engine

    if not isinstance(slot, str):
        return None
    if _engine._subgraph_defs_by_id(workflow).get(str(node.get("type", ""))) is None:
        return None
    try:
        target = _subgraph_write_target(workflow, node.get("id"), slot)
    except ValueError:
        return None  # not a promoted widget either — the input-not-found error stands
    if target is None:
        return None
    nid = node.get("id")
    return ValueError(
        f"input {slot!r} on subgraph instance {nid} is a promoted widget (a value), not a link input — set it with "
        f"`comfy workflow set-widget <file> {nid}.{slot} <value>`. Wiring a live link into the subgraph requires "
        f"promoting a link input in the ComfyUI editor."
    )


def connect(
    workflow: dict,
    graph,
    from_node: Any,
    from_slot: Any,
    to_node: Any,
    to_slot: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    """Wire two nodes, enriching a not-found endpoint error with the list of
    node ids that exist (see :func:`_enrich_resolution_error`)."""
    try:
        return _connect_impl(
            workflow, graph, from_node, from_slot, to_node, to_slot, actor=actor, base_version=base_version
        )
    except ValueError as e:
        raise _enrich_resolution_error(e, workflow, graph) from e


def _connect_impl(
    workflow: dict,
    graph,
    from_node: Any,
    from_slot: Any,
    to_node: Any,
    to_slot: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    for endpoint in (from_node, to_node):
        boundary = _subgraph_boundary_error(workflow, endpoint)
        if boundary is not None:
            raise boundary
    src = _require(workflow, from_node)
    dst = _require(workflow, to_node)
    out_idx, link_type = _resolve_output_slot(src, graph, from_slot)
    try:
        in_idx, grow = _resolve_input_target(dst, graph, to_slot, link_type)
    except ValueError as e:
        promoted = _promoted_widget_error(workflow, dst, to_slot)
        if promoted is not None:
            raise promoted from e
        raise
    # Type-check concrete slots: an output only connects to an input that accepts
    # its type (or a wildcard "*"). Autogrow slots are minted with the source
    # type, so they need no check. Without this, a mis-wire silently clobbers a
    # link.
    if in_idx is not None:
        dst_type = (dst.get("inputs") or [])[in_idx].get("type")
        if not _types_compatible(link_type, dst_type):
            raise ValueError(
                f"type mismatch: {link_type} output of node {from_node} cannot connect to "
                f"{dst_type} input {(dst.get('inputs') or [])[in_idx].get('name')!r} of node {to_node}"
            )
    op = _new_op(
        "connect",
        actor,
        base_version,
        link_id=mint_id(),
        from_node=from_node,
        from_slot=out_idx,
        to_node=to_node,
        to_slot=in_idx,
        link_type=link_type,
    )
    if grow is not None:
        op["grow"] = grow  # autogrow: apply appends this input slot, then wires it
    return apply_op(workflow, op, graph), op


def clear(workflow: dict, *, actor: str = "cli", base_version: int = 0) -> tuple[dict, dict]:
    """Remove every node, link, and group in one op. last_node_id/last_link_id
    are preserved so ids minted after a clear stay monotonic (id reuse would
    let a merge resurrect a deleted node's identity)."""
    removed = [n.get("id") for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    op = _new_op("clear", actor, base_version, removed_nodes=removed)
    return apply_op(workflow, op, None), op


def reset_doc(workflow: dict, *, actor: str = "cli", base_version: int = 0) -> tuple[dict, dict]:
    """Reset the whole document to the empty baseline (op-vocabulary-v1 §1.6).

    Not ``clear``. ``clear`` empties the graph but PRESERVES the id high-water
    marks and the applied-op bookkeeping, so it is an ordinary edit that merges
    with concurrent ops. ``reset_doc`` drops those too: it is a **history
    barrier**, and ops minted against a pre-reset ``base_version`` do not replay
    across it.

    That is why the CLI surface guards it behind an explicit ``--confirm`` and
    why it is standalone-only — there is no safe way to fold "forget everything
    that ever applied" into the middle of a batch.
    """
    removed = [n.get("id") for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    op = _new_op("reset_doc", actor, base_version, removed_nodes=removed)
    return apply_op(workflow, op, None), op


# ---------------------------------------------------------------------------
# Bulk writers — expressing a whole-file replacement as ops (V1-038)
# ---------------------------------------------------------------------------


class NotExpressibleError(ValueError):
    """A graph uses structure the frozen v1 vocabulary cannot express.

    Raised by :func:`replace_ops` INSTEAD of returning a partial batch. A
    partial batch is the dangerous answer: it applies cleanly and leaves a
    document that is not the graph the caller asked for. The caller is expected
    to fall back to whatever whole-document path it had before (the cloud
    agent re-mints), and to say why.
    """


def _inexpressible_reason(workflow: dict) -> str | None:
    """Why ``workflow`` cannot be rebuilt from add_node/connect ops, or None.

    The frozen vocabulary has four batchable kinds and none of them can create a
    subgraph definition, a canvas group, or a reroute point — so a graph that
    carries any of those is not reconstructible from ops, full stop. Enumerated
    positively (a closed list of things we know we CAN'T do) rather than by
    trying and checking, so an unexpressible template fails before it has
    written anything.
    """
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        return "not a frontend-format workflow (no `nodes` list) — only the save/UI format can be op-ified"
    definitions = workflow.get("definitions")
    if isinstance(definitions, dict) and definitions.get("subgraphs"):
        return "the workflow contains a subgraph definition, which no frozen op kind can create"
    if workflow.get("groups"):
        return "the workflow contains canvas groups, which no frozen op kind can create"
    extra = workflow.get("extra")
    if isinstance(extra, dict) and (extra.get("reroutes") or extra.get("linkExtensions")):
        return "the workflow contains reroute points, which no frozen op kind can create"
    for node in workflow["nodes"]:
        if not isinstance(node, dict) or node.get("id") is None or not node.get("type"):
            return "the workflow contains a node with no id or no type"
    for link in workflow.get("links") or []:
        if not isinstance(link, list) or len(link) < 5:
            return "the workflow contains a link that is not a [id, from, from_slot, to, to_slot, type] tuple"
    return None


def _slot_ref(node: dict, slots_key: str, index: Any, alias: str) -> str:
    """`$alias.<slot>` for a spec-form connect, preferring the slot NAME.

    Names are the canonical reference form and survive slot reordering; the
    index is the fallback for a node whose slot list the template omits.
    ``_split_ref_slot`` partitions on the FIRST dot, so a name containing one
    would resolve wrong — those fall back to the index too.
    """
    slots = node.get(slots_key)
    if isinstance(slots, list) and isinstance(index, int) and 0 <= index < len(slots):
        name = (slots[index] or {}).get("name") if isinstance(slots[index], dict) else None
        if isinstance(name, str) and name and "." not in name:
            return f"${alias}.{name}"
    return f"${alias}.{index}"


def _alias_for(class_type: str, used: dict[str, int]) -> str:
    """A deterministic, batch-unique alias for a node — `ksampler`, `ksampler_2`."""
    base = re.sub(r"[^a-z0-9_]", "", str(class_type).lower()) or "node"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base}_{used[base]}"


def replace_ops(old: dict, new: dict, *, actor: str = "cli", base_version: int = 0) -> list[dict]:
    """The stamped op batch that turns ``old`` into ``new``.

    This is what makes a BULK WRITER (a template fetch, a saved-workflow open)
    an attributed, incremental edit instead of a whole-document replacement.
    Without it the only way to land a replaced canvas in a shared document is to
    re-seed it — and §8.6 is explicit that independently re-seeding a base is
    the one thing a replica must never do, because the duplicate identities
    only show up on the first merge.

    Shape: ``delete_node`` for everything currently in ``old`` (in order), then
    ``add_node`` for every node in ``new``, then ``connect`` for every link.
    Widget values need no ``set_widget`` ops — they ride inside the ``add_node``
    payload, which §8.5 makes authoritative at replay.

    **Identity is re-minted, never inherited.** Template graphs are numbered
    from small frontend counters (1, 2, 3…); replaying those ids into a live
    document would reuse identities a concurrent replica may still hold, which
    §1.5 calls out as letting a merge resurrect a deleted node. Every node and
    link gets a fresh ``mint_id`` and every interior reference is remapped onto
    it.

    **Dual-shape on purpose.** Each returned dict is a fully minted op (``op_id``
    / ``actor`` / ``stamp`` + the kind's minted fields) AND carries that kind's
    SPEC keys (``class_type``/``at``/``as``, ``from``/``to``, ``node``). So the
    same array replays through :func:`apply_op` losslessly *and* is accepted
    verbatim by :func:`apply_specs` — one artifact, both consumers. The two are
    not equivalent: ``apply_specs`` re-mints each node from the live catalog, so
    it reproduces the STRUCTURE (classes + wiring) while the op path reproduces
    the graph exactly, widget values included.

    :raises NotExpressibleError: ``new`` uses structure no frozen op can create.
    """
    reason = _inexpressible_reason(new)
    if reason:
        raise NotExpressibleError(reason)

    ops: list[dict] = []
    old_links = [link for link in (old.get("links") or []) if isinstance(link, list) and len(link) >= 5]
    for node in old.get("nodes") or []:
        if not isinstance(node, dict) or node.get("id") is None:
            continue
        nid = node["id"]
        ops.append(
            _new_op(
                "delete_node",
                actor,
                base_version,
                node_id=nid,
                removed_links=[link[0] for link in old_links if link[1] == nid or link[3] == nid],
                # spec key, so apply_specs dispatches the same entry
                node=nid,
            )
        )

    node_ids: dict[Any, int] = {n["id"]: mint_id() for n in new["nodes"]}
    link_ids: dict[Any, int] = {link[0]: mint_id() for link in (new.get("links") or [])}
    aliases: dict[Any, str] = {}
    used: dict[str, int] = {}

    for original in new["nodes"]:
        node = copy.deepcopy(original)
        node["id"] = node_ids[original["id"]]
        for slot in node.get("inputs") or []:
            if isinstance(slot, dict) and slot.get("link") is not None:
                slot["link"] = link_ids.get(slot["link"])
        for slot in node.get("outputs") or []:
            if isinstance(slot, dict) and isinstance(slot.get("links"), list):
                slot["links"] = [link_ids[x] for x in slot["links"] if x in link_ids]
        alias = _alias_for(original.get("type"), used)
        aliases[original["id"]] = alias
        pos = original.get("pos")
        ops.append(
            _new_op(
                "add_node",
                actor,
                base_version,
                node_id=node["id"],
                class_type=original.get("type"),
                pos=pos,
                node=node,
                # spec keys; the node already existed, so a deprecated class replays
                **{"at": pos, "as": alias, "allow_deprecated": True},
            )
        )

    by_original_id = {n["id"]: n for n in new["nodes"]}
    for link in new.get("links") or []:
        lid, from_node, from_slot, to_node, to_slot = link[0], link[1], link[2], link[3], link[4]
        if from_node not in node_ids or to_node not in node_ids:
            # A link to a node the graph does not contain is already broken in
            # the source; dropping it is the faithful translation of a graph the
            # canvas would render with a dangling edge.
            continue
        ops.append(
            _new_op(
                "connect",
                actor,
                base_version,
                link_id=link_ids[lid],
                from_node=node_ids[from_node],
                from_slot=from_slot,
                to_node=node_ids[to_node],
                to_slot=to_slot,
                link_type=link[5] if len(link) > 5 else None,
                # spec keys
                **{
                    "from": _slot_ref(by_original_id[from_node], "outputs", from_slot, aliases[from_node]),
                    "to": _slot_ref(by_original_id[to_node], "inputs", to_slot, aliases[to_node]),
                },
            )
        )
    return ops


def delete_node(
    workflow: dict,
    graph,
    node_id: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    """Delete a node, enriching a not-found error with the list of node ids that
    exist (see :func:`_enrich_resolution_error`)."""
    try:
        return _delete_node_impl(workflow, graph, node_id, actor=actor, base_version=base_version)
    except ValueError as e:
        raise _enrich_resolution_error(e, workflow, graph) from e


def _delete_node_impl(
    workflow: dict,
    graph,
    node_id: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    _require(workflow, node_id)
    removed = [ln[0] for ln in workflow.get("links") or [] if ln[1] == node_id or ln[3] == node_id]
    op = _new_op("delete_node", actor, base_version, node_id=node_id, removed_links=removed)
    return apply_op(workflow, op, graph), op


# ---------------------------------------------------------------------------
# recipes — a parameterized op-batch. A recipe is `{params?, ops:[...]}`; a bare
# list is a param-less batch. `${name}` placeholders in op values are filled from
# `--param`. Validation is strict: a required param with no value, an unknown
# param, or a `${name}` the recipe didn't declare all fail — never a silent blank.
# ---------------------------------------------------------------------------

_PARAM_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class RecipeError(ValueError):
    """A recipe or its parameters are malformed."""


def parse_recipe(doc: Any) -> tuple[list, dict]:
    """Split a recipe document into (ops, params_decl). Accepts a bare op list."""
    if isinstance(doc, list):
        return doc, {}
    if isinstance(doc, dict) and isinstance(doc.get("ops"), list):
        return doc["ops"], (doc.get("params") or {})
    raise RecipeError("recipe must be a JSON array of ops, or an object with an `ops` array")


def resolve_params(params_decl: dict, provided: dict[str, str]) -> dict[str, Any]:
    """Type-coerce provided values against the declared params. Errors on an
    unknown param, or a declared param with neither a value nor a default."""
    unknown = sorted(set(provided) - set(params_decl))
    if unknown:
        raise RecipeError(f"unknown --param {unknown}; recipe declares {sorted(params_decl)}")
    out: dict[str, Any] = {}
    for name, decl in params_decl.items():
        decl = decl if isinstance(decl, dict) else {}
        if name in provided:
            out[name] = _coerce_param(provided[name], decl.get("type", "string"), name)
        elif "default" in decl:
            out[name] = decl["default"]
        else:
            raise RecipeError(f"missing required --param {name!r}")
    return out


def _coerce_param(raw: str, type_name: str, name: str) -> Any:
    if type_name == "int":
        try:
            return int(raw)
        except ValueError as e:
            raise RecipeError(f"--param {name}: expected int, got {raw!r}") from e
    if type_name in ("float", "number"):
        try:
            return float(raw)
        except ValueError as e:
            raise RecipeError(f"--param {name}: expected number, got {raw!r}") from e
    if type_name in ("bool", "boolean"):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise RecipeError(f"--param {name}: expected bool, got {raw!r}")
    return raw  # string (the default)


def substitute_params(ops: list, params: dict[str, Any]) -> list:
    """Replace `${name}` in op values. A value that is exactly `${name}` takes the
    param's real (typed) value; embedded refs interpolate as text. An undeclared
    `${name}` is an error, not a blank."""

    def sub(value: Any) -> Any:
        if isinstance(value, str):
            whole = _PARAM_REF.fullmatch(value)
            if whole:
                return _param(whole.group(1), params)
            return _PARAM_REF.sub(lambda m: str(_param(m.group(1), params)), value)
        if isinstance(value, list):
            return [sub(v) for v in value]
        if isinstance(value, dict):
            return {k: sub(v) for k, v in value.items()}
        return value

    return [sub(op) for op in ops]


def _param(name: str, params: dict[str, Any]) -> Any:
    if name not in params:
        raise RecipeError(f"recipe references undeclared param ${{{name}}}")
    return params[name]


def capture_recipe(workflow: dict, graph, name: str = "captured", lift: dict | None = None) -> tuple[dict, list[dict]]:
    """Project a UI-format graph into a recipe — the op-batch that rebuilds it
    (add_node + non-default set_widget + connect). The inverse of `apply`:
    `apply(empty, capture(wf))` reproduces `wf`. Top-level nodes only.
    Returns ``(recipe, warnings)``.

    UI-only node types (:data:`UI_ONLY_NODE_TYPES`) never reach the API and
    ``add_node`` refuses to mint them, so capturing them verbatim produced
    recipes ``apply`` could not run — one MarkdownNote in the source discarded
    the whole atomic batch. capture therefore SKIPS them, preserving the data
    flow they carried so the rebuilt graph's API prompt is unchanged:

      * links THROUGH ``Reroute`` and ``GetNode``→``SetNode`` chains are
        spliced to the real upstream source;
      * a ``PrimitiveNode``'s value lands as the fed widget's captured value;
      * pure annotations (``Note``/``MarkdownNote``) simply drop.

    Each skipped node (and any link that could not be spliced) is reported in
    ``warnings`` — the recipe rebuilds the executable graph, not the canvas
    decoration.

    `lift` maps `(node_id, widget_name) -> param_name`: those widgets become
    `${param_name}` holes (with a `params` header entry defaulting to the current
    value) even if the value equals the node default — so the fields you want to
    vary are actually parameterizable. No auto-parameterization otherwise."""
    if (workflow.get("definitions") or {}).get("subgraphs"):
        raise RecipeError("capture does not support subgraphs yet — edit/flatten top-level nodes first")
    lift = lift or {}
    all_nodes = [n for n in (workflow.get("nodes") or []) if isinstance(n, dict) and "id" in n]
    by_id = {n["id"]: n for n in all_nodes}
    nodes = [n for n in all_nodes if n.get("type") not in UI_ONLY_NODE_TYPES]
    ui_nodes = [n for n in all_nodes if n.get("type") in UI_ONLY_NODE_TYPES]

    # Validate lift targets up front — no silently-ignored typos.
    for (node_id, widget), _pname in lift.items():
        node = by_id.get(node_id)
        if node is None:
            raise RecipeError(f"--param target node {node_id!r} not in workflow")
        if node.get("type") in UI_ONLY_NODE_TYPES:
            raise RecipeError(
                f"--param target node {node_id} is a UI-only {node.get('type')} — capture skips it (it never reaches the API)"
            )
        if widget not in graph.widget_order_default(node.get("type", "")):
            raise RecipeError(f"--param target {node_id}.{widget!r}: not a widget on {node.get('type')}")

    warnings: list[dict] = []
    for n in ui_nodes:
        warnings.append(
            {
                "code": "ui_only_node_skipped",
                "node_id": n["id"],
                "class_type": n.get("type"),
                "message": (
                    f"{n.get('type')} (id {n['id']}) is UI-only and cannot be rebuilt by apply — skipped; "
                    "data flow through it (if any) is spliced to the real source"
                ),
            }
        )

    links = [ln for ln in (workflow.get("links") or []) if isinstance(ln, list) and len(ln) >= 5]
    # link_id -> (source_id, source_slot). Node identity is compared as a STRING
    # (amendment v1.2) — ids are legitimately either JSON type.
    link_map = {ln[0]: (ln[1], ln[2]) for ln in links if isinstance(ln[0], int)}
    node_by_sid = {str(n["id"]): n for n in all_nodes}

    def _first_input_source(n: dict) -> tuple[Any, Any] | None:
        for inp in n.get("inputs") or []:
            if isinstance(inp, dict):
                lid = inp.get("link")
                if isinstance(lid, int) and lid in link_map:
                    return link_map[lid]
        return None

    # Same upstream-resolution model as workflow_to_api's tracers: hop through
    # Reroute chains and GetNode -> SetNode pairs; a seen-set guards cycles.
    reroute_src: dict[str, tuple[Any, Any]] = {}
    set_src: dict[str, tuple[Any, Any]] = {}
    get_var: dict[str, str] = {}
    prim_val: dict[str, Any] = {}
    for n in ui_nodes:
        t = n.get("type")
        if t == "Reroute":
            src = _first_input_source(n)
            if src is not None:
                reroute_src[str(n["id"])] = src
        elif t in ("SetNode", "GetNode"):
            w = n.get("widgets_values")
            var = w[0] if isinstance(w, list) and w else None
            if not isinstance(var, str) or not var:
                continue
            if t == "GetNode":
                get_var[str(n["id"])] = var
            else:
                src = _first_input_source(n)
                if src is not None:
                    set_src[var] = src
        elif t == "PrimitiveNode":
            w = n.get("widgets_values")
            if isinstance(w, list) and w:
                prim_val[str(n["id"])] = w[0]

    def _trace(src_id: Any, src_slot: Any) -> tuple[Any, Any]:
        seen: set[str] = set()
        while str(src_id) not in seen:
            key = str(src_id)
            seen.add(key)
            if key in reroute_src:
                src_id, src_slot = reroute_src[key]
            elif key in get_var and get_var[key] in set_src:
                src_id, src_slot = set_src[get_var[key]]
            else:
                break
        return src_id, src_slot

    alias_by_sid: dict[str, str] = {}
    counts: dict[str, int] = {}
    for n in nodes:
        slug = re.sub(r"[^a-z0-9]+", "_", str(n.get("type", "node")).lower()).strip("_") or "node"
        counts[slug] = counts.get(slug, 0) + 1
        alias_by_sid[str(n["id"])] = slug if counts[slug] == 1 else f"{slug}_{counts[slug]}"

    # First pass over links: real-target links become connect specs (spliced
    # through UI-only chains); a PrimitiveNode source becomes a widget value on
    # the target (`prim_feeds`) rather than a wire.
    connect_specs: list[dict] = []
    prim_feeds: dict[tuple[str, str], Any] = {}  # (target_sid, widget_name) -> value
    for ln in links:
        _lid, from_id, from_slot, to_id, to_slot = ln[0], ln[1], ln[2], ln[3], ln[4]
        to_node = node_by_sid.get(str(to_id))
        if to_node is None or to_node.get("type") in UI_ONLY_NODE_TYPES:
            # Feeds a UI-only node — its flow is captured when tracing the
            # downstream real consumer, so nothing is lost by skipping here.
            continue
        in_name = _slot_name(to_node.get("inputs"), to_slot)
        src_id, src_slot = _trace(from_id, from_slot)
        if str(src_id) in prim_val:
            prim_feeds[(str(to_id), str(in_name))] = prim_val[str(src_id)]
            continue
        src_node = node_by_sid.get(str(src_id))
        if src_node is None or src_node.get("type") in UI_ONLY_NODE_TYPES:
            warnings.append(
                {
                    "code": "ui_only_link_dropped",
                    "node_id": to_node["id"],
                    "input": str(in_name),
                    "message": (
                        f"link into {to_node.get('type')} (id {to_node['id']}).{in_name} traces back to a UI-only "
                        "node with no real source — dropped"
                    ),
                }
            )
            continue
        out_name = _slot_name(src_node.get("outputs"), src_slot)
        connect_specs.append(
            {
                "op": "connect",
                "from": f"{alias_by_sid[str(src_id)]}.{out_name}",
                "to": f"{alias_by_sid[str(to_id)]}.{in_name}",
            }
        )

    ops: list[dict] = []
    params_header: dict[str, Any] = {}
    for n in nodes:
        alias = alias_by_sid[str(n["id"])]
        class_type = n.get("type")
        add: dict[str, Any] = {"op": "add_node", "class_type": class_type, "as": alias}
        m = graph.node(class_type)
        if m is not None and m.deprecated:
            add["allow_deprecated"] = True
        if n.get("pos"):
            add["at"] = n["pos"]
        if n.get("mode"):
            # mute (2) / bypass (4) change what executes — a recipe that
            # silently revived a bypassed node produced a different API prompt.
            add["mode"] = n["mode"]
        ops.append(add)
        from comfy_cli.cql import engine as _engine

        widgets = _engine._widgets_as_positional(n.get("widgets_values"), graph, class_type)
        order = graph.widget_order_for_node(class_type, widgets)
        defaults = graph.widget_defaults(class_type)
        for i, wname in enumerate(order):
            if i >= len(widgets) and (str(n["id"]), wname) not in prim_feeds:
                continue
            pname = lift.get((n["id"], wname))
            # A PrimitiveNode feeding this widget-input is authoritative over the
            # (possibly stale) serialized widgets_values slot — same precedence
            # the UI→API converter applies.
            value = prim_feeds.pop((str(n["id"]), wname), widgets[i] if i < len(widgets) else None)
            if pname is not None:
                # Explicitly lifted → a ${param} hole, current value as its default.
                ops.append({"op": "set_widget", "node": alias, "widget": wname, "value": f"${{{pname}}}"})
                params_header[pname] = {"type": _widget_param_type(graph, class_type, wname), "default": value}
            elif value != defaults.get(wname):
                # Only widgets that differ from the fresh-node default — add_node fills the rest.
                ops.append({"op": "set_widget", "node": alias, "widget": wname, "value": value})

    for (to_sid, in_name), value in prim_feeds.items():
        target = node_by_sid.get(to_sid, {})
        warnings.append(
            {
                "code": "primitive_feed_unrepresentable",
                "node_id": target.get("id"),
                "input": in_name,
                "message": (
                    f"PrimitiveNode value {value!r} feeds {target.get('type')} (id {target.get('id')}).{in_name}, "
                    "which is not a widget on that node — dropped"
                ),
            }
        )

    ops.extend(connect_specs)
    return {"recipe": name, "params": params_header, "ops": ops}, warnings


def _widget_param_type(graph, class_type: str, widget: str) -> str:
    """Recipe param type for a widget, from its schema port type."""
    m = graph.node(class_type)
    port = next((p for p in (m.inputs if m else []) if p.name == widget), None)
    t = (port.type if port else "").upper()
    if t == "INT":
        return "int"
    if t in ("FLOAT", "NUMBER"):
        return "float"
    if t == "BOOLEAN":
        return "bool"
    return "string"


def _slot_name(slots: Any, idx: Any) -> Any:
    """A link's slot addressed by name where the node declares one, else by index
    (both are valid connect targets)."""
    if isinstance(slots, list) and isinstance(idx, int) and 0 <= idx < len(slots):
        nm = slots[idx].get("name") if isinstance(slots[idx], dict) else None
        if nm:
            return nm
    return idx


# ---------------------------------------------------------------------------
# apply_specs — run a batch of edit specs (add_node/connect/set_widget/delete_node)
# with `as` aliases so later specs reference just-minted nodes. Shared by the
# `apply` and `foreach` commands. Raises on a malformed spec so the caller can keep
# the batch atomic (write nothing on failure).
# ---------------------------------------------------------------------------


def resolve_ref(ref: Any, aliases: dict[str, Any]) -> Any:
    """Map an alias (bare or ``$``-prefixed) to its minted id; pass ints and
    unknown strings through.

    ``$up`` and ``up`` address the same alias — exactly one leading ``$`` is
    stripped before lookup (``$``-prefixed is the canonical documented form; see
    docs/op-vocabulary-v1.md). ``${name}`` is NOT an alias: that shape is
    reserved for recipe parameters (filled by :func:`substitute_params`), so an
    unsubstituted one is rejected loudly here instead of falling through to a
    misleading "node not found"."""
    if isinstance(ref, str):
        if ref.startswith("${"):
            raise ValueError(
                f"{ref!r} looks like an unsubstituted recipe parameter — `${{name}}` is reserved for recipe "
                "params (declare it under `params` and fill it with --param); an alias reference is `$name` "
                "(or bare `name`)"
            )
        if ref.startswith("$"):
            ref = ref[1:]
        if ref in aliases:
            return aliases[ref]
        if ref.lstrip("-").isdigit():
            return int(ref)
    return ref


def _split_ref_slot(spec_val: str, aliases: dict[str, Any]) -> tuple[Any, Any]:
    """Split `<node_or_alias>.<slot>` and resolve the node part."""
    node_part, _, slot = str(spec_val).partition(".")
    return resolve_ref(node_part, aliases), slot


def apply_specs(
    workflow: dict, graph, specs: list, *, actor: str = "cli", base_version: int = 0
) -> tuple[dict, list, dict]:
    """Apply edit specs to ``workflow`` in order. Returns (workflow, ops, aliases)."""
    specs = layout.assign_positions(workflow, graph, specs)
    # Snapshot the inventory BEFORE any op mutates the graph — on failure the
    # caller discards everything below, so this is what actually survives.
    pre_batch_hint = _available_nodes_hint(workflow)
    aliases: dict[str, Any] = {}
    ops: list[dict] = []
    try:
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict) or "op" not in spec:
                raise ValueError(f"spec #{i} must be an object with an 'op' field")
            kind = spec["op"]
            # A missing required field surfaces as a bare KeyError (just the key name);
            # wrap it so the batch/recipe caller learns WHICH spec and op are malformed.
            try:
                if kind == "add_node":
                    workflow, op = add_node(
                        workflow,
                        graph,
                        spec["class_type"],
                        pos=spec.get("at"),
                        mode=spec.get("mode") or 0,
                        actor=actor,
                        base_version=base_version,
                        allow_deprecated=bool(spec.get("allow_deprecated")),
                    )
                    alias = spec.get("as")
                    if alias:
                        # A duplicate alias would silently clobber the earlier node, so a
                        # later `${alias}` reference resolves to the wrong node. Recipes
                        # are generated/templated, so an accidental repeat is plausible —
                        # fail loudly instead.
                        if alias in aliases:
                            raise ValueError(f"spec #{i}: alias {alias!r} is already defined by an earlier spec")
                        aliases[alias] = op["node_id"]
                elif kind == "connect":
                    fn, fs = _split_ref_slot(spec["from"], aliases)
                    tn, ts = _split_ref_slot(spec["to"], aliases)
                    workflow, op = connect(workflow, graph, fn, fs, tn, ts, actor=actor, base_version=base_version)
                elif kind == "set_widget":
                    workflow, op = set_widget(
                        workflow,
                        graph,
                        resolve_ref(spec["node"], aliases),
                        spec["widget"],
                        spec["value"],
                        actor=actor,
                        base_version=base_version,
                    )
                elif kind == "delete_node":
                    workflow, op = delete_node(
                        workflow, graph, resolve_ref(spec["node"], aliases), actor=actor, base_version=base_version
                    )
                elif kind in _NOT_BATCHABLE:
                    # In the frozen vocabulary but standalone-only — surfaced with
                    # its own registered code so the caller learns the standalone
                    # command instead of a generic "unknown op".
                    raise NotBatchableError(i, kind)
                else:
                    raise ValueError(f"spec #{i}: unknown op {kind!r}")
            except KeyError as e:
                raise ValueError(f"spec #{i} ({kind}) is missing required field {e}") from e
            ops.append(op)
    except (NotBatchableError, DeprecatedNodeType):
        # Already carries a registered code and a hint that says what to do
        # instead — don't wrap it into a generic hint.
        raise
    except (ValueError, KeyError) as e:
        err = _rehint_discarded_batch(e, pre_batch_hint)
        # Structured failure position for callers that report a summary
        # receipt (`apply --ack summary`): which spec aborted the batch, its
        # op kind, and how many specs had applied before the abort (all of
        # them then discarded — the batch is atomic). `i`/`spec` are the loop
        # variables at raise time; guard for a non-dict spec.
        err.spec_index = i  # type: ignore[attr-defined]
        err.spec_op = spec.get("op") if isinstance(spec, dict) else None  # type: ignore[attr-defined]
        # The whole batch was discarded — nothing persisted. Reporting the
        # specs that applied-then-were-discarded taught a merge consumer that
        # k-1 ops survived (docs/op-vocabulary-v1.md: "``applied_count`` is
        # always 0 on failure — nothing is written").
        err.applied_count = 0  # type: ignore[attr-defined]
        raise err from e
    return workflow, ops, aliases


# ---------------------------------------------------------------------------
# apply — the deterministic, idempotent replay used by every consumer
# ---------------------------------------------------------------------------


def apply_op(workflow: dict, op: dict, graph) -> dict:
    """Replay one op onto ``workflow`` in place and return it. Idempotent: an
    op whose ``op_id`` was already applied is a no-op."""
    applied = workflow.setdefault("_applied_ops", [])
    if op["op_id"] in applied:
        return workflow
    kind = op["op"]
    # Snapshot the LWW bookkeeping so an exception escaping a handler cannot
    # leave a stamp committed WITHOUT its op_id recorded below. That pairing is
    # the poison state: a retry of the identical op loses to the failed
    # attempt's own stamp and is silently dropped forever.
    stamps_before = dict(workflow.get("_widget_stamps") or {})
    try:
        if kind == "add_node":
            _apply_add_node(workflow, op)
        elif kind == "set_widget":
            _apply_set_widget(workflow, op, graph)
        elif kind == "connect":
            _apply_connect(workflow, op, graph)
        elif kind == "delete_node":
            _apply_delete_node(workflow, op)
        elif kind == "clear":
            _apply_clear(workflow, op)
        elif kind == "reset_doc":
            _apply_reset_doc(workflow, op)
        else:
            raise ValueError(f"unknown op {kind!r}")
    except BaseException:
        if stamps_before or "_widget_stamps" in workflow:
            workflow["_widget_stamps"] = stamps_before
        raise
    # NOT ``applied.append`` — ``_apply_reset_doc`` REPLACES ``_applied_ops``
    # with a fresh list (that is what makes it a history barrier), so the local
    # binding above is stale for that kind and the reset's own op_id would be
    # written into a discarded list. Re-read, so a re-delivered reset_doc is a
    # no-op rather than a second wipe.
    workflow.setdefault("_applied_ops", []).append(op["op_id"])
    return workflow


def _apply_add_node(workflow: dict, op: dict) -> None:
    # Node identity is compared as a STRING everywhere in the apply path
    # (amendment v1.2): ids are legitimately either JSON type, and an exact
    # ``==`` made ``7`` and ``"7"`` two different nodes.
    nodes = workflow.setdefault("nodes", [])
    if any(str(n.get("id")) == str(op["node_id"]) for n in nodes):
        return
    nodes.append(copy.deepcopy(op["node"]))
    # last_node_id is a max-register over INT ids only; a string id (subgraph
    # address, historical workflow) is not comparable and never bumps it.
    if isinstance(op["node_id"], int) and not isinstance(op["node_id"], bool):
        workflow["last_node_id"] = max(workflow.get("last_node_id") or 0, op["node_id"])


def _apply_set_widget(workflow: dict, op: dict, graph) -> None:
    # Last-writer-wins: a lower-stamped concurrent write to this target is
    # dropped, so the surviving value is the same in any apply order.
    if not _lww_gate(workflow, op):
        return
    path = op.get("path")
    if path:
        # Subgraph interior write. A concurrently-deleted instance => no-op
        # (delete wins). Otherwise descend the resolved node path (forking any
        # shared definition en route so a sibling instance can't alias this
        # write) and set the interior widget — the reference resolver the
        # ``slots``/``set-slot`` surface uses, so the two agree by construction.
        if _find_by_str(workflow, path[0]) is None:
            return
        from comfy_cli.cql import engine as _engine

        defs_by_id = _engine._subgraph_defs_by_id(workflow)
        target = _engine._resolve_node_path(workflow, [str(s) for s in path], defs_by_id)
        _engine._write_widget(target, op["inner_widget"], op["value"], graph, extend=False)
        _lww_commit(workflow, op)
        return
    node = _find_by_str(workflow, op["node_id"])
    if node is None:
        return  # target concurrently deleted => no-op (delete wins).
    from comfy_cli.cql import engine as _engine

    widgets = _engine._widgets_as_positional(node.get("widgets_values"), graph, node.get("type", ""))
    node["widgets_values"] = widgets
    idx = _widget_index(graph, node.get("type", ""), op["widget"], widgets)
    if idx >= len(widgets):
        widgets.extend([None] * (idx + 1 - len(widgets)))
    widgets[idx] = op["value"]
    _lww_commit(workflow, op)


def _apply_inputcount_bump(workflow: dict, dst: dict, op: dict, graph, widget: str, value: Any) -> None:
    """Bump a kijai ``inputcount``-family widget as part of applying a connect
    that grew a numbered slot (see ``_inputcount_family_match`` /
    ``_resolve_input_target``). Goes through the SAME last-writer-wins gate
    ``_apply_set_widget`` uses (``_lww_gate``/``_lww_commit``), stamped with
    the connect op's own stamp/op_id — so this widget write shares the
    connect's causal position, and a concurrent explicit
    ``set_widget(..., "inputcount", ...)`` resolves deterministically
    regardless of apply order. A no-op when ``graph`` is unavailable (offline
    edit/merge replay without a catalog) — the slot still grows, just without
    the count bump, which callers with a real catalog never hit."""
    if graph is None:
        return
    widget_op = {
        "op": "set_widget",
        "node_id": op["to_node"],
        "widget": widget,
        "op_id": op["op_id"],
        "stamp": op.get("stamp"),
        "base_version": op.get("base_version"),
    }
    if not _lww_gate(workflow, widget_op):
        return
    from comfy_cli.cql import engine as _engine

    # A VHS-style dict form must be projected, not indexed into: setdefault
    # returned the dict itself, and the positional write below installed an
    # integer key into it — leaving ``inputcount`` stale next to a garbage key.
    widgets = _engine._widgets_as_positional(dst.get("widgets_values"), graph, dst.get("type", ""))
    dst["widgets_values"] = widgets
    idx = _widget_index(graph, dst.get("type", ""), widget, widgets)
    if idx >= len(widgets):
        widgets.extend([None] * (idx + 1 - len(widgets)))
    widgets[idx] = value
    _lww_commit(workflow, widget_op)


def _apply_connect(workflow: dict, op: dict, graph) -> None:
    # Totality: an endpoint concurrently deleted => no crash and no dangling
    # link, so a merge consumer can replay a connect and a delete in either
    # order. Resolve the destination before mutating anything; if it is gone the
    # target slot does not exist and never will (ids are never reused), so there
    # is no register to claim and delete simply wins.
    dst = _find_by_str(workflow, op["to_node"])
    if dst is None:
        return
    grow = op.get("grow")
    if grow is not None:
        # Autogrow is NOT a shared register: every grow mints its own slot keyed
        # by ``grow_id``, so two concurrent grows onto one base both survive and
        # there is nothing to gate (§1.2 / amendment v1.2's carve-out).
        if _find_by_str(workflow, op["from_node"]) is None:
            return
        # Autogrow: grow a concrete slot and wire it. Keyed by ``grow_id`` (the
        # link id) so replay is idempotent AND non-clobbering — a concurrent
        # autogrow that minted the same requested name gets its own fresh slot
        # instead of overwriting this one, so neither connection is lost. The
        # slot's convergence identity is ``grow_id``; its display name stays
        # sequential per the server's ``images.imageN`` convention (or the
        # schema's own element names, when the catalog carries a template).
        ins = dst.setdefault("inputs", [])
        to_idx = next((k for k, i in enumerate(ins) if i.get("grow_id") == op["link_id"]), None)
        if to_idx is None:
            inputcount = grow.get("inputcount")
            if inputcount is not None:
                # Bare-key family (see _next_inputcount_name): a collision must
                # still grow the next free BARE key, never autogrow's dotted
                # base.elemN fallback — that name is meaningless for this family.
                name = _next_inputcount_name(ins, grow["name"])
            else:
                base = str(grow["name"]).split(".", 1)[0]
                template = None if grow.get("widget") else _autogrow_template(graph, dst.get("type", ""), base)
                name = _next_autogrow_name(ins, grow["name"], template)
            entry = {
                "name": name,
                "type": grow["type"],
                "link": None,
                "grow_id": op["link_id"],
            }
            if grow.get("widget"):
                # Mark as a converted widget (ComfyUI's widget→input); value stays
                # in widgets_values for positional alignment, converter uses the link.
                entry["widget"] = {"name": grow["widget"]}
            ins.append(entry)
            to_idx = len(ins) - 1
            if inputcount is not None:
                # Bump using the op's mint-time-planned value (NOT re-derived
                # from a post-collision-renamed slot number): every op's
                # contribution to this LWW register must be a static property
                # of the op, independent of what else has applied first, or
                # the two apply orders' winning stamp would carry DIFFERENT
                # values and the graph would fail to converge (P9). Two
                # concurrent connects that both minted against the same next
                # slot (a genuine same-instant race) both plan the same
                # value, so this still converges for that case; a slot that
                # loses the bare-key naming race to a higher number is a
                # known, accepted LWW-register limitation (not a monotonic
                # counter) — the widget may undercount until the next
                # explicit set_widget or connect on this node corrects it.
                _apply_inputcount_bump(workflow, dst, op, graph, inputcount["widget"], inputcount["value"])
    else:
        to_idx = op["to_slot"]
        ins = dst.get("inputs")
        # Slot drift: a replay against a document whose destination SLOT no
        # longer exists (or never did — a node minted from a different catalog
        # generation) must be as total as a vanished node. There is no register
        # to claim: claiming it here would poison the target — the failed op's
        # own stamp would outrank a retry of the identical op, silently losing
        # the connect forever even after the document is repaired.
        if (
            not isinstance(ins, list)
            or not isinstance(to_idx, int)
            or isinstance(to_idx, bool)
            or to_idx < 0
            or to_idx >= len(ins)
            or not isinstance(ins[to_idx], dict)
        ):
            return
        # --- The concrete-input LWW register (op-vocabulary-v1.md amendment v1.2)
        #
        # A concrete input holds at most one link, so "who occupies this slot" is
        # a SCALAR target — ``("input", to_node, to_slot)`` — resolved by exactly
        # the ``_lww_gate``/``_lww_commit`` pair ``set_widget`` uses. Without the
        # gate the occupant was decided by ARRIVAL ORDER, and composed with
        # delete-wins that produced graphs where a link exists in one
        # interleaving and not in another (found adversarially against the
        # TypeScript port: cloud PR #6722, FINDING 1).
        if not _lww_gate(workflow, op):
            return
        # Claiming the register is UNCONDITIONAL once the gate passes: the prior
        # occupant is retired even if this op then turns out to be a delete-wins
        # no-op below. Deferring the retirement until the link is known to be
        # installable would reintroduce order dependence — whether the incumbent
        # survives would depend on whether the concurrent delete of THIS op's
        # source had arrived yet.
        _lww_commit(workflow, op)
        prev = dst["inputs"][to_idx].get("link")
        if prev is not None and prev != op["link_id"]:
            _remove_link(workflow, prev)
    # Source concurrently deleted => the winning connect leaves the input EMPTY
    # (delete wins over the link, not over the register claim).
    src = _find_by_str(workflow, op["from_node"])
    if src is None:
        return
    # Source SLOT drift gets the same treatment as a deleted source: delete
    # wins over the LINK, not over the register claim — no crash, the input
    # stays empty, and the claim above (concrete branch) stands.
    outs = src.get("outputs")
    from_slot = op["from_slot"]
    if (
        not isinstance(outs, list)
        or not isinstance(from_slot, int)
        or isinstance(from_slot, bool)
        or from_slot < 0
        or from_slot >= len(outs)
        or not isinstance(outs[from_slot], dict)
    ):
        return
    link = [op["link_id"], op["from_node"], op["from_slot"], op["to_node"], to_idx, op["link_type"]]
    links = workflow.setdefault("links", [])
    if not any(ln[0] == op["link_id"] for ln in links):
        links.append(link)
    dst["inputs"][to_idx]["link"] = op["link_id"]
    out_port = outs[from_slot]
    # A real ComfyUI-serialized never-wired output carries `"links": null` — the
    # key EXISTS, so `setdefault` returns the existing `None` instead of
    # installing a fresh list, and the membership check below would raise
    # `TypeError: argument of type 'NoneType' is not iterable`. Check for None
    # explicitly rather than relying on setdefault's "key missing" semantics.
    if out_port.get("links") is None:
        out_port["links"] = []
    out_links = out_port["links"]
    if op["link_id"] not in out_links:
        out_links.append(op["link_id"])


def _remove_link(workflow: dict, link_id: Any) -> None:
    """Drop a link tuple and scrub every input/output reference to it."""
    workflow["links"] = [ln for ln in workflow.get("links") or [] if ln[0] != link_id]
    for n in workflow.get("nodes") or []:
        for inp in n.get("inputs") or []:
            if inp.get("link") == link_id:
                inp["link"] = None
        for out in n.get("outputs") or []:
            if link_id in (out.get("links") or []):
                out["links"] = [lid for lid in out["links"] if lid != link_id]


def _apply_delete_node(workflow: dict, op: dict) -> None:
    node_id = str(op["node_id"])  # node identity is compared as a string (amendment v1.2)
    workflow["nodes"] = [n for n in workflow.get("nodes") or [] if str(n.get("id")) != node_id]
    removed = set(op.get("removed_links") or [])
    kept = [
        ln
        for ln in workflow.get("links") or []
        if ln[0] not in removed and str(ln[1]) != node_id and str(ln[3]) != node_id
    ]
    workflow["links"] = kept
    kept_ids = {ln[0] for ln in kept}
    # Scrub dangling references so no input/output points at a gone link.
    for n in workflow.get("nodes") or []:
        for inp in n.get("inputs") or []:
            if inp.get("link") is not None and inp["link"] not in kept_ids:
                inp["link"] = None
        for out in n.get("outputs") or []:
            out["links"] = [lid for lid in (out.get("links") or []) if lid in kept_ids]


def _apply_clear(workflow: dict, op: dict) -> None:
    workflow["nodes"] = []
    workflow["links"] = []
    if "groups" in workflow:
        workflow["groups"] = []


#: Document-identity keys a ``reset_doc`` keeps. Everything else is discarded:
#: the point of the op is that nothing from the old document survives it. The
#: id stays so the reset document is still THIS workflow, not a new one.
_RESET_DOC_KEEP = ("id",)


def _apply_reset_doc(workflow: dict, op: dict) -> None:
    """Replace the whole document with the empty baseline, bookkeeping included.

    Unlike ``_apply_clear`` this drops ``last_node_id``/``last_link_id``,
    ``_applied_ops`` and ``_widget_stamps`` — the history barrier of §1.6. Ids
    are minted at random in ``[2**40, 2**53)`` (``mint_id``), never allocated
    from the high-water marks, so resetting them to 0 cannot cause id reuse.
    """
    kept = {k: workflow[k] for k in _RESET_DOC_KEEP if k in workflow}
    workflow.clear()
    workflow.update(kept)
    workflow["nodes"] = []
    workflow["links"] = []
    workflow["groups"] = []
    workflow["last_node_id"] = 0
    workflow["last_link_id"] = 0
    workflow["_applied_ops"] = []
    workflow["_widget_stamps"] = {}


# ---------------------------------------------------------------------------
# conflict detection + canonicalization (for ask-to-merge / convergence checks)
# ---------------------------------------------------------------------------


def _write_target(op: dict) -> tuple:
    """The conflict/write target of an op — the LWW register it claims.

    NODE IDS ARE NORMALIZED WITH ``str()`` (amendment v1.2). Node ids are
    legitimately either JSON type — historical workflows carry string ids and
    subgraph addresses are strings like ``"57:3"`` — while every lookup path
    resolves them as strings (``_find_by_str``). Building the target from the
    raw value gave ``7`` and ``"7"`` two different registers for one node, so
    ``_lww_gate`` never compared them and the pair converged by apply order
    (adversarial finding, comfy-multi-player PR #6725). Interior writes already
    normalized their path; every case now matches.
    """
    kind = op["op"]
    if kind == "set_widget":
        # Subgraph writes target the resolved interior path so the flat promoted
        # form (``57.text``) and the nested form (``57/27.text``) that land on the
        # same interior widget share one write target (converge, not clobber).
        if op.get("path"):
            return ("widget", tuple(str(s) for s in op["path"]), op["inner_widget"])
        return ("widget", str(op["node_id"]), op["widget"])
    if kind in ("add_node", "delete_node"):
        return ("node", str(op["node_id"]))
    if kind == "connect":
        grow = op.get("grow")
        if grow is not None:
            # Two autogrow connects onto the same base share a target (their
            # relative order in the batch is the sequence decision the merge
            # consumer must make); distinct bases don't collide.
            return ("input", str(op["to_node"]), "grow", str(grow["name"]).split(".", 1)[0])
        return ("input", str(op["to_node"]), op["to_slot"])
    return (kind,)


def detect_conflict(a: dict, b: dict) -> bool:
    """True iff two ops write the same target incompatibly — the signal V0's
    ask-to-merge raises instead of silently clobbering. Two autogrow connects to
    the same base conflict here (their batch order is undecidable leaderlessly)
    even though :func:`apply_op` keeps both connections and ``canonical`` treats
    their order as immaterial."""
    if _write_target(a) != _write_target(b):
        return False
    if a["op"] == "set_widget" and b["op"] == "set_widget":
        return a.get("value") != b.get("value")
    return True


def _slot_identity(inp: dict) -> tuple:
    """A position-independent identity for an input slot: an autogrown slot is
    keyed by its ``grow_id`` (stable across apply order), a fixed slot by name."""
    if isinstance(inp, dict) and inp.get("grow_id") is not None:
        return ("grow", inp["grow_id"])
    return ("name", inp.get("name") if isinstance(inp, dict) else None)


def canonical(workflow: dict) -> dict:
    """A comparison-stable view: strip apply bookkeeping and normalize every
    order-dependent-but-semantically-immaterial detail away. Two graphs that
    converged are ``canonical``-equal regardless of the order ops were applied in.

    Normalizations: nodes ordered by id; links ordered by id AND their target
    slot resolved from a raw list index to a position-independent identity (so a
    concurrently-grown slot landing at a different index still matches);
    autogrown input slots ordered by ``grow_id`` with their order-dependent
    display name folded out; subgraph definitions ordered by id.
    """
    w = copy.deepcopy(workflow)
    w.pop("_applied_ops", None)
    w.pop("_widget_stamps", None)
    nodes = w.get("nodes")
    # Capture each node's original index -> slot identity BEFORE reordering
    # inputs, so links (which reference the raw index) can be rewritten.
    # Node/link ids are legitimately either JSON type (amendment v1.2) — every
    # key and sort below normalizes with ``str()``: the oracle must be able to
    # COMPARE any legal document, and a raw-typed sort key raised ``TypeError``
    # on the very int/string mix the vocabulary declares legal, while a
    # raw-typed identity key made ``7`` and ``"7"`` miss each other.
    slot_identity: dict[str, dict[int, tuple]] = {}
    if isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            slot_identity[str(n.get("id"))] = {i: _slot_identity(inp) for i, inp in enumerate(n.get("inputs") or [])}
        # Reorder each node's grown slots deterministically (by grow_id) and drop
        # their display name, which is order-dependent (image0 vs image1).
        for n in nodes:
            if not isinstance(n, dict) or not isinstance(n.get("inputs"), list):
                continue
            fixed = [i for i in n["inputs"] if not (isinstance(i, dict) and i.get("grow_id") is not None)]
            grown = sorted(
                (i for i in n["inputs"] if isinstance(i, dict) and i.get("grow_id") is not None),
                key=lambda i: str(i["grow_id"]),
            )
            for i in grown:
                i["name"] = "\x00grow"
            n["inputs"] = fixed + grown
        w["nodes"] = sorted(nodes, key=lambda n: str(n.get("id")))
    links = w.get("links")
    if isinstance(links, list):
        canon = []
        for ln in links:
            ln = list(ln)
            if len(ln) >= 5:
                ident = slot_identity.get(str(ln[3]), {}).get(ln[4]) if isinstance(ln[4], int) else None
                if ident is not None:
                    ln[4] = ident
            canon.append(ln)
        w["links"] = sorted(canon, key=lambda ln: str(ln[0]))
    defs = (w.get("definitions") or {}).get("subgraphs")
    if isinstance(defs, list):
        w["definitions"]["subgraphs"] = sorted(defs, key=lambda sg: str(sg.get("id", "")))
    return w


#: Save-format version this module emits — array-style `links`, matching what
#: `apply_op` builds and what the frontend's zod schema expects.
SAVE_FORMAT_VERSION = 0.4


def _max_id(values) -> int:
    """Largest non-negative int in ``values``; 0 when there is none."""
    best = 0
    for v in values:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > best:
            best = n
    return best


def complete_save_format(workflow: dict) -> dict:
    """Fill the save-format keys a consumer is entitled to assume.

    We emitted ``{nodes, links, last_node_id}`` and omitted ``version`` and
    ``last_link_id``. The frontend's ``validateComfyWorkflow`` zod schema
    requires them, so every consumer had to patch the document before it could
    be used — the cloud agent carried a whole module (`internal/draft/
    normalize.go`) doing exactly this. Completing it at the source deletes that
    for every current and future caller.

    Only ABSENT keys are filled: a document that already declares a version or
    carries its own counters keeps them, so this never rewrites a producer's
    intent. Counters are derived from content, so they cannot under-report an
    id that is actually in use.
    """
    workflow.setdefault("version", SAVE_FORMAT_VERSION)

    if "last_node_id" not in workflow:
        nodes = workflow.get("nodes")
        workflow["last_node_id"] = (
            _max_id(n.get("id") for n in nodes if isinstance(n, dict)) if isinstance(nodes, list) else 0
        )

    if "last_link_id" not in workflow:
        links = workflow.get("links")
        # A link row is [link_id, src, src_slot, tgt, tgt_slot, type].
        workflow["last_link_id"] = (
            _max_id(ln[0] for ln in links if isinstance(ln, list | tuple) and ln) if isinstance(links, list) else 0
        )
    return workflow


def strip_internal(workflow: dict) -> dict:
    """Remove apply-only bookkeeping and complete the save format before serializing.

    Called at every point the document leaves this process, so the two
    guarantees — no internal bookkeeping, no missing save-format keys — hold for
    file writes, ``--stdout`` and batch output alike.
    """
    workflow.pop("_applied_ops", None)
    workflow.pop("_widget_stamps", None)
    return complete_save_format(workflow)


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------


def _build_node(node_id: int, class_type: str, m, graph, pos: list, size: list) -> dict:
    inputs = [{"name": p.name, "type": p.type, "link": None} for p in m.inputs if p.is_link]
    outputs = [{"name": p.name, "type": p.type, "links": []} for p in m.outputs]
    # Widget values in positional order, including dynamic-combo selectors and
    # their sub-widgets — sourced from the engine so add-node matches the converter.
    defaults = graph.widget_defaults(class_type)
    widgets = [defaults.get(name) for name in graph.widget_order_default(class_type)]
    return {
        "id": node_id,
        "type": class_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {},
        "widgets_values": widgets,
    }


def _widget_index(graph, class_type: str, widget: str, widgets_values=None) -> int:
    # Node-aware: expand a dynamic combo's sub-widgets by this node's actual
    # selected key (from ``widgets_values``), not the schema's first key, so the
    # index stays aligned to ``widgets_values`` for the node's real selection.
    order = graph.widget_order_for_node(class_type, widgets_values)
    if widget not in order:
        avail = [w for w in order if w != "control_after_generate"]
        raise ValueError(
            f"widget {widget!r} not found on {class_type}; "
            f"available: {', '.join(avail) if avail else '(none — all inputs are links)'}"
        )
    return order.index(widget)


class FatalFindingError(ValueError):
    """A catalog finding the server will reject — the edit is refused.

    Carries the finding verbatim so the command layer can emit ``code``,
    ``value``, ``field`` and ``did_you_mean`` as ENVELOPE FIELDS. Previously
    these shipped as a soft warning on an ``ok:true`` envelope and every
    consumer had to re-derive fatality by parsing the message text; the agent
    grew ~750 lines of Go doing exactly that.

    Subclasses ValueError so existing ``except ValueError`` handlers still
    catch it — they just lose the structure.
    """

    def __init__(self, finding: dict):
        self.finding = finding
        super().__init__(finding.get("message", finding.get("code", "invalid value")))


def _validate_widget(graph, class_type: str, widget: str, value: Any) -> list[dict]:
    """Shape-validate a widget value (hard error) and collect catalog warnings
    (soft — e.g. unknown COMBO option, out-of-range number)."""
    m = graph.node(class_type)
    if m is None:
        return []
    port = next((p for p in m.inputs if p.name == widget), None)
    if port is None:
        return []
    err = port.validate_shape(value)
    if err:
        raise ValueError(err)
    findings = port.validate_catalog(value)
    # A value the server will reject must not reach the document. `validate`
    # already refused these (`Graph._validate_catalog_value` puts them in
    # `errors`); the edit path was the last surface still writing them and
    # returning ok:true.
    from comfy_cli.cql.engine import SEVERITY_ERROR

    fatal = next((f for f in findings if f.get("severity") == SEVERITY_ERROR), None)
    if fatal is not None:
        raise FatalFindingError(fatal)
    return findings


def _normalize_slot_name(name: Any) -> str:
    """Case/separator-insensitive key for a slot name.

    Lowercased with every run of non-alphanumerics collapsed to a single "_", so
    `IMAGE`/`image`, `model task_id`/`MODEL_TASK_ID` and `Florence2_Model`/
    `florence2_model` all agree. Used only as a FALLBACK after exact matching, and
    only when it identifies exactly one slot.
    """
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _resolve_output_slot(node: dict, graph, slot: Any) -> tuple[int, str]:
    outs = node.get("outputs") or []
    if isinstance(slot, int) or (isinstance(slot, str) and slot.lstrip("-").isdigit()):
        i = int(slot)
        if not (0 <= i < len(outs)):
            raise ValueError(f"output slot {i} out of range for node {node.get('id')}")
        return i, outs[i].get("type", "*")
    for i, o in enumerate(outs):
        if o.get("name") == slot:
            return i, o.get("type", "*")
    # Exact match failed. Accept an unambiguous case/separator variant: an output
    # named `image` addressed as `IMAGE`, or `model task_id` as `MODEL_TASK_ID`.
    # Callers reach for the TYPE string when they were never shown the name, and
    # for these the intent is unambiguous. Guarded to EXACTLY ONE match so a node
    # carrying both `mask` and `MASK` (the only such collision class in the 3573
    # -node catalog) keeps failing rather than being silently guessed.
    want = _normalize_slot_name(slot)
    if want:
        hits = [i for i, o in enumerate(outs) if _normalize_slot_name(o.get("name")) == want]
        if len(hits) == 1:
            i = hits[0]
            return i, outs[i].get("type", "*")
    # Still unmatched. When the node has exactly ONE output there is no
    # ambiguity to guess through — it's the only thing the caller could have
    # meant, even when the requested name is an outright rename rather than a
    # case/separator variant (prod: LUMA_RAY32_KEYFRAME -> 'keyframes',
    # ELEVENLABS_VOICE -> 'voice', IMAGE -> 'images'). Multi-output nodes keep
    # today's behavior: an unmatched name stays ambiguous and errors below.
    if len(outs) == 1:
        return 0, outs[0].get("type", "*")
    names = [o.get("name") for o in outs]
    raise ValueError(f"output {slot!r} not found on node {node.get('id')}; outputs: {names}")


def _resolve_input_slot(node: dict, graph, slot: Any) -> int:
    ins = node.get("inputs") or []
    if isinstance(slot, int) or (isinstance(slot, str) and slot.lstrip("-").isdigit()):
        i = int(slot)
        if not (0 <= i < len(ins)):
            raise ValueError(f"input slot {i} out of range for node {node.get('id')}")
        return i
    for i, inp in enumerate(ins):
        if inp.get("name") == slot:
            return i
    names = [inp.get("name") for inp in ins]
    raise ValueError(f"input {slot!r} not found on node {node.get('id')}; inputs: {names}")


_INPUTCOUNT_KEY_RE = re.compile(r"^(.+)_(\d+)$")

INPUTCOUNT_WIDGET = "inputcount"


def inputcount_family_elements(graph, node_type: str) -> list[str]:
    """The element bases of ``node_type``'s kijai ``inputcount`` family, sorted
    — ``["image"]`` for ImageBatchMulti, ``[]`` for anything that isn't one.

    Both detection signals must be present (see
    :func:`_inputcount_family_match`, which is defined in terms of this): a
    required INT widget named exactly ``inputcount``, PLUS at least one
    ``{elem}_1`` sibling input. Exposed (not private) because the family is a
    *class-level* property of the schema, and exporters — ``nodes
    widget-catalog``, which ships this to the CRDT applier — need to ask about
    a class without first inventing a slot name to probe with.

    Returns ``[]`` when ``graph`` is unavailable (offline edit) or the class
    isn't in the catalog."""
    if graph is None:
        return []
    schema = graph.node(node_type)
    if schema is None:
        return []
    if not any(p.name == INPUTCOUNT_WIDGET and p.type == "INT" and not p.is_link for p in schema.inputs):
        return []
    return sorted({p.name[: -len("_1")] for p in schema.inputs if p.name.endswith("_1")})


def _inputcount_family_match(graph, node_type: str, slot: str) -> tuple[str, int] | None:
    """Detect a kijai ``inputcount``-family numbered key (e.g. ``image_3`` on
    ImageBatchMulti) and split it into ``(elem, n)``. This family is NOT
    autogrow-typed (no ``COMFY_AUTOGROW`` marker) — the schema declares fixed
    inputs plus an ``inputcount`` widget the node reads at runtime, and bare
    1-based keys ARE the correct wire address (unlike autogrow's dotted
    ``base.elemN``).

    Detection signal — pinned against ImageBatchMulti's production
    object_info entry (``services/ingest/data/object_info.json``): the
    schema declares a required INT widget named exactly ``inputcount`` PLUS
    a ``{elem}_1`` sibling input for the requested element (``image_1``,
    ``mask_1``, ``conditioning_1``, ``string_1``, … across the KJNodes
    ``*Multi`` family — ImageBatchMulti, MaskBatchMulti,
    ConditioningMultiCombine, ImageConcatMulti, JoinStringMulti, …). Both
    signals must be present so a coincidentally-named ``foo_3`` input on an
    unrelated node type is never misclassified.

    Returns ``None`` when ``graph`` is unavailable (offline edit), ``slot``
    isn't shaped ``{elem}_<N>``, or the node's schema doesn't carry both
    signals."""
    m = _INPUTCOUNT_KEY_RE.fullmatch(slot)
    if not m:
        return None
    elem, n_str = m.group(1), m.group(2)
    n = int(n_str)
    if n < 1:
        return None
    if elem not in inputcount_family_elements(graph, node_type):
        return None
    return elem, n


def _resolve_input_target(node: dict, graph, slot: Any, elem_type: str | None) -> tuple[int | None, dict | None]:
    """Resolve a connect target. Returns ``(index, None)`` for a concrete input,
    or ``(None, grow)`` where ``grow`` is the input slot to append (autogrow slot,
    or a widget converted to an input).

    - **Autogrow** (``COMFY_AUTOGROW_V3``, e.g. ``BatchImagesNode.images``) declares
      one base input but the server wants one slot key per connection
      (``images.image0``, …). Addressing the base or a dotted ``images.imageN`` key
      grows a concrete slot minted with the source type.
    - **Widget → input** (e.g. ``CreateVideo.fps``): a widget-backed input isn't a
      link slot, so it's converted — a linked input carrying a ``widget`` marker is
      appended. Its value stays in ``widgets_values`` (positional alignment holds);
      the API converter reads the link and skips the widget by name.
    """
    ins = node.get("inputs") or []
    node_type = node.get("type", "")
    # Concrete slot (index or exact name) that is NOT an autogrow base.
    try:
        idx = _resolve_input_slot(node, None, slot)
        if str(ins[idx].get("type", "")).startswith("COMFY_AUTOGROW"):
            base = ins[idx].get("name")
            template = _autogrow_template(graph, node_type, base)
            return None, _plan_autogrow(ins, base, elem_type, template)
        return idx, None
    except ValueError:
        pass
    # Dotted autogrow key (images.image0) or a base that has no concrete slot yet.
    if isinstance(slot, str):
        base = slot.split(".", 1)[0]
        ag = next(
            (i for i in ins if i.get("name") == base and str(i.get("type", "")).startswith("COMFY_AUTOGROW")), None
        )
        if ag is not None:
            template = _autogrow_template(graph, node_type, base)
            grow = _plan_autogrow(ins, base, elem_type, template)  # canonical next sequential slot
            # Addressing the bare base auto-appends. A dotted key is accepted ONLY if
            # it names that exact next slot; an index gap (images.image4), a doubled
            # prefix (images.images.image0), or a stray element (images.foo) would mint
            # a key the server can't map — reject it with the fix instead of growing it.
            if "." in slot and slot != grow["name"]:
                grown = [i.get("name") for i in ins if str(i.get("name", "")).startswith(base + ".")]
                raise ValueError(
                    f"input {slot!r} is not a valid autogrow slot on node {node.get('id')}; "
                    f"autogrow input {base!r} appends one sequential slot per connection "
                    f"(existing: {grown}) — connect to the base {base!r} to auto-append, "
                    f"or use the next free key {grow['name']!r}"
                )
            return None, grow
    # Widget-backed input: convert the widget to a linked input.
    if graph is not None and isinstance(slot, str) and slot in graph.widget_order(node_type):
        return None, {"name": slot, "type": elem_type or "*", "widget": slot}
    # Bare autogrow ELEMENT name (`image1` for base `images`) — the guess agents
    # make on classic batch nodes, and the top workflow-edit failure in alpha
    # traffic. Map it onto the dotted key it implies and hold it to the same
    # next-sequential rule as an explicit dotted target: the canonical next slot
    # grows; anything else is rejected with the base and the exact next free key,
    # instead of the generic not-found that never mentions autogrow at all.
    if isinstance(slot, str):
        for ag in ins:
            base = ag.get("name")
            if not base or not str(ag.get("type", "")).startswith("COMFY_AUTOGROW"):
                continue
            template = _autogrow_template(graph, node_type, base)
            if not _autogrow_bare_slot_pattern(base, template).fullmatch(slot):
                continue
            grow = _plan_autogrow(ins, base, elem_type, template)
            if f"{base}.{slot}" == grow["name"]:
                return None, grow
            grown = [i.get("name") for i in ins if str(i.get("name", "")).startswith(base + ".")]
            raise ValueError(
                f"input {slot!r} addresses autogrow input {base!r} on node {node.get('id')} "
                f"but is not the next sequential slot (existing: {grown}) — connect to the "
                f"base {base!r} to auto-append, or use the next free key {grow['name']!r}"
            )
    # kijai `inputcount` family (ImageBatchMulti, MaskBatchMulti, …) — see
    # _inputcount_family_match. Bare 1-based keys are the correct address;
    # growing one must also bump the `inputcount` widget (carried on `grow`
    # and applied through the same LWW-stamped path set_widget uses, see
    # _apply_connect) or the node never reads the new slot.
    if isinstance(slot, str):
        fam = _inputcount_family_match(graph, node_type, slot)
        if fam is not None:
            elem, n = fam
            existing = [i for i in ins if re.fullmatch(rf"{re.escape(elem)}_\d+", str(i.get("name", "")))]
            next_n = len(existing) + 1
            if n == next_n:
                return None, {
                    "name": slot,
                    "type": elem_type or "*",
                    "inputcount": {"widget": "inputcount", "value": n},
                }
            grown = [i.get("name") for i in existing]
            next_key = f"{elem}_{next_n}"
            raise ValueError(
                f"input {slot!r} addresses inputcount input {elem!r} on node {node.get('id')} "
                f"but is not the next sequential slot (existing: {grown}) — inputcount nodes "
                f"grow sequentially; use the next free key {next_key!r}"
            )
    names = [i.get("name") for i in ins]
    raise ValueError(f"input {slot!r} not found on node {node.get('id')}; inputs: {names}")


def _autogrow_bare_slot_pattern(base: str, template: dict | None) -> re.Pattern:
    """A regex recognizing a bare element name (no ``base.`` prefix, e.g.
    ``image1`` for base ``images``) as plausibly addressing this autogrow
    input, so a guessed bare name resolves to the actionable fix rather than a
    generic not-found. Matches the schema's element vocabulary when known —
    any literal ``names`` entry, or its ``{names[-1]}N`` overflow form, or
    ``{prefix}N`` — else the historical ``{stem}N`` pluralization guess when
    ``template`` is None."""
    if template:
        names = template.get("names")
        if names:
            alts = "|".join(re.escape(n) for n in names)
            return re.compile(rf"(?:{alts})|{re.escape(names[-1])}\d+")
        prefix = template.get("prefix")
        if prefix:
            return re.compile(re.escape(prefix) + r"\d+")
    stem = base[:-1] if base.endswith("s") else base
    return re.compile(re.escape(stem) + r"\d+")


def _plan_autogrow(ins: list, base: str, elem_type: str | None, template: dict | None = None) -> dict:
    """The canonical next autogrow slot for ``base``. Element name comes from
    the node schema when known — ``template["names"][N]`` verbatim, else
    ``f"{prefix}{N}"`` (0-based) — falling back to the historical
    ``{base}.{base[:-1]}{N}`` heuristic only when ``template`` is unavailable
    (schema unavailable: offline edit, catalog miss). Callers validate any
    explicitly requested key against this name before growing."""
    taken = {str(i.get("name", "")) for i in ins}
    elem = _autogrow_elem_name(base, _first_free_autogrow_index(taken, base, template), template)
    return {"name": f"{base}.{elem}", "type": elem_type or "*"}
