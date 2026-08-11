"""Pure-Python CQL graph engine.

Parses ComfyUI's ``object_info.json``, builds an indexed compatibility graph,
and exposes upstream/downstream, path-finding, validation, annotations,
and widget-order resolution.

Port of ``github.com/Comfy-Org/cql/nodegraph`` (Go).
"""

from __future__ import annotations

import copy
import difflib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid as _uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from comfy_cli.cql._net import is_loopback_host
from comfy_cli.http import NoRedirectHandler, build_http_only_opener

# ---------------------------------------------------------------------------
# Types — mirrors nodegraph/types.go
# ---------------------------------------------------------------------------

_IMPLICIT_WIDGET_TYPES = frozenset({"STRING", "INT", "FLOAT", "NUMBER", "BOOLEAN", "COMBO"})

# Work budget for ``Graph.search_paths``: the number of frontier states it will
# expand before giving up and reporting ``truncated``. A full cloud catalog has
# thousands of nodes, so an unreachable target must fail fast rather than walk
# the whole type lattice.
_MAX_PATH_SEARCH_STATES = 20_000


@dataclass
class PortOptions:
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: Any = None
    multiline: bool = False
    control_after_generate: bool = False
    force_input: bool = False


@dataclass
class Port:
    name: str
    type: str
    required: bool = False
    is_link: bool = False
    enum_values: list[Any] = field(default_factory=list)  # preserves the option's real type (int combos stay int)
    # True when object_info shipped an explicit choice list for this input —
    # *including an empty one*. An empty declared list means "the server has
    # zero options here" (a model folder with no files installed), which is a
    # different statement from "the choices aren't in object_info at all"
    # (remote/dynamic combos, whose options the frontend fetches at runtime).
    # Only the former is safe to validate against.
    enum_declared: bool = False
    options: PortOptions = field(default_factory=PortOptions)
    # The verbatim ``INPUT_TYPES`` spec this port was parsed from. Retained
    # (by reference — no copy) because a dynamic combo's sub-input schema lives
    # in the spec's ``options`` blocks and is NOT recoverable from the parsed
    # fields above. Output ports carry ``None``.
    raw_spec: Any = None

    @property
    def is_autogrow(self) -> bool:
        """V3 autogrow input (e.g. BatchImagesNode.images): the schema declares
        ONE input, but the server expects autogrown slot keys —
        ``images.image0``, ``images.image1``, … one per connection."""
        return self.type.startswith("COMFY_AUTOGROW")

    @property
    def is_dynamic_combo(self) -> bool:
        """V3 dynamic combo (e.g. ``COMFY_DYNAMICCOMBO_V3``): the schema declares
        ONE selector input, but the option the selector names carries its own
        ``INPUT_TYPES`` block, which the frontend — and ``convert_ui_to_api`` —
        lower into dotted slot keys (``model.size_preset``, ``model.width``, …).

        Deliberately the same test the converter applies
        (``workflow_to_api._is_widget_input``), so validation expands exactly the
        tree the converter lowered."""
        return self.type.startswith("COMFY_") and "COMBO" in self.type

    def autogrow_slot_example(self) -> str:
        """Best-effort slot-key example for hints. The element name comes from
        the node's V3 definition and isn't in object_info; the observed server
        convention is the singular of the input name (images → image0)."""
        stem = self.name[:-1] if self.name.endswith("s") else self.name
        return f"{self.name}.{stem}0, {self.name}.{stem}1, …"

    def validate_shape(self, value: Any) -> str | None:
        """Hard-reject on JSON-shape mismatch. Returns error message or None."""
        if self.type == "INT":
            if isinstance(value, bool) or not isinstance(value, int | float):
                return f"{self.name}: expected INT, got {type(value).__name__} {value!r}"
            if isinstance(value, float) and value != int(value):
                return f"{self.name}: expected integer, got {value}"
        elif self.type in ("FLOAT", "NUMBER"):
            if isinstance(value, bool) or not isinstance(value, int | float):
                return f"{self.name}: expected {self.type}, got {type(value).__name__}"
        elif self.type == "STRING":
            if not isinstance(value, str):
                return f"{self.name}: expected STRING (string), got {type(value).__name__}"
        elif self.type == "COMBO":
            # COMBO options are usually strings, but the server also ships
            # int-valued combos (e.g. LTXV `duration`/`fps`). Accept any
            # scalar here; membership is the catalog enum check's job. Only
            # bool and container/None shapes are a true mismatch.
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                return f"{self.name}: expected COMBO (string or number), got {type(value).__name__}"
        elif self.type == "BOOLEAN":
            if not isinstance(value, bool):
                return f"{self.name}: expected BOOLEAN, got {type(value).__name__}"
        return None

    def validate_catalog(self, value: Any) -> list[dict]:
        """Soft checks against catalog snapshot. Returns warnings list."""
        if self.validate_shape(value) is not None:
            return []
        warnings: list[dict] = []
        if self.type == "COMBO" and self.enum_values:
            # Membership compares on the stringified form BOTH ways, so a value
            # matches its option regardless of int/str (`8` ↔ "8", `8.0` ↔ "8").
            # This keeps validate lenient (never false-warns on a real value)
            # while the displayed schema keeps the option's true type. The
            # warning carries the FULL valid list (typed) so a rejection tells
            # the agent exactly what to pick — no truncation, no guessing.
            candidates = {str(value)}
            if isinstance(value, float) and value.is_integer():
                candidates.add(str(int(value)))
            enum_str = {str(e) for e in self.enum_values}
            if not (candidates & enum_str):
                warnings.append(
                    {
                        "code": "unknown_enum_value",
                        "field": self.name,
                        "message": f"{value!r} not in {len(self.enum_values)} known options for {self.name}",
                        "valid_options": list(self.enum_values),
                    }
                )
        elif self.type == "COMBO" and self.enum_declared:
            # The server declared this field's choices and shipped NONE of them:
            # the folder backing it is empty — no models (or inputs) installed.
            # An empty option list is STRONGER evidence the value is unavailable
            # than a populated one, not weaker: the server rejects every value
            # against an empty list (`execution.validate_inputs` →
            # "Value not in list"), so skipping the check here just defers the
            # failure to run time — and it fails hardest on the fresh install
            # this check exists to serve, where the big downloads (UNETLoader,
            # CLIPLoader) have no built-in options to compare against.
            # Ports whose options are not in object_info at all keep
            # `enum_declared=False` and stay unconstrained (see the field).
            warnings.append(
                {
                    "code": "no_options_available",
                    "field": self.name,
                    "message": f"{value!r} is unavailable: the server reports 0 installed options for {self.name}",
                    "valid_options": [],
                }
            )
        if self.type in ("INT", "FLOAT", "NUMBER") and isinstance(value, int | float):
            if self.options.min is not None and value < self.options.min:
                warnings.append(
                    {
                        "code": "below_min",
                        "field": self.name,
                        "message": f"{self.name}={value} below catalog min {self.options.min}",
                    }
                )
            if self.options.max is not None and value > self.options.max:
                warnings.append(
                    {
                        "code": "above_max",
                        "field": self.name,
                        "message": f"{self.name}={value} above catalog max {self.options.max}",
                    }
                )
        return warnings


@dataclass
class Morphism:
    id: str
    display_name: str = ""
    description: str = ""
    category: str = ""
    inputs: list[Port] = field(default_factory=list)
    outputs: list[Port] = field(default_factory=list)
    is_output_node: bool = False
    is_api_node: bool = False
    deprecated: bool = False
    experimental: bool = False
    search_aliases: list[str] = field(default_factory=list)
    pack: str = ""
    labels: list[str] = field(default_factory=list)
    cloud_disabled: bool = False

    def output_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.outputs:
            if p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def input_link_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.inputs:
            if p.is_link and p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def required_link_types(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in self.inputs:
            if p.is_link and p.required and p.type not in seen:
                seen.add(p.type)
                out.append(p.type)
        return out

    def has_input(self, t: str) -> bool:
        return any(p.is_link and p.type == t for p in self.inputs)

    def has_output(self, t: str) -> bool:
        return any(p.type == t for p in self.outputs)

    def can_apply(self, available: set[str]) -> bool:
        return all(t in available for t in self.required_link_types())


# ---------------------------------------------------------------------------
# Parsing — mirrors nodegraph/parse.go
# ---------------------------------------------------------------------------


def _is_link(type_id: str, is_enum: bool, force_input: bool) -> bool:
    """Determine if an input participates in typed wiring (link) or is inline (widget)."""
    if is_enum:
        return False
    if type_id in _IMPLICIT_WIDGET_TYPES and not force_input and type_id != "*":
        return False
    return True


def _derive_pack(python_module: str) -> str:
    if not python_module:
        return "core"
    if (
        python_module.startswith("nodes")
        or python_module.startswith("comfy_extras")
        or python_module.startswith("comfy.comfy_types")
    ):
        return "core"
    if python_module.startswith("custom_nodes."):
        parts = python_module.split(".", 3)
        if len(parts) >= 2:
            return parts[1]
    return "core"


def _parse_port_options(opts_raw: dict) -> PortOptions:
    return PortOptions(
        min=opts_raw.get("min"),
        max=opts_raw.get("max"),
        step=opts_raw.get("step"),
        default=opts_raw.get("default"),
        multiline=bool(opts_raw.get("multiline", False)),
        control_after_generate=_control_after_generate_set(opts_raw.get("control_after_generate")),
        force_input=bool(opts_raw.get("forceInput", False)),
    )


def _control_after_generate_set(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val != "" and val != "false"
    return True


def _parse_input_spec(spec: Any) -> tuple[str, bool, list[Any], PortOptions, bool]:
    """Returns (type_id, is_enum, enum_values, options, enum_declared).

    ``enum_declared`` is True when the spec shipped an explicit choice list —
    *including an empty one* — so validation can tell "this server has zero
    options for the field" from "this field's options aren't in object_info".
    It is deliberately separate from ``is_enum`` (which stays truthiness-based)
    because ``is_enum`` also decides link-vs-widget, and an empty option list
    must not move a port between those.
    """
    if isinstance(spec, str):
        return spec, False, [], PortOptions(), False

    if not isinstance(spec, list) or len(spec) == 0:
        return "UNKNOWN", False, [], PortOptions(), False

    opts_raw = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    port_opts = _parse_port_options(opts_raw)

    first = spec[0]
    if isinstance(first, str):
        # V3 / partner-API combo dialect: the type is the literal string
        # "COMBO" (or a dynamic-combo type) and the choices live in the
        # options dict, e.g. ["COMBO", {"options": ["480p", "720p"]}].
        # Without this, dict-form combos lose their enum and validate can't
        # enum-check them — exactly the partner nodes (ByteDance, BFL, …)
        # where the choices array is the precision check.
        options = opts_raw.get("options")
        if isinstance(options, list) and all(_is_scalar_choice(v) for v in options):
            # Keep each option's real type: an int-valued combo (Sora-2/LTXV
            # `duration`) must stay [4, 8, 12], not ["4","8","12"], so `nodes
            # show` is truthful and agents pass the type the cloud accepts.
            # An EMPTY list is a declared-but-unpopulated combo — still not an
            # enum for wiring purposes (`is_enum` stays False, exactly as
            # before), but flagged as declared so validate can say "0 options
            # installed" instead of silently skipping the check.
            return first, bool(options), list(options), port_opts, True
        # No usable `options` key at all: a remote/dynamic combo whose choices
        # the frontend fetches at runtime. Unknowable here — stay unconstrained.
        return first, False, [], port_opts, False

    if isinstance(first, list):
        # Same: preserve the option types for the classic list-form combo. The
        # list IS the declaration, so an empty one (a model folder with nothing
        # installed) is declared-but-empty, not unconstrained.
        return "COMBO", True, list(first), port_opts, True

    return "UNKNOWN", False, [], port_opts, False


def _is_scalar_choice(v: Any) -> bool:
    """A combo option is enumerable only if it's a scalar. Dynamic combos
    (COMFY_DYNAMICCOMBO_V3) carry dict options describing sub-inputs — those
    are not membership choices and must not be flattened into enum_values."""
    return isinstance(v, str | int | float) and not isinstance(v, bool)


def _ordered_names(raw: dict, order: list[str] | None) -> list[str]:
    """Return input names in declared order, falling back to alphabetical."""
    seen: set[str] = set()
    out: list[str] = []
    for name in order or []:
        if name in raw and name not in seen:
            out.append(name)
            seen.add(name)
    for name in sorted(raw.keys()):
        if name not in seen:
            out.append(name)
    return out


def _parse_inputs(raw: dict, order: list[str] | None, required: bool) -> list[Port]:
    ports: list[Port] = []
    for name in _ordered_names(raw, order):
        spec = raw[name]
        type_id, is_enum, enum_values, opts, enum_declared = _parse_input_spec(spec)
        ports.append(
            Port(
                name=name,
                type=type_id,
                required=required,
                is_link=_is_link(type_id, is_enum, opts.force_input),
                enum_values=enum_values,
                enum_declared=enum_declared,
                options=opts,
                raw_spec=spec,
            )
        )
    return ports


def _unmarshal_string_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val]
    return []


def _parse_morphism(node_id: str, raw: dict) -> Morphism:
    input_block = raw.get("input") or {}
    input_order = raw.get("input_order") or {}
    req_raw = input_block.get("required") or {}
    opt_raw = input_block.get("optional") or {}
    req_order = input_order.get("required")
    opt_order = input_order.get("optional")

    inputs = _parse_inputs(req_raw, req_order, required=True)
    inputs += _parse_inputs(opt_raw, opt_order, required=False)

    raw_outputs = raw.get("output") or []
    output_names = _unmarshal_string_list(raw.get("output_name"))
    outputs: list[Port] = []
    for i, out in enumerate(raw_outputs):
        name = output_names[i] if i < len(output_names) else ""
        t = out if isinstance(out, str) else "COMBO"
        outputs.append(Port(name=name, type=t, required=True, is_link=True))

    # These are declared `str` and every consumer treats them as one (`.lower()`,
    # `.startswith()`, markup escaping). /object_info is server-supplied and a
    # custom node can put any JSON type here, so coerce rather than trust the
    # annotation — an int category used to crash `nodes search` with a raw
    # AttributeError instead of the structured error the command otherwise emits.
    return Morphism(
        id=node_id,
        display_name=str(raw.get("display_name") or node_id),
        description=str(raw.get("description") or ""),
        category=str(raw.get("category") or ""),
        inputs=inputs,
        outputs=outputs,
        is_output_node=bool(raw.get("output_node", False)),
        is_api_node=bool(raw.get("api_node", False)),
        deprecated=bool(raw.get("deprecated", False)),
        experimental=bool(raw.get("experimental", False)),
        search_aliases=_unmarshal_string_list(raw.get("search_aliases")),
        pack=_derive_pack(str(raw.get("python_module") or "")),
    )


# ---------------------------------------------------------------------------
# Annotations — mirrors nodegraph/annotations.go
# ---------------------------------------------------------------------------


def parse_supported_nodes(data: bytes) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse supported_nodes.yaml → (node_pack, node_labels)."""
    try:
        import yaml

        cfg = yaml.safe_load(data)
    except Exception:
        return {}, {}
    if not isinstance(cfg, dict):
        return {}, {}
    node_pack: dict[str, str] = {}
    node_labels: dict[str, list[str]] = {}
    for pack in cfg.get("node_packs") or []:
        if not isinstance(pack, dict):
            continue
        pack_name = pack.get("name", "")
        for node_name, labels in (pack.get("node_labels") or {}).items():
            node_pack[node_name] = pack_name
            node_labels[node_name] = list(labels) if isinstance(labels, list) else []
    return node_pack, node_labels


def parse_disable_config(data: bytes) -> set[str]:
    """Parse cloud_disable_config.yaml → set of labels that disable nodes."""
    try:
        import yaml

        cfg = yaml.safe_load(data)
    except Exception:
        return set()
    if not isinstance(cfg, dict):
        return set()
    disable = cfg.get("disable_nodes") or {}
    labels: set[str] = set()
    for rule in disable.get("or") or []:
        if isinstance(rule, dict):
            for label, enabled in rule.items():
                if enabled:
                    labels.add(label)
    return labels


# ---------------------------------------------------------------------------
# Graph — mirrors nodegraph/graph.go
# ---------------------------------------------------------------------------


class Graph:
    """Indexed compatibility graph over ComfyUI node classes."""

    def __init__(self) -> None:
        self._nodes: dict[str, Morphism] = {}
        self._producers: dict[str, list[Morphism]] = defaultdict(list)
        self._consumers: dict[str, list[Morphism]] = defaultdict(list)
        self._types: set[str] = set()
        self._annotated = False
        # Lazily-computed closure of types obtainable without wiring anything in
        # (see ``free_types``). Invalidated implicitly: graphs are built once.
        self._free_types: frozenset[str] | None = None
        # The raw ``/object_info`` payload this graph was built from. Retained
        # verbatim so callers that also need to lower a UI-format workflow to
        # API format (``convert_ui_to_api``) can reuse it without a second fetch.
        self._raw: dict[str, Any] = {}

    @property
    def object_info(self) -> dict[str, Any]:
        """The raw ``/object_info`` dict this graph was built from (``{}`` if
        the graph was constructed without one).

        Read-only: this is the graph's live internal schema state, returned by
        reference to avoid copying a large payload. Callers (e.g. the validate
        command handing it to ``convert_ui_to_api``) must not mutate it.
        """
        return self._raw

    @classmethod
    def from_object_info(cls, object_info: dict[str, Any]) -> Graph:
        if not isinstance(object_info, dict):
            raise LoadError(
                "object_info must be a JSON object",
                details={"top_level_type": type(object_info).__name__},
            )
        g = cls()
        g._raw = object_info
        for node_id, raw in object_info.items():
            if not isinstance(raw, dict):
                continue
            m = _parse_morphism(node_id, raw)
            g._nodes[m.id] = m
            for t in m.output_types():
                g._producers[t].append(m)
                g._types.add(t)
            for t in m.input_link_types():
                g._consumers[t].append(m)
                g._types.add(t)
        # Sort indexes for deterministic output
        for t in g._producers:
            g._producers[t].sort(key=lambda m: m.id)
        for t in g._consumers:
            g._consumers[t].sort(key=lambda m: m.id)
        return g

    # -- Annotation --

    def annotate(
        self,
        supported_nodes_yaml: bytes | None = None,
        cloud_disable_yaml: bytes | None = None,
    ) -> None:
        node_pack: dict[str, str] = {}
        node_labels: dict[str, list[str]] = {}
        disable_labels: set[str] = set()

        if supported_nodes_yaml:
            node_pack, node_labels = parse_supported_nodes(supported_nodes_yaml)
        if cloud_disable_yaml:
            disable_labels = parse_disable_config(cloud_disable_yaml)

        for nid, m in self._nodes.items():
            if nid in node_pack:
                m.pack = node_pack[nid]
            if nid in node_labels:
                m.labels = node_labels[nid]
            m.cloud_disabled = any(label in disable_labels for label in m.labels)
        self._annotated = True

    # -- Lookup --

    def node(self, name: str) -> Morphism | None:
        # A malformed workflow can supply an unhashable class_type (list/dict);
        # dict.get on an unhashable key raises TypeError, so screen it out here
        # rather than crash the reachability walk / lookups (BE-3406 hardening).
        if not isinstance(name, str):
            return None
        return self._nodes.get(name)

    def all_nodes(self) -> list[Morphism]:
        return sorted(self._nodes.values(), key=lambda m: m.id)

    def node_count(self) -> int:
        return len(self._nodes)

    # -- Traversal --

    def upstream(self, name: str) -> list[Morphism]:
        m = self._nodes.get(name)
        if m is None:
            return []
        seen: set[str] = set()
        result: list[Morphism] = []
        for t in m.input_link_types():
            for producer in self._producers.get(t, []):
                if producer.id != name and producer.id not in seen:
                    seen.add(producer.id)
                    result.append(producer)
        result.sort(key=lambda m: m.id)
        return result

    def downstream(self, name: str) -> list[Morphism]:
        m = self._nodes.get(name)
        if m is None:
            return []
        seen: set[str] = set()
        result: list[Morphism] = []
        for t in m.output_types():
            for consumer in self._consumers.get(t, []):
                if consumer.id != name and consumer.id not in seen:
                    seen.add(consumer.id)
                    result.append(consumer)
        result.sort(key=lambda m: m.id)
        return result

    def cloud_enabled_nodes(self) -> list[Morphism]:
        """All nodes that are enabled on Comfy Cloud."""
        return sorted([m for m in self._nodes.values() if not m.cloud_disabled], key=lambda m: m.id)

    def packs(self) -> list[str]:
        """All known pack names, sorted."""
        return sorted(set(m.pack for m in self._nodes.values() if m.pack))

    def free_types(self) -> frozenset[str]:
        """Types obtainable without wiring anything in — the fixpoint closure
        over nodes whose required link inputs are already satisfied (loaders,
        primitives, text-to-X API nodes, and whatever those unlock).

        The exact walker uses this to decide whether a step's *other* required
        inputs (a ``VAE`` for ``VAEDecode``, say) could be supplied by a support
        node. Support nodes are reported per path rather than routed through, so
        they never masquerade as steps on the requested path.
        """
        if self._free_types is None:
            free: set[str] = set()
            changed = True
            while changed:
                changed = False
                for m in self._nodes.values():
                    if not m.can_apply(free):
                        continue
                    for t in m.output_types():
                        if t != "*" and t not in free:
                            free.add(t)
                            changed = True
            self._free_types = frozenset(free)
        return self._free_types

    def search_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        exact: bool = True,
        max_depth: int = 6,
        max_paths: int = 10,
        max_states: int = _MAX_PATH_SEARCH_STATES,
    ) -> dict:
        """Routed paths from ``from_type`` to ``to_type``, with honest bounds.

        Every step consumes the type the previous step produced — the first step
        consumes ``from_type`` — through a **declared link input of that type**,
        so a node that merely owns a widget *named* like the type (the COMBO
        ``model`` on the partner-API image nodes) is never routed through. Path
        length (the number of steps) is bounded by ``max_depth``.

        In ``exact`` mode a step is only taken when the node's other required
        link inputs are satisfiable — from types produced earlier on the path, or
        from a support node needing no wiring of its own (``free_types``). Those
        support inputs are reported per path under ``support`` instead of being
        spliced into ``steps``. Loose mode skips the satisfiability check and
        reports no support.

        Returns ``{"paths", "truncated", "truncated_by", "depth_limited",
        "collapsed", "not_searched", "not_searched_reason"}``:

        - ``not_searched`` — the walk declined the query outright and never ran,
          so the empty result is an abstention rather than an answer.
          ``not_searched_reason`` says which: ``"same_type"`` (FROM and TO are
          the same type — self-returning routes exist but this walker cannot
          represent them) or ``"degenerate_bounds"`` (``max_depth`` or
          ``max_paths`` below 1, a bound no path can satisfy).
        - ``truncated`` — the walk stopped early (``max_paths`` reached, or the
          internal state budget exhausted), so paths exist that are not listed.
        - ``depth_limited`` — the frontier was still expanding at ``max_depth``,
          so longer paths may exist beyond the requested bound.
        - ``collapsed`` — the walk reached some intermediate state by more than
          one route and explored it only once, so alternate chains through that
          state are not listed. Reachability is unaffected (the surviving route
          explores exactly the same continuations), which is why an **empty**
          result with all four flags false is a proof that no path exists —
          but a non-empty one is a sample of the routes, not the full set.

        A caller may only treat the listing as exhaustive when all four are
        false. Each errs toward true: hitting ``max_paths`` exactly is reported
        as truncated even when nothing further existed, and a revisited state is
        reported as collapsed even when its alternate route led nowhere.
        """
        result: dict = {
            "paths": [],
            "truncated": False,
            "truncated_by": None,
            "depth_limited": False,
            "collapsed": False,
            "not_searched": False,
            "not_searched_reason": None,
        }
        # Query shapes the walk declines outright. The empty result they yield is
        # an abstention, not a proof, so it has to say so — otherwise it reads
        # as "no route exists" with every limit flag reassuringly false.
        if from_type == to_type:
            # Self-returning routes are real (``MODEL -> LoraLoader -> MODEL``),
            # but the walk cannot represent them: the no-op rule below drops any
            # step whose output type equals its input type, and for a same-type
            # query that is the terminal step. Declining is the honest option.
            result["not_searched"] = True
            result["not_searched_reason"] = "same_type"
            return result
        if max_depth < 1 or max_paths < 1:
            result["not_searched"] = True
            result["not_searched_reason"] = "degenerate_bounds"
            return result

        free = self.free_types() if exact else frozenset()
        paths: list[dict] = result["paths"]
        # state: (current_type, types produced by the path so far, steps[])
        queue: list[tuple[str, frozenset[str], list[dict]]] = [(from_type, frozenset(), [])]
        visited: set[tuple[str, frozenset[str]]] = {(from_type, frozenset())}
        states = 0

        while queue and len(paths) < max_paths:
            next_queue: list[tuple[str, frozenset[str], list[dict]]] = []
            for cur_type, produced, steps in queue:
                consumers = self._consumers.get(cur_type, [])
                if len(steps) >= max_depth:
                    if consumers:
                        result["depth_limited"] = True
                    continue
                available = free | produced | {from_type}
                for consumer in consumers:
                    if exact and not consumer.can_apply(available):
                        continue
                    outs = [t for t in consumer.output_types() if t != "*"]
                    # Loose mode ignores availability, so keeping ``produced``
                    # empty there collapses the visited key back to the type
                    # alone — the pruning loose path-finding has always used.
                    new_produced = produced | frozenset(outs) if exact else produced
                    for out_t in outs:
                        if out_t == cur_type:
                            continue
                        step = {"node": consumer.id, "input_type": cur_type, "output_type": out_t}
                        new_steps = steps + [step]
                        if out_t == to_type:
                            paths.append(self._path_record(from_type, to_type, new_steps, free if exact else None))
                            if len(paths) >= max_paths:
                                result["truncated"] = True
                                result["truncated_by"] = "max_paths"
                                return result
                            continue
                        key = (out_t, new_produced)
                        if key in visited:
                            # A second route into a state already queued. Its
                            # continuations are covered by the first one, so
                            # dropping it costs no reachability — but the chains
                            # it would have printed are lost, so the listing can
                            # no longer be called complete.
                            result["collapsed"] = True
                            continue
                        if states >= max_states:
                            result["truncated"] = True
                            result["truncated_by"] = "max_states"
                            return result
                        states += 1
                        visited.add(key)
                        next_queue.append((out_t, new_produced, new_steps))
            queue = next_queue
        return result

    def find_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 4,
        max_paths: int = 10,
    ) -> list[dict]:
        """Loose (routing-only) paths — see ``search_paths``."""
        return self.search_paths(from_type, to_type, exact=False, max_depth=max_depth, max_paths=max_paths)["paths"]

    def exact_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 6,
        max_paths: int = 10,
    ) -> list[dict]:
        """Satisfiability-aware paths — see ``search_paths``."""
        return self.search_paths(from_type, to_type, exact=True, max_depth=max_depth, max_paths=max_paths)["paths"]

    def _path_record(self, from_type: str, to_type: str, steps: list[dict], free: frozenset[str] | None) -> dict:
        record = {"from": from_type, "to": to_type, "steps": steps}
        if free is not None:
            record["support"] = self._support_for(from_type, steps, free)
        return record

    def _support_for(self, from_type: str, steps: list[dict], free: frozenset[str]) -> list[dict]:
        """Required link inputs a routed path needs *besides* the routed type,
        each with a node that can supply it without wiring of its own."""
        available: set[str] = {from_type}
        support: list[dict] = []
        seen: set[str] = set()
        for step in steps:
            m = self._nodes.get(step["node"])
            if m is None:
                continue
            for t in m.required_link_types():
                if t in available or t in seen:
                    continue
                seen.add(t)
                support.append({"type": t, "node": self._free_producer(t, free)})
            available.update(m.output_types())
        return support

    def _free_producer(self, type_id: str, free: frozenset[str]) -> str | None:
        """A node producing ``type_id`` that needs no incoming links, preferring
        one with no link inputs at all. ``None`` when the type can only be
        obtained by wiring something up first."""
        if type_id not in free:
            return None
        producers = self._producers.get(type_id, [])
        for m in producers:
            if not m.required_link_types():
                return m.id
        for m in producers:
            if m.can_apply(free):
                return m.id
        return None

    # -- Browse --

    def list_types(self) -> list[str]:
        return sorted(self._types)

    def category_tree(self) -> dict:
        """Build a hierarchical category tree with node counts."""
        counts: dict[str, int] = defaultdict(int)
        for m in self._nodes.values():
            if m.category:
                counts[m.category] += 1

        root: dict[str, Any] = {"FullPath": "", "Count": 0, "Children": {}}
        for path, count in sorted(counts.items()):
            parts = path.split("/")
            node = root
            for i, part in enumerate(parts):
                full = "/".join(parts[: i + 1])
                if part not in node["Children"]:
                    node["Children"][part] = {"FullPath": full, "Count": 0, "Children": {}}
                child = node["Children"][part]
                child["Count"] += count
                node = child
        return {"Root": root}

    # -- Widget order --

    def widget_order(self, class_name: str) -> list[str]:
        m = self._nodes.get(class_name)
        if m is None:
            return []
        order: list[str] = []
        for p in m.inputs:
            if p.is_link:
                continue
            order.append(p.name)
            if p.options.control_after_generate:
                order.append("control_after_generate")
        return order

    # -- Validation --

    def validate_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Validate an API-format workflow. Returns {valid, errors, warnings}."""
        errors: list[dict] = []
        warnings: list[dict] = []
        all_names = list(self._nodes.keys())
        # No-outputs check: the server rejects any prompt with zero output nodes
        # (execution.py:1155-1162, prompt_no_outputs). Track whether any
        # recognized node is an output node.
        has_output_node = False
        # Server parity: ComfyUI's validate_prompt only validates output nodes
        # and their transitive input ancestors — any node not reachable from an
        # output is pruned and never validated (execution.py). Restrict the
        # promoted hard checks (required-input presence, autogrow-required, and
        # below_min/above_max ranges) to that reachable set, so a
        # disconnected/incomplete node the server would silently drop isn't
        # hard-rejected here. Edge/shape/enum checks are left as-is (pre-existing
        # behavior, out of scope for this change).
        reachable = _output_reachable_node_ids(workflow, self)

        for node_id, node_data in workflow.items():
            # `_meta` is the compose/run provenance block (schema/blueprint/items),
            # stripped before submit — not a node and not a mistake. `comfy workflow compose`
            # adds it itself, so warning here is self-inflicted noise.
            if node_id == "_meta":
                continue
            if not isinstance(node_data, dict):
                warnings.append(
                    {
                        "node_id": node_id,
                        "field": node_id,
                        "code": "non_node_key",
                        "message": f"key {node_id!r} is not a workflow node (expected a dict with class_type)",
                    }
                )
                continue
            class_type = node_data.get("class_type", "")
            # A non-string class_type (list/dict from malformed JSON) is unhashable
            # and would crash the self._nodes.get(class_type) lookup below; treat it
            # as absent so it flows to the structured non_node_key path instead.
            if not isinstance(class_type, str):
                class_type = ""
            if not class_type:
                warnings.append(
                    {
                        "node_id": node_id,
                        "field": node_id,
                        "code": "non_node_key",
                        "message": f"key {node_id!r} has no class_type and will be ignored by the server",
                    }
                )
                continue

            m = self._nodes.get(class_type)
            if m is None:
                close = difflib.get_close_matches(class_type, all_names, n=3, cutoff=0.6)
                errors.append(
                    {
                        "node_id": node_id,
                        "code": "unknown_class_type",
                        "message": f"class_type {class_type!r} not found in object_info",
                        "hint": f"did you mean: {', '.join(close)}?"
                        if close
                        else "run `comfy nodes search <name>` to find available classes",
                        "suggestions": close,
                    }
                )
                continue

            if m.is_output_node:
                has_output_node = True

            port_by_name = {p.name: p for p in m.inputs}
            # V3 autogrow inputs are declared once (e.g. `images`) but wired as
            # slot keys (`images.image0`, `images.image1`, …). Track which
            # autogrow ports actually received a slot so the required-but-empty
            # case surfaces here instead of as a cryptic server reject.
            autogrow_ports = {p.name: p for p in m.inputs if p.is_autogrow}
            autogrow_seen: set[str] = set()
            node_inputs = node_data.get("inputs")
            # A truthy non-dict `inputs` (e.g. a string/list from malformed JSON)
            # sails through `or {}` and crashes `.items()`; treat it as empty so
            # required-input checks flag the absence instead of raising.
            if not isinstance(node_inputs, dict):
                node_inputs = {}
            for input_name, value in node_inputs.items():
                if autogrow_ports and "." in input_name:
                    base = input_name.split(".", 1)[0]
                    if base in autogrow_ports:
                        autogrow_seen.add(base)
                if input_name in autogrow_ports and isinstance(value, list) and len(value) == 2:
                    port = autogrow_ports[input_name]
                    errors.append(
                        {
                            "node_id": node_id,
                            "field": input_name,
                            "code": "autogrow_bare_input",
                            "message": (
                                f"input {input_name!r} is an autogrow input ({port.type}) and cannot be "
                                f"wired as a single connection — the server expects one slot key per "
                                f"connection"
                            ),
                            "hint": f"wire one key per connection: {port.autogrow_slot_example()} "
                            f'(e.g. "{input_name}.{input_name[:-1] if input_name.endswith("s") else input_name}0": '
                            f"[{value[0]!r}, {value[1]!r}])",
                        }
                    )
                    continue
                # Link references: [source_node_id, output_index]
                if isinstance(value, list) and len(value) == 2:
                    src_id = str(value[0])
                    out_idx = value[1] if isinstance(value[1], int) else None

                    # (i) source node exists in workflow
                    src_data = workflow.get(src_id)
                    if not isinstance(src_data, dict) or not src_data.get("class_type"):
                        errors.append(
                            {
                                "node_id": node_id,
                                "field": input_name,
                                "code": "dangling_edge",
                                "message": f"input {input_name!r} references node {src_id!r} which does not exist",
                                "hint": f"add node {src_id!r} to the workflow, or rewire this input to an existing node",
                            }
                        )
                        continue

                    src_class = src_data["class_type"]
                    # Route through the guarded lookup: a referenced node with an
                    # unhashable class_type (malformed JSON) would otherwise crash
                    # the dict.get here.
                    src_m = self.node(src_class)
                    if src_m is None:
                        # Source class_type already flagged by the outer loop
                        continue

                    # (ii) output index in range
                    if out_idx is None or out_idx < 0 or out_idx >= len(src_m.outputs):
                        valid_indices = ", ".join(f"[{i}]={p.type}" for i, p in enumerate(src_m.outputs))
                        errors.append(
                            {
                                "node_id": node_id,
                                "field": input_name,
                                "code": "output_index_out_of_range",
                                "message": (
                                    f"input {input_name!r} references {src_class}[{value[1]}] "
                                    f"but {src_class} has {len(src_m.outputs)} output(s)"
                                ),
                                "hint": f"valid indices for {src_class}: {valid_indices}",
                            }
                        )
                        continue

                    # (iii) type compatibility — advisory only.
                    # ComfyUI allows cross-type wiring via reroutes, converters,
                    # and wildcard ports; the server is the authoritative validator.
                    port = port_by_name.get(input_name)
                    if port is not None:
                        src_type = src_m.outputs[out_idx].type
                        dst_type = port.type
                        if src_type != "*" and dst_type != "*" and src_type != dst_type:
                            # Find the correct index for the expected type
                            correct = [f"[{i}]" for i, p in enumerate(src_m.outputs) if p.type == dst_type]
                            hint = (
                                f"use {src_class}{correct[0]} instead"
                                if correct
                                else f"run `comfy nodes ls --produces {dst_type}` to find a source"
                            )
                            warnings.append(
                                {
                                    "node_id": node_id,
                                    "field": input_name,
                                    "code": "edge_type_mismatch",
                                    "message": (
                                        f"input {input_name!r} expects {dst_type} but "
                                        f"{src_class}[{out_idx}] produces {src_type}"
                                    ),
                                    "hint": hint,
                                }
                            )
                    continue

                port = port_by_name.get(input_name)
                if port is None:
                    continue
                # Shape check (hard error)
                shape_err = port.validate_shape(value)
                if shape_err:
                    errors.append(
                        {
                            "node_id": node_id,
                            "field": input_name,
                            "code": "shape_mismatch",
                            "message": shape_err,
                            "hint": f"expected {port.type}; check the value type",
                        }
                    )
                    continue
                # Catalog checks (enum membership, etc.). Range violations are a
                # hard reject only on a node the server will actually run; on a
                # pruned (output-unreachable) node they stay advisory warnings.
                cat_errors, cat_warnings = _validate_catalog_value(
                    node_id, class_type, input_name, port, value, range_is_error=node_id in reachable
                )
                errors.extend(cat_errors)
                warnings.extend(cat_warnings)

            # Required-presence checks apply only to output-reachable nodes: the
            # server prunes unreachable nodes without validating them, so
            # enforcing required inputs on a disconnected node over-rejects a
            # prompt the server would run (BE-3406).
            if node_id in reachable:
                errors.extend(_check_autogrow_required(node_id, autogrow_ports, autogrow_seen, node_data))
                errors.extend(_check_required_present(node_id, m, node_data))
                dyn_errors, dyn_warnings = _check_dynamic_combos(node_id, class_type, m, node_data)
                errors.extend(dyn_errors)
                warnings.extend(dyn_warnings)

        # No-outputs check: the server rejects any prompt with zero output
        # nodes (execution.py:1155-1162, prompt_no_outputs) — including an
        # empty/node-less prompt. Suppress it only when an unknown_class_type
        # error is present: that node could be the real (custom) output node we
        # just can't see, so the missing-output message would be misleading
        # noise on top of the unknown-class error the user must resolve first.
        has_unknown_class = any(e.get("code") == "unknown_class_type" for e in errors)
        if not has_output_node and not has_unknown_class:
            errors.append(
                {
                    # workflow-level error: no owning node, hence None (keeps the
                    # node_id/field keys every other error carries, for schema
                    # consistency).
                    "node_id": None,
                    "field": None,
                    "code": "prompt_no_outputs",
                    "message": "workflow has no output nodes — the server will reject it (prompt_no_outputs)",
                    "hint": "add an output node such as SaveImage/PreviewImage",
                }
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # -- Workflow slot editing --

    def get_template_schema(self, template_id: str, workflow: dict) -> dict:
        """Extract the slot manifest from a frontend-format workflow."""
        return {"id": template_id, "slots": _extract_frontend_slots(workflow, self)}

    def apply_slots(self, workflow: dict, overrides: dict[str, Any]) -> tuple[dict, list[dict]]:
        """Apply slot overrides. Returns (modified_workflow, warnings)."""
        import copy

        wf = copy.deepcopy(workflow)
        warnings: list[dict] = []
        for addr, value in overrides.items():
            warnings.extend(_apply_one_slot(wf, addr, value, self))
        return wf, warnings

    def expand_variations(self, workflow: dict, variations: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Apply N override sets, return N independent workflow copies."""
        results: list[dict] = []
        all_warnings: list[dict] = []
        for overrides in variations:
            modified, warnings = self.apply_slots(workflow, overrides)
            results.append(modified)
            all_warnings.extend(warnings)
        return results, all_warnings

    # -- Source loading (local / cloud / file) --

    @classmethod
    def load(
        cls,
        *,
        mode: str = "local",
        input_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        supported_nodes_yaml: bytes | None = None,
        cloud_disable_yaml: bytes | None = None,
    ) -> Graph:
        """Unified entry point: resolve object_info, build graph, annotate.

        Both local and cloud are the same: fetch ``/object_info`` from the
        resolved target. The only differences are base URL, path prefix,
        and auth headers — all handled by ``_load_from_target()``.

        Resolution order:
          1. ``input_path`` → read from local JSON file
          2. ``mode`` → resolve a Target via the CLI's routing chain,
             fetch ``/object_info`` over HTTP (local or cloud).

        Security: loopback-only guard for local, HTTPS-only for cloud,
        bounded read (64 MB), no-redirect policy.
        """
        if input_path is not None:
            raw = _load_from_file(input_path)
        else:
            raw = _load_from_target(mode=mode, host=host, port=port)

        g = cls.from_object_info(raw)
        if supported_nodes_yaml or cloud_disable_yaml:
            g.annotate(supported_nodes_yaml, cloud_disable_yaml)
        else:
            # ``--input <dump>`` is the offline path: the caller handed us a
            # local file precisely so nothing goes over the wire. An incidental
            # annotation lookup must not be the one thing that reaches out.
            g._try_default_annotations(allow_network=input_path is None)
        return g

    def _try_default_annotations(self, *, allow_network: bool = True) -> None:
        """Load node annotation data from Comfy-Org/comfy-complete.

        Resolves via :mod:`comfy_cli.cql.annotations_source`, which prefers a
        TTL-fresh local cache, falls back to a live fetch from the public repo
        (bounded, negative-cached, and skipped entirely when ``allow_network``
        is false), and finally to the package-bundled snapshot — so the data
        stays fresh without a ``pip install -U`` while remaining offline-safe.

        The annotations enrich every node with:
          - pack membership (which custom-node pack it belongs to)
          - behavioral labels (ReadsArbitraryFile, NetworkAccess, etc.)
          - cloud_disabled (whether this node is disabled on cloud)

        Useful for BOTH local and cloud: an agent building a workflow on a
        local server still needs to know which nodes will work on cloud.
        Local-only custom nodes not in comfy-complete simply get no labels
        and cloud_disabled=False (safe default).
        """
        try:
            from comfy_cli.cql import annotations_source

            sup, dis = annotations_source.load_annotation_bytes(allow_network=allow_network)
            if sup or dis:
                self.annotate(sup, dis)
        except Exception:
            pass  # missing data / network is non-fatal

    # -- Serialization helpers for CLI compat --

    def morphism_to_dict(self, m: Morphism) -> dict[str, Any]:
        return {
            "id": m.id,
            "name": m.id,
            "display_name": m.display_name,
            "description": m.description,
            "category": m.category,
            "output_types": m.output_types(),
            "output_node": m.is_output_node,
            "is_api_node": m.is_api_node,
            "deprecated": m.deprecated,
            "pack": m.pack,
            "labels": m.labels,
            "cloud_disabled": m.cloud_disabled,
            "inputs": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "is_link": p.is_link,
                    "section": "required" if p.required else "optional",
                    "choices": p.enum_values,
                    "options": {
                        "min": p.options.min,
                        "max": p.options.max,
                        "step": p.options.step,
                        "default": p.options.default,
                    },
                    # Autogrow inputs wire as one slot key per connection;
                    # surface that here so `nodes show` is self-documenting.
                    **({"autogrow": True, "wire_as": p.autogrow_slot_example()} if p.is_autogrow else {}),
                }
                for p in m.inputs
            ],
            "outputs": [{"name": p.name, "type": p.type} for p in m.outputs],
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
#
# Context-independent checks factored out of Graph.validate_workflow so the
# driver loop reads as the connection/class_type walk it is. Each returns the
# error/warning dicts for the caller to append — no shared state is threaded.


def _output_reachable_node_ids(workflow: dict[str, Any], graph: Graph) -> set[str]:
    """Node ids the server would actually validate: output nodes and their
    transitive input ancestors.

    Mirrors ComfyUI's execution.py::validate_prompt, which validates only the
    output nodes (``OUTPUT_NODE``) and everything reachable by walking their
    input links backward — any node not reachable from an output is pruned and
    never validated. We reproduce that reachable set so the promoted hard checks
    (required_input_missing, autogrow_no_slots, below_min/above_max) don't
    reject a disconnected node the server would silently drop.

    An input value shaped ``[source_node_id, output_index]`` is a link edge (the
    same predicate the per-input link walk uses); we follow those edges backward
    from every output node. A reference to a node absent from the workflow (a
    dangling edge, flagged separately) simply isn't traversed.
    """
    reachable: set[str] = set()
    stack: list[str] = []
    for node_id, node_data in workflow.items():
        if node_id == "_meta" or not isinstance(node_data, dict):
            continue
        m = graph.node(node_data.get("class_type", ""))
        if m is not None and m.is_output_node and node_id not in reachable:
            reachable.add(node_id)
            stack.append(node_id)
    while stack:
        node_data = workflow.get(stack.pop())
        if not isinstance(node_data, dict):
            continue
        node_inputs = node_data.get("inputs")
        # Guard against a truthy non-dict `inputs` from malformed JSON, which
        # would slip past `or {}` and raise AttributeError on `.values()`.
        if not isinstance(node_inputs, dict):
            continue
        for value in node_inputs.values():
            if isinstance(value, list) and len(value) == 2:
                src_id = str(value[0])
                if src_id in workflow and src_id not in reachable:
                    reachable.add(src_id)
                    stack.append(src_id)
    return reachable


def _validate_catalog_value(
    node_id: str, class_type: str, input_name: str, port: Port, value: Any, *, range_is_error: bool = True
) -> tuple[list[dict], list[dict]]:
    """Enum-membership and other catalog checks for one scalar input value.

    Returns (errors, warnings): unknown-enum and out-of-range values are hard
    errors (the server rejects them); every other catalog finding is a
    namespaced warning.

    ``range_is_error`` gates the below_min/above_max promotion (BE-3406): the
    server only range-checks nodes it actually runs, so on a pruned
    (output-unreachable) node these are demoted back to advisory warnings.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    for w in port.validate_catalog(value):
        if w["code"] == "unknown_enum_value":
            top = port.enum_values[:8]
            errors.append(
                {
                    "node_id": node_id,
                    "field": input_name,
                    "code": "unknown_enum_value",
                    "message": w["message"],
                    "hint": f"valid options include: {', '.join(str(v) for v in top)}"
                    + (
                        f" (and {len(port.enum_values) - 8} more — see valid_options)"
                        if len(port.enum_values) > 8
                        else ""
                    ),
                    "suggestions": port.enum_values[:20],
                    # full, typed list — never truncated, so the agent
                    # can pick a real value instead of guessing.
                    "valid_options": list(port.enum_values),
                }
            )
        elif w["code"] == "no_options_available":
            # Declared-but-empty option list: the server has nothing installed
            # for this field, so it rejects EVERY value (value_not_in_list
            # against an empty list — execution.py:1035-1067). Same hard-error
            # tier as unknown_enum_value, and for the same reason; the only
            # difference is that there is no option to suggest. This is the
            # fresh-install case the membership check used to skip in silence:
            # UNETLoader/CLIPLoader ship no built-in options, so on a bare
            # install the two largest missing downloads went unreported while a
            # half-populated install got warned.
            errors.append(
                {
                    "node_id": node_id,
                    "field": input_name,
                    "code": "no_options_available",
                    "message": w["message"],
                    "hint": (
                        f"the server has no files installed for {input_name!r} on {class_type} — "
                        f"install it (e.g. `comfy model download`) or point this input at an installed file"
                    ),
                    "suggestions": [],
                    "valid_options": [],
                }
            )
        elif w["code"] in ("below_min", "above_max"):
            # The server hard-rejects out-of-range values
            # (value_smaller_than_min / value_bigger_than_max, execution.py:1008-1033),
            # so promote these to errors — same as unknown_enum_value above. One
            # theoretical over-strictness: the server skips its built-in range
            # checks for args covered by a node's custom VALIDATE_INPUTS
            # (execution.py:1007); we accept that trade-off exactly as the
            # unknown_enum_value hard error does.
            bound = port.options.min if w["code"] == "below_min" else port.options.max
            op = ">=" if w["code"] == "below_min" else "<="
            entry = {
                "node_id": node_id,
                "field": input_name,
                "code": w["code"],
                "message": w["message"],
                "hint": f"use a value {op} {bound}",
            }
            # Only a hard reject on a node the server will run; on a pruned node
            # it stays advisory (matching pre-promotion behavior for that node).
            if range_is_error:
                errors.append(entry)
            else:
                # Demoted to an advisory warning: match the fully-qualified
                # `field` schema every other warning uses (preflight renders
                # w["field"]), rather than leaking the bare input_name.
                entry["field"] = f"{node_id}.{class_type}.{input_name}"
                warnings.append(entry)
        else:
            w["field"] = f"{node_id}.{class_type}.{w['field']}"
            warnings.append(w)
    return errors, warnings


def _check_required_present(node_id: str, m: Morphism, node_data: dict) -> list[dict]:
    """Required inputs that are absent from ``node_data["inputs"]`` entirely.

    Mirrors the server (execution.py:884-900): any required input the frontend
    didn't serialize is a hard reject (required_input_missing). The per-input
    loop only inspects keys that ARE present, so this catches the absent ones.
    Skipped, because each is handled by its own path (and would otherwise
    double-error): autogrow ports (``_check_autogrow_required``), dynamic combos
    (``_check_dynamic_combos``, which reports the absent selector itself so it can
    also name the valid options), and COMFY_DYNAMICSLOT (always optional
    server-side anyway). The server has NO exemption for required inputs that
    carry a default — the frontend always serializes widget values, so absence
    is a genuine authoring error; we do not skip ports with defaults.
    """
    present = node_data.get("inputs") or {}
    errors: list[dict] = []
    for port in m.inputs:
        if not port.required or port.is_autogrow:
            continue
        if port.is_dynamic_combo or port.type.startswith("COMFY_DYNAMICSLOT"):
            continue
        if port.name in present:
            continue
        errors.append(
            {
                "node_id": node_id,
                "field": port.name,
                "code": "required_input_missing",
                "message": (
                    f"required input {port.name!r} is missing — "
                    f"the server will reject this node (required_input_missing)"
                ),
                "hint": f"add {port.name!r} to inputs"
                + (
                    f" (e.g. a {port.type} value)"
                    if not port.is_link
                    else f" (wire a {port.type} link: [<node_id>, <output_index>])"
                ),
            }
        )
    return errors


# A dynamic combo's option may declare another dynamic combo among its
# sub-inputs. Mirrors ``workflow_to_api._MAX_DYNAMIC_COMBO_DEPTH`` so validation
# walks the same bounded tree the converter expanded.
_MAX_DYNAMIC_COMBO_DEPTH = 16


def _dynamic_combo_options(spec: Any) -> list[dict]:
    """Every option block a dynamic-combo spec declares, in schema order.

    Shape: ``["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": …, "inputs":
    {"required": {…}, "optional": {…}}}, …]}]``. Non-dict entries are dropped
    rather than raising — object_info is server-supplied and we never want a
    malformed option block to take down validation of the whole workflow.
    (Counterpart of ``workflow_to_api._dynamic_combo_selected_subs``, which
    resolves the same structure for the conversion side.)
    """
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return []
    options_meta = spec[1] if isinstance(spec[1], dict) else {}
    return [o for o in (options_meta.get("options") or []) if isinstance(o, dict)]


def _check_dynamic_combos(node_id: str, class_type: str, m: Morphism, node_data: dict) -> tuple[list[dict], list[dict]]:
    """Validate every dynamic-combo input against the option its selector names.

    Mirrors the server: ``DynamicCombo._expand_schema_for_dynamic``
    (comfy_api/latest/_io.py) reads the submitted selector, finds the option
    whose ``key`` equals it, and folds THAT option's inputs — and only that
    one's — into the node's finalized required/optional sets under dotted names
    (``model.width``). ``validate_inputs`` (execution.py:886-900) then walks
    those finalized names, so a missing one is a real ``required_input_missing``
    and a present one gets the same type/range/enum checks as any input.

    object_info never declares those dotted keys, so the driver loop skips them
    (``port_by_name`` misses) and ``_check_required_present`` exempts the parent
    selector. Without this walk a workflow that omits or mistypes a sub-input
    validates clean and is then rejected at ``/prompt`` — validation giving
    false confidence right before a paid run.
    """
    errors: list[dict] = []
    warnings: list[dict] = []
    present = node_data.get("inputs") or {}
    for port in m.inputs:
        if not port.is_dynamic_combo:
            continue
        e, w = _check_dynamic_combo_input(node_id, class_type, port.name, port.raw_spec, port.required, present)
        errors.extend(e)
        warnings.extend(w)
    return errors, warnings


def _check_dynamic_combo_input(
    node_id: str,
    class_type: str,
    name: str,
    spec: Any,
    required: bool,
    present: dict,
    depth: int = 0,
) -> tuple[list[dict], list[dict]]:
    """One dynamic-combo input: resolve its selected option, check its sub-inputs."""
    errors: list[dict] = []
    warnings: list[dict] = []
    options = _dynamic_combo_options(spec)
    keys = [o.get("key") for o in options]

    if name not in present:
        # ``_check_required_present`` deliberately exempts the dynamic types
        # ("handled by its own path") — this is that path, so the absent
        # selector is reported here and nowhere else (no double-error).
        #
        # NOTE the failure is *later* than a plain missing input. With no
        # selector the server's schema expansion is a no-op
        # (``DynamicCombo._expand_schema_for_dynamic`` returns early), so the
        # selector never enters ``valid_inputs`` and ``/prompt`` does NOT emit
        # ``required_input_missing`` for it — the node's inputs are dropped and
        # it fails at EXECUTION instead, i.e. after a paid node has been
        # entered. Still a hard error here: the workflow cannot run.
        if required:
            errors.append(
                {
                    "node_id": node_id,
                    "field": name,
                    "code": "required_input_missing",
                    "message": (
                        f"required dynamic-combo input {name!r} is missing — the server cannot resolve "
                        f"which option's sub-inputs apply, so this node fails at execution"
                    ),
                    "hint": f"set {name!r} to one of its options: "
                    + ", ".join(str(k) for k in keys[:8])
                    + (f" (and {len(keys) - 8} more — see valid_options)" if len(keys) > 8 else ""),
                    "suggestions": keys[:20],
                    "valid_options": keys,
                }
            )
        return errors, warnings

    selected = present[name]
    if isinstance(selected, list) and len(selected) == 2:
        # Wired as a link: which option expands is only known at execution time,
        # so there is no static sub-input set to check. The edge itself is
        # already validated by the driver loop.
        return errors, warnings

    option = next((o for o in options if o.get("key") == selected), None)
    if option is None:
        # Strict ``==`` on the key — the same test the server
        # (``DynamicCombo._expand_schema_for_dynamic``) and the converter both
        # apply, so all three agree on which option expands.
        #
        # Same late failure as the absent selector above: an unmatched key
        # expands to nothing, so the server drops this node's inputs rather
        # than rejecting the prompt, and the run dies at execution. The
        # converter ALSO failed to expand it, silently misaligning every
        # following widget value — surfacing that is the whole point.
        errors.append(
            {
                "node_id": node_id,
                "field": name,
                "code": "unknown_enum_value",
                "message": (
                    f"{selected!r} not in {len(keys)} known options for {name} — its sub-inputs "
                    f"cannot be resolved, so this node fails at execution"
                ),
                "hint": f"valid options: {', '.join(str(k) for k in keys[:8])}"
                + (f" (and {len(keys) - 8} more — see valid_options)" if len(keys) > 8 else ""),
                "suggestions": keys[:20],
                "valid_options": keys,
            }
        )
        return errors, warnings

    if depth >= _MAX_DYNAMIC_COMBO_DEPTH:
        return errors, warnings  # pathological nesting — the converter stops here too

    sub_def = option.get("inputs")
    if not isinstance(sub_def, dict):
        return errors, warnings
    for section, sub_required in (("required", True), ("optional", False)):
        section_def = sub_def.get(section)
        if not isinstance(section_def, dict):
            continue
        for sub_name, sub_spec in section_def.items():
            e, w = _check_dynamic_combo_sub(
                node_id, class_type, f"{name}.{sub_name}", sub_spec, sub_required, present, depth
            )
            errors.extend(e)
            warnings.extend(w)
    return errors, warnings


def _check_dynamic_combo_sub(
    node_id: str,
    class_type: str,
    dotted: str,
    sub_spec: Any,
    sub_required: bool,
    present: dict,
    depth: int,
) -> tuple[list[dict], list[dict]]:
    """Presence + shape + catalog checks for one expanded sub-input.

    The sub-input spec is a plain ``INPUT_TYPES`` entry, so it goes through the
    same ``_parse_input_spec`` / :class:`Port` machinery as a top-level input and
    inherits identical shape and enum/range semantics.
    """
    type_id, is_enum, enum_values, opts, enum_declared = _parse_input_spec(sub_spec)
    port = Port(
        name=dotted,
        type=type_id,
        required=sub_required,
        is_link=_is_link(type_id, is_enum, opts.force_input),
        enum_values=enum_values,
        enum_declared=enum_declared,
        options=opts,
        raw_spec=sub_spec,
    )

    if port.is_dynamic_combo:
        # Nested dynamic combo: its own selector/presence rules apply one level down.
        return _check_dynamic_combo_input(node_id, class_type, dotted, sub_spec, sub_required, present, depth + 1)

    if port.is_autogrow:
        # An autogrow sub-input wires as `<dotted>.<slot>` keys and routinely
        # declares `min: 0` even inside the `required` section (Seedream's
        # `model.images`), so absence is NOT a server reject — the converter
        # emits no key at all for a zero-slot autogrow. Nothing to presence- or
        # shape-check here.
        return [], []

    if dotted not in present:
        if not sub_required:
            return [], []
        return [
            {
                "node_id": node_id,
                "field": dotted,
                "code": "required_input_missing",
                "message": (
                    f"required input {dotted!r} is missing — the server will reject this node (required_input_missing)"
                ),
                "hint": f"add {dotted!r} to inputs"
                + (
                    f" (e.g. a {port.type} value)"
                    if not port.is_link
                    else f" (wire a {port.type} link: [<node_id>, <output_index>])"
                ),
            }
        ], []

    value = present[dotted]
    if isinstance(value, list) and len(value) == 2:
        # A wired sub-input — the driver loop already ran the dangling-edge and
        # output-index checks on this same key.
        return [], []

    shape_err = port.validate_shape(value)
    if shape_err:
        return [
            {
                "node_id": node_id,
                "field": dotted,
                "code": "shape_mismatch",
                "message": shape_err,
                "hint": f"expected {port.type}; check the value type",
            }
        ], []

    return _validate_catalog_value(node_id, class_type, dotted, port, value)


def _check_autogrow_required(
    node_id: str, autogrow_ports: dict[str, Port], autogrow_seen: set[str], node_data: dict
) -> list[dict]:
    """Required autogrow inputs that received no connected slots.

    The server would reject such a node, so surface it here instead of as a
    cryptic downstream reject.
    """
    inputs = node_data.get("inputs") or {}
    errors: list[dict] = []
    for base, port in autogrow_ports.items():
        if port.required and base not in autogrow_seen and base not in inputs:
            errors.append(
                {
                    "node_id": node_id,
                    "field": base,
                    "code": "autogrow_no_slots",
                    "message": (
                        f"required autogrow input {base!r} has no connected slots — the server will reject this node"
                    ),
                    "hint": f"wire one key per connection: {port.autogrow_slot_example()}",
                }
            )
    return errors


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

_MAX_OBJECT_INFO_BYTES = 64 * 1024 * 1024


_opener = build_http_only_opener(NoRedirectHandler())


class LoadError(Exception):
    """Failed to load object_info from a source."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def _load_from_file(path: str) -> dict[str, Any]:
    """Read object_info from a local JSON file."""
    from pathlib import Path

    p = Path(path).expanduser()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise LoadError(f"cannot read object_info: {p}: {e}", details={"path": str(p)}) from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LoadError(f"invalid JSON in {p}: {e}", details={"path": str(p)}) from e


def _load_from_target(*, mode: str = "local", host: str | None = None, port: int | None = None) -> dict[str, Any]:
    """Fetch /object_info from the resolved target — local or cloud.

    Both paths are the same HTTP fetch; only the base URL, path prefix,
    and auth headers differ. The Target abstraction handles all three.

    Security:
      - Local: loopback-only guard (warn on non-loopback)
      - Cloud: HTTPS enforced by Target + auth headers
      - Both: bounded read (64 MB), no-redirect policy
    """
    from comfy_cli.target import resolve_target

    target = resolve_target(where=mode, host=host, port=port)
    url = target.url("object_info")

    # Loopback guard for local targets
    if not target.is_cloud:
        parsed_host = urllib.parse.urlsplit(url).hostname or ""
        if not is_loopback_host(parsed_host):
            raise LoadError(
                f"Refusing to fetch object_info from non-loopback host {parsed_host!r} "
                f"in local mode (potential SSRF). Use --where cloud for remote targets."
            )

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")

    # Auth headers (cloud only — local has no auth)
    from comfy_cli.http import target_auth_headers

    for k, v in target_auth_headers(target).items():
        req.add_header(k, v)

    try:
        with _opener.open(req, timeout=30.0) as resp:
            raw = resp.read(_MAX_OBJECT_INFO_BYTES + 1)
            if len(raw) > _MAX_OBJECT_INFO_BYTES:
                raise LoadError(
                    f"response exceeds {_MAX_OBJECT_INFO_BYTES} bytes",
                    details={"url": url},
                )
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        hint = "run `comfy cloud login`" if target.is_cloud else "run `comfy launch` first"
        raise LoadError(
            f"HTTP {e.code} from {url}: {body[:200]}",
            details={"url": url, "status": e.code, "hint": hint},
        ) from e
    except urllib.error.URLError as e:
        hint = "run `comfy cloud login`" if target.is_cloud else "run `comfy launch` first"
        raise LoadError(
            f"cannot reach {url}: {e.reason if hasattr(e, 'reason') else e}",
            details={"url": url, "mode": mode, "hint": hint},
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise LoadError(f"invalid response from {url}: {e}", details={"url": url}) from e


# ---------------------------------------------------------------------------
# Slot editing helpers — port of nodegraph/frontend.go + runtemplate.go
# ---------------------------------------------------------------------------

# Defends slot recursion against pathological / cyclic subgraph nesting.
_MAX_SUBGRAPH_DEPTH = 32


# Delimiter that separates subgraph-nesting levels in a slot address. The
# final ``.`` separates the (possibly nested) node path from the input name.
# Examples:
#   ``3.seed``          top-level node 3, widget ``seed``
#   ``10/9.prompt``     subgraph instance 10 → interior node 9, widget ``prompt``
#   ``10/3/7.value``    instance 10 → interior subgraph node 3 → interior node 7
# UUID subgraph class_types contain ``-`` but never ``/`` or top-level ``.`` in
# an instance id, so the delimiters stay unambiguous.
_SUBGRAPH_PATH_SEP = "/"


def _subgraph_defs_by_id(workflow: dict) -> dict[str, dict]:
    """Index subgraph definitions so an instance's ``type`` resolves to its def.

    A subgraph *instance* node's ``type`` is normally the UUID ``id`` of its
    definition, so the UUID is the primary key. Real ComfyUI saves can also
    carry several distinct defs sharing the cosmetic ``name`` "New Subgraph";
    keying by name alone would silently map instances onto the wrong def, so id
    always wins. We still register ``name`` as a *fallback* key (only when it
    doesn't shadow an id and isn't ambiguous across defs) to support older
    name-typed templates that predate UUID ids.
    """
    defs = (workflow.get("definitions") or {}).get("subgraphs") or []
    by_id: dict[str, dict] = {}
    name_counts: dict[str, int] = {}
    name_first: dict[str, dict] = {}
    for sg in defs:
        if not isinstance(sg, dict):
            continue
        sg_id = sg.get("id")
        if isinstance(sg_id, str) and sg_id:
            by_id[sg_id] = sg
        name = sg.get("name")
        if isinstance(name, str) and name:
            name_counts[name] = name_counts.get(name, 0) + 1
            name_first.setdefault(name, sg)
    for name, count in name_counts.items():
        if count == 1 and name not in by_id:
            by_id[name] = name_first[name]
    return by_id


def _node_widget_slots(node: dict, prefix: str, graph: Graph) -> list[dict]:
    """Surface a regular node's widget inputs as slots under ``prefix``.

    ``prefix`` is the addressable node path (``"3"`` at top level, ``"10/9"``
    inside a subgraph). Returns one slot per widget input the schema knows
    about. Returns ``[]`` for nodes whose type isn't in object_info.
    """
    node_type = node.get("type", "")
    m = graph.node(node_type)
    if m is None:
        return []
    order = graph.widget_order(node_type)
    widgets = node.get("widgets_values") or []
    slots: list[dict] = []
    for port in m.inputs:
        if port.is_link:
            continue
        try:
            idx = order.index(port.name)
        except ValueError:
            continue
        current = widgets[idx] if idx < len(widgets) else None
        slots.append(
            {
                "address": f"{prefix}.{port.name}",
                "name": port.name,
                "type": port.type,
                "current_value": current,
                "instance_id": prefix,
                "node_type": node_type,
            }
        )
    return slots


def _extract_frontend_slots(workflow: dict, graph: Graph) -> list[dict]:
    """Walk workflow nodes and extract tweakable slots.

    For every node we surface its widget inputs at ``<nodePath>.<input>``. When
    a node is a *subgraph instance* (its ``type`` is a UUID matching a def under
    ``definitions.subgraphs``) we additionally recurse INTO the definition and
    surface every interior node's widget inputs under a nested, instance-scoped
    address (``<instanceId>/<interiorId>.<input>``, recursing for deeper
    subgraphs). This is what lets an agent slot-edit a fetched gallery template
    whose editable prompt/seed/image live inside an opaque UUID subgraph.

    Curated subgraph ``inputs[]`` (the proxy parameter list) are still exposed
    at ``<instanceId>.<name>`` for backward compatibility, but the recursion
    means an agent is never stranded when those proxies are missing or dangle
    (fetched templates routinely point proxyWidgets at deleted interior ids).
    """
    defs_by_id = _subgraph_defs_by_id(workflow)
    slots: list[dict] = []
    seen_addrs: set[str] = set()

    def add(slot: dict) -> None:
        addr = slot["address"]
        if addr in seen_addrs:
            return
        seen_addrs.add(addr)
        slots.append(slot)

    def walk(nodes: list, prefix: str, depth: int) -> None:
        if depth > _MAX_SUBGRAPH_DEPTH:
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", ""))
            node_path = f"{prefix}{_SUBGRAPH_PATH_SEP}{node_id}" if prefix else node_id
            node_type = node.get("type", "")
            sg = defs_by_id.get(node_type)
            if sg is None:
                for slot in _node_widget_slots(node, node_path, graph):
                    add(slot)
                continue

            # Subgraph instance. A *curated* template (every declared proxy input
            # resolves to a live interior widget) keeps its clean, hand-picked
            # parameter view — we surface only its declared inputs and do NOT
            # recurse, so the agent sees the intended surface. When the proxies
            # are missing or dangling (the norm for fetched gallery templates,
            # whose proxyWidgets point at deleted interior ids) we recurse into
            # the definition so the real editable inner inputs are reachable.
            declared, fully_curated = _declared_subgraph_slots(node, sg, node_id, graph)
            for slot in declared:
                add(slot)
            if not fully_curated:
                walk(sg.get("nodes") or [], node_path, depth + 1)

    walk(workflow.get("nodes") or [], "", 0)
    return slots


# Sentinel returned by _resolve_proxy_value when a proxy entry is genuinely
# unresolvable (interior node missing, widget name not in the node's order, or
# index past the end of widgets_values).  Callers must use ``is _UNRESOLVED``
# to distinguish this from a legitimately-null widget value (e.g. seed saved as
# None, or an optional image input that has not yet been set).
_UNRESOLVED = object()


def _declared_subgraph_slots(instance: dict, sg: dict, instance_id: str, graph: Graph) -> tuple[list[dict], bool]:
    """Build slots for a subgraph instance's curated proxy inputs.

    Returns ``(slots, fully_curated)`` where ``fully_curated`` is True only when
    the instance declares at least one input and EVERY declared input resolves
    to a real interior widget value (so the curated surface is complete and the
    caller can skip recursion).
    """
    declared: list[dict] = []
    inputs = sg.get("inputs") or []
    any_declared = False
    all_resolved = True
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        inp_name = inp.get("name", "")
        if not inp_name:
            continue
        any_declared = True
        current = _resolve_proxy_value(instance, sg, inp_name, graph)
        if current is _UNRESOLVED:
            all_resolved = False
            continue
        inp_type = inp.get("type", {})
        declared.append(
            {
                "address": f"{instance_id}.{inp_name}",
                "name": inp_name,
                "type": inp_type if isinstance(inp_type, str) else str(inp_type),
                "current_value": current,
                "instance_id": instance_id,
                "node_type": instance.get("type", ""),
            }
        )
    return declared, (any_declared and all_resolved)


def _resolve_proxy_value(instance: dict, subgraph: dict, input_name: str, graph: Graph):
    """Navigate proxyWidgets to find the current widget value.

    Returns the widget's current value (which may be ``None`` for a
    legitimately-null widget) or the module-level ``_UNRESOLVED`` sentinel when
    the proxy entry points at a missing/dangling interior node, when the widget
    name is absent from the node's widget order, or when the index is past the
    end of ``widgets_values``.  Callers must use ``is _UNRESOLVED`` to test for
    the unresolvable case so that a real ``None`` value is preserved.
    """
    proxy = (instance.get("properties") or {}).get("proxyWidgets") or []
    for entry in proxy:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        name = entry[1] if isinstance(entry[1], str) else str(entry[1])
        if name != input_name:
            continue
        interior_id = str(entry[0])
        for inode in subgraph.get("nodes") or []:
            if not isinstance(inode, dict) or str(inode.get("id", "")) != interior_id:
                continue
            interior_class = inode.get("type", "")
            order = graph.widget_order(interior_class)
            try:
                idx = order.index(name)
            except ValueError:
                return _UNRESOLVED
            widgets = inode.get("widgets_values") or []
            return widgets[idx] if idx < len(widgets) else _UNRESOLVED
        break
    return _UNRESOLVED


def _write_widget(node: dict, input_name: str, value: Any, graph: Graph, *, extend: bool) -> list[dict]:
    """Write ``value`` into ``node``'s ``widgets_values`` slot for ``input_name``.

    Validates against the node's schema and returns catalog warnings. ``extend``
    pads a short widget list for top-level direct edits (matches prior behavior);
    interior subgraph nodes always carry a full widget list and are not padded.
    """
    node_type = node.get("type", "")
    m = graph.node(node_type)
    if m is None:
        raise ValueError(f"unknown node type {node_type!r} for node {node.get('id')}")
    order = graph.widget_order(node_type)
    try:
        widget_idx = order.index(input_name)
    except ValueError:
        avail = [n for n in order if n != "control_after_generate"]
        raise ValueError(
            f"widget {input_name!r} not found on {node_type}; "
            f"available widgets: {', '.join(avail) if avail else '(none — all inputs are links)'}"
        )
    widgets = node.get("widgets_values") or []
    if widget_idx >= len(widgets):
        if not extend:
            raise ValueError(f"widget index {widget_idx} out of range for {node_type}")
        widgets.extend([None] * (widget_idx + 1 - len(widgets)))

    warnings: list[dict] = []
    port = next((p for p in m.inputs if p.name == input_name), None)
    if port:
        err = port.validate_shape(value)
        if err:
            raise ValueError(err)
        warnings = port.validate_catalog(value)

    widgets[widget_idx] = value
    node["widgets_values"] = widgets
    return warnings


def _resolve_node_path(workflow: dict, segments: list[str], defs_by_id: dict[str, dict]) -> dict:
    """Walk a ``/``-separated node path into (possibly nested) subgraphs.

    The first segment names a top-level node; each subsequent segment names an
    interior node of the subgraph definition the previous segment instantiated.
    Returns the resolved (mutable) node dict, or raises ValueError describing
    the first hop that couldn't be found.

    Isolation: every non-terminal hop is forked (if its subgraph definition is
    shared) before descending, so interior writes at any depth can't alias
    sibling instances — not just the first hop.
    """
    nodes = workflow.get("nodes") or []
    node = next((n for n in nodes if isinstance(n, dict) and str(n.get("id", "")) == segments[0]), None)
    if node is None:
        raise ValueError(f"node {segments[0]} not found in workflow")
    for seg in segments[1:]:
        # ``node`` is a non-terminal hop we are about to descend into: fork its
        # shared definition first so the write below this hop stays isolated.
        _isolate_shared_subgraph(workflow, node, defs_by_id)
        defs_by_id = _subgraph_defs_by_id(workflow)  # rebuild: node.type may have changed
        sg = defs_by_id.get(node.get("type", ""))
        if sg is None:
            raise ValueError(f"node {node.get('id')} is not a subgraph; cannot descend to {seg!r}")
        inner = next((n for n in (sg.get("nodes") or []) if isinstance(n, dict) and str(n.get("id", "")) == seg), None)
        if inner is None:
            raise ValueError(f"interior node {seg} not found in subgraph {sg.get('id')}")
        node = inner
    return node


def _count_instances(workflow: dict, def_id: str) -> int:
    """Count nodes (top-level + interior-of-definitions) instantiating ``def_id``."""
    count = 0
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and str(n.get("type", "")) == def_id:
            count += 1
    for sg in (workflow.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            for n in sg.get("nodes") or []:
                if isinstance(n, dict) and str(n.get("type", "")) == def_id:
                    count += 1
    return count


def _isolate_shared_subgraph(workflow: dict, instance: dict, defs_by_id: dict[str, dict]) -> None:
    """If ``instance``'s subgraph definition is shared with another instance,
    deep-copy it under a fresh id and repoint ``instance`` so an interior write
    can't alias sibling instances. No-op when the instance already owns its def.
    """
    def_id = str(instance.get("type", ""))
    sg = defs_by_id.get(def_id)
    if sg is None or _count_instances(workflow, def_id) <= 1:
        return
    new_sg = copy.deepcopy(sg)
    new_id = str(_uuid.uuid4())
    new_sg["id"] = new_id
    workflow.setdefault("definitions", {}).setdefault("subgraphs", []).append(new_sg)
    instance["type"] = new_id


def _apply_one_slot(workflow: dict, addr: str, value: Any, graph: Graph) -> list[dict]:
    """Apply a single slot override. Returns warnings. Raises ValueError on hard errors.

    Address forms (see ``_extract_frontend_slots`` / ``_SUBGRAPH_PATH_SEP``):
      * ``<id>.<input>``                 — top-level node widget (direct mode).
      * ``<id>.<declaredInput>``         — curated subgraph proxy input, routed
                                           through ``proxyWidgets`` to its interior
                                           node (legacy template mode).
      * ``<instanceId>/<innerId>.<input>`` (and deeper) — a widget on an interior
                                           node reached by descending into the
                                           subgraph definition(s).
    """
    if "." not in addr:
        raise ValueError(f"invalid slot address {addr!r} (expected 'instance_id.input_name')")
    # Node paths use '/' as separator; node IDs are numeric or UUID (no '.').
    # Input names may legitimately contain dots (e.g. 'images.image0').
    # Always split on the FIRST dot so multi-dot input names are preserved.
    node_path, input_name = addr.split(".", 1)
    segments = node_path.split(_SUBGRAPH_PATH_SEP)

    defs_by_id = _subgraph_defs_by_id(workflow)

    # --- Nested form: descend the subgraph path and write the interior widget. ---
    if len(segments) > 1:
        # _resolve_node_path forks every shared definition along the path (each
        # non-terminal hop) before the terminal write, so sibling instances at
        # any nesting depth stay independent.
        target = _resolve_node_path(workflow, segments, defs_by_id)
        return _write_widget(target, input_name, value, graph, extend=False)

    instance_id = segments[0]
    nodes = workflow.get("nodes") or []
    instance = next((n for n in nodes if isinstance(n, dict) and str(n.get("id", "")) == instance_id), None)
    if instance is None:
        raise ValueError(f"node {instance_id} not found in workflow")

    node_type = instance.get("type", "")
    sg = defs_by_id.get(node_type)

    # --- Curated subgraph proxy input (legacy ``<id>.<declaredInput>``). ---
    if sg is not None:
        proxy = (instance.get("properties") or {}).get("proxyWidgets") or []
        interior_id = None
        for entry in proxy:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            name = entry[1] if isinstance(entry[1], str) else str(entry[1])
            if name == input_name:
                interior_id = str(entry[0])
                break
        if interior_id is None:
            raise ValueError(
                f"no proxyWidget mapping for {addr}; "
                f"address an interior input directly, e.g. {instance_id}/<innerId>.<input> "
                f"(run `comfy workflow slots` to list them)"
            )
        inode = next(
            (n for n in (sg.get("nodes") or []) if isinstance(n, dict) and str(n.get("id", "")) == interior_id),
            None,
        )
        if inode is None:
            raise ValueError(f"interior node {interior_id} not found in subgraph")
        return _write_widget(inode, input_name, value, graph, extend=False)

    # --- Direct mode: regular top-level node. ---
    return _write_widget(instance, input_name, value, graph, extend=True)
