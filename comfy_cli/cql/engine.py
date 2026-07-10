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
from comfy_cli.http import NoRedirectHandler

# ---------------------------------------------------------------------------
# Types — mirrors nodegraph/types.go
# ---------------------------------------------------------------------------

_IMPLICIT_WIDGET_TYPES = frozenset({"STRING", "INT", "FLOAT", "NUMBER", "BOOLEAN", "COMBO"})


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
    options: PortOptions = field(default_factory=PortOptions)

    @property
    def is_autogrow(self) -> bool:
        """V3 autogrow input (e.g. BatchImagesNode.images): the schema declares
        ONE input, but the server expects autogrown slot keys —
        ``images.image0``, ``images.image1``, … one per connection."""
        return self.type.startswith("COMFY_AUTOGROW")

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
    needs_gpu: bool = True  # default True per Go

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


def _parse_input_spec(spec: Any) -> tuple[str, bool, list[Any], PortOptions]:
    """Returns (type_id, is_enum, enum_values, options)."""
    if isinstance(spec, str):
        return spec, False, [], PortOptions()

    if not isinstance(spec, list) or len(spec) == 0:
        return "UNKNOWN", False, [], PortOptions()

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
        if isinstance(options, list) and options and all(_is_scalar_choice(v) for v in options):
            # Keep each option's real type: an int-valued combo (Sora-2/LTXV
            # `duration`) must stay [4, 8, 12], not ["4","8","12"], so `nodes
            # show` is truthful and agents pass the type the cloud accepts.
            return first, True, list(options), port_opts
        return first, False, [], port_opts

    if isinstance(first, list):
        # Same: preserve the option types for the classic list-form combo.
        return "COMBO", True, list(first), port_opts

    return "UNKNOWN", False, [], port_opts


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
        type_id, is_enum, enum_values, opts = _parse_input_spec(spec)
        ports.append(
            Port(
                name=name,
                type=type_id,
                required=required,
                is_link=_is_link(type_id, is_enum, opts.force_input),
                enum_values=enum_values,
                options=opts,
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

    return Morphism(
        id=node_id,
        display_name=raw.get("display_name") or node_id,
        description=raw.get("description") or "",
        category=raw.get("category") or "",
        inputs=inputs,
        outputs=outputs,
        is_output_node=bool(raw.get("output_node", False)),
        is_api_node=bool(raw.get("api_node", False)),
        deprecated=bool(raw.get("deprecated", False)),
        experimental=bool(raw.get("experimental", False)),
        search_aliases=_unmarshal_string_list(raw.get("search_aliases")),
        pack=_derive_pack(raw.get("python_module") or ""),
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


def parse_no_gpu_nodes(data: bytes) -> set[str]:
    """Parse no_gpu_nodes.json → set of CPU-only node IDs."""
    try:
        cfg = json.loads(data)
    except Exception:
        return set()
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        return set()
    return set(cfg.get("no_gpu_nodes") or [])


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

    @classmethod
    def from_object_info(cls, object_info: dict[str, Any]) -> Graph:
        if not isinstance(object_info, dict):
            raise LoadError(
                "object_info must be a JSON object",
                details={"top_level_type": type(object_info).__name__},
            )
        g = cls()
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
        no_gpu_json: bytes | None = None,
    ) -> None:
        node_pack: dict[str, str] = {}
        node_labels: dict[str, list[str]] = {}
        disable_labels: set[str] = set()
        no_gpu: set[str] = set()

        if supported_nodes_yaml:
            node_pack, node_labels = parse_supported_nodes(supported_nodes_yaml)
        if cloud_disable_yaml:
            disable_labels = parse_disable_config(cloud_disable_yaml)
        if no_gpu_json:
            no_gpu = parse_no_gpu_nodes(no_gpu_json)

        for nid, m in self._nodes.items():
            if nid in node_pack:
                m.pack = node_pack[nid]
            if nid in node_labels:
                m.labels = node_labels[nid]
            m.cloud_disabled = any(label in disable_labels for label in m.labels)
            m.needs_gpu = nid not in no_gpu
        self._annotated = True

    # -- Lookup --

    def node(self, name: str) -> Morphism | None:
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

    def pack_nodes(self, pack: str) -> list[Morphism]:
        """All nodes belonging to a custom-node pack (case-insensitive)."""
        p = pack.lower()
        return sorted([m for m in self._nodes.values() if m.pack.lower() == p], key=lambda m: m.id)

    def label_nodes(self, label: str) -> list[Morphism]:
        """All nodes carrying a specific behavioral label."""
        return sorted([m for m in self._nodes.values() if label in m.labels], key=lambda m: m.id)

    def cloud_disabled_nodes(self) -> list[Morphism]:
        """All nodes that are disabled on Comfy Cloud."""
        return sorted([m for m in self._nodes.values() if m.cloud_disabled], key=lambda m: m.id)

    def cloud_enabled_nodes(self) -> list[Morphism]:
        """All nodes that are enabled on Comfy Cloud."""
        return sorted([m for m in self._nodes.values() if not m.cloud_disabled], key=lambda m: m.id)

    def api_nodes(self) -> list[Morphism]:
        """All partner API nodes."""
        return sorted([m for m in self._nodes.values() if m.is_api_node], key=lambda m: m.id)

    def output_nodes(self) -> list[Morphism]:
        """All terminal output nodes (SaveImage, etc.)."""
        return sorted([m for m in self._nodes.values() if m.is_output_node], key=lambda m: m.id)

    def packs(self) -> list[str]:
        """All known pack names, sorted."""
        return sorted(set(m.pack for m in self._nodes.values() if m.pack))

    def known_labels(self) -> list[str]:
        """All known labels, sorted."""
        labels: set[str] = set()
        for m in self._nodes.values():
            labels.update(m.labels)
        return sorted(labels)

    def find_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 4,
        max_paths: int = 10,
    ) -> list[dict]:
        """BFS multi-hop path finding from one type to another."""
        if from_type == to_type:
            return []
        # queue items: (current_type, steps[])
        queue: list[tuple[str, list[dict]]] = [(from_type, [])]
        visited: set[str] = {from_type}
        paths: list[dict] = []

        while queue and len(paths) < max_paths:
            next_queue: list[tuple[str, list[dict]]] = []
            for cur_type, steps in queue:
                if len(steps) >= max_depth:
                    continue
                for consumer in self._consumers.get(cur_type, []):
                    for out_t in consumer.output_types():
                        if out_t == cur_type:
                            continue
                        step = {"node": consumer.id, "input_type": cur_type, "output_type": out_t}
                        new_steps = steps + [step]
                        if out_t == to_type:
                            paths.append({"from": from_type, "to": to_type, "steps": new_steps})
                            if len(paths) >= max_paths:
                                return paths
                        elif out_t not in visited and len(new_steps) < max_depth:
                            visited.add(out_t)
                            next_queue.append((out_t, new_steps))
            queue = next_queue
        return paths

    def exact_paths(
        self,
        from_type: str,
        to_type: str,
        *,
        max_depth: int = 6,
        max_paths: int = 10,
    ) -> list[dict]:
        """Satisfiability-aware BFS: each step's required link inputs must be
        available from types produced by prior steps."""
        if from_type == to_type:
            return []
        # state: (available_types_frozenset, steps[])
        initial: frozenset[str] = frozenset({from_type})
        queue: list[tuple[frozenset[str], list[dict]]] = [(initial, [])]
        visited: set[frozenset[str]] = {initial}
        paths: list[dict] = []

        while queue and len(paths) < max_paths:
            next_queue: list[tuple[frozenset[str], list[dict]]] = []
            for available, steps in queue:
                if len(steps) >= max_depth:
                    continue
                for m in sorted(self._nodes.values(), key=lambda m: m.id):
                    if not m.can_apply(available):
                        continue
                    new_outs = [t for t in m.output_types() if t not in available and t != "*"]
                    if not new_outs:
                        continue
                    # Pick one representative input type this node consumes from available
                    input_type = ""
                    for t in m.required_link_types():
                        if t in available:
                            input_type = t
                            break
                    for out_t in new_outs:
                        step = {"node": m.id, "input_type": input_type, "output_type": out_t}
                        new_steps = steps + [step]
                        new_avail = available | frozenset(new_outs)
                        if out_t == to_type:
                            # ``from_type`` seeds ``available`` and the set only
                            # grows, so every reachable path originates from it by
                            # construction — no extra consumption guard needed.
                            paths.append({"from": from_type, "to": to_type, "steps": new_steps})
                            if len(paths) >= max_paths:
                                return paths
                        elif new_avail not in visited and len(new_steps) < max_depth:
                            visited.add(new_avail)
                            next_queue.append((new_avail, new_steps))
            queue = next_queue
        return paths

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

        for node_id, node_data in workflow.items():
            # `_meta` is the compose/run provenance block (schema/blueprint/items),
            # stripped before submit — not a node and not a mistake. `comfy compose`
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

            port_by_name = {p.name: p for p in m.inputs}
            # V3 autogrow inputs are declared once (e.g. `images`) but wired as
            # slot keys (`images.image0`, `images.image1`, …). Track which
            # autogrow ports actually received a slot so the required-but-empty
            # case surfaces here instead of as a cryptic server reject.
            autogrow_ports = {p.name: p for p in m.inputs if p.is_autogrow}
            autogrow_seen: set[str] = set()
            for input_name, value in (node_data.get("inputs") or {}).items():
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
                    src_m = self._nodes.get(src_class)
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
                # Catalog checks (enum membership, etc.)
                cat_errors, cat_warnings = _validate_catalog_value(node_id, class_type, input_name, port, value)
                errors.extend(cat_errors)
                warnings.extend(cat_warnings)

            errors.extend(_check_autogrow_required(node_id, autogrow_ports, autogrow_seen, node_data))

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
        host: str = "127.0.0.1",
        port: int = 8188,
        supported_nodes_yaml: bytes | None = None,
        cloud_disable_yaml: bytes | None = None,
        no_gpu_json: bytes | None = None,
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
        if supported_nodes_yaml or cloud_disable_yaml or no_gpu_json:
            g.annotate(supported_nodes_yaml, cloud_disable_yaml, no_gpu_json)
        else:
            g._try_default_annotations()
        return g

    def _try_default_annotations(self) -> None:
        """Load bundled annotation files from ``comfy_cli.cql.data``.

        These ship as package data (40 KB total) from Comfy-Org/comfy-complete.
        They enrich every node with:
          - pack membership (which custom-node pack it belongs to)
          - behavioral labels (ReadsArbitraryFile, NetworkAccess, etc.)
          - cloud_disabled (whether this node is disabled on cloud)

        Useful for BOTH local and cloud: an agent building a workflow on a
        local server still needs to know which nodes will work on cloud.
        Local-only custom nodes not in comfy-complete simply get no labels
        and cloud_disabled=False (safe default).
        """
        try:
            from importlib import resources

            data_pkg = resources.files("comfy_cli.cql.data")
            sup = (data_pkg / "supported_nodes.yaml").read_bytes()
            dis = (data_pkg / "cloud_disable_config.yaml").read_bytes()
            nogpu = (data_pkg / "no_gpu_nodes.json").read_bytes()
            self.annotate(sup, dis, nogpu)
        except Exception:
            pass  # missing package data is non-fatal

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
            "needs_gpu": m.needs_gpu,
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


def _validate_catalog_value(
    node_id: str, class_type: str, input_name: str, port: Port, value: Any
) -> tuple[list[dict], list[dict]]:
    """Enum-membership and other catalog checks for one scalar input value.

    Returns (errors, warnings): an unknown enum value is a hard error carrying
    the valid options; every other catalog finding is a namespaced warning.
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
        else:
            w["field"] = f"{node_id}.{class_type}.{w['field']}"
            warnings.append(w)
    return errors, warnings


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


_opener = urllib.request.build_opener(NoRedirectHandler())


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


def _load_from_target(*, mode: str = "local", host: str = "127.0.0.1", port: int = 8188) -> dict[str, Any]:
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
    if target.is_cloud:
        if target.api_key:
            req.add_header("X-API-Key", target.api_key)
        elif target.auth_token:
            req.add_header("Authorization", f"Bearer {target.auth_token}")

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
