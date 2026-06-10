"""Typed fragment-based workflow composition — the domain core.

A FRAGMENT is a workflow-JSON file with a ``_fragment`` metadata header that
declares its typed inputs, outputs, and parameters. A BLUEPRINT (YAML) lists
fragments to instantiate, how to bind their inputs, and how to override their
parameters. Composing a blueprint produces ONE API-format workflow ready to
submit with ``comfy run``.

The big idea: tested, reusable subgraph fragments — "workflow as code." An
agent can ship a 4-stage video pipeline by writing a 10-line blueprint instead
of hand-merging four 100-node JSONs and rewiring edges.

Format — ``fragments/<name>.json``
----------------------------------
::

    {
      "_fragment": {
        "name":        "image_blend",
        "version":     "1",
        "description": "Blend two images using a configurable mode and factor.",
        "terminal":    false,
        "inputs":  {"image1": {"type":"IMAGE", "binds":"10.image1"}, ...},
        "outputs": {"image":  {"type":"IMAGE", "from":"10", "port":0}},
        "params":  {"blend_factor": {"type":"FLOAT", "binds":"10.blend_factor",
                                     "default":0.5}, ...}
      },
      "10": {"class_type":"ImageBlend", "inputs":{...}, "_meta":{...}},
      ...
    }

Blueprint — ``blueprints/<name>.yaml``
--------------------------------------
::

    output_prefix: outputs/my_pipeline
    pipeline:
      - fragment: text_card
        alias:    headline
        inputs:   {destination_image: inputs/base.png,
                   source_mask:       inputs/mask.png}
        params:   {text_prompt: "BREAKING NEWS"}

      - fragment: text_card
        alias:    subhead
        inputs:   {destination_image: $headline.image,
                   source_mask:       inputs/sub_mask.png}
        params:   {text_prompt: "...details..."}

This module is pure value-in, value-out: it reads fragment files and returns
plain dicts. It does no rendering and knows nothing about Typer or error
codes — the CLI shell in ``command/workflow_fragments.py`` wraps it.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Input modalities the composer can materialize from a bare file path by
# injecting a loader node. Everything else (MODEL, CONDITIONING, LATENT, VAE,
# and any custom socket type) is "graph-only": valid as a fragment input, but
# it can only be fed by a cross-step ref (`$alias.output`), never a path.
LOADABLE_INPUT_TYPES = {"IMAGE", "MASK", "AUDIO", "VIDEO"}
KNOWN_PARAM_TYPES = {"STRING", "INT", "FLOAT", "BOOL", "COMBO"}

# A ComfyUI socket type is UPPER_SNAKE_CASE (IMAGE, MODEL, CONTROL_NET, ...).
# We accept any such token as an input type so fragments can model the full
# ComfyUI graph, including custom-node sockets — not just a fixed modality set.
_SOCKET_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# A step alias reads like a variable name; this also keeps `$alias.output`
# refs unambiguous (a stray `:` or `.` can't masquerade as an alias).
_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FragmentError(Exception):
    """A fragment file is malformed or fails schema validation."""

    def __init__(self, message: str, *, path: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.path = path
        self.hint = hint


class BlueprintError(Exception):
    """A blueprint is malformed or references a fragment that won't compose."""

    def __init__(self, message: str, *, step_alias: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.step_alias = step_alias
        self.hint = hint


# ---------------------------------------------------------------------------
# Fragment model
# ---------------------------------------------------------------------------


@dataclass
class FragmentPort:
    name: str
    type: str
    binds: str | None = None  # for inputs and params: "<node_id>.<input_name>"
    from_node: str | None = None  # for outputs
    port: int = 0
    default: Any = None
    has_default: bool = False


@dataclass
class Fragment:
    """A parsed fragment: metadata + interior nodes."""

    name: str
    version: str
    description: str
    terminal: bool
    inputs: dict[str, FragmentPort] = field(default_factory=dict)
    outputs: dict[str, FragmentPort] = field(default_factory=dict)
    params: dict[str, FragmentPort] = field(default_factory=dict)
    nodes: dict[str, dict] = field(default_factory=dict)
    source_path: str = ""


def _parse_port(name: str, spec: dict, role: str) -> FragmentPort:
    """Parse one entry from inputs/outputs/params. ``role`` ∈ {input, output, param}."""
    if not isinstance(spec, dict):
        raise FragmentError(f"{role} {name!r}: expected an object, got {type(spec).__name__}")
    t = spec.get("type")
    if not isinstance(t, str):
        raise FragmentError(f"{role} {name!r}: missing or non-string `type`")
    if role == "input" and not _SOCKET_TYPE_RE.match(t):
        raise FragmentError(
            f"{role} {name!r}: type {t!r} is not a valid ComfyUI socket type",
            hint="use an UPPER_SNAKE_CASE socket type, e.g. IMAGE, MODEL, CONDITIONING, LATENT, VAE",
        )
    if role == "param" and t not in KNOWN_PARAM_TYPES:
        raise FragmentError(
            f"{role} {name!r}: type {t!r} not in {sorted(KNOWN_PARAM_TYPES)}",
        )

    port = FragmentPort(name=name, type=t)
    if role in ("input", "param"):
        binds = spec.get("binds")
        if not isinstance(binds, str) or "." not in binds:
            raise FragmentError(
                f"{role} {name!r}: `binds` must be '<node_id>.<input_name>' (got {binds!r})",
            )
        port.binds = binds
    elif role == "output":
        frm = spec.get("from")
        if not isinstance(frm, str):
            raise FragmentError(f"output {name!r}: `from` must be a string node id (got {frm!r})")
        port.from_node = frm
        port.port = int(spec.get("port", 0))
    if role == "param" and "default" in spec:
        port.default = spec["default"]
        port.has_default = True
    return port


def parse_fragment(data: dict, *, source_path: str = "") -> Fragment:
    """Parse a fragment JSON dict into a typed ``Fragment``.

    Raises ``FragmentError`` on any schema violation.
    """
    if not isinstance(data, dict):
        raise FragmentError("fragment JSON must be an object", path=source_path)
    meta = data.get("_fragment")
    if not isinstance(meta, dict):
        raise FragmentError(
            "missing `_fragment` metadata header",
            path=source_path,
            hint="every fragment file must declare a `_fragment` object with name/inputs/outputs/params",
        )

    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise FragmentError("`_fragment.name` is required (non-empty string)", path=source_path)

    inputs_raw = meta.get("inputs") or {}
    outputs_raw = meta.get("outputs") or {}
    params_raw = meta.get("params") or {}
    for label, raw in (("inputs", inputs_raw), ("outputs", outputs_raw), ("params", params_raw)):
        if not isinstance(raw, dict):
            raise FragmentError(f"`_fragment.{label}` must be an object", path=source_path)

    frag = Fragment(
        name=name,
        version=str(meta.get("version", "1")),
        description=str(meta.get("description", "")),
        terminal=bool(meta.get("terminal", False)),
        source_path=source_path,
    )
    for n, spec in inputs_raw.items():
        frag.inputs[n] = _parse_port(n, spec, "input")
    for n, spec in outputs_raw.items():
        frag.outputs[n] = _parse_port(n, spec, "output")
    for n, spec in params_raw.items():
        frag.params[n] = _parse_port(n, spec, "param")

    # interior nodes: every top-level key besides _fragment
    for k, v in data.items():
        if k == "_fragment":
            continue
        if not isinstance(v, dict) or "class_type" not in v:
            raise FragmentError(
                f"interior key {k!r}: expected a node object with `class_type`",
                path=source_path,
            )
        frag.nodes[k] = v
    if not frag.nodes:
        raise FragmentError("fragment has no interior nodes", path=source_path)

    # cross-check: all `binds` and `from` reference real interior nodes
    for port in list(frag.inputs.values()) + list(frag.params.values()):
        if port.binds is None:
            raise FragmentError(
                f"input/param {port.name!r} is missing `binds`",
                path=source_path,
            )
        node_id = port.binds.split(".", 1)[0]
        if node_id not in frag.nodes:
            raise FragmentError(
                f"`binds` points to missing interior node {node_id!r} (in {port.name!r})",
                path=source_path,
            )
    for port in frag.outputs.values():
        if port.from_node not in frag.nodes:
            raise FragmentError(
                f"output {port.name!r}: `from` points to missing interior node {port.from_node!r}",
                path=source_path,
            )
    return frag


def load_fragment(path: Path) -> Fragment:
    """Read a fragment JSON file. Raises FragmentError on I/O or schema failure."""
    if not path.is_file():
        raise FragmentError(f"fragment file not found: {path}", path=str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise FragmentError(f"unable to read fragment: {e}", path=str(path)) from e
    except json.JSONDecodeError as e:
        raise FragmentError(f"fragment file is not valid JSON: {e}", path=str(path)) from e
    return parse_fragment(data, source_path=str(path))


def resolve_fragment_name(name: str, lib_dir: Path) -> Path:
    """``name`` may be a bare name (``text_card`` → ``<lib>/text_card.json``) or a path."""
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate
    return (lib_dir / f"{name}.json").expanduser()


# ---------------------------------------------------------------------------
# Pipeline composer
# ---------------------------------------------------------------------------


@dataclass
class _StepOutput:
    """Where a step's named output landed in the merged workflow."""

    node_id: str
    port: int
    type: str


class Pipeline:
    """Composes fragments + per-step bindings into one API-format workflow.

    Pure value-in, value-out: load fragments, parse a blueprint, return the
    merged workflow dict. The Typer command wraps this for I/O.
    """

    def __init__(self) -> None:
        self.workflow: dict[str, dict] = {}
        self.next_id: int = 100
        self.outputs: dict[str, dict[str, _StepOutput]] = {}
        self.last_step_terminal: bool = False

    # -- ID allocation -------------------------------------------------------

    def _alloc(self, n: int) -> int:
        start = self.next_id
        self.next_id += n + 50  # buffer to keep renumbering deterministic
        return start

    # -- Loader-node helpers -------------------------------------------------

    def _add_loader(self, class_type: str, input_name: str, path: str, *, title: str) -> str:
        node_id = str(self._alloc(1))
        self.workflow[node_id] = {
            "class_type": class_type,
            "_meta": {"title": title},
            "inputs": {input_name: path},
        }
        return node_id

    def _add_image_to_mask(self, image_ref: list) -> str:
        node_id = str(self._alloc(1))
        self.workflow[node_id] = {
            "class_type": "ImageToMask",
            "_meta": {"title": "→ mask"},
            "inputs": {"image": image_ref, "channel": "red"},
        }
        return node_id

    def _resolve_input(self, value: Any, decl_type: str, *, step_alias: str, in_name: str):
        """Return the [node_id, port] (or literal) the input should bind to."""
        # Cross-step ref — wires to a prior step's output, whatever its type.
        if isinstance(value, str) and value.startswith("$"):
            ref = value[1:]
            if "." not in ref:
                raise BlueprintError(
                    f"[{step_alias}] input {in_name!r}: cross-step ref must be '$alias.output_name' (got {value!r})",
                    step_alias=step_alias,
                )
            alias, output_name = ref.split(".", 1)
            if not _ALIAS_RE.match(alias) or not output_name:
                raise BlueprintError(
                    f"[{step_alias}] input {in_name!r}: malformed cross-step ref {value!r}",
                    step_alias=step_alias,
                    hint="a cross-step ref is '$alias.output_name' — e.g. $headline.image",
                )
            if alias not in self.outputs:
                raise BlueprintError(
                    f"[{step_alias}] input {in_name!r}: unknown alias {alias!r}",
                    step_alias=step_alias,
                    hint=f"available aliases: {sorted(self.outputs.keys())}",
                )
            if output_name not in self.outputs[alias]:
                raise BlueprintError(
                    f"[{step_alias}] input {in_name!r}: alias {alias!r} has no output {output_name!r}",
                    step_alias=step_alias,
                    hint=f"alias {alias!r} exposes: {sorted(self.outputs[alias].keys())}",
                )
            out = self.outputs[alias][output_name]
            return [out.node_id, out.port]

        # STRING passes through as a literal (any scalar — int/float included).
        if decl_type == "STRING":
            return value

        if not isinstance(value, str):
            raise BlueprintError(
                f"[{step_alias}] input {in_name!r}: type {decl_type!r} needs a file path "
                f"or a cross-step ref, got {type(value).__name__}",
                step_alias=step_alias,
            )

        # Loadable modalities — materialize the path with the right loader node.
        if decl_type == "IMAGE":
            return [self._add_loader("LoadImage", "image", value, title=f"load {Path(value).name}"), 0]
        if decl_type == "MASK":
            load_id = self._add_loader("LoadImage", "image", value, title=f"load {Path(value).name}")
            return [self._add_image_to_mask([load_id, 0]), 0]
        if decl_type == "AUDIO":
            return [self._add_loader("LoadAudio", "audio", value, title=f"load {Path(value).name}"), 0]
        if decl_type == "VIDEO":
            return [self._add_loader("LoadVideo", "video", value, title=f"load {Path(value).name}"), 0]

        # Graph-only socket types (MODEL, CONDITIONING, LATENT, VAE, custom):
        # there is no loader to inject — they must come from a prior step.
        raise BlueprintError(
            f"[{step_alias}] input {in_name!r}: type {decl_type!r} can't be loaded from a path "
            f"({value!r}); feed it from a prior step with a cross-step ref",
            step_alias=step_alias,
            hint="only IMAGE/MASK/AUDIO/VIDEO accept a file path; wire everything else via $alias.output_name",
        )

    # -- Add one step --------------------------------------------------------

    def add_step(self, fragment: Fragment, alias: str, inputs: dict, params: dict) -> None:
        if not _ALIAS_RE.match(alias):
            raise BlueprintError(
                f"alias {alias!r} is not a valid identifier",
                step_alias=alias,
                hint="aliases read like a variable name: letters/digits/_/-, starting with a letter or _",
            )
        # Validate inputs/params presence
        for in_name in fragment.inputs:
            if in_name not in inputs:
                raise BlueprintError(
                    f"[{alias}] missing required input {in_name!r}",
                    step_alias=alias,
                    hint=f"fragment {fragment.name!r} requires: {sorted(fragment.inputs.keys())}",
                )
        full_params: dict[str, Any] = {}
        for p_name, port in fragment.params.items():
            if p_name in params:
                full_params[p_name] = params[p_name]
            elif port.has_default:
                full_params[p_name] = port.default
            else:
                raise BlueprintError(
                    f"[{alias}] missing required param {p_name!r} (no default)",
                    step_alias=alias,
                )
        # Unknown keys → fail loud, so typos don't silently no-op.
        extra_inputs = set(inputs) - set(fragment.inputs)
        if extra_inputs:
            raise BlueprintError(
                f"[{alias}] unknown inputs: {sorted(extra_inputs)}",
                step_alias=alias,
                hint=f"fragment declares inputs: {sorted(fragment.inputs.keys())}",
            )
        extra_params = set(params) - set(fragment.params)
        if extra_params:
            raise BlueprintError(
                f"[{alias}] unknown params: {sorted(extra_params)}",
                step_alias=alias,
                hint=f"fragment declares params: {sorted(fragment.params.keys())}",
            )
        if alias in self.outputs:
            raise BlueprintError(f"alias {alias!r} used by a previous step", step_alias=alias)

        # Deep-copy interior nodes, remap IDs, apply params + inputs
        offset = self._alloc(len(fragment.nodes))
        remap = {old: str(int(old) + offset) for old in fragment.nodes.keys()}
        new_nodes: dict[str, dict] = {}
        for old_id, node in fragment.nodes.items():
            new_node = copy.deepcopy(node)
            for input_name, value in list(new_node.get("inputs", {}).items()):
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    if value[0] in remap:
                        new_node["inputs"][input_name] = [remap[value[0]], value[1]]
            new_nodes[remap[old_id]] = new_node

        for p_name, port in fragment.params.items():
            if port.binds is None:
                raise FragmentError(f"param {port.name!r} is missing `binds`")
            old_id, input_name = port.binds.split(".", 1)
            new_nodes[remap[old_id]]["inputs"][input_name] = full_params[p_name]
        for in_name, port in fragment.inputs.items():
            if port.binds is None:
                raise FragmentError(f"input {port.name!r} is missing `binds`")
            resolved = self._resolve_input(inputs[in_name], port.type, step_alias=alias, in_name=in_name)
            old_id, input_name = port.binds.split(".", 1)
            new_nodes[remap[old_id]]["inputs"][input_name] = resolved

        self.workflow.update(new_nodes)

        # Record outputs (keep the declared type so the composer can pick a
        # save node later without re-reading the fragment from disk).
        self.outputs[alias] = {}
        for o_name, port in fragment.outputs.items():
            if port.from_node is None:
                raise FragmentError(f"output {port.name!r} is missing `from`")
            self.outputs[alias][o_name] = _StepOutput(node_id=remap[port.from_node], port=port.port, type=port.type)
        self.last_step_terminal = fragment.terminal

    # -- Save-node convenience ----------------------------------------------

    def add_save(self, output: _StepOutput, output_type: str, *, prefix: str) -> None:
        if output_type == "VIDEO":
            class_type, ref_key = "SaveVideo", "video"
            inputs = {
                ref_key: [output.node_id, output.port],
                "filename_prefix": prefix,
                "format": "mp4",
                "codec": "h264",
            }
        else:
            class_type, ref_key = "SaveImage", "images"
            inputs = {ref_key: [output.node_id, output.port], "filename_prefix": prefix}
        node_id = str(self._alloc(1))
        self.workflow[node_id] = {
            "class_type": class_type,
            "_meta": {"title": f"save composed final ({output_type.lower()})"},
            "inputs": inputs,
        }


# ---------------------------------------------------------------------------
# Blueprint parsing + compose entry points
# ---------------------------------------------------------------------------

# `$item` / `$item.<field>` placeholders in a `foreach` pipeline template are
# replaced with the current item's value before the step is composed. A bare
# `$item` yields the whole item; `$item.<field>` indexes a mapping item.
_ITEM_RE = re.compile(r"^\$item(?:\.([A-Za-z_][A-Za-z0-9_-]*))?$")


def _validate_pipeline_steps(steps: Any) -> list[dict]:
    """Validate a `pipeline:` list and return it (each step is a mapping)."""
    if not isinstance(steps, list) or not steps:
        raise BlueprintError("blueprint must have a non-empty `pipeline:` list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise BlueprintError(f"pipeline[{i}]: each step must be a mapping")
        name = step.get("fragment")
        alias = step.get("alias")
        if not isinstance(name, str) or not isinstance(alias, str):
            raise BlueprintError(f"pipeline[{i}]: each step must declare `fragment` and `alias` strings")
    return steps


def _substitute_item(value: Any, item: Any, *, ns: str, alias_refs: bool = True) -> Any:
    """Resolve `$item` placeholders in a blueprint value against ``item``.

    A bare ``$item`` becomes the whole item; ``$item.<field>`` indexes a mapping
    item. Cross-step refs (``$alias.output``) are namespaced with ``ns`` so each
    item's branch wires to its own copies, never another item's. Non-``$`` values
    pass through untouched. Containers (lists/dicts) are recursed.

    ``alias_refs`` controls whether ``$alias.output`` cross-step refs are
    namespaced. Set to ``False`` for params, which are literal widget values and
    must never be alias-rewritten.
    """
    if isinstance(value, str) and value.startswith("$"):
        m = _ITEM_RE.match(value)
        if m:
            field = m.group(1)
            if field is None:
                return item
            if not isinstance(item, dict):
                raise BlueprintError(
                    f"`$item.{field}` requires a mapping item, got {type(item).__name__}",
                    hint="use `$item` (whole value) for scalar foreach items",
                )
            if field not in item:
                raise BlueprintError(
                    f"foreach item has no field {field!r} (referenced as `$item.{field}`)",
                    hint=f"item exposes: {sorted(item.keys())}",
                )
            return item[field]
        # A `$alias.output` cross-step ref — namespace its alias to this branch.
        # Only valid for INPUTS; params are literal widget values, never refs.
        if alias_refs and "." in value:
            alias, _, output = value[1:].partition(".")
            if _ALIAS_RE.match(alias):
                return f"${ns}__{alias}.{output}"
        return value
    if isinstance(value, list):
        return [_substitute_item(v, item, ns=ns, alias_refs=alias_refs) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_item(v, item, ns=ns, alias_refs=alias_refs) for k, v in value.items()}
    return value


def _add_branch(
    pipeline: Pipeline,
    steps: list[dict],
    *,
    lib_dir: Path,
    item: Any = None,
    ns: str | None = None,
) -> tuple[str, bool, list[str]]:
    """Add one pipeline instantiation to ``pipeline``.

    When ``item``/``ns`` are given the steps are treated as a ``foreach`` template:
    ``$item.*`` placeholders are bound to ``item`` and every alias (its own +
    cross-step refs) is prefixed with ``ns`` so the branch can't collide with or
    wire into another item's branch. Returns ``(final_alias, last_terminal,
    used_fragments)``.
    """
    used_fragments: list[str] = []
    final_alias = ""
    for step in steps:
        name = step["fragment"]
        alias = step["alias"]
        inputs = step.get("inputs") or {}
        params = step.get("params") or {}
        if ns is not None:
            alias = f"{ns}__{alias}"
            inputs = {k: _substitute_item(v, item, ns=ns) for k, v in inputs.items()}
            params = {k: _substitute_item(v, item, ns=ns, alias_refs=False) for k, v in params.items()}
        fragment = load_fragment(resolve_fragment_name(name, lib_dir))
        used_fragments.append(fragment.name)
        pipeline.add_step(fragment=fragment, alias=alias, inputs=inputs, params=params)
        final_alias = alias
    return final_alias, pipeline.last_step_terminal, used_fragments


def _auto_save(pipeline: Pipeline, final_alias: str, *, terminal: bool, prefix: str) -> dict | None:
    """Append a save node for a branch's final IMAGE/VIDEO output, unless terminal."""
    if terminal:
        return None
    chosen = None
    for out in pipeline.outputs[final_alias].values():
        if out.type in ("IMAGE", "VIDEO"):
            chosen = out
            break
    if chosen is None:
        return None
    pipeline.add_save(chosen, chosen.type, prefix=prefix)
    return {"type": chosen.type, "prefix": prefix}


def compose_blueprint(blueprint: dict, *, lib_dir: Path) -> tuple[dict, dict]:
    """Compose a single-graph (no ``foreach``) blueprint into an API workflow.

    Returns ``(workflow, summary)`` where ``summary`` describes the composition
    (step count, node count, final-save action). Raises ``BlueprintError`` /
    ``FragmentError`` on any failure. For ``foreach`` blueprints (and chunking),
    use :func:`compose_blueprints`, which returns one entry per emitted graph.
    """
    if not isinstance(blueprint, dict):
        raise BlueprintError("blueprint must be a YAML mapping")
    steps = _validate_pipeline_steps(blueprint.get("pipeline"))

    pipeline = Pipeline()
    final_alias, terminal, used_fragments = _add_branch(pipeline, steps, lib_dir=lib_dir)
    save_action = _auto_save(
        pipeline, final_alias, terminal=terminal, prefix=str(blueprint.get("output_prefix", "composed"))
    )

    summary = {
        "steps": len(steps),
        "nodes": len(pipeline.workflow),
        "fragments_used": used_fragments,
        "final_alias": final_alias,
        "save_action": save_action,
    }
    return pipeline.workflow, summary


def _resolve_foreach_items(spec: Any, *, blueprint_dir: Path | None) -> list[Any]:
    """Resolve a ``foreach:`` value (inline list or ``{$ref: path.yaml}``) to items."""
    if isinstance(spec, dict) and "$ref" in spec:
        ref = spec["$ref"]
        if not isinstance(ref, str):
            raise BlueprintError("foreach `$ref` must be a path string")
        if blueprint_dir is None:
            raise BlueprintError(
                "foreach `$ref` cannot be resolved without a blueprint directory",
                hint="`$ref` is resolved relative to the blueprint file; compose from a file, not an in-memory dict",
            )
        ref_path = (blueprint_dir / ref).expanduser()
        if not ref_path.is_file():
            raise BlueprintError(f"foreach `$ref` file not found: {ref_path}")
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise BlueprintError("PyYAML is required to resolve foreach `$ref`") from e
        try:
            spec = yaml.safe_load(ref_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise BlueprintError(f"foreach `$ref` file is not valid YAML: {e}") from e
    if not isinstance(spec, list) or not spec:
        raise BlueprintError(
            "foreach must resolve to a non-empty list of items",
            hint="provide an inline list or `{$ref: items.yaml}` pointing at a YAML list",
        )
    return spec


def _item_namespace(item: Any, index: int) -> str:
    """A unique, alias-safe namespace token for one item's branch."""
    if isinstance(item, dict) and isinstance(item.get("id"), str | int):
        raw = re.sub(r"[^A-Za-z0-9_]", "_", str(item["id"]))
        if raw:
            return f"i{index}_{raw}"
    return f"i{index}"


def _item_prefix(base: str, item: Any, index: int) -> str:
    """Per-item terminal/output prefix — uses the item id when present, else index."""
    if isinstance(item, dict) and item.get("id") is not None:
        return f"{base}/{item['id']}"
    return f"{base}/{index}"


def compose_blueprints(
    blueprint: dict,
    *,
    lib_dir: Path,
    blueprint_dir: Path | None = None,
) -> list[tuple[dict, dict]]:
    """Compose a blueprint into one or more API workflows.

    Without ``foreach:`` this returns a single ``[(workflow, summary)]`` — the
    same graph :func:`compose_blueprint` produces. With ``foreach:`` the
    ``pipeline:`` is a TEMPLATE instantiated once per item; all instantiations
    land in ONE graph as independent branches (the engine parallelizes them).
    ``$item`` / ``$item.<field>`` in step ``inputs``/``params`` bind to the item;
    each branch gets a unique alias namespace so the copies never collide.

    ``chunk: N`` (only when set) splits the items into ``ceil(items/N)`` graphs
    for memory limits; the default is a single graph. ``foreach: {$ref: x.yaml}``
    loads items from a YAML list resolved relative to ``blueprint_dir``.
    """
    if not isinstance(blueprint, dict):
        raise BlueprintError("blueprint must be a YAML mapping")
    steps = _validate_pipeline_steps(blueprint.get("pipeline"))

    if "foreach" not in blueprint:
        return [compose_blueprint(blueprint, lib_dir=lib_dir)]

    items = _resolve_foreach_items(blueprint["foreach"], blueprint_dir=blueprint_dir)
    base_prefix = str(blueprint.get("output_prefix", "composed"))

    chunk = blueprint.get("chunk")
    if chunk is None:
        chunk_size = len(items)
    else:
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise BlueprintError("`chunk:` must be a positive integer", hint="omit `chunk` for a single graph")
        chunk_size = chunk

    enumerated = list(enumerate(items))
    batches = [enumerated[i : i + chunk_size] for i in range(0, len(enumerated), chunk_size)]
    total_graphs = len(batches)

    results: list[tuple[dict, dict]] = []
    for batch in batches:
        pipeline = Pipeline()
        save_actions: list[dict] = []
        used_fragments: list[str] = []
        for index, item in batch:
            ns = _item_namespace(item, index)
            final_alias, terminal, frags = _add_branch(pipeline, steps, lib_dir=lib_dir, item=item, ns=ns)
            used_fragments.extend(frags)
            sa = _auto_save(pipeline, final_alias, terminal=terminal, prefix=_item_prefix(base_prefix, item, index))
            if sa:
                save_actions.append(sa)
        summary = {
            "steps": len(steps),
            "items": len(batch),
            "total_items": len(items),
            "graphs": total_graphs,
            "nodes": len(pipeline.workflow),
            "fragments_used": used_fragments,
            "save_actions": save_actions,
        }
        results.append((pipeline.workflow, summary))
    return results
