"""``comfy workflow compose`` and ``comfy workflow fragment`` — typed fragment-based composition.

A FRAGMENT is a workflow-JSON file with a ``_fragment`` metadata header that
declares its typed inputs, outputs, and parameters. A RECIPE (YAML) lists
fragments to instantiate, how to bind their inputs, and how to override their
parameters. Composing a recipe produces ONE API-format workflow ready to
submit with ``comfy run``.

The big idea: tested, reusable subgraph fragments — "workflow as code." An
agent can ship a 4-stage video pipeline by writing a 10-line recipe instead
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

Recipe — ``recipes/<name>.yaml``
--------------------------------
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
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint

KNOWN_INPUT_TYPES = {"IMAGE", "MASK", "AUDIO", "VIDEO", "STRING"}
KNOWN_PARAM_TYPES = {"STRING", "INT", "FLOAT", "BOOL", "COMBO"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FragmentError(Exception):
    """A fragment file is malformed or fails schema validation."""

    def __init__(self, message: str, *, path: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.path = path
        self.hint = hint


class RecipeError(Exception):
    """A recipe is malformed or references a fragment that won't compose."""

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
    binds: str | None = None   # for inputs and params: "<node_id>.<input_name>"
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
    if role == "input" and t not in KNOWN_INPUT_TYPES:
        raise FragmentError(
            f"{role} {name!r}: type {t!r} not in {sorted(KNOWN_INPUT_TYPES)}",
            hint="extend KNOWN_INPUT_TYPES if you need a new modality",
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


class Pipeline:
    """Composes fragments + per-step bindings into one API-format workflow.

    Pure value-in, value-out: load fragments, parse recipe, return the merged
    workflow dict. The Typer command wraps this for I/O.
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
        if not isinstance(value, str):
            # Treat non-string values as literals (e.g. int/float for STRING-like passthrough).
            # Only valid for STRING right now.
            if decl_type == "STRING":
                return value
            raise RecipeError(
                f"[{step_alias}] input {in_name!r}: expected a string binding for type "
                f"{decl_type!r}, got {type(value).__name__}",
                step_alias=step_alias,
            )

        if value.startswith("$"):
            ref = value[1:]
            if "." not in ref:
                raise RecipeError(
                    f"[{step_alias}] input {in_name!r}: cross-step ref must be "
                    f"'$alias.output_name' (got {value!r})",
                    step_alias=step_alias,
                )
            alias, output_name = ref.split(".", 1)
            if alias not in self.outputs:
                raise RecipeError(
                    f"[{step_alias}] input {in_name!r}: unknown alias {alias!r}",
                    step_alias=step_alias,
                    hint=f"available aliases: {sorted(self.outputs.keys())}",
                )
            if output_name not in self.outputs[alias]:
                raise RecipeError(
                    f"[{step_alias}] input {in_name!r}: alias {alias!r} has no output "
                    f"{output_name!r}",
                    step_alias=step_alias,
                    hint=f"alias {alias!r} exposes: {sorted(self.outputs[alias].keys())}",
                )
            out = self.outputs[alias][output_name]
            return [out.node_id, out.port]

        # Literal — auto-wrap based on declared type.
        if decl_type == "STRING":
            return value
        if decl_type == "AUDIO":
            return [self._add_loader("LoadAudio", "audio", value, title=f"load {Path(value).name}"), 0]
        if decl_type == "VIDEO":
            return [self._add_loader("LoadVideo", "video", value, title=f"load {Path(value).name}"), 0]
        load_id = self._add_loader("LoadImage", "image", value, title=f"load {Path(value).name}")
        if decl_type == "MASK":
            return [self._add_image_to_mask([load_id, 0]), 0]
        return [load_id, 0]

    # -- Add one step --------------------------------------------------------

    def add_step(self, fragment: Fragment, alias: str, inputs: dict, params: dict) -> None:
        # Validate inputs/params presence
        for in_name in fragment.inputs:
            if in_name not in inputs:
                raise RecipeError(
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
                raise RecipeError(
                    f"[{alias}] missing required param {p_name!r} (no default)",
                    step_alias=alias,
                )
        # Unknown keys → fail loud, so typos don't silently no-op.
        extra_inputs = set(inputs) - set(fragment.inputs)
        if extra_inputs:
            raise RecipeError(
                f"[{alias}] unknown inputs: {sorted(extra_inputs)}",
                step_alias=alias,
                hint=f"fragment declares inputs: {sorted(fragment.inputs.keys())}",
            )
        extra_params = set(params) - set(fragment.params)
        if extra_params:
            raise RecipeError(
                f"[{alias}] unknown params: {sorted(extra_params)}",
                step_alias=alias,
                hint=f"fragment declares params: {sorted(fragment.params.keys())}",
            )
        if alias in self.outputs:
            raise RecipeError(f"alias {alias!r} used by a previous step", step_alias=alias)

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
            old_id, input_name = port.binds.split(".", 1)
            new_nodes[remap[old_id]]["inputs"][input_name] = full_params[p_name]
        for in_name, port in fragment.inputs.items():
            resolved = self._resolve_input(inputs[in_name], port.type, step_alias=alias, in_name=in_name)
            old_id, input_name = port.binds.split(".", 1)
            new_nodes[remap[old_id]]["inputs"][input_name] = resolved

        self.workflow.update(new_nodes)

        # Record outputs
        self.outputs[alias] = {}
        for o_name, port in fragment.outputs.items():
            self.outputs[alias][o_name] = _StepOutput(node_id=remap[port.from_node], port=port.port)
        self.last_step_terminal = fragment.terminal

    # -- Save-node convenience ----------------------------------------------

    def add_save(self, output: _StepOutput, output_type: str, *, prefix: str) -> None:
        if output_type == "VIDEO":
            class_type, ref_key = "SaveVideo", "video"
            inputs = {ref_key: [output.node_id, output.port],
                      "filename_prefix": prefix, "format": "mp4", "codec": "h264"}
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
# Recipe parsing + compose entry point
# ---------------------------------------------------------------------------


def compose_recipe(recipe: dict, *, lib_dir: Path) -> tuple[dict, dict]:
    """Compose a parsed recipe dict + fragment library directory into an API workflow.

    Returns ``(workflow, summary)`` where ``summary`` describes the composition
    (step count, node count, final-save action). Raises ``RecipeError`` /
    ``FragmentError`` on any failure.
    """
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be a YAML mapping")
    steps = recipe.get("pipeline")
    if not isinstance(steps, list) or not steps:
        raise RecipeError("recipe must have a non-empty `pipeline:` list")

    pipeline = Pipeline()
    used_fragments: list[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RecipeError(f"pipeline[{i}]: each step must be a mapping")
        name = step.get("fragment")
        alias = step.get("alias")
        if not isinstance(name, str) or not isinstance(alias, str):
            raise RecipeError(f"pipeline[{i}]: each step must declare `fragment` and `alias` strings")
        fragment_path = resolve_fragment_name(name, lib_dir)
        fragment = load_fragment(fragment_path)
        used_fragments.append(fragment.name)
        pipeline.add_step(
            fragment=fragment,
            alias=alias,
            inputs=step.get("inputs") or {},
            params=step.get("params") or {},
        )

    final_alias = steps[-1].get("alias")
    save_action = None
    if not pipeline.last_step_terminal:
        # Pick first IMAGE or VIDEO output of the final step
        final_outs = pipeline.outputs[final_alias]
        chosen = None
        for o_name, out in final_outs.items():
            # Look at the fragment to find this output's type
            last_step = steps[-1]
            last_frag = load_fragment(resolve_fragment_name(last_step["fragment"], lib_dir))
            if o_name in last_frag.outputs:
                t = last_frag.outputs[o_name].type
                if t in ("IMAGE", "VIDEO"):
                    chosen = (out, t)
                    break
        if chosen:
            output, out_type = chosen
            prefix = str(recipe.get("output_prefix", "composed"))
            pipeline.add_save(output, out_type, prefix=prefix)
            save_action = {"type": out_type, "prefix": prefix}

    summary = {
        "steps": len(steps),
        "nodes": len(pipeline.workflow),
        "fragments_used": used_fragments,
        "final_alias": final_alias,
        "save_action": save_action,
    }
    return pipeline.workflow, summary


# ---------------------------------------------------------------------------
# Typer surface
# ---------------------------------------------------------------------------

fragment_app = typer.Typer(no_args_is_help=True, help="Inspect and validate workflow fragments.")


def _default_lib_dir(override: str | None) -> Path:
    """Resolve ``--lib`` → Path. Default is ``./fragments`` in cwd."""
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "fragments"


@tracking.track_command("workflow")
def compose_cmd(
    recipe: Annotated[Path, typer.Argument(help="Recipe YAML file.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", show_default=False, help="Output workflow JSON path. Defaults to <recipe>.compiled.json"),
    ] = None,
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Fragment library directory. Defaults to ./fragments"),
    ] = None,
):
    """Compose a YAML recipe of fragments into a single API-format workflow."""
    renderer = get_renderer()
    if not recipe.is_file():
        renderer.error(code="recipe_not_found", message=f"Recipe file not found: {recipe}")
        raise typer.Exit(code=1)

    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        renderer.error(code="recipe_yaml_unavailable", message="PyYAML is required for `compose`",
                       hint="install with: pip install pyyaml")
        raise typer.Exit(code=1) from e

    try:
        recipe_data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        renderer.error(code="recipe_invalid_yaml", message=f"Recipe is not valid YAML: {e}")
        raise typer.Exit(code=1) from e

    lib_dir = _default_lib_dir(lib)
    try:
        workflow, summary = compose_recipe(recipe_data, lib_dir=lib_dir)
    except FragmentError as e:
        renderer.error(code="fragment_invalid", message=str(e),
                       hint=e.hint or "", details={"path": e.path})
        raise typer.Exit(code=1) from e
    except RecipeError as e:
        renderer.error(code="recipe_invalid", message=str(e),
                       hint=e.hint or "", details={"step_alias": e.step_alias})
        raise typer.Exit(code=1) from e

    out_path = out or recipe.with_suffix(".compiled.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

    payload = {
        "recipe": str(recipe),
        "out": str(out_path),
        **summary,
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] composed [bold]{out_path}[/bold]")
        rprint(f"  steps     : {summary['steps']}")
        rprint(f"  nodes     : {summary['nodes']}")
        rprint(f"  fragments : {', '.join(summary['fragments_used'])}")
        if summary.get("save_action"):
            sa = summary["save_action"]
            rprint(f"  saving    : {sa['type']} → '{sa['prefix']}'")
    renderer.emit(payload, command="workflow compose")


@fragment_app.command("ls", help="List fragments in a library directory.")
@tracking.track_command("workflow")
def fragment_ls_cmd(
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    if not lib_dir.is_dir():
        renderer.error(
            code="fragment_lib_not_found",
            message=f"Fragment library directory not found: {lib_dir}",
            hint="create ./fragments/ or pass --lib <dir>",
        )
        raise typer.Exit(code=1)

    rows: list[dict] = []
    errors: list[dict] = []
    for path in sorted(lib_dir.glob("*.json")):
        try:
            frag = load_fragment(path)
        except FragmentError as e:
            errors.append({"path": str(path), "error": str(e)})
            continue
        rows.append({
            "name": frag.name,
            "version": frag.version,
            "description": frag.description,
            "inputs": list(frag.inputs.keys()),
            "outputs": list(frag.outputs.keys()),
            "params": list(frag.params.keys()),
            "terminal": frag.terminal,
            "path": str(path),
        })

    payload = {"lib": str(lib_dir), "count": len(rows), "fragments": rows, "errors": errors}
    if renderer.is_pretty():
        if not rows and not errors:
            rprint("[dim]No fragments found.[/dim]")
        for f in rows:
            rprint(f"[bold]{f['name']}[/bold]  v{f['version']}  "
                   f"in={','.join(f['inputs']) or '∅'}  "
                   f"out={','.join(f['outputs']) or '∅'}  "
                   f"params={','.join(f['params']) or '∅'}")
            if f["description"]:
                rprint(f"  [dim]{f['description']}[/dim]")
        for e in errors:
            rprint(f"[red]✗ {e['path']}: {e['error']}[/red]")
    renderer.emit(payload, command="workflow fragment ls")


@fragment_app.command("show", help="Show a fragment's metadata, ports, and interior node count.")
@tracking.track_command("workflow")
def fragment_show_cmd(
    fragment: Annotated[str, typer.Argument(help="Fragment name (looked up in --lib) or path to .json.")],
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    path = resolve_fragment_name(fragment, lib_dir)
    try:
        frag = load_fragment(path)
    except FragmentError as e:
        renderer.error(code="fragment_invalid", message=str(e),
                       hint=e.hint or "", details={"path": e.path})
        raise typer.Exit(code=1) from e

    payload = {
        "path": str(path),
        "name": frag.name,
        "version": frag.version,
        "description": frag.description,
        "terminal": frag.terminal,
        "inputs": {n: {"type": p.type, "binds": p.binds} for n, p in frag.inputs.items()},
        "outputs": {n: {"type": p.type, "from": p.from_node, "port": p.port} for n, p in frag.outputs.items()},
        "params": {n: {"type": p.type, "binds": p.binds,
                       **({"default": p.default} if p.has_default else {})}
                   for n, p in frag.params.items()},
        "node_count": len(frag.nodes),
    }
    if renderer.is_pretty():
        rprint(f"[bold]{frag.name}[/bold]  v{frag.version}")
        if frag.description:
            rprint(f"  [dim]{frag.description}[/dim]")
        rprint(f"  terminal: {frag.terminal}  |  interior nodes: {len(frag.nodes)}")
        if frag.inputs:
            rprint("  [bold]inputs[/bold]")
            for n, p in frag.inputs.items():
                rprint(f"    {n}: {p.type}  → {p.binds}")
        if frag.outputs:
            rprint("  [bold]outputs[/bold]")
            for n, p in frag.outputs.items():
                rprint(f"    {n}: {p.type}  ← {p.from_node}[{p.port}]")
        if frag.params:
            rprint("  [bold]params[/bold]")
            for n, p in frag.params.items():
                d = f"  (default={p.default!r})" if p.has_default else ""
                rprint(f"    {n}: {p.type}  → {p.binds}{d}")
    renderer.emit(payload, command="workflow fragment show")


@fragment_app.command("validate", help="Validate that a fragment file is well-formed.")
@tracking.track_command("workflow")
def fragment_validate_cmd(
    fragment: Annotated[str, typer.Argument(help="Fragment name (looked up in --lib) or path to .json.")],
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    path = resolve_fragment_name(fragment, lib_dir)
    try:
        frag = load_fragment(path)
    except FragmentError as e:
        renderer.error(
            code="fragment_invalid",
            message=str(e),
            hint=e.hint or "",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e

    payload = {
        "path": str(path),
        "valid": True,
        "name": frag.name,
        "node_count": len(frag.nodes),
        "ports": {"inputs": len(frag.inputs), "outputs": len(frag.outputs), "params": len(frag.params)},
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] {path}  ({frag.name} v{frag.version}, {len(frag.nodes)} nodes)")
    renderer.emit(payload, command="workflow fragment validate")
