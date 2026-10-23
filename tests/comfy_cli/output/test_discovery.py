"""Phase 2: ``comfy discover`` produces a complete self-describing document.

These are the contract tests: an agent should be able to call ``comfy
--json discover`` once and get everything it needs without consulting source.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"
SRC_ROOT = Path(__file__).resolve().parents[3] / "comfy_cli"


def _validator_for(schema_name: str) -> jsonschema.Validator:
    schema = json.loads((SCHEMAS_DIR / schema_name).read_text())
    store: dict[str, dict] = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        if s.get("$id"):
            store[s["$id"]] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def _run_cli(args: list[str], env: dict | None = None) -> dict:
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    if env:
        proc_env.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    assert result.stdout.strip(), f"empty stdout. stderr={result.stderr!r}"
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


def test_discover_envelope_validates():
    envelope = _run_cli(["--json", "discover"])
    _validator_for("envelope.json").validate(envelope)
    assert envelope["ok"] is True
    assert envelope["command"] == "discover"


def test_discover_payload_validates():
    envelope = _run_cli(["--json", "discover"])
    _validator_for("discover.json").validate(envelope["data"])


def test_discover_includes_all_shipped_schemas():
    envelope = _run_cli(["--json", "discover"])
    data = envelope["data"]
    shipped = {p.stem for p in SCHEMAS_DIR.glob("*.json")}
    declared = set(data["schemas"].keys())
    assert shipped == declared


def test_discover_annotates_commands_with_schema():
    envelope = _run_cli(["--json", "discover"])
    cmds = envelope["data"]["commands"]
    assert cmds["env"]["output_schema"] == "env.json"
    assert cmds["which"]["output_schema"] == "which.json"
    assert cmds["run"]["output_schema"] == "run.json"
    assert cmds["run"]["stream_event_schema"] == "run_event.json"
    assert cmds["discover"]["output_schema"] == "discover.json"
    assert cmds["docs"]["subcommands"]["search"]["output_schema"] == "docs.json"
    assert cmds["docs"]["subcommands"]["status"]["output_schema"] == "docs.json"
    assert cmds["docs"]["subcommands"]["show"]["output_schema"] == "docs.json"


def test_discover_lists_run_prompt_and_set_options():
    # `comfy run --prompt`/`--set` must be visible on the agent
    # surface. The options are auto-introspected into the commands tree.
    envelope = _run_cli(["--json", "discover"])
    run_params = envelope["data"]["commands"]["run"]["params"]
    flags = {flag for p in run_params for flag in (p.get("flags") or [])}
    assert "--prompt" in flags
    assert "--set" in flags
    # --workflow is now optional (omit it to use the bundled default).
    workflow_param = next(p for p in run_params if "--workflow" in (p.get("flags") or []))
    assert workflow_param["required"] is False


def test_discover_includes_error_codes_from_markdown():
    envelope = _run_cli(["--json", "discover"])
    codes = {row["code"] for row in envelope["data"]["error_codes"]}
    # Spot-check a representative set across phases.
    assert "cancelled" in codes
    assert "workflow_not_found" in codes
    assert "cloud_not_configured" in codes
    # `cql_unavailable` was removed when the Python grammar layer was deleted.
    assert "cql_no_graph" in codes  # the loader's "no source available" survives


def test_discover_capabilities_flags():
    envelope = _run_cli(["--json", "discover"])
    caps = envelope["data"]["capabilities"]
    assert caps["json_envelope"] is True
    assert caps["json_stream"] is True
    assert caps["cancellation"] is True
    assert caps["cql"] is True
    assert caps["where_routing"] is True
    assert "local" in caps["where_targets"]
    assert "cloud" in caps["where_targets"]


def test_discover_schemas_only_strips_command_tree():
    envelope = _run_cli(["--json", "discover", "--schemas-only"])
    data = envelope["data"]
    assert "schemas" in data
    assert "commands" not in data
    assert "error_codes" not in data


def test_models_and_templates_registered():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    for cmd in (
        # Canonical singular `model` noun.
        "comfy model search",
        "comfy model show",
        "comfy model list-folders",
        "comfy model list-folder",
        # Deprecated plural alias — still registered (envelopes carry `models …`).
        "comfy models search",
        "comfy models show",
        "comfy models list-folders",
        "comfy models list-folder",
        "comfy templates ls",
        "comfy templates show",
        "comfy templates fetch",
    ):
        assert cmd in COMMAND_SCHEMAS, cmd


def test_discover_exposes_output_contract_versions():
    """Agents negotiate shape on `output_contract`, not the CLI version."""
    from comfy_cli.output.renderer import ENVELOPE_SCHEMA, EVENT_SCHEMA

    envelope = _run_cli(["--json", "discover"])
    contract = envelope["data"]["output_contract"]
    assert contract == {"envelope": ENVELOPE_SCHEMA, "event": EVENT_SCHEMA}
    assert contract == {"envelope": "envelope/1", "event": "event/1"}
    # The envelope itself carries the discriminator + version.
    assert envelope["schema"] == "envelope/1"
    assert envelope["type"] == "envelope"


# ---------------------------------------------------------------------------
# Registration ratchet: every `renderer.emit(..., command="X")` call site must
# either be registered in COMMAND_SCHEMAS or sit in the frozen allowlist
# below. The allowlist may only shrink — new commands must register a schema.
# (Scan style mirrors tests/comfy_cli/output/test_error_code_registry.py.)
# ---------------------------------------------------------------------------

# Commands that emitted envelopes before schema registration was enforced.
# Do NOT add to this set: register the command in COMMAND_SCHEMAS instead.
LEGACY_UNREGISTERED: frozenset[str] = frozenset(
    {
        "agent-review",
        "cloud set-base-url",
        "cloud set-key",
        "feedback",
        "generate emit-workflow",
        "jobs cancel",
        "setup",
        "welcome",
    }
)


def _iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _parse_sources(root: Path) -> list[tuple[Path, ast.Module]]:
    trees: list[tuple[Path, ast.Module]] = []
    for path in _iter_python_files(root):
        try:
            trees.append((path, ast.parse(path.read_text(encoding="utf-8"))))
        except (OSError, SyntaxError):
            continue
    return trees


def _string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _Forwarder:
    """A helper that emits on its caller's behalf: it takes a ``command``
    parameter and passes that same name straight through to
    ``.emit(command=command)``.

    The literal therefore lives at the *call site*, which is why a scan that
    only looks at ``.emit`` arguments sees an ``ast.Name`` and records nothing.
    """

    __slots__ = ("pos_index", "default")

    def __init__(self, pos_index: int | None, default: str | None) -> None:
        self.pos_index = pos_index
        self.default = default


def _own_body(fn: ast.AST):
    """Walk ``fn``'s own statements, not those of anything nested inside it.

    ``ast.walk`` would descend into nested functions, lambdas and classes, so an
    outer function whose inner closure happens to have its own ``command``
    parameter would be misread as a forwarder.
    """
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _rebinds_command(fn: ast.AST) -> bool:
    """True if ``fn`` assigns to the name ``command`` anywhere in its own body.

    A rebound parameter (``command = f"x {command}"``) means the value that
    reaches ``.emit`` is no longer the caller's literal, so the call site must
    not be trusted.
    """
    for node in _own_body(fn):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = [node.optional_vars] if node.optional_vars else []
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id == "command":
                    return True
    return False


def _module_key(path: Path) -> str:
    return str(path.resolve())


def _resolve_import(node: ast.ImportFrom, current: Path) -> Path | None:
    """Map a ``from X import ...`` onto a file inside the scanned tree."""
    if node.level:
        base = current.resolve().parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = node.module.split(".") if node.module else []
    else:
        if not node.module:
            return None
        parts = node.module.split(".")
        if not parts or parts[0] != SRC_ROOT.name:
            return None
        base = SRC_ROOT.parent
    target = base
    for part in parts:
        target = target / part
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.exists():
            return candidate.resolve()
    return None


def _imported_names(tree: ast.Module, path: Path) -> dict[str, str]:
    """``local name -> defining module key`` for ``from ... import name`` forms."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        source = _resolve_import(node, path)
        if source is None:
            continue
        for alias in node.names:
            if alias.name != "*":
                names[alias.asname or alias.name] = _module_key(source)
    return names


def _collect_command_forwarders(trees) -> dict[tuple[str, str], _Forwarder]:
    """Pass 1: find functions that forward a ``command`` parameter into ``.emit``.

    Keyed by ``(defining module, function name)``. Qualifying the key is what
    lets pass 2 refuse a same-named function from an unrelated module instead of
    attributing its call sites to this forwarder.
    """
    forwarders: dict[tuple[str, str], _Forwarder] = {}
    for path, tree in trees:
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = fn.args
            positional = [a.arg for a in (*args.posonlyargs, *args.args)]
            keyword_only = [a.arg for a in args.kwonlyargs]
            if "command" not in positional and "command" not in keyword_only:
                continue
            forwards = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "emit"
                and any(
                    kw.arg == "command" and isinstance(kw.value, ast.Name) and kw.value.id == "command"
                    for kw in node.keywords
                )
                for node in _own_body(fn)
            )
            if not forwards or _rebinds_command(fn):
                continue
            pos_index = positional.index("command") if "command" in positional else None
            default: str | None = None
            if pos_index is not None:
                # Defaults right-align onto the positional parameters.
                offset = pos_index - (len(positional) - len(args.defaults))
                if offset >= 0:
                    default = _string_literal(args.defaults[offset])
            else:
                default = _string_literal(args.kw_defaults[keyword_only.index("command")])
            forwarders[(_module_key(path), fn.name)] = _Forwarder(pos_index, default)
    return forwarders


def _scan_command_literals(trees) -> dict[str, list[Path]]:
    """Record the ``command="X"`` string literals this scan can prove reach ``.emit``.

    Exactly two shapes are recognised, and recorded identically:

    1. a literal passed directly to ``.emit(command="X")``;
    2. a literal passed to a *one-hop* forwarder — a function that takes a
       ``command`` parameter and passes that name through unchanged.

    Deliberately NOT covered, so no caller mistakes this for a total scan:
    multi-hop chains, and any hop that transforms the value. For example
    ``deploy._run_move(..., "promote")`` → ``_render_move(command=command)`` →
    ``.emit(command=f"deploy {command}")`` builds its name with an f-string, so
    ``deploy promote``/``deploy rollback`` remain unchecked by the ratchet.
    Non-literal arguments are skipped rather than guessed at.

    Purely static — nothing here imports or executes ``comfy_cli``.
    """
    forwarders = _collect_command_forwarders(trees)
    found: dict[str, list[Path]] = {}
    for path, tree in trees:
        module = _module_key(path)
        imported = _imported_names(tree, path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "emit":
                for kw in node.keywords:
                    if kw.arg == "command" and (literal := _string_literal(kw.value)) is not None:
                        found.setdefault(literal, []).append(path)
                continue
            # Followed only through a bare-name call that resolves to a forwarder
            # defined in this module or explicitly imported into it. An unrelated
            # `obj.emit_selected(...)`, or a same-named `_finish` from another
            # module, is therefore never attributed to a forwarder.
            if not isinstance(func, ast.Name):
                continue
            forwarder = forwarders.get((module, func.id))
            if forwarder is None and func.id in imported:
                forwarder = forwarders.get((imported[func.id], func.id))
            if forwarder is None:
                continue
            explicit = [kw for kw in node.keywords if kw.arg == "command"]
            if explicit:
                if (literal := _string_literal(explicit[0].value)) is not None:
                    found.setdefault(literal, []).append(path)
                continue
            # A positional `*args` splat only obscures the binding if it sits at
            # or before `command`'s index; it can never bind a keyword-only one.
            first_starred = next((i for i, a in enumerate(node.args) if isinstance(a, ast.Starred)), None)
            if forwarder.pos_index is not None:
                shadowed = first_starred is not None and first_starred <= forwarder.pos_index
                if not shadowed and len(node.args) > forwarder.pos_index:
                    if (literal := _string_literal(node.args[forwarder.pos_index])) is not None:
                        found.setdefault(literal, []).append(path)
                    continue
                if shadowed:
                    continue
            # No `command` argument at all: the signature default is what gets
            # emitted — unless a `**kwargs` splat could be supplying one.
            splatted = any(kw.arg is None for kw in node.keywords)
            if forwarder.default is not None and not splatted:
                found.setdefault(forwarder.default, []).append(path)
    return found


def _collect_emitted_commands() -> dict[str, list[Path]]:
    """AST-scan comfy_cli for the ``command="X"`` literals reaching ``.emit``
    directly or through a one-hop forwarder (see ``_scan_command_literals``)."""
    return _scan_command_literals(_parse_sources(SRC_ROOT))


@pytest.fixture(scope="module")
def emitted_commands() -> dict[str, list[Path]]:
    return _collect_emitted_commands()


def _scan_snippet(source: str) -> dict[str, list[Path]]:
    return _scan_command_literals([(Path("snippet.py"), ast.parse(source))])


# --- positive controls for the collector itself -----------------------------
# The ratchet above was blind for as long as it existed: every structured
# workflow edit passes its command name to a local `_finish` helper, so the
# `.emit(command=...)` argument is an `ast.Name` and the literal sits one frame
# up at the call site. Five such forwarders across four files hid 11 command
# literals; two were unregistered (`workflow clear`, `workflow reset-doc`) and
# one was registered under the WRONG name (`delete-node` emitting
# `"workflow delete"`). These tests watch the collector find a forwarded
# literal, so the guard is not trusted on the strength of never having fired.


def test_collector_finds_a_literal_forwarded_through_a_helper():
    found = _scan_snippet(
        """
def _finish(renderer, payload, command: str) -> None:
    renderer.emit(payload, command=command)

def cmd():
    _finish(renderer, {}, "widget poke")
"""
    )
    assert "widget poke" in found


def test_collector_finds_a_forwarded_literal_passed_by_keyword():
    found = _scan_snippet(
        """
def _emit_terminal(renderer, payload, *, command: str) -> None:
    renderer.emit(payload, command=command)

def cmd():
    _emit_terminal(renderer, {}, command="widget poke")
"""
    )
    assert "widget poke" in found


def test_collector_uses_a_forwarders_literal_default_when_the_call_omits_it():
    found = _scan_snippet(
        """
def shared(workflow, *, command: str = "widget poke") -> None:
    renderer.emit(workflow, command=command)

def cmd():
    shared(workflow)
"""
    )
    assert "widget poke" in found


def test_collector_ignores_a_non_literal_forwarded_argument():
    """Conservative by design: an unresolvable argument is skipped, not guessed."""
    found = _scan_snippet(
        """
def _finish(renderer, payload, command: str) -> None:
    renderer.emit(payload, command=command)

def cmd(name):
    _finish(renderer, {}, name)
"""
    )
    assert found == {}


def test_collector_ignores_a_helper_that_does_not_forward_into_emit():
    found = _scan_snippet(
        """
def log_it(renderer, command: str) -> None:
    renderer.debug(command)

def cmd():
    log_it(renderer, "widget poke")
"""
    )
    assert found == {}


def test_collector_ignores_a_command_rebound_before_the_emit():
    """A transformed value is not the caller's literal, so don't record it."""
    found = _scan_snippet(
        """
def helper(renderer, payload, command: str) -> None:
    command = f"deploy {command}"
    renderer.emit(payload, command=command)

def cmd():
    helper(renderer, {}, "promote")
"""
    )
    assert found == {}


def test_collector_ignores_an_outer_function_whose_nested_closure_forwards():
    """`ast.walk` would descend into the closure and misread the outer signature."""
    found = _scan_snippet(
        """
def outer(renderer, command: str) -> None:
    def inner(payload, command: str) -> None:
        renderer.emit(payload, command=command)

    register(inner)

def cmd():
    outer(renderer, "widget poke")
"""
    )
    assert found == {}


def test_collector_keeps_a_keyword_only_default_through_a_positional_splat():
    """A positional splat cannot bind a keyword-only parameter, so the default holds."""
    found = _scan_snippet(
        """
def helper(*args, command: str = "widget poke") -> None:
    renderer.emit(args, command=command)

def cmd(values):
    helper(*values)
"""
    )
    assert found == {"widget poke": [Path("snippet.py")]}


def test_collector_keeps_a_literal_when_a_splat_follows_the_command_argument():
    found = _scan_snippet(
        """
def helper(renderer, command: str, *rest) -> None:
    renderer.emit(rest, command=command)

def cmd(extra):
    helper(renderer, "widget poke", *extra)
"""
    )
    assert "widget poke" in found


def test_collector_skips_a_splat_that_could_shadow_the_command_position():
    found = _scan_snippet(
        """
def helper(renderer, command: str = "widget poke") -> None:
    renderer.emit({}, command=command)

def cmd(values):
    helper(*values)
"""
    )
    assert found == {}


def test_collector_does_not_attribute_a_same_named_function_from_another_module():
    """Forwarders are keyed by defining module, so `tracking._finish`-style
    name collisions in an unrelated module are not followed."""
    forwarder = """
def _finish(renderer, payload, command: str) -> None:
    renderer.emit(payload, command=command)
"""
    unrelated = """
def _finish(run):
    return run.close()

def cmd():
    _finish("not a command")
"""
    found = _scan_command_literals(
        [
            (Path("emitter.py"), ast.parse(forwarder)),
            (Path("unrelated.py"), ast.parse(unrelated)),
        ]
    )
    assert found == {}


# Every forwarder the scan must keep resolving, as ``(module, function)``.
# Only `_finish`/`_emit_op` have forwarder-only commands to ratchet on, so
# without this the other three could stop resolving with nothing failing.
EXPECTED_FORWARDERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("comfy_cli/command/workflow_edit.py", "_finish"),
        ("comfy_cli/command/workflow_edit.py", "_emit_op"),
        ("comfy_cli/command/jobs.py", "_emit_terminal"),
        ("comfy_cli/selector.py", "emit_selected"),
        ("comfy_cli/command/workflow.py", "validate_api_workflow"),
    }
)


def test_every_known_command_forwarder_still_resolves():
    """Ratchet against silent coverage loss across all five forwarders."""
    detected = {
        (Path(module).relative_to(SRC_ROOT.parent).as_posix(), name)
        for module, name in _collect_command_forwarders(_parse_sources(SRC_ROOT))
    }
    missing = sorted(EXPECTED_FORWARDERS - detected)
    assert not missing, (
        f"command forwarders no longer detected: {missing}\n"
        "Their call-site literals are now invisible to the registration ratchet."
    )


def test_workflow_edit_commands_are_reachable_only_through_the_forwarder(emitted_commands):
    """Ratchet keeping the forwarder pass wired to the real source tree.

    Every one of these is emitted exclusively via ``workflow_edit._finish`` /
    ``_emit_op`` or ``jobs._emit_terminal``, so a regression that stops
    following forwarders drops them all and this fails before the registration
    check can go quiet again.
    """
    forwarded_only = [
        "jobs watch",
        "workflow add-node",
        "workflow clear",
        "workflow connect",
        "workflow delete-node",
        "workflow insert-workflow",
        "workflow reset-doc",
        "workflow set-node-field",
        "workflow set-widget",
    ]
    missing = sorted(cmd for cmd in forwarded_only if cmd not in emitted_commands)
    assert not missing, f"collector stopped following command literals through forwarding helpers: {missing}"


def test_every_emitted_command_registers_a_schema(emitted_commands):
    """If this fails: you added a `renderer.emit(command="X")` call without
    registering X in COMMAND_SCHEMAS. Register it (comfy_cli/discovery.py),
    don't grow LEGACY_UNREGISTERED."""
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert emitted_commands, "AST scan found no emit(command=...) call sites — scanner broken?"
    unregistered = {
        cmd: [str(p.relative_to(SRC_ROOT.parent)) for p in paths]
        for cmd, paths in sorted(emitted_commands.items())
        if f"comfy {cmd}" not in COMMAND_SCHEMAS and cmd not in LEGACY_UNREGISTERED
    }
    assert not unregistered, (
        f"Commands emitting envelopes without a registered schema:\n{unregistered}\n"
        "Add each to COMMAND_SCHEMAS in comfy_cli/discovery.py."
    )


def test_legacy_unregistered_only_shrinks(emitted_commands):
    """Every allowlisted command must still exist in source. If this fails the
    command was removed or registered — delete it from LEGACY_UNREGISTERED so
    the allowlist ratchets down."""
    from comfy_cli.discovery import COMMAND_SCHEMAS

    stale = sorted(
        cmd for cmd in LEGACY_UNREGISTERED if cmd not in emitted_commands or f"comfy {cmd}" in COMMAND_SCHEMAS
    )
    assert not stale, f"Stale LEGACY_UNREGISTERED entries (no longer emitted, or now registered): {stale}"


def test_discover_pretty_mode_shows_counts():
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "--no-json", "discover"],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    # Panel-rendered pretty output: section headers + counts.
    assert "Commands" in result.stdout
    assert "Schemas" in result.stdout
    assert "Capabilities" in result.stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def test_every_registered_schema_names_a_shipped_file():
    """Every ``COMMAND_SCHEMAS`` / ``STREAM_EVENT_SCHEMAS`` value must name a
    file in comfy_cli/schemas/; a schema-file rename that leaves a stale
    mapping would make every envelope for that command unvalidatable."""
    from comfy_cli.discovery import COMMAND_SCHEMAS, STREAM_EVENT_SCHEMAS

    missing = sorted(
        name
        for name in {*COMMAND_SCHEMAS.values(), *STREAM_EVENT_SCHEMAS.values()}
        if not (SCHEMAS_DIR / f"{name}.json").is_file()
    )
    assert not missing, f"registered schemas with no shipped file: {missing}"
