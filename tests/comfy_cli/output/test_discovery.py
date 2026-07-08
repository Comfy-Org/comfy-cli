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


def test_discover_lists_run_prompt_and_set_options():
    # BE-2535: `comfy run --prompt`/`--set` must be visible on the agent
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


def _collect_emitted_commands() -> dict[str, list[Path]]:
    """AST-scan comfy_cli for ``.emit(..., command="X")`` literal call sites."""
    found: dict[str, list[Path]] = {}
    for path in _iter_python_files(SRC_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "emit"):
                continue
            for kw in node.keywords:
                if kw.arg == "command" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    found.setdefault(kw.value.value, []).append(path)
    return found


@pytest.fixture(scope="module")
def emitted_commands() -> dict[str, list[Path]]:
    return _collect_emitted_commands()


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
