"""Validate that envelope / error / help schemas are well-formed and that
real output from migrated commands validates against the appropriate schema.

This is the regression gate the plan calls out: every --json-capable command
ships a schema and its output must pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / name).read_text())


def _validator_for(name: str) -> jsonschema.Validator:
    schema = _load_schema(name)
    # Build a local store keyed by $id AND by filename so refs like
    # "error.json" resolve without hitting the network.
    store = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        sid = s.get("$id")
        if sid:
            store[sid] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


@pytest.mark.parametrize(
    "schema_name",
    [
        "envelope.json",
        "error.json",
        "help.json",
        "env.json",
        "which.json",
        "run.json",
        "run_event.json",
    ],
)
def test_schemas_are_well_formed(schema_name):
    schema = _load_schema(schema_name)
    # Will raise if the schema itself is invalid.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_envelope_schema_declares_contract_version():
    """envelope.json pins the discriminator + version the renderer emits."""
    from comfy_cli.output.renderer import ENVELOPE_SCHEMA

    props = _load_schema("envelope.json")["properties"]
    assert props["schema"]["const"] == ENVELOPE_SCHEMA == "envelope/1"
    assert props["type"]["const"] == "envelope"


def test_run_event_schema_declares_contract_version():
    from comfy_cli.output.renderer import EVENT_SCHEMA

    props = _load_schema("run_event.json")["properties"]
    assert props["schema"]["const"] == EVENT_SCHEMA == "event/1"


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
    assert result.stdout.strip(), f"expected JSON on stdout, got nothing.\nstderr: {result.stderr}"
    # Last non-empty line is the envelope (handles --json-stream).
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


def test_env_json_validates():
    envelope = _run_cli(["--json", "env"])
    env_validator = _validator_for("envelope.json")
    env_validator.validate(envelope)
    # And the data sub-document validates against env.json.
    data_validator = _validator_for("env.json")
    data_validator.validate(envelope["data"])
    assert envelope["command"] == "env"
    assert envelope["ok"] is True


def test_which_json_validates():
    envelope = _run_cli(["--json", "which"])
    _validator_for("envelope.json").validate(envelope)
    _validator_for("which.json").validate(envelope["data"])


def test_help_json_validates():
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "--help-json"],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    doc = json.loads(result.stdout)
    # --help-json now wraps in the standard envelope when running in JSON mode
    # (which is the default for non-TTY subprocesses). The bare help doc lives
    # under `data` — validate both layers so we lock the new contract.
    if isinstance(doc, dict) and {"ok", "command", "data"} <= doc.keys():
        _validator_for("envelope.json").validate(doc)
        _validator_for("help.json").validate(doc["data"])
    else:
        # Pretty-mode emits the bare doc directly to stdout.
        _validator_for("help.json").validate(doc)


def test_non_tty_auto_selects_json():
    """A subprocess (no TTY) defaults to JSON without --json being passed.

    This is the agent-out-of-the-box case: Claude Code / Cursor / etc. shell
    out and read stdout; they never see a TTY, so the renderer flips to
    JSON without the agent having to opt in.
    """
    envelope = _run_cli(["which"])
    _validator_for("envelope.json").validate(envelope)


def test_no_json_forces_pretty():
    """When stdout is not a TTY but --no-json is set, we should NOT get JSON."""
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "--no-json", "which"],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    # Panel-rendered pretty output: workspace section header is enough to
    # confirm we're not emitting JSON.
    assert "workspace" in result.stdout.lower()
    # And it must not be valid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def _bare_comfy(extra_args=None, drop_agent_markers=True):
    proc_env = os.environ.copy()
    proc_env.setdefault("NO_COLOR", "1")
    if drop_agent_markers:
        for k in ("CLAUDECODE", "AI_AGENT", "COMFY_USER_AGENT", "COMFY_OUTPUT"):
            proc_env.pop(k, None)
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *(extra_args or [])],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )


def test_bare_comfy_pipe_shows_human_banner_not_json():
    """A human piping bare `comfy` (non-TTY, no agent markers) gets the human
    welcome banner — not a useless JSON welcome envelope. Auto-JSON-on-non-TTY is
    for real commands/agents, not the welcome screen."""
    result = _bare_comfy()
    assert "comfy setup" in result.stdout, result.stdout[:300]
    # The banner is not a JSON envelope.
    try:
        json.loads(result.stdout)
        parsed = True
    except json.JSONDecodeError:
        parsed = False
    assert not parsed, f"expected the pretty banner, got JSON: {result.stdout[:200]}"


def test_bare_comfy_explicit_json_still_emits_welcome_envelope():
    """`--json` (explicit) keeps the machine welcome envelope."""
    result = _bare_comfy(["--json"])
    doc = json.loads([ln for ln in result.stdout.splitlines() if ln.strip()][-1])
    assert doc["command"] == "welcome"
    assert doc["ok"] is True


def test_bare_comfy_real_agent_keeps_json_welcome():
    """A real detected agent (CLAUDECODE) still gets the machine welcome envelope —
    only the non-TTY *pipe* case is overridden to the human banner, so the agent
    contract (JSON on stdout) is preserved."""
    proc_env = os.environ.copy()
    for k in ("AI_AGENT", "COMFY_USER_AGENT", "COMFY_OUTPUT"):
        proc_env.pop(k, None)
    proc_env["CLAUDECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli"],
        capture_output=True,
        text=True,
        env=proc_env,
        check=False,
    )
    doc = json.loads([ln for ln in result.stdout.splitlines() if ln.strip()][-1])
    assert doc["command"] == "welcome"
    assert doc["ok"] is True
