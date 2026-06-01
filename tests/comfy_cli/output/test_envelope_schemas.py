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
