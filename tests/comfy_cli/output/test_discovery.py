"""Phase 2: ``comfy discover`` produces a complete self-describing document.

These are the contract tests: an agent should be able to call ``comfy
--json discover`` once and get everything it needs without consulting source.
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
