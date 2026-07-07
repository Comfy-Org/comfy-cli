"""Self-discovery for agents.

``comfy discover --json`` emits a single document describing the entire CLI
surface plus the JSON Schemas the CLI's structured outputs adhere to.

Why this exists: a fresh agent should be able to ``comfy discover`` once and
have everything it needs — command tree, schemas, error codes, capability
flags — without scraping ``--help`` text or reading source.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from comfy_cli.help_json import build_help_json
from comfy_cli.output.renderer import ENVELOPE_SCHEMA, EVENT_SCHEMA

# Maps fully-qualified command paths to the schema name (without .json) that
# the command's envelope ``data`` field validates against. Phase 2 covers the
# commands that already emit structured output. New commands should register
# here as they migrate.
COMMAND_SCHEMAS: dict[str, str] = {
    "comfy env": "env",
    "comfy which": "which",
    "comfy run": "run",
    "comfy discover": "discover",
    "comfy auth list": "auth",
    "comfy auth set": "auth",
    "comfy auth remove": "auth",
    "comfy cloud login": "auth",
    "comfy cloud logout": "auth",
    "comfy cloud whoami": "auth",
    "comfy jobs ls": "jobs",
    "comfy jobs status": "jobs",
    "comfy jobs watch": "jobs",
    "comfy jobs wait": "jobs_wait",
    # help / validation
    "comfy help": "help",
    "comfy validate": "workflow",
    # nodes introspection
    "comfy nodes ls": "nodes",
    "comfy nodes show": "nodes",
    "comfy nodes search": "nodes",
    "comfy nodes upstream": "nodes",
    "comfy nodes downstream": "nodes",
    "comfy nodes path": "nodes",
    "comfy nodes types": "nodes",
    "comfy nodes categories": "nodes",
    "comfy nodes refresh": "nodes",
    # workflow editing
    "comfy workflow slots": "workflow",
    "comfy workflow set-slot": "workflow",
    "comfy workflow vary": "workflow",
    # workflow cloud CRUD + fragment composition
    "comfy workflow list": "workflow",
    "comfy workflow get": "workflow",
    "comfy workflow save": "workflow",
    "comfy preview": "preview",
    "comfy workflow delete": "workflow",
    "comfy workflow compose": "workflow",
    "comfy workflow decompose": "workflow",
    "comfy workflow fragment ls": "workflow",
    "comfy workflow fragment show": "workflow",
    "comfy workflow fragment validate": "workflow",
    # skill management
    "comfy skills install": "skill",
    "comfy skills uninstall": "skill",
    "comfy skills show": "skill",
    "comfy skills status": "skill",
    "comfy skills validate": "skill",
    # `comfy skill` is the hidden singular alias; envelopes from the skills
    # group carry the singular form in `command`, so both spellings register.
    "comfy skill install": "skill",
    "comfy skill uninstall": "skill",
    "comfy skill list": "skill",
    "comfy skill show": "skill",
    "comfy skill status": "skill",
    # model discovery (all asset types: checkpoints, loras, controlnets, vae, ...)
    "comfy models search": "models",
    "comfy models show": "models",
    "comfy models list-folders": "models",
    "comfy models list-folder": "models",
    # template gallery
    "comfy templates ls": "templates",
    "comfy templates show": "templates",
    "comfy templates fetch": "templates",
    "comfy templates refresh": "templates",
    # file transfer
    "comfy upload": "transfer",
    "comfy download": "transfer",
    # project convention
    "comfy project init": "project",
    "comfy project status": "project",
    "comfy assets push": "assets",
    # config
    "comfy set-default": "set_default",
    "comfy version": "version",
    # background server logs
    "comfy logs": "logs",
}


# Streaming event schemas for commands that emit NDJSON
# (e.g. `comfy --json-stream jobs watch <id>`).


# Streaming commands additionally emit per-line events that validate against a
# different schema. Keyed by command path.
STREAM_EVENT_SCHEMAS: dict[str, str] = {
    "comfy run": "run_event",
    "comfy jobs watch": "run_event",
}


def _read_schema(name: str) -> dict[str, Any]:
    pkg = resources.files("comfy_cli.schemas")
    text = (pkg / f"{name}.json").read_text(encoding="utf-8")
    return json.loads(text)


def list_schema_names() -> list[str]:
    pkg = resources.files("comfy_cli.schemas")
    out: list[str] = []
    for child in pkg.iterdir():
        n = child.name
        if n.endswith(".json"):
            out.append(n[:-5])
    return sorted(out)


def load_all_schemas() -> dict[str, dict[str, Any]]:
    """Return ``{schema_name: {name, title, schema}}`` for every shipped schema."""
    bundle: dict[str, dict[str, Any]] = {}
    for name in list_schema_names():
        schema = _read_schema(name)
        bundle[name] = {
            "name": f"{name}.json",
            "title": schema.get("title", name),
            "schema": schema,
        }
    return bundle


def load_error_codes() -> list[dict[str, Any]]:
    """Return the error-code list for the discovery envelope.

    Sourced from the typed registry in :mod:`comfy_cli.error_codes`, which is
    the contract agents branch on.
    """
    from comfy_cli import error_codes

    return [dict(row) for row in error_codes.as_discover_rows()]


def build_discovery(app: Any, *, prog_name: str = "comfy", version: str = "") -> dict[str, Any]:
    """Build the discovery document.

    ``app`` is the root Typer app. Kept untyped to avoid importing typer at
    module import; the call site already has it.
    """
    help_doc = build_help_json(app, prog_name=prog_name)

    schemas = load_all_schemas()

    # Annotate each leaf command with its schema (when known) so agents don't
    # have to cross-reference the schemas map.
    _annotate_schemas(help_doc["commands"], path=[prog_name])

    capabilities = {
        "json_envelope": True,
        "json_stream": True,
        "cancellation": True,
        "cql": True,
        "where_routing": True,
        "where_targets": ["local", "cloud"],
        # Comfy Cloud uses OAuth (Authorization Code + PKCE); third-party
        # services that don't speak OAuth keep the API-key path.
        "cloud_oauth": True,
        "auth_providers": {
            "oauth": ["comfy-cloud"],
            "api_key": ["civitai", "huggingface"],
        },
    }

    return {
        "prog": prog_name,
        "version": version,
        # Versioned machine-output contract (see comfy_cli/output/renderer.py
        # for the constants and the bump rule). Agents negotiate shape on
        # these, not on the CLI `version` above.
        "output_contract": {"envelope": ENVELOPE_SCHEMA, "event": EVENT_SCHEMA},
        "commands": help_doc["commands"],
        "root": help_doc["root"],
        "schemas": schemas,
        "command_schemas": dict(COMMAND_SCHEMAS),
        "stream_event_schemas": dict(STREAM_EVENT_SCHEMAS),
        "error_codes": load_error_codes(),
        "capabilities": capabilities,
    }


def _annotate_schemas(commands: dict[str, Any], *, path: list[str]) -> None:
    for name, entry in commands.items():
        fqp = " ".join(path + [name])
        schema_name = COMMAND_SCHEMAS.get(fqp)
        if schema_name:
            entry["output_schema"] = f"{schema_name}.json"
        stream_schema = STREAM_EVENT_SCHEMAS.get(fqp)
        if stream_schema:
            entry["stream_event_schema"] = f"{stream_schema}.json"
        subs = entry.get("subcommands")
        if subs:
            _annotate_schemas(subs, path=path + [name])
