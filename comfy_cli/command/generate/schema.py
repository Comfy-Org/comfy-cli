"""Convert an openapi requestBody schema into CLI flag definitions, and parse
user-supplied argv against those flags.

This is the equivalent of fal-ai's `genmedia run <id> --param value` UX: the
schema for each endpoint drives which flags are valid, their types, and their
help text. We accept inline JSON for object/array params and treat fields with
``format: binary`` as file-path inputs that get streamed via multipart.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_cli.command.generate.spec import Endpoint


@dataclass
class FlagDef:
    name: str  # openapi property name; CLI flag = "--" + name
    kind: str  # "string" | "integer" | "number" | "boolean" | "enum" | "object" | "array" | "binary"
    required: bool
    description: str = ""
    default: Any = None
    enum: list[str] = field(default_factory=list)
    item_kind: str | None = None  # for arrays: kind of items ("binary", "string", ...)


class SchemaError(ValueError):
    pass


def _classify(prop: dict[str, Any]) -> tuple[str, str | None]:
    """Return (kind, item_kind). item_kind only set when kind == 'array'."""
    if "enum" in prop and prop.get("type", "string") == "string":
        return "enum", None
    t = prop.get("type")
    if t == "string" and prop.get("format") == "binary":
        return "binary", None
    if t == "string":
        return "string", None
    if t == "integer":
        return "integer", None
    if t == "number":
        return "number", None
    if t == "boolean":
        return "boolean", None
    if t == "array":
        items = prop.get("items") or {}
        if items.get("format") == "binary":
            return "array", "binary"
        return "array", items.get("type", "string")
    if t == "object" or "oneOf" in prop or "anyOf" in prop or "allOf" in prop:
        return "object", None
    # Fallback — treat as string.
    return "string", None


def flags_for(endpoint: Endpoint) -> list[FlagDef]:
    schema = endpoint.request_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: list[FlagDef] = []
    for name, prop in props.items():
        if not isinstance(prop, dict):
            continue
        kind, item_kind = _classify(prop)
        out.append(
            FlagDef(
                name=name,
                kind=kind,
                required=name in required,
                description=str(prop.get("description") or "").strip(),
                default=prop.get("default"),
                enum=list(prop.get("enum") or []),
                item_kind=item_kind,
            )
        )
    return out


def _coerce(flag: FlagDef, raw: str) -> Any:
    """Convert a string value from argv into its typed form, raising SchemaError
    with a clear message on failure."""
    if flag.kind == "string":
        return raw
    if flag.kind == "enum":
        if flag.enum and raw not in flag.enum:
            raise SchemaError(f"--{flag.name}: {raw!r} is not one of {flag.enum}")
        return raw
    if flag.kind == "integer":
        try:
            return int(raw)
        except ValueError as e:
            raise SchemaError(f"--{flag.name}: expected integer, got {raw!r}") from e
    if flag.kind == "number":
        try:
            return float(raw)
        except ValueError as e:
            raise SchemaError(f"--{flag.name}: expected number, got {raw!r}") from e
    if flag.kind == "boolean":
        v = raw.lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
        raise SchemaError(f"--{flag.name}: expected boolean (true/false), got {raw!r}")
    if flag.kind in ("object", "array"):
        # For arrays of binary, the raw value is a comma-separated list of paths
        # or a JSON array of paths.
        if flag.kind == "array" and flag.item_kind == "binary":
            try:
                parsed = json.loads(raw) if raw.startswith("[") else [p.strip() for p in raw.split(",")]
            except json.JSONDecodeError as e:
                raise SchemaError(f"--{flag.name}: invalid file list: {e}") from e
            return [Path(p).expanduser() for p in parsed]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise SchemaError(
                f"--{flag.name}: expected JSON {flag.kind}, got {raw!r}. "
                'Wrap the value in quotes, e.g. --{n} \'{{"k":"v"}}\''.format(n=flag.name)
            ) from e
    if flag.kind == "binary":
        return Path(raw).expanduser()
    raise SchemaError(f"--{flag.name}: unknown kind {flag.kind!r}")  # unreachable


def parse_args(flags: list[FlagDef], argv: list[str]) -> dict[str, Any]:
    """Parse ``argv`` against the given flag list. Returns {name: typed_value}.

    Recognized forms:
      --name value
      --name=value
      --name           (only for boolean flags; means True)
      --no-name        (only for boolean flags; means False)
    """
    by_name = {f.name: f for f in flags}
    # Also accept dash-separated aliases so `--no-X` matching can't collide with
    # underscored names.
    by_dashed = {f.name.replace("_", "-"): f for f in flags}

    values: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise SchemaError(f"Unexpected positional argument: {token!r}")
        body = token[2:]
        raw: str | None = None
        if "=" in body:
            body, raw = body.split("=", 1)
        flag = by_name.get(body) or by_dashed.get(body)
        if flag is None and body.startswith("no-"):
            candidate = body[3:]
            f = by_name.get(candidate) or by_dashed.get(candidate)
            if f and f.kind == "boolean":
                values[f.name] = False
                i += 1
                continue
        if flag is None:
            raise SchemaError(f"Unknown flag: {token!r}. Run `comfy generate schema <model>` to list params.")
        if flag.kind == "boolean" and raw is None:
            # Look ahead — only consume next token if it parses as boolean.
            if i + 1 < len(argv) and argv[i + 1].lower() in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raw = argv[i + 1]
                i += 1
            else:
                values[flag.name] = True
                i += 1
                continue
        if raw is None:
            if i + 1 >= len(argv):
                raise SchemaError(f"--{flag.name}: missing value")
            raw = argv[i + 1]
            i += 2
        else:
            i += 1
        values[flag.name] = _coerce(flag, raw)

    missing = [f.name for f in flags if f.required and f.name not in values]
    if missing:
        joined = ", ".join(f"--{m}" for m in missing)
        raise SchemaError(f"Missing required argument(s): {joined}")
    return values


def help_text(endpoint: Endpoint, flags: list[FlagDef]) -> str:
    """Produce a human-readable help block describing a model and its flags.
    The caller is expected to already print a one-line ``Model:`` header, so we
    skip restating the id here."""
    lines: list[str | None] = [
        f"  {endpoint.summary}" if endpoint.summary else None,
        f"  partner: {endpoint.partner}    style: {endpoint.category}    "
        f"content-type: {endpoint.request_content_type}    "
        f"mode: {'async (' + endpoint.polling + ')' if endpoint.polling else 'sync'}",
        "",
        "Parameters (use as `--name value`):",
    ]
    if not flags:
        lines.append("  (no parameters)")
    for f in flags:
        marker = "  *" if f.required else "   "
        type_str = f.kind
        if f.kind == "enum":
            type_str = "enum=" + "|".join(f.enum)
        if f.kind == "array" and f.item_kind:
            type_str = f"array<{f.item_kind}>"
        head = f"{marker} --{f.name} <{type_str}>"
        lines.append(head)
        if f.description:
            lines.append(f"      {f.description}")
        if f.default is not None:
            lines.append(f"      default: {f.default!r}")
    lines.append("")
    lines.append("Common options:")
    lines.append("  --download <path>  Save outputs locally. Supports {request_id}, {index}, {ext}.")
    lines.append("  --async            Submit and return job id without waiting.")
    lines.append("  --json             Emit raw JSON response instead of pretty output.")
    lines.append("  --timeout <sec>    Override sync-poll timeout (default 300).")
    lines.append("  --api-key <key>    Override COMFY_API_KEY env var.")
    return "\n".join(line for line in lines if line is not None)


def example_invocation(endpoint: Endpoint, flags: list[FlagDef], display_name: str | None = None) -> str:
    """A copy-paste invocation snippet showing required args."""
    parts = ["comfy generate", display_name or endpoint.id]
    for f in flags:
        if not f.required:
            continue
        if f.kind == "binary":
            parts.extend([f"--{f.name}", "./input.png"])
        elif f.kind == "enum":
            parts.extend([f"--{f.name}", f.enum[0] if f.enum else "VALUE"])
        elif f.kind == "string":
            parts.extend([f"--{f.name}", shlex.quote("...")])
        elif f.kind in ("object", "array"):
            parts.extend([f"--{f.name}", shlex.quote("{}")])
        else:
            parts.extend([f"--{f.name}", "0"])
    return " ".join(parts)
