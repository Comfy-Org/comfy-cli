"""Pre-submit checks: fetch the server's ``object_info`` and validate the
workflow before we burn a queue slot.

Three concerns live here: HTTP fetch with renderer-routed error reporting,
pure-Python workflow validation (unknown class_types, shape drift), and
partner-API node detection (used to inject the right credential into
``extra_data`` for local submits — cloud handles this server-side).
"""

from __future__ import annotations

import json
import urllib.error
from urllib import request

import typer

from comfy_cli.command.run.loader import _MAX_BODY_PREVIEW
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint

# Partner-API nodes live under `partner/...` in ComfyUI/cloud object_info
# (e.g. `partner/video/ByteDance`). The category prefix is only the fallback —
# the authoritative signal is the `api_node: true` flag.
PARTNER_NODE_CATEGORY_PREFIXES = ("partner/",)


def fetch_object_info(host, port, timeout):
    """GET ``/object_info`` from the running ComfyUI server.

    The response describes every loaded node class's input schema and is what
    the converter uses to map widget values to input names, fill defaults, etc.

    Failures go through ``renderer.error(...)`` (error envelope in JSON modes,
    red panel in pretty mode) and raise ``typer.Exit(code=1)``.
    """
    renderer = get_renderer()
    url = f"http://{host}:{port}/object_info"
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(64 * 1024 * 1024)
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace").strip()
        renderer.error(
            code="object_info_unavailable",
            message=f"Failed to fetch /object_info (HTTP {e.code})",
            hint="check the ComfyUI server logs; restart the server",
            details={"status": e.code, "body": body_text[:_MAX_BODY_PREVIEW]},
        )
        raise typer.Exit(code=1) from e
    except urllib.error.URLError as e:
        renderer.error(
            code="connection_error",
            message=f"Failed to fetch /object_info from {host}:{port}: {e.reason}",
            hint="override with --host / --port",
        )
        raise typer.Exit(code=1) from e
    except TimeoutError as e:
        renderer.error(
            code="connection_error",
            message=f"Failed to fetch /object_info from {host}:{port}: timed out after {timeout}s",
            hint="override with --host / --port, or raise --timeout",
        )
        raise typer.Exit(code=1) from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        renderer.error(
            code="object_info_unavailable",
            message="Server returned invalid JSON for /object_info",
            hint="check that the host:port really is a ComfyUI server",
            details={"status": 200, "body": body.decode("utf-8", errors="replace")[:_MAX_BODY_PREVIEW]},
        )
        raise typer.Exit(code=1) from e


def _preflight_validate(renderer, workflow: dict, object_info: dict, *, target_label: str = "server") -> None:
    """Pre-submit validation via the pure-Python CQL engine.

    Checks unknown class_types, input shape mismatches, and catalog drift.
    Raises typer.Exit(1) on hard errors. Prints warnings in pretty mode.
    Skips silently when object_info is empty (server unreachable — fail open).
    """
    if not object_info:
        return

    from comfy_cli.cql.engine import Graph

    graph = Graph.from_object_info(object_info)
    validation = graph.validate_workflow(workflow)
    if not validation.get("valid", True):
        errors = validation.get("errors", [])
        hint_parts = []
        for e in errors[:5]:
            line = f"node {e.get('node_id') or '?'}: {e.get('message', '')}"
            suggestions = e.get("suggestions") or []
            if suggestions:
                line += f" (did you mean: {', '.join(suggestions)}?)"
            hint_parts.append(line)
        renderer.error(
            code="workflow_unknown_nodes",
            message=f"Workflow has {len(errors)} validation error(s) against {target_label}",
            hint="\n".join(hint_parts),
            details={"errors": errors, "warnings": validation.get("warnings", [])},
        )
        raise typer.Exit(code=1)

    warnings = validation.get("warnings", [])
    if warnings and renderer.is_pretty():
        for w in warnings:
            pprint(f"[yellow]⚠ {w.get('field', '?')}: {w.get('message', '')}[/yellow]")


def _fetch_object_info(host: str, port: int) -> dict:
    """Fetch object_info for partner-node detection + validation. Fail open."""
    try:
        from comfy_cli.cql.engine import _load_from_target

        return _load_from_target(mode="local", host=host, port=port)
    except Exception:  # noqa: BLE001 — fail open
        return {}


def _detect_partner_nodes(workflow: dict, object_info: dict) -> list[str]:
    """Return sorted unique class_types in ``workflow`` that are partner-API
    nodes (Veo, Kling, BFL, Gemini, ByteDance, Bria, etc.). Pure function —
    tests pass object_info directly.

    Detection is primarily the authoritative ``api_node: true`` flag in
    object_info, with a ``partner/...`` category-prefix fallback for servers
    that don't surface the flag.

    A partner-API node call requires an ``api_key_comfy_org`` (or the
    OAuth-equivalent ``auth_token_comfy_org``) in ``extra_data`` at
    submit time. Without it the local server happily accepts the
    workflow at /prompt and then fails the actual node at execute time
    with ``Unauthorized: Please login first to use this node`` — a
    sharp edge observed during the Veo3 video run.
    """
    used: set[str] = set()
    for node in workflow.values():
        if isinstance(node, dict):
            ct = node.get("class_type")
            if isinstance(ct, str):
                used.add(ct)
    out: list[str] = []
    for ct in used:
        info = object_info.get(ct) or {}
        if not isinstance(info, dict):
            continue
        if info.get("api_node") is True:
            out.append(ct)
            continue
        category = info.get("category")
        if isinstance(category, str) and category.startswith(PARTNER_NODE_CATEGORY_PREFIXES):
            out.append(ct)
    return sorted(out)
