"""Build a CQL-shaped graph dict from sources.

Sources, in priority order:

1. A local file (``--input path``). May be:
   - A raw ``object_info`` JSON dump (the response from ``/object_info``).
   - An API-format workflow JSON.
   - An already-shaped CQL graph (``{"nodes": [...], "inputs": [...]}``).
2. A local ComfyUI server's ``/object_info`` endpoint (``--host`` / ``--port``).

The loader is intentionally permissive: anything dict-shaped that looks like
one of those formats is normalized into ``{"nodes": [...], "inputs": [...],
"categories": [...]}`` so the engine can run uniformly.

This module performs only local I/O. Network calls hit ``http://host:port``
and are short-circuited when no host is provided.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from comfy_cli.cql.errors import CQLRuntimeError

# Cap raw bytes read from disk or the network. Real `object_info` dumps are a
# few MB; anything past 256 MiB is almost certainly a wrong path or a hostile
# server and would just OOM the CLI before json.loads even fails.
MAX_INPUT_BYTES = 256 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects — a 302 from /object_info to elsewhere is suspicious
    and would expose us to a different server than the user asked to query."""

    def http_error_301(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_303 = http_error_307 = http_error_308 = http_error_301


_LOADER_OPENER = urllib.request.build_opener(_NoRedirect())


def load_graph(
    *,
    input_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    if input_path:
        return _load_from_file(input_path)
    if host and port:
        return _load_from_server(host, int(port), timeout=timeout)
    raise CQLRuntimeError(
        "no graph source available",
        details={"hint": "pass --input <path> or --host/--port pointing at a ComfyUI server"},
    )


def _load_from_file(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        size = p.stat().st_size
    except OSError as e:
        raise CQLRuntimeError(f"cannot stat {p}: {e}") from e
    if size > MAX_INPUT_BYTES:
        raise CQLRuntimeError(
            f"{p} is {size} bytes, exceeds MAX_INPUT_BYTES={MAX_INPUT_BYTES}",
            details={"hint": "shrink the input or raise MAX_INPUT_BYTES in cql.loader"},
        )
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise CQLRuntimeError(f"cannot read {p}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CQLRuntimeError(f"{p} is not valid JSON: {e}") from e
    return normalize(data)


def _load_from_server(host: str, port: int, *, timeout: float) -> dict[str, Any]:
    url = f"http://{host}:{port}/object_info"
    # Refuse anything that isn't a localhost-ish target — we don't want CQL
    # silently sending traffic to a remote box. (Cloud CQL goes through its
    # own path; this loader is local-only by design.)
    parsed = urllib.parse.urlsplit(url)
    if (parsed.hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("127."):
        raise CQLRuntimeError(
            f"refusing non-loopback CQL server target: {host}",
            details={"hint": "pass --input <path> for remote object_info dumps"},
        )
    try:
        with _LOADER_OPENER.open(url, timeout=timeout) as resp:
            # Bounded read so a misbehaving server can't OOM us.
            raw = resp.read(MAX_INPUT_BYTES + 1)
            if len(raw) > MAX_INPUT_BYTES:
                raise CQLRuntimeError(
                    f"server response exceeds MAX_INPUT_BYTES={MAX_INPUT_BYTES}",
                    details={"host": host, "port": port},
                )
            data = json.loads(raw)
    except urllib.error.URLError as e:
        raise CQLRuntimeError(
            f"failed to reach {url}: {e.reason if hasattr(e, 'reason') else e}",
            details={"host": host, "port": port},
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise CQLRuntimeError(f"server returned invalid object_info: {e}") from e
    return normalize(data)


# ---- normalization --------------------------------------------------------


def normalize(data: Any) -> dict[str, Any]:
    """Turn any supported input into ``{nodes, inputs, categories}``."""
    if not isinstance(data, dict):
        raise CQLRuntimeError("expected a JSON object at the top level")

    # Already CQL-shaped — trust it.
    if any(isinstance(data.get(k), list) for k in ("nodes", "inputs", "categories")):
        graph: dict[str, Any] = {
            "nodes": list(data.get("nodes") or []),
            "inputs": list(data.get("inputs") or []),
            "categories": list(data.get("categories") or []),
        }
        return graph

    if _looks_like_object_info(data):
        return _from_object_info(data)
    if _looks_like_api_workflow(data):
        return _from_api_workflow(data)

    raise CQLRuntimeError(
        "unrecognized graph shape",
        details={"keys_sample": sorted(list(data.keys()))[:10]},
    )


def _looks_like_object_info(data: dict[str, Any]) -> bool:
    # /object_info maps "ClassName" -> { "input": {...}, "category": "...",
    # "display_name": "...", "description": "...", "output": [...], ... }
    if not data:
        return False
    for v in data.values():
        if isinstance(v, dict) and ("input" in v or "category" in v):
            return True
        break
    return False


def _looks_like_api_workflow(data: dict[str, Any]) -> bool:
    if not data:
        return False
    first = next(iter(data.values()))
    return isinstance(first, dict) and "class_type" in first


def _from_object_info(data: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    categories: dict[str, int] = {}

    for class_name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        category = raw.get("category")
        node = {
            "name": class_name,
            "display_name": raw.get("display_name") or class_name,
            "category": category,
            "description": raw.get("description"),
            "output_node": bool(raw.get("output_node", False)),
            "output_types": list(raw.get("output") or []),
        }
        nodes.append(node)
        if category:
            categories[category] = categories.get(category, 0) + 1

        sections = raw.get("input") or {}
        if isinstance(sections, dict):
            for section, body in sections.items():  # "required" / "optional" / "hidden"
                if not isinstance(body, dict):
                    continue
                for input_name, spec in body.items():
                    inputs.append(_normalize_input(class_name, section, input_name, spec))

    return {
        "nodes": nodes,
        "inputs": inputs,
        "categories": [{"name": k, "node_count": v} for k, v in sorted(categories.items())],
    }


def _normalize_input(class_name: str, section: str, name: str, spec: Any) -> dict[str, Any]:
    type_name: Any = None
    options: dict[str, Any] = {}
    choices: list[Any] = []
    if isinstance(spec, list) and spec:
        type_name = spec[0]
        if isinstance(type_name, list):
            choices = list(type_name)
            type_name = "ENUM"
        if len(spec) > 1 and isinstance(spec[1], dict):
            options = dict(spec[1])
    elif isinstance(spec, str):
        type_name = spec
    return {
        "node": class_name,
        "section": section,
        "name": name,
        "type": type_name,
        "choices": choices,
        "options": options,
    }


def _from_api_workflow(data: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for nid, node in data.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        title = (node.get("_meta") or {}).get("title") if isinstance(node.get("_meta"), dict) else None
        nodes.append(
            {
                "id": nid,
                "name": class_type or "?",
                "class_type": class_type,
                "title": title,
                "category": None,
            }
        )
        raw_inputs = node.get("inputs") or {}
        if isinstance(raw_inputs, dict):
            for in_name, value in raw_inputs.items():
                ref = isinstance(value, list) and len(value) == 2
                inputs.append(
                    {
                        "node_id": nid,
                        "node": class_type,
                        "name": in_name,
                        "value": None if ref else value,
                        "ref_node": value[0] if ref else None,
                        "ref_slot": value[1] if ref else None,
                        "is_reference": ref,
                    }
                )
    return {"nodes": nodes, "inputs": inputs, "categories": []}
