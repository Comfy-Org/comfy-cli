"""The ONE ``--select`` projection grammar over envelope ``data`` payloads.

This module is the single selector implementation for the CLI (V1-011 / C4):
the four heaviest read commands (``templates ls``, ``nodes show``,
``workflow slots``, ``generate list``) accept ``--select <expr>`` and project
their JSON payload through it. No second dialect will ever be added — keep the
grammar exactly this small.

Grammar (gjson-style dot paths):

  - **dot path** — ``a.b.c`` walks object keys.
  - **array index** — ``a.0.b`` indexes into an array (non-negative decimal).
    On an object, a digit segment is an ordinary key lookup.
  - **array wildcard** — ``items.#.name`` maps the rest of the path over every
    array element and returns the array of matches; per-element misses are
    dropped. ``items.#`` alone returns the whole array. A wildcard whose
    remainder matches zero elements of a non-empty array is a miss; over an
    empty array it matches and returns ``[]``. Wildcards compose
    (``rows.#.tags.#``).
  - **multi-select** — ``name,inputs`` splits on commas and returns an object
    keyed by each sub-expression that matched. It is a miss only when every
    part misses.

There is no escaping: keys containing ``.``, ``,`` or ``#`` cannot be
addressed. Malformed expressions (empty, empty segment, empty part) are
reported as a miss, never an error — the CLI fails open (see
``selected_payload``): the command still succeeds and returns a bounded key
inventory of the full payload plus a ``select_no_match`` advisory so the
caller can correct the expression from what it just learned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# Hard bound on the serialized fail-open inventory (~1-2KB per V1-011).
_INVENTORY_MAX_BYTES = 2048
# (top-level key cap, nested key cap) attempts, largest first; the first
# rendering that fits under the byte bound wins.
_INVENTORY_CAPS = ((40, 16), (16, 6), (6, 0))

WILDCARD = "#"


def select(data: Any, expr: str) -> tuple[Any, bool]:
    """Evaluate ``expr`` against ``data``. Pure; never raises on bad input.

    Returns ``(result, matched)``. ``matched`` is False for both a malformed
    expression and a well-formed one that matched nothing — the caller's
    fail-open path treats them identically.
    """
    if not isinstance(expr, str) or not expr.strip():
        return None, False
    parts = [p.strip() for p in expr.split(",")]
    if len(parts) > 1:
        out: dict[str, Any] = {}
        for part in parts:
            result, matched = _select_one(data, part)
            if matched:
                out[part] = result
        if out:
            return out, True
        return None, False
    return _select_one(data, parts[0])


def _select_one(data: Any, path: str) -> tuple[Any, bool]:
    if not path:
        return None, False
    segments = path.split(".")
    if any(seg == "" for seg in segments):
        return None, False
    return _walk(data, segments)


def _walk(current: Any, segments: list[str]) -> tuple[Any, bool]:
    if not segments:
        return current, True
    seg, rest = segments[0], segments[1:]
    if seg == WILDCARD:
        if not isinstance(current, list):
            return None, False
        if not rest:
            return list(current), True
        out = []
        for element in current:
            result, matched = _walk(element, rest)
            if matched:
                out.append(result)
        if out or not current:
            return out, True
        return None, False
    if isinstance(current, Mapping):
        if seg in current:
            return _walk(current[seg], rest)
        return None, False
    if isinstance(current, list):
        if seg.isdigit():
            index = int(seg)
            if index < len(current):
                return _walk(current[index], rest)
        return None, False
    return None, False


# ---------------------------------------------------------------------------
# Fail-open inventory
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _capped_keys(mapping: Mapping, cap: int) -> list[str]:
    keys = [str(k) for k in mapping]
    if cap and len(keys) > cap:
        return keys[:cap] + [f"…+{len(keys) - cap} more"]
    return keys if cap else []


def _describe(value: Any, nested_cap: int) -> Any:
    """One level of shape for a top-level value: type, size, and (for
    objects / arrays-of-objects) one level of keys."""
    if isinstance(value, Mapping):
        desc: dict[str, Any] = {"type": "object", "size": len(value)}
        if nested_cap:
            desc["keys"] = _capped_keys(value, nested_cap)
        return desc
    if isinstance(value, list):
        desc = {"type": "array", "size": len(value)}
        if nested_cap and value and isinstance(value[0], Mapping):
            desc["item_keys"] = _capped_keys(value[0], nested_cap)
        return desc
    return {"type": _type_name(value)}


def _inventory(data: Any, top_cap: int, nested_cap: int) -> Any:
    if isinstance(data, Mapping):
        keys = list(data)
        inv: dict[str, Any] = {str(k): _describe(data[k], nested_cap) for k in keys[:top_cap]}
        if len(keys) > top_cap:
            inv["…"] = f"+{len(keys) - top_cap} more keys"
        return inv
    return _describe(data, nested_cap)


def key_inventory(data: Any) -> Any:
    """A bounded (~2KB serialized) shape summary of ``data``: top-level keys,
    value types, sizes for objects/arrays, one nested level of keys."""
    inv: Any = None
    for caps in _INVENTORY_CAPS:
        inv = _inventory(data, *caps)
        if len(_dumps(inv).encode("utf-8")) <= _INVENTORY_MAX_BYTES:
            return inv
    return inv


# ---------------------------------------------------------------------------
# Shared emit path for the four --select commands
# ---------------------------------------------------------------------------


def _dumps(obj: Any) -> str:
    # Same serialization convention as the envelope writer (renderer
    # _write_json_line): compact-ish, non-ASCII passthrough, best-effort
    # coercion for stray non-JSON types.
    from comfy_cli.output.renderer import _json_default

    return json.dumps(obj, default=_json_default, ensure_ascii=False)


def _num_bytes(obj: Any) -> int:
    return len(_dumps(obj).encode("utf-8"))


def selected_payload(payload: Any, expr: str) -> tuple[Any, bool, dict[str, int]]:
    """Apply ``expr`` to a command's full ``data`` payload.

    Returns ``(data, matched, meta)`` where ``data`` is what the envelope
    should carry (the selected slice, or — fail-open — the key inventory plus
    a ``select_no_match`` advisory in ``warnings``), and ``meta`` holds the
    envelope's sibling byte counts: ``selected_bytes`` (serialized emitted
    slice) and ``total_bytes`` (serialized full payload).
    """
    result, matched = select(payload, expr)
    if matched:
        data: Any = result
    else:
        from comfy_cli import error_codes

        registered = error_codes.get("select_no_match")
        data = {
            "inventory": key_inventory(payload),
            "warnings": [
                {
                    "code": "select_no_match",
                    "message": f"--select {expr!r} matched nothing in the payload",
                    "hint": registered.hint if registered else None,
                }
            ],
        }
    meta = {"selected_bytes": _num_bytes(data), "total_bytes": _num_bytes(payload)}
    return data, matched, meta


def emit_selected(renderer: Any, payload: Any, expr: str, *, command: str) -> None:
    """Render/emit a command payload through ``--select``.

    JSON modes: one envelope whose ``data`` is the selected slice and which
    carries sibling ``selected_bytes`` / ``total_bytes`` fields. Pretty mode:
    the selected slice pretty-printed as JSON (bare strings printed plain), or
    — fail-open — a yellow advisory plus the key inventory. Exit code is the
    caller's (always 0): a miss is never an error.
    """
    data, matched, meta = selected_payload(payload, expr)
    if renderer.is_pretty():
        if matched:
            if isinstance(data, str):
                # A selected bare string is almost always feeding a shell /
                # human eyeball; don't wrap it in JSON quotes. markup=False so
                # payload text can't be interpreted as Rich tags.
                renderer.console().print(data, markup=False)
            else:
                renderer.console().print_json(_dumps(data))
        else:
            warning = data["warnings"][0]
            renderer.warn(warning["message"], hint=warning["hint"])
            renderer.console().print_json(_dumps(data["inventory"]))
        return
    renderer.emit(data, command=command, extra=meta)
