"""The ONE ``--select`` projection grammar over envelope ``data`` payloads.

This module is the single selector implementation for the CLI (V1-011 / C4):
the four heaviest read commands (``templates ls``, ``nodes show``,
``workflow slots``, ``generate list``) accept ``--select <expr>`` and project
their JSON payload through it. No second dialect will ever be added — the
grammar is gjson's path syntax, kept to the subset below.

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
  - **row query** (gjson's) — ``items.#(<cond>)#`` keeps the array elements
    that satisfy ``<cond>`` (the rest of the path maps over them, like ``#``);
    ``items.#(<cond>)`` is the FIRST such element (the rest of the path walks
    it). ``<cond>`` is ``<key> <op> <value>``, ``<op> <value>`` for an array of
    scalars, or a bare ``<key>`` (the element has it, non-null). ``<key>`` is
    a dot path inside the element; ``<op>`` is one of ``==`` ``!=`` ``<``
    ``<=`` ``>`` ``>=`` ``%`` (glob match, ``*`` and ``?``) ``!%``; ``<value>``
    is a ``"double-quoted"`` string (``\\"`` escapes a quote), a number,
    ``true``, ``false`` or ``null``. ``==``/``!=`` compare a number and a
    string by their text (``instance_id=="5"`` matches ``5``); the order
    operators compare two numbers or two strings, never a mix. Zero matching
    elements is an answer, not a miss: ``#(…)#`` returns ``[]``; ``#(…)`` with
    no match is a miss. Dots and commas inside a row query are its own
    (``#(value=="a.b,c")``).

Outside a row query there is no escaping: keys containing ``.``, ``,`` or
``#`` cannot be addressed. Malformed expressions (empty, empty segment, empty part) are
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
    parts = _split_top(expr, ",")
    if parts is None:
        return None, False
    parts = [p.strip() for p in parts]
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


def _split_top(text: str, sep: str) -> list[str] | None:
    """Split ``text`` on ``sep`` outside any ``#(...)`` row query and outside
    quoted strings in one. ``None`` for an unbalanced query or quote."""
    parts: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"' and depth:
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    if depth or in_str:
        return None
    parts.append(text[start:])
    return parts


def _select_one(data: Any, path: str) -> tuple[Any, bool]:
    if not path:
        return None, False
    segments = _split_top(path, ".")
    if segments is None or any(seg == "" for seg in segments):
        return None, False
    return _walk(data, segments)


_QUERY_OPS = ("==", "!=", "<=", ">=", "!%", "<", ">", "%")


def _parse_query(seg: str) -> tuple[tuple[list[str], str | None, Any], bool] | None:
    """Parse a ``#(<cond>)`` / ``#(<cond>)#`` segment to
    ``((key_path, op, value), all_matches)``; ``None`` when ``seg`` is not a
    well-formed row query."""
    if not seg.startswith("#("):
        return None
    if seg.endswith(")#"):
        body, every = seg[2:-2], True
    elif seg.endswith(")"):
        body, every = seg[2:-1], False
    else:
        return None
    # The operator is the first one outside a quoted value: values are always
    # quoted when they are strings, so a key never contains a quote.
    quote = body.find('"')
    head = body if quote < 0 else body[:quote]
    op_at, op = -1, None
    for candidate in _QUERY_OPS:
        at = head.find(candidate)
        if at >= 0 and (op_at < 0 or at < op_at or (at == op_at and len(candidate) > len(op))):
            op_at, op = at, candidate
    if op is None:
        key = body.strip()
        if not key or '"' in key:
            return None
        path = key.split(".")
        return ((path, None, None), every) if all(path) else None
    key = body[:op_at].strip()
    raw = body[op_at + len(op) :].strip()
    path = key.split(".") if key else []
    if not all(path):
        return None
    ok, value = _parse_value(raw)
    if not ok:
        return None
    if op in ("%", "!%") and not isinstance(value, str):
        return None
    return (path, op, value), every


def _parse_value(raw: str) -> tuple[bool, Any]:
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        try:
            value = json.loads(raw)
        except ValueError:
            return False, None
        return isinstance(value, str), value
    if raw in ("true", "false", "null"):
        return True, {"true": True, "false": False, "null": None}[raw]
    try:
        number = json.loads(raw)
    except ValueError:
        return False, None
    if isinstance(number, int | float) and not isinstance(number, bool):
        return True, number
    return False, None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _glob(pattern: str, text: str) -> bool:
    """gjson's match: ``*`` any run, ``?`` one character, everything else literal."""
    import re

    regex = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
    return re.fullmatch(regex, text, flags=re.DOTALL) is not None


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _satisfies(element: Any, cond: tuple[list[str], str | None, Any]) -> bool:
    path, op, want = cond
    got: Any = element
    for key in path:
        if not isinstance(got, Mapping) or key not in got:
            return False
        got = got[key]
    if op is None:
        return got is not None
    if op in ("%", "!%"):
        if not isinstance(got, str):
            return False
        return _glob(want, got) is (op == "%")
    if op in ("==", "!="):
        if _is_number(got) and _is_number(want):
            equal = got == want
        elif isinstance(got, Mapping | list) or isinstance(want, Mapping | list):
            equal = False
        elif got is None or want is None or isinstance(got, bool) or isinstance(want, bool):
            equal = type(got) is type(want) and got == want
        else:
            equal = _text(got) == _text(want)
        return equal is (op == "==")
    if _is_number(got) and _is_number(want):
        a, b = got, want
    elif isinstance(got, str) and isinstance(want, str):
        a, b = got, want
    else:
        return False
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _walk(current: Any, segments: list[str]) -> tuple[Any, bool]:
    if not segments:
        return current, True
    seg, rest = segments[0], segments[1:]
    if seg.startswith("#("):
        parsed = _parse_query(seg)
        if parsed is None or not isinstance(current, list):
            return None, False
        cond, every = parsed
        if not every:
            for element in current:
                if _satisfies(element, cond):
                    return _walk(element, rest)
            return None, False
        kept = [element for element in current if _satisfies(element, cond)]
        if not rest:
            return kept, True
        out = []
        for element in kept:
            result, matched = _walk(element, rest)
            if matched:
                out.append(result)
        return out, True
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
