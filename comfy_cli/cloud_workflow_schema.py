"""Matching frontend canvas schemas plus exact input-only refinement checks.

The generated structural schemas remain byte-identical to their source export.
Runtime annotations restore refinements that Zod's JSON Schema export omits;
opaque workflow metadata and widget objects are never traversed as graph nodes.
"""

from __future__ import annotations

import functools
import json
import re
from importlib.resources import files

from jsonschema import Draft202012Validator, FormatChecker

_SAFE_INTEGER = 9007199254740991
_JS_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_REPO_ID = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?\Z", re.ASCII)
_GITHUB_USER = re.compile(r"(?!-)(?!.*--)[a-zA-Z0-9-]+(?<!-)\Z", re.ASCII)
_SEMVER = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([\da-z-]+(?:\.[\da-z-]+)*))?(?:\+([\da-z-]+(?:\.[\da-z-]+)*))?\Z",
    re.ASCII,
)
_HASH = re.compile(r"[0-9a-f]{4,40}\Z", re.IGNORECASE | re.ASCII)


def _slot_index(value) -> bool:
    if not isinstance(value, str):
        return True  # Existing integer schema enforces numeric safe-int bounds.
    text = value.lstrip(_JS_WHITESPACE)
    if text.startswith(("+", "-")):
        text = text[1:]
    if text.startswith(("0x", "0X")):
        return re.match(r"[0-9a-fA-F]", text[2:]) is not None
    # parseInt accepts a digit prefix even if it rounds/overflows to Infinity.
    return re.match(r"[0-9]", text) is not None


def _repo_id(value) -> bool:
    # Python regex '$' also accepts a trailing LF; the frontend's JS check
    # does not. Fullmatch preserves the source predicate across engines.
    return not isinstance(value, str) or (1 <= len(value) <= 100 and _REPO_ID.fullmatch(value) is not None)


def _aux_id(value) -> bool:
    if not isinstance(value, str):
        return True
    parts = value.split("/")
    return (
        len(parts) == 2
        and 1 <= len(parts[0]) <= 39
        and 1 <= len(parts[1]) <= 100
        and _GITHUB_USER.fullmatch(parts[0]) is not None
        and _REPO_ID.fullmatch(parts[1]) is not None
    )


def _version(value) -> bool:
    if not isinstance(value, str):
        return True
    if value == "unknown":
        return True
    stripped = value[1:] if value.startswith("v") else value
    return _HASH.fullmatch(stripped) is not None or _SEMVER.fullmatch(stripped) is not None


def _link_id(value) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value) is None:
        return False
    if len(value.lstrip("-")) > 16:
        return False
    return abs(int(value)) <= _SAFE_INTEGER


def _guid(value) -> bool:
    return (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value) is not None
    )


def _uri(value) -> bool:
    if not isinstance(value, str):
        return True  # The generated schema owns the string-type check.
    try:
        # Load native code only when a canvas actually declares a model URL.
        from ada_url import URL

        # Browser URL() receives a USVString: merge valid UTF-16 pairs and
        # replace lone surrogates with U+FFFD. Do not rewrite the stored graph.
        scalar_value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
        return URL.can_parse(scalar_value)  # No base: relative references fail.
    except (ImportError, OSError, ValueError):
        return False


def _annotate(schema: dict) -> None:
    if "$ref" in schema and (not isinstance(schema["$ref"], str) or not schema["$ref"].startswith("#")):
        raise ValueError("Remote graph schema references are not permitted")
    props = schema.get("properties")
    if isinstance(props, dict):
        for key in ("slot_index", "origin_slot", "target_slot"):
            if isinstance(props.get(key), dict):
                props[key]["format"] = "comfy-slot-index"
        if isinstance(props.get("cnr_id"), dict):
            props["cnr_id"]["format"] = "comfy-cnr-id"
        if isinstance(props.get("aux_id"), dict):
            props["aux_id"]["format"] = "comfy-aux-id"
        if isinstance(props.get("ver"), dict):
            props["ver"]["format"] = "comfy-pack-version"
        presentation = props.get("linkPresentation")
        if isinstance(presentation, dict) and isinstance(presentation.get("propertyNames"), dict):
            presentation["propertyNames"]["format"] = "comfy-link-id"
        links = props.get("links")
        item = links.get("items") if isinstance(links, dict) else None
        items = item.get("prefixItems") if isinstance(item, dict) else None
        if isinstance(items, list) and len(items) == 6:
            for index in (2, 4):
                items[index]["format"] = "comfy-slot-index"
    for value in schema.values():
        if isinstance(value, dict):
            _annotate(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _annotate(item)


@functools.lru_cache(maxsize=2)
def _validator(version: str) -> Draft202012Validator:
    schema = json.loads(files("comfy_cli.schemas").joinpath(f"cloud_workflow_{version}.json").read_text())
    _annotate(schema)
    checker = FormatChecker()
    checks = {
        "uuid": _guid,
        "uri": _uri,
        "comfy-slot-index": _slot_index,
        "comfy-cnr-id": _repo_id,
        "comfy-aux-id": _aux_id,
        "comfy-pack-version": _version,
        "comfy-link-id": _link_id,
    }
    for name, check in checks.items():
        checker.checks(name, raises=(ValueError, TypeError))(check)

    def known_formats(item):
        if isinstance(item, dict):
            if "format" in item and item["format"] not in checks:
                raise ValueError("Unsupported graph schema format")
            for value in item.values():
                known_formats(value)
        elif isinstance(item, list):
            for value in item:
                known_formats(value)

    known_formats(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=checker)


def is_supported_canvas(graph: object) -> bool:
    """Whether the pinned frontend consumer accepts this JSON canvas input."""
    if not isinstance(graph, dict):
        return False
    try:
        json.dumps(graph, allow_nan=False)
        return _validator("1_0" if graph.get("version") == 1 else "0_4").is_valid(graph)
    except (ValueError, TypeError, RecursionError, OSError):
        return False
