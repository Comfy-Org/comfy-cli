"""Canonical reader and writer for the ``comfy-build/1`` file format."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, TypeAlias

import yaml
from typing_extensions import assert_never

from comfy_cli.file_utils import atomic_write_text

SPEC_SCHEMA: Final = "comfy-build/1"
_TOP_LEVEL_KEYS: Final = ("schema", "id", "name", "description", "syncedRevision", "definition")
_DEFINITION_KEYS: Final = (
    "schema",
    "baseComfyVersion",
    "models",
    "customNodes",
    "pipDependencies",
    "environment",
)
_SORT_KEYS: Final = {
    "models": ("type", "filename", "sha256"),
    "customNodes": ("name", "id", "repository"),
}

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
BuildSpec: TypeAlias = JsonObject
SpecFormat: TypeAlias = Literal["yaml", "json"]


class BuildSpecError(Exception):
    message: str
    path: Path | None

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.message = message
        self.path = path
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path is not None else self.message


class BuildSpecInvalidError(BuildSpecError):
    code = "build_spec_invalid"


class BuildSpecWriteError(BuildSpecError):
    code = "build_spec_write_error"


class _LiteralString(str):
    pass


class _CanonicalDumper(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, indentless=False)

    def determine_block_hints(self, text: str) -> str:
        hints = super().determine_block_hints(text)
        if text.endswith("\n") and "+" not in hints:
            return f"{hints}+"
        return hints


def _represent_literal(dumper: _CanonicalDumper, value: _LiteralString):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


_CanonicalDumper.add_representer(_LiteralString, _represent_literal)


def _invalid(message: str, path: Path | None) -> BuildSpecInvalidError:
    return BuildSpecInvalidError(message, path=path)


def _parse_json_value(value, *, location: str, path: Path | None) -> JsonValue:
    match value:
        case None | str() | bool() | int():
            return value
        case float() if math.isfinite(value):
            return value
        case float():
            raise _invalid(f"{location} contains a non-finite number", path)
        case list():
            return [
                _parse_json_value(item, location=f"{location}[{index}]", path=path) for index, item in enumerate(value)
            ]
        case dict():
            parsed: JsonObject = {}
            for key, item in value.items():
                match key:
                    case str():
                        parsed[key] = _parse_json_value(item, location=f"{location}.{key}", path=path)
                    case _:
                        raise _invalid(f"{location} contains a non-string mapping key", path)
            return parsed
        case _:
            raise _invalid(f"{location} contains a value that is not JSON-compatible", path)


def _parse_json_object(value, *, location: str, path: Path | None) -> JsonObject:
    parsed = _parse_json_value(value, location=location, path=path)
    match parsed:
        case dict() as mapping:
            return mapping
        case _:
            raise _invalid(f"{location} must be a mapping", path)


def _normalized_sort_key(entry: JsonObject, keys: tuple[str, ...], *, location: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for key in keys:
        match entry.get(key):
            case None:
                normalized.append("")
            case str() as value:
                normalized.append(value)
            case value:
                raise BuildSpecInvalidError(f"{location}.{key} must be a string or null, got {type(value).__name__}")
    return tuple(normalized)


def _sorted_entries(value: JsonValue, collection: str) -> list[JsonValue]:
    match value:
        case list() as entries:
            keyed: list[tuple[tuple[str, ...], JsonObject]] = []
            for index, entry in enumerate(entries):
                match entry:
                    case dict():
                        key = _normalized_sort_key(
                            entry, _SORT_KEYS[collection], location=f"definition.{collection}[{index}]"
                        )
                        keyed.append((key, entry))
                    case _:
                        raise BuildSpecInvalidError(f"definition.{collection}[{index}] must be a mapping")
            keyed.sort(key=lambda item: item[0])
            for previous, current in zip(keyed, keyed[1:]):
                if previous[0] == current[0]:
                    raise BuildSpecInvalidError(
                        f"definition.{collection} contains duplicate sort identity {current[0]!r}"
                    )
            return [entry for _, entry in keyed]
        case _:
            raise BuildSpecInvalidError(f"definition.{collection} must be a list")


def _canonicalize_value(value: JsonValue, path: tuple[str, ...]) -> JsonValue:
    match value:
        case dict() as mapping:
            return _canonicalize_mapping(mapping, path)
        case list() as items:
            return [_canonicalize_value(item, (*path, str(index))) for index, item in enumerate(items)]
        case str() if path == ("definition", "pipDependencies"):
            return _LiteralString(value.replace("\r\n", "\n").replace("\r", "\n"))
        case _:
            return value


def _canonicalize_mapping(mapping: Mapping[str, JsonValue], path: tuple[str, ...]) -> JsonObject:
    if path == ():
        known_keys = _TOP_LEVEL_KEYS
    elif path == ("definition",):
        known_keys = _DEFINITION_KEYS
    else:
        known_keys = ()

    ordered: JsonObject = {}
    for key in (*known_keys, *sorted(set(mapping) - set(known_keys))):
        if key not in mapping:
            continue
        value = mapping[key]
        if path == ("definition",) and key in _SORT_KEYS:
            value = _sorted_entries(value, key)
        ordered[key] = _canonicalize_value(value, (*path, key))
    return ordered


def canonicalize_build_spec(spec: Mapping[str, JsonValue], *, source: Path | None = None) -> BuildSpec:
    """Return a recursively ordered copy after enforcing the ``comfy-build/1`` boundary."""
    document = _parse_json_object(dict(spec), location="$", path=source)

    match document.get("schema"):
        case "comfy-build/1":
            pass
        case unsupported:
            raise _invalid(f"unsupported build spec schema {unsupported!r}; expected {SPEC_SCHEMA!r}", source)
    for key in ("name", "description"):
        if not isinstance(document.get(key), str):
            raise _invalid(f"{key} must be a string", source)
    for key in ("id", "syncedRevision"):
        if key not in document:
            document[key] = None
        else:
            match document[key]:
                case None | str():
                    pass
                case _:
                    raise _invalid(f"{key} must be a string or null", source)
    if not isinstance(document.get("definition"), dict):
        raise _invalid("definition must be a mapping", source)
    return _canonicalize_mapping(document, ())


def serialize_build_spec(spec: Mapping[str, JsonValue], *, format: SpecFormat) -> str:
    """Serialize a spec to canonical YAML or JSON with exactly one final LF."""
    canonical = canonicalize_build_spec(spec)
    match format:
        case "json":
            return json.dumps(canonical, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        case "yaml":
            output = yaml.dump(
                canonical,
                Dumper=_CanonicalDumper,
                allow_unicode=True,
                default_flow_style=False,
                explicit_end=False,
                explicit_start=False,
                line_break="\n",
                sort_keys=False,
                width=4096,
            )
            if output.endswith("...\n"):
                output = output[:-4]
            output = output.replace('  pipDependencies: ""\n', "  pipDependencies: |-\n", 1)
            return output.rstrip("\n") + "\n"
        case unreachable:
            assert_never(unreachable)


def read_build_spec(path: Path) -> BuildSpec:
    """Read YAML or JSON from ``path`` and return its canonical in-memory shape."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _invalid(f"could not read build spec: {error}", path) from error

    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            for event in yaml.parse(text, Loader=yaml.SafeLoader):
                if isinstance(event, yaml.events.AliasEvent) or getattr(event, "anchor", None) is not None:
                    raise _invalid("YAML anchors and aliases are not supported", path)
            raw = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise _invalid(f"could not parse build spec: {error}", path) from error

    document = _parse_json_object(raw, location="$", path=path)
    return canonicalize_build_spec(document, source=path)


def write_build_spec(path: Path, spec: Mapping[str, JsonValue]) -> None:
    """Atomically write ``spec`` using the format selected by ``path`` suffix."""
    format: SpecFormat = "json" if path.suffix.lower() == ".json" else "yaml"
    content = serialize_build_spec(spec, format=format)
    try:
        atomic_write_text(path, content, fsync=True)
    except OSError as error:
        raise BuildSpecWriteError(f"could not write build spec: {error}", path=path) from error
