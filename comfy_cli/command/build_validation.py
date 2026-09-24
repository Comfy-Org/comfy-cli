from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit

from typing_extensions import assert_never

from comfy_cli.command.build_paths import BuildPaths, resolve_local_path
from comfy_cli.command.build_spec import SPEC_SCHEMA, BuildSpecInvalidError, JsonObject, JsonValue

_AUTHORING_FIELDS: Final = frozenset({"source", "localPath", "localDigest", "localSizeBytes"})
MODEL_SOURCES: Final = ("blobId", "sourceUri")
NODE_SOURCES: Final = ("blobId", "registryVersion", "repository")
MODEL_RESOLVE_BATCH_SIZE: Final = 32

# comfy-builder's own model rules (``definition.validateModels`` and
# ``common.ValidModelDir``), so a spec its release cut would refuse is refused here,
# before any upload, with every problem at once. The reasons are the builder's words,
# so what this prints reads the same as what a save or a cut would say.
_SAFE_SEGMENT: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MAX_MODEL_DIR: Final = 255
_RESERVED_MODEL_ROOTS: Final = frozenset({"configs", "custom_nodes"})
_WINDOWS_RESERVED: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
)
_MODEL_DIR_REASON: Final = "must be a model directory under models/ (e.g. checkpoints, insightface/models/antelopev2)"


class ModelLookupState(str, Enum):
    CANDIDATE_FOUND = "candidate_found"
    NONE_FOUND = "none_found"
    LOOKUP_ERROR = "lookup_error"
    NOT_LOOKUPABLE = "not_lookupable"


@dataclass(frozen=True, slots=True)
class ModelLookup:
    index: int
    filename: str | None
    state: ModelLookupState
    candidates: tuple[JsonObject, ...] = ()
    error: str | None = None

    def as_json(self) -> JsonObject:
        result: JsonObject = {
            "entry": f"definition.models[{self.index}]",
            "filename": self.filename,
            "state": self.state.value,
        }
        if self.candidates:
            result["candidates"] = [deepcopy(candidate) for candidate in self.candidates]
        if self.error is not None:
            result["error"] = self.error
        return result


class ModelResolver(Protocol):
    def __call__(self, filenames: list[str]) -> list[JsonObject]: ...


@dataclass(frozen=True, slots=True)
class LocalPathContext:
    location: str
    root: Path
    kind: Literal["model", "node"]


def _entries(definition: JsonObject, collection: str) -> list[JsonObject]:
    value = definition.get(collection, [])
    if not isinstance(value, list):
        raise BuildSpecInvalidError(f"definition.{collection} must be a list")
    entries: list[JsonObject] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise BuildSpecInvalidError(f"definition.{collection}[{index}] must be a mapping")
        entries.append(entry)
    return entries


def _set_sources(entry: JsonObject, fields: tuple[str, ...], *, location: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for field in fields:
        value = entry.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise BuildSpecInvalidError(f"{location}.{field} must be a string or null")
        if value.strip():
            sources[field] = value
    return sources


def _project_model(entry: JsonObject, *, location: str) -> JsonObject:
    sources = _set_sources(entry, MODEL_SOURCES, location=location)
    projected = deepcopy(entry)
    for field in (*_AUTHORING_FIELDS, *MODEL_SOURCES):
        projected.pop(field, None)
    for winner in MODEL_SOURCES:
        if winner in sources:
            projected[winner] = sources[winner]
            break
    return projected


def _project_node(entry: JsonObject, *, location: str) -> JsonObject:
    sources = _set_sources(entry, NODE_SOURCES, location=location)
    projected = deepcopy(entry)
    for field in (*_AUTHORING_FIELDS, *NODE_SOURCES):
        projected.pop(field, None)
    for winner in NODE_SOURCES:
        if winner not in sources:
            continue
        projected[winner] = sources[winner]
        if winner in {"blobId", "registryVersion"}:
            projected.pop("gitRef", None)
            projected.pop("commit", None)
        break
    return projected


def project_wire_definition(definition: JsonObject) -> JsonObject:
    """Return the exclusion-based builder wire copy using D-I source precedence."""
    projected = deepcopy(definition)
    if "models" in definition:
        projected["models"] = [
            _project_model(entry, location=f"definition.models[{index}]")
            for index, entry in enumerate(_entries(definition, "models"))
        ]
    if "customNodes" in definition:
        projected["customNodes"] = [
            _project_node(entry, location=f"definition.customNodes[{index}]")
            for index, entry in enumerate(_entries(definition, "customNodes"))
        ]
    return projected


def _required_string(entry: JsonObject, field: str, *, location: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BuildSpecInvalidError(f"{location}.{field} must be a non-empty string")
    return value


def _optional_string(entry: JsonObject, field: str, *, location: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BuildSpecInvalidError(f"{location}.{field} must be a string or null")
    return value


def _validate_local_path(entry: JsonObject, context: LocalPathContext) -> None:
    local_path = _optional_string(entry, "localPath", location=context.location)
    source = _optional_string(entry, "source", location=context.location)
    if local_path is None:
        if source == "local":
            raise BuildSpecInvalidError(f"{context.location}.localPath is required when source is 'local'")
        return
    if not local_path.strip():
        raise BuildSpecInvalidError(f"{context.location}.localPath must be a non-empty string")
    resolved = resolve_local_path(context.root, local_path, entry=context.location)
    match context.kind:
        case "model":
            correct_kind = resolved.is_file()
            expected = "file"
        case "node":
            correct_kind = resolved.is_dir()
            expected = "directory"
        case unreachable:
            assert_never(unreachable)
    if not correct_kind:
        raise BuildSpecInvalidError(f"{context.location}.localPath must resolve to a {expected}: {local_path!r}")


def _validate_authoring_definition(definition: JsonObject, paths: BuildPaths) -> None:
    schema = definition.get("schema")
    if schema is not None and not isinstance(schema, str):
        raise BuildSpecInvalidError("definition.schema must be a string or null")
    for index, model in enumerate(_entries(definition, "models")):
        location = f"definition.models[{index}]"
        _required_string(model, "type", location=location)
        _optional_string(model, "filename", location=location)
        _validate_local_path(model, LocalPathContext(location, paths.models_dir, "model"))
    for index, node in enumerate(_entries(definition, "customNodes")):
        location = f"definition.customNodes[{index}]"
        _required_string(node, "name", location=location)
        _validate_local_path(node, LocalPathContext(location, paths.custom_nodes_dir, "node"))


def _validate_wire_sources(original: JsonObject, projected: JsonObject, collection: str) -> None:
    fields = MODEL_SOURCES if collection == "models" else NODE_SOURCES
    originals = _entries(original, collection)
    wire_entries = _entries(projected, collection)
    for index, (authoring_entry, wire_entry) in enumerate(zip(originals, wire_entries)):
        effective = _set_sources(wire_entry, fields, location=f"definition.{collection}[{index}]")
        if len(effective) > 1:
            raise BuildSpecInvalidError(f"definition.{collection}[{index}] has multiple effective builder sources")
        if effective or authoring_entry.get("source") == "local":
            continue
        raise BuildSpecInvalidError(f"definition.{collection}[{index}] has no effective builder source")


def _valid_model_dir(value: str, directories: frozenset[str] | None) -> bool:
    """``common.ValidModelDir``: a vetted directory, or a relative path that can only
    land inside models/. A case variant of a vetted directory ("Loras") is a typo, and
    only the builder's list can tell one, so without *directories* it passes here and
    the save warns about it instead."""
    if directories and value in directories:
        return True
    if not value or len(value) > _MAX_MODEL_DIR:
        return False
    if directories and value.lower() in {directory.lower() for directory in directories}:
        return False
    segments = value.split("/")
    if segments[0].lower() in _RESERVED_MODEL_ROOTS:
        return False
    for segment in segments:
        if not _SAFE_SEGMENT.fullmatch(segment) or segment.endswith("."):
            return False
        if segment.partition(".")[0].upper() in _WINDOWS_RESERVED:
            return False
    return True


def _valid_filename(value: str) -> bool:
    return _SAFE_SEGMENT.fullmatch(value) is not None and ".." not in value


def _https_url(value: str) -> bool:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.netloc)


def _lacks_extension(uri: str) -> bool:
    """Whether the name the builder derives from *uri* (Go's ``path.Base`` of its
    path) has no extension, as a Civitai ``/api/download/models/<id>`` link does."""
    path = urlsplit(uri.strip()).path
    if not path:
        return False  # path.Base("") is ".", which has one
    base = path.rstrip("/").rsplit("/", 1)[-1] or "/"
    return "." not in base


def model_rule_problems(
    definition: JsonObject, projected: JsonObject, directories: frozenset[str] | None = None
) -> list[dict[str, str]]:
    """Every model entry the builder's cut would refuse, as ``{field, reason,
    model}``. ``models[<n>]`` counts the spec as it is read, which sorts the models,
    so ``model`` names the entry (its filename, link or local path) for a person
    looking for it in a file they ordered themselves. A ``source: local`` entry's
    link and sha256 are left alone: push replaces both with what it uploads."""
    problems: list[dict[str, str]] = []
    model = ""

    def refuse(field: str, reason: str) -> None:
        problems.append({"field": field, "reason": reason, "model": model})

    for index, (entry, wire) in enumerate(zip(_entries(definition, "models"), _entries(projected, "models"))):
        location = f"definition.models[{index}]"
        model = next(
            (
                value
                for key in ("filename", "sourceUri", "localPath")
                if isinstance(value := entry.get(key), str) and value.strip()
            ),
            "",
        )
        model_type = entry.get("type")
        if isinstance(model_type, str) and not _valid_model_dir(model_type, directories):
            refuse(f"{location}.type", _MODEL_DIR_REASON)
        filename = entry.get("filename")
        has_filename = isinstance(filename, str) and bool(filename.strip())
        if has_filename and not _valid_filename(filename):
            refuse(f"{location}.filename", "must be a safe filename")
        if entry.get("source") == "local":
            continue
        uri = wire.get("sourceUri")
        if isinstance(uri, str) and uri.strip():
            if not _https_url(uri):
                refuse(f"{location}.sourceUri", "must be an https URL")
            elif not has_filename and _lacks_extension(uri):
                refuse(f"{location}.filename", "sourceUri has no file extension; set an explicit filename")
        sha256 = entry.get("sha256")
        if isinstance(sha256, str) and sha256.strip() and not _SHA256.fullmatch(sha256.strip().lower()):
            refuse(f"{location}.sha256", "must be a 64-character sha256")
    return problems


def _problems_message(problems: list[dict[str, str]]) -> str:
    count = len(problems)
    lines = [
        f"  {problem['field']}" + (f" ({problem['model']})" if problem["model"] else "") + f": {problem['reason']}"
        for problem in problems
    ]
    heading = f"{count} problem{'s' if count != 1 else ''} the builder would refuse this build for:"
    return "\n".join([heading, *lines])


def validate_local_build_spec(
    spec: JsonObject, paths: BuildPaths, *, model_directories: frozenset[str] | None = None
) -> JsonObject:
    """Run authoring validation, then return the validated normalized wire copy.

    *model_directories* is the builder's vetted list, when the caller has signed in
    and read it; it is what tells a case variant of a folder from a new one."""
    if spec.get("schema") != SPEC_SCHEMA:
        raise BuildSpecInvalidError(f"unsupported build spec schema {spec.get('schema')!r}; expected {SPEC_SCHEMA!r}")
    definition: JsonValue = spec.get("definition")
    if not isinstance(definition, dict):
        raise BuildSpecInvalidError("definition must be a mapping")
    _validate_authoring_definition(definition, paths)
    projected = project_wire_definition(definition)
    _validate_wire_sources(definition, projected, "models")
    _validate_wire_sources(definition, projected, "customNodes")
    problems = model_rule_problems(definition, projected, model_directories)
    if problems:
        raise BuildSpecInvalidError(_problems_message(problems), issues=problems)
    return projected


def _lookup_result(index: int, filename: str, response: JsonValue | None) -> ModelLookup:
    location = f"definition.models[{index}]"
    if not isinstance(response, dict):
        return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned no result")
    echoed = response.get("filename")
    if echoed != filename:
        return ModelLookup(
            index,
            filename,
            ModelLookupState.LOOKUP_ERROR,
            error=f"lookup returned {echoed!r} for {location}",
        )
    error = response.get("error")
    if error is not None:
        if not isinstance(error, str):
            return ModelLookup(
                index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned a malformed error"
            )
        if error.strip():
            return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error=error)
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return ModelLookup(index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned malformed candidates")
    parsed: list[JsonObject] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return ModelLookup(
                index, filename, ModelLookupState.LOOKUP_ERROR, error="lookup returned a malformed candidate"
            )
        parsed.append(candidate)
    state = ModelLookupState.CANDIDATE_FOUND if parsed else ModelLookupState.NONE_FOUND
    return ModelLookup(index, filename, state, candidates=tuple(parsed))


def lookup_public_model_sources(definition: JsonObject, resolver: ModelResolver) -> list[ModelLookup]:
    """Resolve lookupable model filenames in 32-item batches and restore spec order."""
    models = _entries(definition, "models")
    lookupable: list[tuple[int, str]] = []
    results: dict[int, ModelLookup] = {}
    for index, model in enumerate(models):
        filename = model.get("filename")
        if isinstance(filename, str) and filename.strip():
            lookupable.append((index, filename))
        else:
            results[index] = ModelLookup(index, None, ModelLookupState.NOT_LOOKUPABLE)

    for start in range(0, len(lookupable), MODEL_RESOLVE_BATCH_SIZE):
        batch = lookupable[start : start + MODEL_RESOLVE_BATCH_SIZE]
        response = resolver([filename for _, filename in batch])
        for offset, (index, filename) in enumerate(batch):
            item = response[offset] if offset < len(response) else None
            results[index] = _lookup_result(index, filename, item)
    return [results[index] for index in range(len(models))]
