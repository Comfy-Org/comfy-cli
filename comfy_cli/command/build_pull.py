"""Identity-preserving reconciliation for ``comfy build pull``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from typing_extensions import assert_never

from comfy_cli.command.build_push import normalize_repository_identity
from comfy_cli.command.build_spec import BuildSpecError, BuildSpecInvalidError, JsonObject, JsonValue

IdentityTier: TypeAlias = Literal["sha256", "model_location", "blobId", "id", "repository", "name"]

_MODEL_LOCAL_FIELDS: Final = frozenset({"source", "localPath", "blobId", "sha256", "sizeBytes"})
_NODE_LOCAL_FIELDS: Final = frozenset({"source", "localPath", "blobId", "localDigest", "localSizeBytes"})
_MODEL_KNOWN_FIELDS: Final = _MODEL_LOCAL_FIELDS | {"type", "filename", "sourceUri"}
_NODE_KNOWN_FIELDS: Final = _NODE_LOCAL_FIELDS | {
    "name",
    "id",
    "repository",
    "gitRef",
    "commit",
    "registryVersion",
    "sourceUri",
}
_DEFINITION_KNOWN_FIELDS: Final = frozenset(
    {
        "schema",
        "baseComfyVersion",
        "baseImage",
        "models",
        "customNodes",
        "pipDependencies",
        "environment",
        "modelPolicy",
        "partnerNodePolicy",
        "customNodePolicy",
    }
)
# `schema` names the definition's own format, not build state: the builder has no
# concept of it, so it is the one field exempt from the round-trip check below.
_UNSYNCED_DEFINITION_FIELDS: Final = frozenset({"schema"})


class UnsyncedDefinitionError(BuildSpecError):
    code = "build_pull_unsynced_definition"

    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields
        super().__init__(
            f"the fetched Build's definition omits {', '.join(fields)}, which the local spec sets; "
            "pulling would delete them"
        )


def _entries(definition: JsonObject, collection: str, *, side: str) -> list[JsonObject]:
    values = definition.get(collection, [])
    if not isinstance(values, list):
        raise BuildSpecInvalidError(f"{side} definition.{collection} must be a list")
    entries: list[JsonObject] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise BuildSpecInvalidError(f"{side} definition.{collection}[{index}] must be a mapping")
        entries.append(value)
    return entries


def _text(entry: JsonObject, field: str, *, location: str) -> str | None:
    value = entry.get(field)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise BuildSpecInvalidError(f"{location}.{field} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _identity(entry: JsonObject, tier: IdentityTier, *, location: str) -> tuple[str, ...] | None:
    match tier:
        case "sha256" | "blobId" | "id" | "name":
            value = _text(entry, tier, location=location)
            return (value,) if value is not None else None
        case "model_location":
            model_type = _text(entry, "type", location=location)
            filename = _text(entry, "filename", location=location)
            return (model_type, filename) if model_type is not None and filename is not None else None
        case "repository":
            repository = _text(entry, "repository", location=location)
            return (normalize_repository_identity(repository),) if repository is not None else None
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class _Candidates:
    local: frozenset[int]
    server: frozenset[int]


@dataclass(frozen=True, slots=True)
class _FieldPolicy:
    known: frozenset[str]
    local: frozenset[str]


class _TieredMatcher:
    """Mutable greedy-match accumulator; matched indexes leave every later tier."""

    def __init__(self, local: list[JsonObject], server: list[JsonObject], collection: str) -> None:
        self.local = local
        self.server = server
        self.collection = collection
        self.local_unmatched = set(range(len(local)))
        self.server_unmatched = set(range(len(server)))
        self.local_by_server: dict[int, int] = {}

    def candidates(self) -> _Candidates:
        return _Candidates(frozenset(self.local_unmatched), frozenset(self.server_unmatched))

    def groups(
        self, tier: IdentityTier, candidates: _Candidates
    ) -> tuple[dict[tuple[str, ...], list[int]], dict[tuple[str, ...], list[int]]]:
        local: dict[tuple[str, ...], list[int]] = {}
        server: dict[tuple[str, ...], list[int]] = {}
        for index in candidates.local:
            key = _identity(self.local[index], tier, location=f"local definition.{self.collection}[{index}]")
            if key is not None:
                local.setdefault(key, []).append(index)
        for index in candidates.server:
            key = _identity(self.server[index], tier, location=f"server definition.{self.collection}[{index}]")
            if key is not None:
                server.setdefault(key, []).append(index)
        return local, server

    def pair(self, local_index: int, server_index: int) -> None:
        self.local_unmatched.remove(local_index)
        self.server_unmatched.remove(server_index)
        self.local_by_server[server_index] = local_index

    def ambiguity(self, tier: IdentityTier, key: tuple[str, ...], candidates: _Candidates) -> BuildSpecInvalidError:
        local = ", ".join(f"local definition.{self.collection}[{index}]" for index in sorted(candidates.local))
        server = ", ".join(f"server definition.{self.collection}[{index}]" for index in sorted(candidates.server))
        return BuildSpecInvalidError(
            f"ambiguous pull identity at {self.collection}.{tier}={key!r}: {local or 'no local entry'}; "
            f"{server or 'no server entry'}"
        )

    def run(self, tier: IdentityTier, candidates: _Candidates | None = None) -> None:
        active = candidates or self.candidates()
        local_groups, server_groups = self.groups(tier, active)
        for key in sorted(local_groups.keys() & server_groups.keys()):
            local_indexes = local_groups[key]
            server_indexes = server_groups[key]
            group = _Candidates(frozenset(local_indexes), frozenset(server_indexes))
            if len(local_indexes) != 1 or len(server_indexes) != 1:
                raise self.ambiguity(tier, key, group)
            self.pair(local_indexes[0], server_indexes[0])


def _match_models(local: list[JsonObject], server: list[JsonObject]) -> dict[int, int]:
    matcher = _TieredMatcher(local, server, "models")
    local_digests, server_digests = matcher.groups("sha256", matcher.candidates())
    for digest in sorted(local_digests.keys() & server_digests.keys()):
        local_group = frozenset(local_digests[digest])
        server_group = frozenset(server_digests[digest])
        if len(local_group) == 1 and len(server_group) == 1:
            matcher.pair(next(iter(local_group)), next(iter(server_group)))
            continue
        matcher.run("model_location", _Candidates(local_group, server_group))
        remaining = _Candidates(local_group & matcher.local_unmatched, server_group & matcher.server_unmatched)
        if not remaining.local or not remaining.server:
            continue
        if len(remaining.local) == 1 and len(remaining.server) == 1:
            matcher.pair(next(iter(remaining.local)), next(iter(remaining.server)))
            continue
        raise matcher.ambiguity("sha256", digest, remaining)
    matcher.run("model_location")
    return matcher.local_by_server


def _match_nodes(local: list[JsonObject], server: list[JsonObject]) -> dict[int, int]:
    matcher = _TieredMatcher(local, server, "customNodes")
    for tier in ("blobId", "id", "repository", "name"):
        matcher.run(tier)
    return matcher.local_by_server


def _merge_entry(
    local: JsonObject,
    server: JsonObject,
    policy: _FieldPolicy,
) -> JsonObject:
    merged = deepcopy(server)
    for key, value in local.items():
        if key not in merged and (key not in policy.known or value is None):
            merged[key] = deepcopy(value)
    for field in policy.local:
        if field in local:
            merged[field] = deepcopy(local[field])
        else:
            merged.pop(field, None)
    merged.pop("sourceUri", None)
    local_source_uri = local.get("sourceUri")
    if isinstance(local_source_uri, str) and local_source_uri.strip():
        merged["sourceUri"] = local_source_uri
    return merged


def _merge_collection(
    local: JsonObject, server: JsonObject, collection: Literal["models", "customNodes"]
) -> list[JsonValue]:
    local_entries = _entries(local, collection, side="local")
    server_entries = _entries(server, collection, side="server")
    match collection:
        case "models":
            matches = _match_models(local_entries, server_entries)
            policy = _FieldPolicy(_MODEL_KNOWN_FIELDS, _MODEL_LOCAL_FIELDS)
        case "customNodes":
            matches = _match_nodes(local_entries, server_entries)
            policy = _FieldPolicy(_NODE_KNOWN_FIELDS, _NODE_LOCAL_FIELDS)
        case unreachable:
            assert_never(unreachable)
    return [
        deepcopy(entry)
        if server_index not in matches
        else _merge_entry(local_entries[matches[server_index]], entry, policy)
        for server_index, entry in enumerate(server_entries)
    ]


def merge_pull_definition(local: JsonObject, server: JsonObject) -> JsonObject:
    """Merge a server-owned definition onto local authoring/cache fields by identity."""
    merged = deepcopy(server)
    for key, value in local.items():
        if key not in _DEFINITION_KNOWN_FIELDS and key not in merged:
            merged[key] = deepcopy(value)
    merged["models"] = _merge_collection(local, server, "models")
    merged["customNodes"] = _merge_collection(local, server, "customNodes")
    for key in _UNSYNCED_DEFINITION_FIELDS & set(local):
        merged.setdefault(key, deepcopy(local[key]))
    dropped = tuple(sorted(set(local) - set(merged)))
    if dropped:
        raise UnsyncedDefinitionError(dropped)
    return merged


@dataclass(frozen=True, slots=True)
class PulledSpec:
    """The spec `pull` writes. ``definition`` is the very object under
    ``spec["definition"]``, carried alongside so a caller that has to diff or
    project it does not have to re-narrow it out of a ``JsonValue``."""

    spec: JsonObject
    definition: JsonObject


def merge_pulled_spec(local_spec: JsonObject, remote: JsonObject, build_id: str) -> PulledSpec:
    """Return the atomically writable local spec for one fetched Build."""
    local_definition = local_spec.get("definition")
    server_definition = remote.get("definition")
    if not isinstance(local_definition, dict) or not isinstance(server_definition, dict):
        raise BuildSpecInvalidError("both the local spec and fetched Build need a definition mapping")
    name = remote.get("name")
    # `description` is `*string` + `omitempty` on the builder, so an empty one
    # arrives absent, not as `""`. Same state on the wire; default, don't refuse.
    description = remote.get("description")
    if description is None:
        description = ""
    revision = remote.get("updatedAt")
    if not isinstance(name, str) or not isinstance(description, str) or not isinstance(revision, str) or not revision:
        raise BuildSpecInvalidError("the fetched Build needs string name, description and updatedAt fields")
    definition = merge_pull_definition(local_definition, server_definition)
    merged = deepcopy(local_spec)
    merged.update(
        {
            "id": build_id,
            "name": name,
            "description": description,
            "syncedRevision": revision,
            "definition": definition,
        }
    )
    return PulledSpec(merged, definition)
