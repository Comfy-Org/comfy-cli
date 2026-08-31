"""Identity-preserving reconciliation for ``comfy build pull``."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from typing_extensions import assert_never

from comfy_cli.command.build_push import normalize_repository_identity
from comfy_cli.command.build_spec import BuildSpecError, BuildSpecInvalidError, JsonObject, JsonValue
from comfy_cli.command.build_validation import MODEL_SOURCES, NODE_SOURCES

IdentityTier: TypeAlias = Literal["sha256", "model_location", "blobId", "id", "repository", "name"]

_SIZE_FIELDS: Final = frozenset({"sizeBytes", "localSizeBytes"})
# Metadata describing whatever the entry's source resolves to, rather than the
# entry itself, and which the builder therefore enforces against it. `sizeBytes`
# is deliberately absent: the builder's Model carries no size column, so pairing
# it with a source states nothing, and restricting it would only delete a size
# the scanner knew. A node's git coordinates *are* such a claim — `_project_node`
# keeps them on the wire when `repository` wins — though nothing fills them
# today, both sitting outside `_NODE_LOCAL_FIELDS`.
_MODEL_SOURCE_DEPENDENT: Final = frozenset({"sha256"})
_NODE_SOURCE_DEPENDENT: Final = frozenset({"gitRef", "commit"})
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
# Authoring-only fields. The builder has no typed field for either — zero
# references in its `definition/` package — so no builder-owned write path can
# produce one, and their absence says nothing about build state. It *can* echo
# them: `Definition` is a free-form map stored and returned verbatim, so a Build
# last written by `comfy build push` carries both back. The exemption is for the
# other case — a Build last written by a client that has no field for them, from
# which they return absent and would otherwise read as a missed round trip.
_UNSYNCED_DEFINITION_FIELDS: Final = frozenset({"schema", "environment"})


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
    sources: frozenset[str]
    # Metadata that describes the bytes a source points at rather than the entry
    # itself, and so is only meaningful paired with the source it was taken from.
    source_dependent: frozenset[str]
    fillable: frozenset[str]


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


def _states(entry: JsonObject, field: str) -> bool:
    """Whether *entry* makes a usable statement about *field*.

    A text field has to be a non-blank string — the guard the pre-merge
    ``sourceUri`` handling already applied. Carrying a ``"   "`` or a stray
    number through a pull writes a spec that fails ``project_wire_definition``
    on the very next push, which is a worse outcome than dropping it here.
    """
    value = entry.get(field)
    if field in _SIZE_FIELDS:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str) and bool(value.strip())


def _merge_entry(
    local: JsonObject,
    server: JsonObject,
    policy: _FieldPolicy,
) -> JsonObject:
    """Merge one server entry onto its local match, by who owns the entry.

    Only a ``source: "local"`` entry is described by bytes on this machine, so
    only there do the local ``blobId``/digest/size fields outrank the server's.
    For any other source the server holds the entry's content identity, and
    letting local win reverted a server-side switch from a private blob to a
    public ``sourceUri`` on the next push, or dropped the server's blob
    reference and hash under a local public entry.

    The non-local side *fills* rather than replaces, so re-owning these fields
    cannot itself delete authoring data — with one exclusion that matters. The
    source fields are a precedence group, not independent values:
    ``project_wire_definition`` emits the first of ``MODEL_SOURCES`` /
    ``NODE_SOURCES`` an entry names and drops the rest. Filling a stale local
    ``blobId`` beside a server ``sourceUri`` would therefore not merely add a
    field, it would *outrank* the server's and reinstate the exact revert this
    merge exists to prevent. So once the server names any source, local fills
    none of them; a server that names no source at all is the only case where
    absence is not a statement.

    ``source_dependent`` travels in that same group. ``sha256`` describes the
    bytes a source points at, not the entry, so pairing a local hash with a
    source the server has since changed states an integrity claim that was never
    true — and the builder enforces it, failing the pull's damage at deploy
    staging with a checksum mismatch rather than here. Dropping a hash is safe
    (it is optional); keeping a wrong one is not.
    """
    merged = deepcopy(server)
    for key, value in local.items():
        if key not in merged and (key not in policy.known or value is None):
            merged[key] = deepcopy(value)
    if local.get("source") == "local":
        for field in policy.local:
            if field in local:
                merged[field] = deepcopy(local[field])
            else:
                merged.pop(field, None)
        merged.pop("sourceUri", None)
        if _states(local, "sourceUri"):
            merged["sourceUri"] = deepcopy(local["sourceUri"])
        return merged
    fillable = policy.fillable
    if any(_states(merged, field) for field in policy.sources):
        fillable -= policy.sources | policy.source_dependent
    for field in sorted(fillable):
        if _states(local, field) and not _states(merged, field):
            merged[field] = deepcopy(local[field])
    return merged


def _merge_collection(
    local: JsonObject, server: JsonObject, collection: Literal["models", "customNodes"]
) -> list[JsonValue]:
    local_entries = _entries(local, collection, side="local")
    server_entries = _entries(server, collection, side="server")
    match collection:
        case "models":
            matches = _match_models(local_entries, server_entries)
            policy = _FieldPolicy(
                _MODEL_KNOWN_FIELDS,
                _MODEL_LOCAL_FIELDS,
                frozenset(MODEL_SOURCES),
                _MODEL_SOURCE_DEPENDENT,
                # `sourceUri` is fillable for both, though only a model's is a
                # *source*: a node has no public resolution path, so nothing here
                # writes one and it can outrank nothing. It stays fillable so a
                # hand-authored one is carried rather than silently deleted.
                _MODEL_LOCAL_FIELDS | {"sourceUri"},
            )
        case "customNodes":
            matches = _match_nodes(local_entries, server_entries)
            policy = _FieldPolicy(
                _NODE_KNOWN_FIELDS,
                _NODE_LOCAL_FIELDS,
                frozenset(NODE_SOURCES),
                _NODE_SOURCE_DEPENDENT,
                _NODE_LOCAL_FIELDS | {"sourceUri"},
            )
        case unreachable:
            assert_never(unreachable)
    return [
        deepcopy(entry)
        if server_index not in matches
        else _merge_entry(local_entries[matches[server_index]], entry, policy)
        for server_index, entry in enumerate(server_entries)
    ]


def _carries_data(value: JsonValue) -> bool:
    """Whether losing *value* would actually lose anything.

    An empty value is not evidence the build was never synced. The two create
    paths that build a definition server-side — ``from_snapshot`` and
    ``from_workflow`` — store it through ``ToMap``, which is ``json.Marshal`` of
    a typed struct with ``omitempty``, so a pin-less snapshot's empty
    ``pipDependencies`` is stored *absent* rather than blank. A local ``""``
    against that Build then read as a missed round trip while there was nothing
    to lose.

    Non-empty values still refuse, which is the guard doing its job: scan-
    captured pins are real data, and a Build that lacks them has not carried
    them. Adopting such a Build still requires pushing first.
    """
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple)):
        return bool(value)
    return True


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
    dropped = tuple(sorted(key for key in set(local) - set(merged) if _carries_data(local[key])))
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
