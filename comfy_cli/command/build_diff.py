"""Reconciling a fresh local scan against a build spec's stored ``definition``.

``comfy build update`` rescans the installation and has to answer two questions
from one pass over the same data: *what will change* — the diff it renders and
confirms — and *what the new document is* — the ``definition`` it writes.

**The rescan is merged, not substituted.** Everything the scanners produce comes
from the scan; everything else is carried over from the stored entry. That
distinction is load-bearing, not a nicety:

- ``push`` caches ``blobId`` on an uploaded entry and may persist a resolver's
  ``sourceUri``; ``pull`` merges server state and unknown keys back in
  (plan decisions D-I). Substituting the definition wholesale would delete all
  of it, so ``pull`` → ``update`` would churn the spec every single time.
- Conversely the merge starts from the *scanned* entry, so a fact the scan no
  longer reports — a pack that stopped being a git checkout, say — genuinely
  disappears rather than lingering as stale metadata.

**A content change invalidates the cache it backs.** ``blobId`` names bytes the
builder already holds; once a model's ``sha256`` or a node's ``localDigest``
moves, that blob is the wrong bytes and a persisted ``sourceUri`` is the wrong
URL. Both are dropped here so the next ``push`` re-uploads instead of silently
shipping what the user replaced (plan decisions D-I, "derived-field rules").

Diff and merge live together because they share one notion of entry identity
and because ``update`` diffs the merge *result* against the stored document —
which is what makes "empty diff" mean exactly "the file will not change".

Pure: no Typer, no filesystem, no renderer beyond the one rendering helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from comfy_cli.command.build_spec import JsonObject, JsonValue

Change: TypeAlias = Literal["added", "removed", "changed"]
Status: TypeAlias = Literal["changed", "unchanged"]

__all__ = [
    "CategoryDiff",
    "DefinitionDiff",
    "EntryDiff",
    "diff_definitions",
    "merge_definition",
    "render_definition_diff",
    "summarize_definition_diff",
]

#: The row markers the build design fixes for the pretty table (lines 466-471).
_MARKS: Final[Mapping[Change, str]] = {"added": "+", "removed": "-", "changed": "~"}

#: Every change a collection entry can undergo, in the order both the counts
#: object and the summary line report them.
_CHANGES: Final[tuple[Change, ...]] = ("added", "removed", "changed")

#: The two ``definition`` collections and the fields identifying one entry in
#: each. Identity is what a person means by "the same model" / "the same pack":
#: a model is its placement target, a node is its folder. Deliberately NARROWER
#: than the canonical sort key in ``build_spec`` (which also carries ``sha256``
#: / ``id`` / ``repository``) — a model whose bytes moved must read as one
#: CHANGED row, not as an addition beside a removal.
_IDENTITY: Final[Mapping[str, tuple[str, ...]]] = {
    "models": ("type", "filename"),
    "customNodes": ("name",),
}

#: Every key the scanners own, per collection (``build.scan_models`` and
#: ``build.scan_custom_nodes``). A key outside these sets was written by
#: something other than a scan — ``push``, ``pull``, or a hand edit — and the
#: merge carries it over untouched.
_SCANNED_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "models": frozenset({"type", "filename", "localPath", "sha256", "sizeBytes", "source"}),
    "customNodes": frozenset(
        {
            "name",
            "localPath",
            "source",
            "repository",
            "gitRef",
            "id",
            "registryVersion",
            "localDigest",
            "localSizeBytes",
        }
    ),
}

#: The field whose value IS an entry's content identity. A model has no
#: ``localDigest``; its ``sha256`` plays that role.
_CONTENT_IDENTITY: Final[Mapping[str, str]] = {"models": "sha256", "customNodes": "localDigest"}

#: Cached references a content change makes wrong. Only a model can carry a
#: resolver-derived ``sourceUri``; a node has no public resolution path.
_INVALIDATED_BY_CONTENT: Final[Mapping[str, tuple[str, ...]]] = {
    "models": ("blobId", "sourceUri"),
    "customNodes": ("blobId",),
}

#: Definition-level keys the scan owns. Everything else at that level — an
#: unknown key a newer builder introduced, say — survives the rescan.
_SCANNED_DEFINITION_KEYS: Final[frozenset[str]] = frozenset(
    {"schema", "models", "customNodes", "baseComfyVersion", "pipDependencies", "environment"}
)

#: Distinguishes "absent" from a stored ``null`` when comparing scalars.
_ABSENT: Final = object()


@dataclass(frozen=True, slots=True)
class EntryDiff:
    """One affected ``models[]`` / ``customNodes[]`` entry."""

    change: Change
    name: str
    fields: tuple[str, ...] = ()

    def as_json(self) -> dict[str, object]:
        return {"change": self.change, "name": self.name, "fields": list(self.fields)}


@dataclass(frozen=True, slots=True)
class CategoryDiff:
    """Every affected entry in one collection, in identity order."""

    entries: tuple[EntryDiff, ...]

    def count(self, change: Change) -> int:
        return sum(1 for entry in self.entries if entry.change == change)

    def counts(self) -> dict[str, int]:
        return {change: self.count(change) for change in _CHANGES}

    def as_json(self) -> dict[str, object]:
        return {**self.counts(), "entries": [entry.as_json() for entry in self.entries]}


@dataclass(frozen=True, slots=True)
class DefinitionDiff:
    """Per-category counts plus the affected entries — the design's shape."""

    collections: Mapping[str, CategoryDiff]
    scalars: Mapping[str, Status]

    @property
    def is_empty(self) -> bool:
        return not any(category.entries for category in self.collections.values()) and all(
            status == "unchanged" for status in self.scalars.values()
        )

    def as_json(self) -> dict[str, object]:
        """The table's information, structurally — an agent parses no table.

        Collections carry counts plus entries; single-valued categories carry
        the bare ``"changed"``/``"unchanged"`` the design uses for them
        (builder-cli-design lines 451-458).
        """
        payload: dict[str, object] = {name: diff.as_json() for name, diff in self.collections.items()}
        payload.update(self.scalars)
        return payload

    def as_drift(self) -> dict[str, object]:
        """The same comparison as ``as_json``, reported as the design's drift.

        ``status`` answers *whether* the installation has moved and by how much,
        so it carries the per-category counts alone (builder-cli-design lines
        451-458). The affected entries stay with ``update``'s diff, which is
        about to act on them; a report nobody acts on does not need the list.
        """
        payload: dict[str, object] = {name: diff.counts() for name, diff in self.collections.items()}
        payload.update(self.scalars)
        return payload


def _entries(definition: Mapping[str, JsonValue], collection: str) -> list[JsonObject]:
    """The collection's entries.

    ``read_build_spec`` has already proven every entry is a mapping and the
    collection a list (``build_spec._sorted_entries`` refuses anything else),
    so the only shape this has to cope with is the key being absent.
    """
    value = definition.get(collection)
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def _identity(entry: JsonObject, collection: str) -> tuple[str, ...]:
    return tuple(str(entry.get(field) or "") for field in _IDENTITY[collection])


def _index(entries: Sequence[JsonObject], collection: str) -> dict[tuple[str, ...], list[JsonObject]]:
    """Group entries by identity.

    A group rather than a single entry because a hand-edited spec can hold two
    models sharing ``(type, filename)``; pairing them positionally keeps such a
    file diffable instead of turning it into a crash or an arbitrary winner.
    """
    grouped: dict[tuple[str, ...], list[JsonObject]] = {}
    for entry in entries:
        grouped.setdefault(_identity(entry, collection), []).append(entry)
    return grouped


def _merge_entry(stored: JsonObject, scanned: JsonObject, collection: str) -> JsonObject:
    """The scanned entry, plus every key the scanner does not own."""
    carried = {key: value for key, value in stored.items() if key not in _SCANNED_KEYS[collection]}
    identity_field = _CONTENT_IDENTITY[collection]
    if stored.get(identity_field) != scanned.get(identity_field):
        for field in _INVALIDATED_BY_CONTENT[collection]:
            carried.pop(field, None)
    return {**scanned, **carried}


def _merge_collection(
    stored: Sequence[JsonObject],
    scanned: Sequence[JsonObject],
    collection: str,
) -> list[JsonObject]:
    previous = _index(stored, collection)
    taken: dict[tuple[str, ...], int] = {}
    merged: list[JsonObject] = []
    for entry in scanned:
        identity = _identity(entry, collection)
        group = previous.get(identity, [])
        position = taken.get(identity, 0)
        taken[identity] = position + 1
        merged.append(_merge_entry(group[position], entry, collection) if position < len(group) else entry)
    return merged


def merge_definition(stored: Mapping[str, JsonValue], scanned: Mapping[str, JsonValue]) -> JsonObject:
    """Return the ``definition`` a rescan produces against ``stored``.

    The scan wins on every fact it reports; the stored document keeps the
    derived and unknown keys it alone knows about. An entry present only in the
    scan is added verbatim; an entry the scan no longer sees is dropped, since
    the whole command means "recompute the spec from the local installation".
    """
    carried = {key: value for key, value in stored.items() if key not in _SCANNED_DEFINITION_KEYS}
    merged: JsonObject = {**scanned, **carried}
    for collection in _IDENTITY:
        merged[collection] = list(
            _merge_collection(_entries(stored, collection), _entries(scanned, collection), collection)
        )
    return merged


def _changed_fields(stored: JsonObject, updated: JsonObject) -> tuple[str, ...]:
    keys = stored.keys() | updated.keys()
    return tuple(sorted(key for key in keys if stored.get(key) != updated.get(key)))


def _diff_collection(stored: Sequence[JsonObject], updated: Sequence[JsonObject], collection: str) -> CategoryDiff:
    previous = _index(stored, collection)
    current = _index(updated, collection)
    entries: list[EntryDiff] = []
    for identity in sorted(previous.keys() | current.keys()):
        before = previous.get(identity, [])
        after = current.get(identity, [])
        name = "/".join(part for part in identity if part) or "?"
        entries.extend(
            EntryDiff("changed", name, fields)
            for fields in (_changed_fields(old, new) for old, new in zip(before, after))
            if fields
        )
        entries.extend(EntryDiff("added", name) for _ in after[len(before) :])
        entries.extend(EntryDiff("removed", name) for _ in before[len(after) :])
    return CategoryDiff(entries=tuple(entries))


def _diff_scalars(stored: Mapping[str, JsonValue], updated: Mapping[str, JsonValue]) -> dict[str, Status]:
    """Every non-collection category, so an unknown key is reported too."""
    keys = (stored.keys() | updated.keys()) - set(_IDENTITY)
    return {
        key: ("unchanged" if stored.get(key, _ABSENT) == updated.get(key, _ABSENT) else "changed")
        for key in sorted(keys)
    }


def diff_definitions(stored: Mapping[str, JsonValue], updated: Mapping[str, JsonValue]) -> DefinitionDiff:
    """Compare the document about to be written against the one on disk."""
    return DefinitionDiff(
        collections={
            collection: _diff_collection(_entries(stored, collection), _entries(updated, collection), collection)
            for collection in _IDENTITY
        },
        scalars=_diff_scalars(stored, updated),
    )


def summarize_definition_diff(diff: DefinitionDiff) -> str:
    """The one-line summary printed under the table and echoed in the prompt."""
    if diff.is_empty:
        return "no changes"
    parts = [
        f"{name} +{category.count('added')} -{category.count('removed')} ~{category.count('changed')}"
        for name, category in diff.collections.items()
    ]
    parts.extend(name for name, status in diff.scalars.items() if status == "changed")
    return ", ".join(parts)


def render_definition_diff(renderer, diff: DefinitionDiff) -> None:
    """The pretty view: a Rich table grouped by category, then the summary."""
    from rich.table import Table

    table = Table(title="Pending spec changes")
    table.add_column("", style="bold", no_wrap=True)
    table.add_column("category", style="cyan", no_wrap=True)
    table.add_column("entry", style="white")
    table.add_column("fields", style="dim")
    for name, category in diff.collections.items():
        for entry in category.entries:
            table.add_row(_MARKS[entry.change], name, entry.name, ", ".join(entry.fields))
    for name, status in diff.scalars.items():
        if status == "changed":
            table.add_row(_MARKS["changed"], name, "", "")
    renderer.console().print(table)
    renderer.print(summarize_definition_diff(diff))
