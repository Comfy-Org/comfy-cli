"""Local cache reconciliation and upload planning for ``comfy build push``."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol
from urllib.parse import urlsplit

from typing_extensions import assert_never

from comfy_cli.command.build_package import package_node
from comfy_cli.command.build_paths import BuildPaths, resolve_local_path
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, JsonValue
from comfy_cli.command.build_validation import project_wire_definition

_SCP_REPOSITORY: Final = re.compile(r"^(?P<user>[^@/]+)@(?P<host>[^:/]+):(?P<path>.+)$")


class ModelDigest(Protocol):
    def __call__(self, path: Path) -> str: ...


class BlobClient(Protocol):
    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str | None]: ...

    def upload_blob(self, upload_url: str, path: Path) -> None: ...


class Persist(Protocol):
    def __call__(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PushUpload:
    """One blob this push still owes the builder, and the file that holds it.

    ``path`` is the bytes to PUT: the model file itself for a ``model``, the
    packaged archive for a ``node_zip``. It is ``None`` only when the plan was
    drawn without a ``package_dir`` — `--dry-run` and `pull` want the count and
    the byte total, and materializing archives neither will send is the memory
    cost this planner exists to avoid.
    """

    kind: Literal["model", "node_zip"]
    collection: Literal["models", "customNodes"]
    index: int
    filename: str
    sha256: str
    size_bytes: int
    path: Path | None


@dataclass(frozen=True, slots=True)
class SkippedSymlink:
    """One symlink packaging left out of a node archive, and where it lives.

    ``location`` points into the **local** spec's definition, which is the order
    `init`, `update` and `push` go on to report. It is not portable to a
    definition assembled somewhere else: `build pull` ships the server's node
    order and omits local nodes the server lacks, so it must renumber these rows
    against what it actually emits (``build._relocate_skipped_symlinks``).
    ``local_path`` needs no such fixup and identifies the node either way.
    """

    location: str
    local_path: str
    member: str

    def as_row(self) -> JsonObject:
        return {"location": self.location, "localPath": self.local_path, "member": self.member}


@dataclass(frozen=True, slots=True)
class PushPreparation:
    spec: JsonObject
    uploads: tuple[PushUpload, ...]
    stale_source_uri_models: tuple[int, ...]
    skipped_symlinks: tuple[SkippedSymlink, ...] = ()

    @property
    def definition(self) -> JsonObject:
        definition = self.spec["definition"]
        if not isinstance(definition, dict):
            raise BuildSpecInvalidError("definition must be a mapping")
        return definition


@dataclass(frozen=True, slots=True)
class PublicNodeIdentity:
    kind: Literal["registry", "repository"]
    value: str
    version: str | None = None


def _entries(definition: JsonObject, collection: str) -> list[JsonObject]:
    values = definition.get(collection, [])
    if not isinstance(values, list):
        raise BuildSpecInvalidError(f"definition.{collection} must be a list")
    entries: list[JsonObject] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise BuildSpecInvalidError(f"definition.{collection}[{index}] must be a mapping")
        entries.append(value)
    return entries


def _local_path(entry: JsonObject, root: Path, *, location: str) -> Path:
    value = entry.get("localPath")
    if not isinstance(value, str) or not value:
        raise BuildSpecInvalidError(f"{location}.localPath is required when source is 'local'")
    return resolve_local_path(root, value, entry=location)


def _has_text(entry: JsonObject, key: str) -> bool:
    value = entry.get(key)
    return isinstance(value, str) and bool(value.strip())


def prepare_push(
    spec: JsonObject, paths: BuildPaths, model_digest: ModelDigest, *, package_dir: Path | None = None
) -> PushPreparation:
    """Reconcile the spec against local bytes and plan what still has to upload.

    ``package_dir`` is the scratch directory node archives are written into, and
    naming one is what makes the returned plan *uploadable*. Callers that want
    only the reconciled spec and the upload totals — `--dry-run`, and `pull`,
    which reads ``spec`` and never ``uploads`` — leave it unset, and every
    archive is discarded as soon as its digest is known.
    """
    prepared = deepcopy(spec)
    definition = prepared.get("definition")
    if not isinstance(definition, dict):
        raise BuildSpecInvalidError("definition must be a mapping")
    uploads: list[PushUpload] = []
    stale_source_uris: list[int] = []
    skipped_symlinks: list[SkippedSymlink] = []

    for index, model in enumerate(_entries(definition, "models")):
        if model.get("source") != "local":
            continue
        location = f"definition.models[{index}]"
        path = _local_path(model, paths.models_dir, location=location)
        try:
            digest = model_digest(path)
            size = path.stat().st_size
        except OSError as error:
            raise BuildSpecInvalidError(f"{location} could not be read: {error}") from error
        if model.get("sha256") != digest:
            model.pop("blobId", None)
            if "sourceUri" in model:
                stale_source_uris.append(index)
                model.pop("sourceUri", None)
        model["sha256"] = digest
        model["sizeBytes"] = size
        if not _has_text(model, "blobId") and not _has_text(model, "sourceUri"):
            uploads.append(
                PushUpload("model", "models", index, str(model.get("filename") or path.name), digest, size, path)
            )

    for index, node in enumerate(_entries(definition, "customNodes")):
        if node.get("source") != "local":
            continue
        location = f"definition.customNodes[{index}]"
        path = _local_path(node, paths.custom_nodes_dir, location=location)
        # One build, one identity: packaging twice lets the directory change
        # between calls and pairs these bytes with another archive's digest.
        # `NodePackageError` propagates: it already names the node directory the
        # user must fix, which a `BuildSpecInvalidError` here would relabel with
        # the spec file's path at the call site.
        package = package_node(path, package_dir / f"node-{index}.zip" if package_dir is not None else None)
        digest, size = package.sha256, package.size_bytes
        # `_local_path` above already proved localPath is a non-empty str.
        local_path = str(node["localPath"])
        skipped_symlinks.extend(SkippedSymlink(location, local_path, member) for member in package.skipped_symlinks)
        if node.get("localDigest") != digest:
            node.pop("blobId", None)
        node["localDigest"] = digest
        node["localSizeBytes"] = size
        if not _has_text(node, "blobId"):
            name = str(node.get("name") or f"node-{index}")
            uploads.append(
                PushUpload("node_zip", "customNodes", index, f"{name}.zip", digest, size, package.archive_path)
            )

    return PushPreparation(prepared, tuple(uploads), tuple(stale_source_uris), tuple(skipped_symlinks))


def pending_uploads(preparation: PushPreparation) -> tuple[PushUpload, ...]:
    """The planned uploads whose bytes the builder does not already hold.

    Read against the *current* definition rather than the plan, because
    ``upload_assets`` writes each ``blobId`` back into it as the blob lands. A
    node archive was previously exempt from that re-read, so asking a second
    time — after a partial run, or simply to count what is left — reported every
    completed node as still pending.
    """
    definition = preparation.definition
    entries = {collection: _entries(definition, collection) for collection in ("models", "customNodes")}
    return tuple(
        upload
        for upload in preparation.uploads
        if _still_unresolved(entries[upload.collection][upload.index], upload.kind)
    )


def _still_unresolved(entry: JsonObject, kind: Literal["model", "node_zip"]) -> bool:
    """Whether *entry* still needs its bytes uploaded.

    The two kinds resolve differently, exactly as ``prepare_push`` plans them: a
    model is satisfied by a public ``sourceUri`` as well as a ``blobId``, while a
    packaged node archive is only ever satisfied by a ``blobId``.
    """
    if _has_text(entry, "blobId"):
        return False
    match kind:
        case "model":
            return not _has_text(entry, "sourceUri")
        case "node_zip":
            return True
        case unreachable:
            assert_never(unreachable)


def unresolved_models(preparation: PushPreparation) -> list[JsonObject]:
    models = _entries(preparation.definition, "models")
    indexes = {upload.index for upload in preparation.uploads if upload.kind == "model"}
    return [model for index, model in enumerate(models) if index in indexes and _still_unresolved(model, "model")]


def spec_without_stale_source_uris(original: JsonObject, preparation: PushPreparation) -> JsonObject:
    cleaned = deepcopy(original)
    definition = cleaned.get("definition")
    if not isinstance(definition, dict):
        raise BuildSpecInvalidError("definition must be a mapping")
    models = _entries(definition, "models")
    for index in preparation.stale_source_uri_models:
        models[index].pop("sourceUri", None)
    return cleaned


def upload_assets(preparation: PushPreparation, client: BlobClient, persist: Persist | None = None) -> int:
    """Upload every pending blob, checkpointing each id the moment it lands.

    The spec file is this command's resume store — ``prepare_push`` skips any
    entry that already carries a ``blobId`` — so *when* it is written decides
    whether an interrupted push resumes or starts over. ``persist`` is the
    caller's atomic write of the reconciled spec, and it runs per blob rather
    than once at the end because the builder mints a fresh id per
    ``create_blob`` for content it does not already hold, so an id lost with the
    process is not merely re-uploaded — it orphans the bytes stored under the
    first one.

    Returns the number of blobs whose bytes were actually transferred. A
    builder that already holds the content answers ``create_blob`` with the
    existing id and no upload URL, so the pending count is an upper bound.
    """
    uploaded = 0
    uploads = pending_uploads(preparation)
    definition = preparation.definition
    for upload in uploads:
        if upload.path is None:
            # A ValueError, not a BuildSpecInvalidError: `build._builder_call`
            # wraps this call and maps the former to a CLI error, while the
            # latter is not in its taxonomy and would escape as a traceback.
            raise ValueError(
                f"{upload.filename} was planned without an archive to upload; "
                "prepare_push needs a package_dir to materialize node archives"
            )
        blob_id, upload_url = client.create_blob(upload.kind, upload.filename, upload.sha256, upload.size_bytes)
        if upload_url is not None:
            client.upload_blob(upload_url, upload.path)
            uploaded += 1
        _entries(definition, upload.collection)[upload.index]["blobId"] = blob_id
        if persist is not None:
            persist()
    return uploaded


def normalize_repository_identity(repository: str) -> str:
    raw = repository.strip()
    scp = _SCP_REPOSITORY.fullmatch(raw)
    if scp is not None:
        host = scp.group("host").lower()
        path = scp.group("path")
    else:
        try:
            parsed = urlsplit(raw)
            host = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            return raw.rstrip("/").removesuffix(".git")
        if not host:
            return raw.rstrip("/").removesuffix(".git")
        host += f":{port}" if port is not None else ""
        path = parsed.path
    normalized_path = path.strip("/").removesuffix(".git")
    return f"https://{host}/{normalized_path}"


def public_node_projection(definition: JsonObject) -> list[JsonObject]:
    original = _entries(definition, "customNodes")
    node_values: list[JsonValue] = [node for node in original]
    projected = project_wire_definition({"customNodes": node_values})
    wire = _entries(projected, "customNodes")
    return [
        node
        for source, node in zip(original, wire)
        if source.get("source") != "local" and (_has_text(node, "registryVersion") or _has_text(node, "repository"))
    ]


def public_node_identities(nodes: list[JsonObject]) -> set[PublicNodeIdentity]:
    identities: set[PublicNodeIdentity] = set()
    for index, node in enumerate(nodes):
        if _has_text(node, "registryVersion"):
            node_id = node.get("id")
            version = node.get("registryVersion")
            if not isinstance(node_id, str) or not node_id.strip() or not isinstance(version, str):
                raise BuildSpecInvalidError(f"public customNodes[{index}] needs id + registryVersion")
            identities.add(PublicNodeIdentity("registry", node_id, version))
        elif _has_text(node, "repository"):
            repository = node["repository"]
            assert isinstance(repository, str)
            identities.add(PublicNodeIdentity("repository", normalize_repository_identity(repository)))
    return identities
