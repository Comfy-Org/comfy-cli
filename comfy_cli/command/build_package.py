"""Deterministic local custom-node packaging."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final, NoReturn

_EXCLUDED_DIRECTORIES: Final = frozenset({".git", "__pycache__"})
_FIXED_MTIME: Final = (1980, 1, 1, 0, 0, 0)
_FIXED_MODE: Final = stat.S_IFREG | 0o644
# Bounds the resident cost of one member and of one hashing read. A node that
# vendors a multi-GB weight file is the case this exists for.
_CHUNK_BYTES: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class NodePackageError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"cannot package {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class NodePackage:
    """One node's archive, its content identity, and what packaging left out.

    Identity travels *with* the archive rather than being recomputed from the
    path, so a caller can never pair one archive with another archive's digest.

    ``archive_path`` is ``None`` when the caller named no destination: the
    archive was still produced (there is no way to know a node's identity
    without it) but into a file discarded on return, so a caller that wants only
    ``sha256``/``size_bytes`` pays no disk and no memory for bytes it will not
    send.
    """

    archive_path: Path | None
    sha256: str
    size_bytes: int
    skipped_symlinks: tuple[str, ...]


def _member_name(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _unreadable(root: Path, path: Path | str | None, error: OSError) -> NodePackageError:
    """Turn any traversal or read failure into the packaging error contract.

    A member that *holds content* and cannot be read aborts the archive rather
    than shrinking it. A silently short archive would be hashed, written into
    the spec as the node's ``localDigest``, and released — surfacing only as an
    ImportError inside a deployed container, arbitrarily far from the
    unreadable file that caused it.

    This is about readability, not exhaustiveness: ``_package_files`` also
    leaves out entries that carry no packageable bytes at all — sockets, FIFOs,
    device nodes — and those are dropped without a report, because unlike a
    symlink none of them can stand in for vendored code.
    """
    detail = error.strerror or str(error)
    if path is None:
        return NodePackageError(root, f"could not be read: {detail}")
    target = Path(path)
    if target == root:
        return NodePackageError(root, f"could not be read: {detail}")
    return NodePackageError(root, f"{_member_name(root, target)} could not be read: {detail}")


def _excluded_member(root: Path, destination: Path | None) -> frozenset[str]:
    """The member name(s) *destination* occupies inside *root*, if any.

    The tree is walked *before* the archive is opened, so a destination that
    already exists under ``root`` would otherwise be collected as a member and
    then truncated by ``open("wb+")`` — and ``_write_archive`` would read the
    very file it is appending to, which never reaches EOF. The archive grows
    until the disk does not.

    Excluding it also makes a repeat packaging idempotent: the digest covers the
    node's content rather than the output of the previous run. Nothing is
    excluded when the destination is outside ``root`` — which is every caller in
    the tree, since ``prepare_push`` packages into a ``TemporaryDirectory`` — so
    no already-minted ``localDigest`` moves.

    Matched on the resolved path so that a relative path, a ``..`` segment or a
    symlinked parent cannot smuggle the destination past the comparison, *and*
    on the literal one so that a destination lexically inside the root but
    symlinked out of it is still recognised as the archive rather than reported
    to the user as vendored content packaging left behind.
    """
    if destination is None:
        return frozenset()
    names: set[str] = set()
    for root_path, destination_path in (
        (os.path.realpath(root), os.path.realpath(destination)),
        (root, destination),
    ):
        try:
            names.add(Path(destination_path).relative_to(root_path).as_posix())
        except ValueError:
            continue
    return frozenset(names)


def _package_files(root: Path, excluded: frozenset[str] = frozenset()) -> tuple[list[tuple[str, Path]], list[str]]:
    """Return ``(members, skipped_symlinks)`` for *root*, or raise.

    Symlinks are excluded from the archive by design (they are not portable
    content), but they are *reported* rather than dropped in silence — a node
    that vendors its dependencies through a symlink would otherwise package to
    a near-empty zip and succeed.

    ``excluded`` holds the member name(s) the archive is being written to; see
    ``_excluded_member``.
    """
    members: list[tuple[str, Path]] = []
    skipped: list[str] = []

    def fail(error: OSError) -> NoReturn:
        raise _unreadable(root, error.filename, error)

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=fail, followlinks=False):
        directory = Path(dirpath)
        kept: list[str] = []
        for name in sorted(dirnames):
            if name in _EXCLUDED_DIRECTORIES:
                continue
            path = directory / name
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise _unreadable(root, path, error) from error
            if stat.S_ISLNK(mode):
                skipped.append(_member_name(root, path))
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            path = directory / name
            # Above the lstat on purpose: an unreadable *output* file is not
            # this node's content, so it must not raise `_unreadable`.
            if _member_name(root, path) in excluded:
                continue
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise _unreadable(root, path, error) from error
            if stat.S_ISLNK(mode):
                skipped.append(_member_name(root, path))
            elif stat.S_ISREG(mode):
                members.append((_member_name(root, path), path))
    members.sort(key=lambda member: member[0])
    skipped.sort()
    return members, skipped


def _copy_declared(source: IO[bytes], target: IO[bytes], declared: int) -> int:
    """Copy from *source* to *target*, returning the byte count, never past *declared*.

    Bounding the read is what stops a member that grows while it is being
    archived from being followed forever: an unbounded copy against a file some
    other writer is still extending never reaches EOF, and the archive grows
    until the disk does. The caller compares the return against *declared*, so a
    file that changed under packaging aborts rather than landing in the archive
    at a length its header does not describe.
    """
    remaining = declared
    while remaining > 0:
        chunk = source.read(min(_CHUNK_BYTES, remaining))
        if not chunk:
            break
        target.write(chunk)
        remaining -= len(chunk)
    return declared - remaining


def _write_archive(root: Path, members: Sequence[tuple[str, Path]], stream: IO[bytes]) -> tuple[str, int]:
    """Stream the archive into *stream* and return ``(sha256, size_bytes)``.

    Two passes over one file rather than one pass over a buffer. The digest has
    to cover the *finished* zip, and the writer seeks backwards to fix up each
    member's header once its CRC and size are known, so bytes hashed on their
    way in are not the bytes the archive ends up holding. Re-reading costs one
    chunk of memory; buffering costs the whole archive, twice over once
    ``getvalue()`` copies it.

    Every field the determinism guarantee rests on is set here exactly as the
    buffered ``writestr`` set it, so the digest is unchanged for identical
    input — existing specs carry ``localDigest`` values minted the old way, and
    a drifted digest would invalidate every stored ``blobId``.
    """
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for member_name, member_path in members:
            info = zipfile.ZipInfo(member_name, date_time=_FIXED_MTIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _FIXED_MODE << 16
            try:
                # `writestr` set this from the payload it was handed, and the
                # zip64 decision reads it: left at 0 a member past 4 GiB aborts
                # with a RuntimeError instead of widening its header.
                # Held separately: closing the member writer rewrites
                # `info.file_size` to whatever was actually written.
                declared = member_path.stat().st_size
                info.file_size = declared
                with member_path.open("rb") as source, archive.open(info, "w") as target:
                    copied = _copy_declared(source, target, declared)
                    still_reading = bool(source.read(1))
            except OSError as error:
                raise _unreadable(root, member_path, error) from error
            if copied != declared or still_reading:
                raise NodePackageError(
                    root, f"{_member_name(root, member_path)} changed size while it was being packaged"
                )
    size_bytes = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest(), size_bytes


def package_node(path: Path, destination: Path | None = None) -> NodePackage:
    """Build one custom-node archive and its identity in a single pass.

    The archive is never resident in memory. ``destination`` names where the
    caller wants to keep it; with ``None`` it is still built — a node's identity
    is the digest of its bytes, so there is no shortcut — but into a file that is
    unlinked on return. ``prepare_push`` packages *every* local node before
    uploading any of them, so an archive retained per node makes peak cost the
    sum of the whole workspace's vendored content.
    """
    root = path.expanduser()
    try:
        mode = os.lstat(root).st_mode
    except OSError as error:
        raise _unreadable(root, root, error) from error
    if stat.S_ISLNK(mode):
        raise NodePackageError(root, "node root must not be a symlink")
    if not stat.S_ISDIR(mode):
        raise NodePackageError(root, "node root must be a directory")

    members, skipped = _package_files(root, _excluded_member(root, destination))
    if destination is None:
        with tempfile.TemporaryFile(prefix="comfy-node-package-") as scratch:
            sha256, size_bytes = _write_archive(root, members, scratch)
        return NodePackage(None, sha256, size_bytes, tuple(skipped))
    try:
        stream = destination.open("wb+")
    except OSError as error:
        detail = error.strerror or str(error)
        raise NodePackageError(root, f"archive could not be written to {destination}: {detail}") from error
    with stream:
        sha256, size_bytes = _write_archive(root, members, stream)
    return NodePackage(destination, sha256, size_bytes, tuple(skipped))
