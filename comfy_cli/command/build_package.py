"""Deterministic local custom-node packaging."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

_EXCLUDED_DIRECTORIES: Final = frozenset({".git", "__pycache__"})
_FIXED_MTIME: Final = (1980, 1, 1, 0, 0, 0)
_FIXED_MODE: Final = stat.S_IFREG | 0o644


@dataclass(frozen=True, slots=True)
class NodePackageError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"cannot package {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class NodePackage:
    """One node's archive, its content identity, and what packaging left out.

    Identity travels *with* the bytes rather than being recomputed from the
    path, so a caller can never pair one archive with another archive's digest.
    """

    payload: bytes
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


def _package_files(root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Return ``(members, skipped_symlinks)`` for *root*, or raise.

    Symlinks are excluded from the archive by design (they are not portable
    content), but they are *reported* rather than dropped in silence — a node
    that vendors its dependencies through a symlink would otherwise package to
    a near-empty zip and succeed.
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


def package_node(path: Path) -> NodePackage:
    """Build one custom-node archive and its identity in a single pass."""
    root = path.expanduser()
    try:
        mode = os.lstat(root).st_mode
    except OSError as error:
        raise _unreadable(root, root, error) from error
    if stat.S_ISLNK(mode):
        raise NodePackageError(root, "node root must not be a symlink")
    if not stat.S_ISDIR(mode):
        raise NodePackageError(root, "node root must be a directory")

    members, skipped = _package_files(root)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for member_name, member_path in members:
            info = zipfile.ZipInfo(member_name, date_time=_FIXED_MTIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _FIXED_MODE << 16
            try:
                contents = member_path.read_bytes()
            except OSError as error:
                raise _unreadable(root, member_path, error) from error
            archive.writestr(info, contents)
    payload = output.getvalue()
    return NodePackage(payload, hashlib.sha256(payload).hexdigest(), len(payload), tuple(skipped))
