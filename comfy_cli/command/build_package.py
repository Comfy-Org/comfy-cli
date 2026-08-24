"""Deterministic local custom-node packaging."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_EXCLUDED_DIRECTORIES: Final = frozenset({".git", "__pycache__"})
_FIXED_MTIME: Final = (1980, 1, 1, 0, 0, 0)
_FIXED_MODE: Final = stat.S_IFREG | 0o644


@dataclass(frozen=True, slots=True)
class NodePackageError(ValueError):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"cannot package {self.path}: {self.reason}"


def _package_files(root: Path) -> list[tuple[str, Path]]:
    members: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if name not in _EXCLUDED_DIRECTORIES and not (directory / name).is_symlink()
        )
        for name in sorted(filenames):
            path = directory / name
            if name.endswith(".pyc") or path.is_symlink():
                continue
            try:
                mode = path.stat(follow_symlinks=False).st_mode
            except OSError:
                continue
            if stat.S_ISREG(mode):
                members.append((path.relative_to(root).as_posix(), path))
    members.sort(key=lambda member: member[0])
    return members


def build_node_package(path: Path) -> bytes:
    """Return the canonical ZIP_STORED archive for one custom-node directory."""
    root = path.expanduser()
    if root.is_symlink():
        raise NodePackageError(root, "node root must not be a symlink")
    if not root.is_dir():
        raise NodePackageError(root, "node root must be a directory")

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for member_name, member_path in _package_files(root):
            info = zipfile.ZipInfo(member_name, date_time=_FIXED_MTIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _FIXED_MODE << 16
            archive.writestr(info, member_path.read_bytes())
    return output.getvalue()


def node_content_identity(path: Path) -> tuple[str, int]:
    """Return ``(sha256, size)`` for a node's canonical package."""
    package = build_node_package(path)
    return hashlib.sha256(package).hexdigest(), len(package)
