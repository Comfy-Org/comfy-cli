from __future__ import annotations

import io
import os
import stat
import zipfile
from pathlib import Path

import pytest

from comfy_cli.command.build_package import build_node_package, node_content_identity


def _node_tree(root: Path) -> Path:
    node = root / "node"
    (node / "nested").mkdir(parents=True)
    (node / "__init__.py").write_bytes(b"NODE")
    (node / "nested" / "data.txt").write_bytes(b"DATA")
    return node


def test_package_is_byte_identical_across_repeated_builds(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)

    # When
    first = build_node_package(node)
    second = build_node_package(node)

    # Then
    assert first == second


def test_content_identity_is_independent_of_absolute_root(tmp_path: Path) -> None:
    # Given
    first_node = _node_tree(tmp_path / "first")
    second_node = _node_tree(tmp_path / "different" / "depth")

    # When
    first = node_content_identity(first_node)
    second = node_content_identity(second_node)

    # Then
    assert first == second


def test_content_identity_changes_when_file_content_changes(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    before = node_content_identity(node)

    # When
    (node / "nested" / "data.txt").write_bytes(b"CHANGED")
    after = node_content_identity(node)

    # Then
    assert after != before


def test_package_has_exact_members_and_fixed_metadata(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    (node / ".git").mkdir()
    (node / ".git" / "config").write_text("secret", encoding="utf-8")
    (node / "__pycache__").mkdir()
    (node / "__pycache__" / "cached.pyc").write_bytes(b"cache")
    (node / "root.pyc").write_bytes(b"cache")
    (node / "target.txt").write_text("target", encoding="utf-8")
    os.symlink(node / "target.txt", node / "linked.txt")
    os.symlink(node / "nested", node / "linked-dir")

    # When
    archive = build_node_package(node)

    # Then
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        assert package.namelist() == ["__init__.py", "nested/data.txt", "target.txt"]
        for member in package.infolist():
            assert member.compress_type == zipfile.ZIP_STORED
            assert member.date_time == (1980, 1, 1, 0, 0, 0)
            assert stat.S_IFMT(member.external_attr >> 16) == stat.S_IFREG
            assert stat.S_IMODE(member.external_attr >> 16) == 0o644


def test_symlinked_node_root_is_rejected(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    linked_root = tmp_path / "linked-node"
    os.symlink(node, linked_root)

    # When / Then
    with pytest.raises(ValueError, match="node root.*symlink"):
        build_node_package(linked_root)
