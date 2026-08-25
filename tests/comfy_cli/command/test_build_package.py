from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from comfy_cli.command.build_package import NodePackageError, package_node

posix_permissions = pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
    reason="needs POSIX mode bits that root ignores",
)


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
    first = package_node(node).payload
    second = package_node(node).payload

    # Then
    assert first == second


def test_content_identity_is_independent_of_absolute_root(tmp_path: Path) -> None:
    # Given
    first_node = _node_tree(tmp_path / "first")
    second_node = _node_tree(tmp_path / "different" / "depth")

    # When
    first = package_node(first_node).sha256
    second = package_node(second_node).sha256

    # Then
    assert first == second


def test_content_identity_changes_when_file_content_changes(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    before = package_node(node).sha256

    # When
    (node / "nested" / "data.txt").write_bytes(b"CHANGED")
    after = package_node(node).sha256

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
    archive = package_node(node).payload

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
        package_node(linked_root)


def test_identity_describes_the_payload_it_is_returned_with(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)

    # When
    package = package_node(node)

    # Then
    assert package.sha256 == hashlib.sha256(package.payload).hexdigest()
    assert package.size_bytes == len(package.payload)


def test_skipped_symlinks_are_reported_rather_than_dropped_in_silence(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    (node / "target.txt").write_text("target", encoding="utf-8")
    os.symlink(node / "target.txt", node / "linked.txt")
    os.symlink(node / "nested", node / "vendor")

    # When
    package = package_node(node)

    # Then
    assert package.skipped_symlinks == ("linked.txt", "vendor")
    with zipfile.ZipFile(io.BytesIO(package.payload)) as archive:
        assert "linked.txt" not in archive.namelist()
        assert "vendor/data.txt" not in archive.namelist()


@posix_permissions
def test_an_untraversable_directory_aborts_instead_of_shrinking_the_archive(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    vendor = node / "vendor"
    vendor.mkdir()
    (vendor / "lib.py").write_bytes(b"LIB")
    os.chmod(vendor, 0o000)

    # When / Then
    try:
        with pytest.raises(NodePackageError, match="vendor could not be read"):
            package_node(node)
    finally:
        os.chmod(vendor, 0o755)


@posix_permissions
def test_an_unreadable_file_raises_the_packaging_error_rather_than_escaping(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    secret = node / "nested" / "secret.py"
    secret.write_bytes(b"SECRET")
    os.chmod(secret, 0o000)

    # When / Then
    try:
        with pytest.raises(NodePackageError, match="nested/secret.py could not be read"):
            package_node(node)
    finally:
        os.chmod(secret, 0o644)
