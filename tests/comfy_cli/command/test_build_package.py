from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import tempfile
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


def _archive_bytes(node: Path, destination: Path) -> bytes:
    package = package_node(node, destination)
    assert package.archive_path == destination
    return destination.read_bytes()


def _buffered_archive(root: Path) -> bytes:
    """The pre-streaming implementation, kept here as an independent oracle.

    Specs already in the wild carry ``localDigest`` values this exact algorithm
    minted, and a digest that drifted would invalidate every ``blobId`` stored
    beside them. Pinning that against a second call to the code under test would
    prove only that it agrees with itself, so the guarantee is pinned against a
    reimplementation of the old algorithm over the same input.
    """
    members = sorted((path.relative_to(root).as_posix(), path) for path in root.rglob("*") if path.is_file())
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for member_name, member_path in members:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, member_path.read_bytes())
    return output.getvalue()


def test_package_is_byte_identical_across_repeated_builds(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)

    # When
    first = _archive_bytes(node, tmp_path / "first.zip")
    second = _archive_bytes(node, tmp_path / "second.zip")

    # Then
    assert first == second


def test_streaming_to_a_destination_reproduces_the_buffered_digest(tmp_path: Path) -> None:
    """`localDigest` values minted before packaging streamed must still verify."""
    # Given: a tree whose member order differs between path sort and name sort,
    # which is where a rewritten walk would silently reorder the archive.
    node = _node_tree(tmp_path)
    (node / "a").mkdir()
    (node / "a" / "b.txt").write_bytes(b"NESTED")
    (node / "a.txt").write_bytes(b"SIBLING")
    expected = _buffered_archive(node)
    destination = tmp_path / "node.zip"

    # When
    package = package_node(node, destination)

    # Then
    assert destination.read_bytes() == expected
    assert package.sha256 == hashlib.sha256(expected).hexdigest()
    assert package.size_bytes == len(expected)


def test_packaging_without_a_destination_retains_no_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity needs the bytes; keeping them is what `prepare_push` cannot afford."""
    # Given
    node = _node_tree(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    expected = hashlib.sha256(_buffered_archive(node)).hexdigest()

    # When
    package = package_node(node)

    # Then
    assert package.archive_path is None
    assert package.sha256 == expected
    assert list(scratch.iterdir()) == []


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
    archive = _archive_bytes(node, tmp_path / "node.zip")

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


def test_identity_describes_the_archive_it_is_returned_with(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    destination = tmp_path / "node.zip"

    # When
    package = package_node(node, destination)

    # Then
    written = destination.read_bytes()
    assert package.sha256 == hashlib.sha256(written).hexdigest()
    assert package.size_bytes == len(written)


def test_skipped_symlinks_are_reported_rather_than_dropped_in_silence(tmp_path: Path) -> None:
    # Given
    node = _node_tree(tmp_path)
    (node / "target.txt").write_text("target", encoding="utf-8")
    os.symlink(node / "target.txt", node / "linked.txt")
    os.symlink(node / "nested", node / "vendor")

    # When
    destination = tmp_path / "node.zip"
    package = package_node(node, destination)

    # Then
    assert package.skipped_symlinks == ("linked.txt", "vendor")
    with zipfile.ZipFile(destination) as archive:
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
        os.chmod(vendor, 0o700)


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
        os.chmod(secret, 0o600)
