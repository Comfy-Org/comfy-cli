from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from comfy_cli.command.build_digest_cache import ModelDigestCache, cache_path


class _CountingDigest:
    """The real hash, plus a count of how many times the file was actually read."""

    def __init__(self) -> None:
        self.reads: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.reads.append(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "base.safetensors"
    path.write_bytes(b"MODEL")
    return path


def test_an_unchanged_file_is_hashed_once_across_separate_commands(model: Path, tmp_path: Path) -> None:
    """The memo has to outlive the process, or a no-op re-push re-reads everything."""
    # Given
    digest = _CountingDigest()
    memo = tmp_path / "digests.json"
    expected = hashlib.sha256(b"MODEL").hexdigest()

    # When
    first = ModelDigestCache(digest, memo)(model)
    second = ModelDigestCache(digest, memo)(model)

    # Then
    assert (first, second) == (expected, expected)
    assert digest.reads == [model]


@pytest.mark.parametrize(
    "replacement",
    [pytest.param(b"CHANGED-AND-LONGER", id="different size"), pytest.param(b"OTHER", id="identical size")],
)
def test_edited_content_is_hashed_again(model: Path, tmp_path: Path, replacement: bytes) -> None:
    # Given
    digest = _CountingDigest()
    memo = tmp_path / "digests.json"
    cache = ModelDigestCache(digest, memo)
    cache(model)

    # When
    model.write_bytes(replacement)
    reread = ModelDigestCache(digest, memo)(model)

    # Then
    assert reread == hashlib.sha256(replacement).hexdigest()
    assert digest.reads == [model, model]


def test_a_moved_mtime_alone_is_enough_to_hash_again(model: Path, tmp_path: Path) -> None:
    """The stamp is compared whole: this is a cache, never a content bypass."""
    # Given
    digest = _CountingDigest()
    memo = tmp_path / "digests.json"
    ModelDigestCache(digest, memo)(model)

    # When
    stamp = model.stat()
    os.utime(model, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    ModelDigestCache(digest, memo)(model)

    # Then
    assert digest.reads == [model, model]


def test_a_corrupt_memo_falls_back_to_hashing(model: Path, tmp_path: Path) -> None:
    # Given
    digest = _CountingDigest()
    memo = tmp_path / "digests.json"
    memo.write_bytes(b"\xff not json at all")

    # When
    result = ModelDigestCache(digest, memo)(model)

    # Then
    assert result == hashlib.sha256(b"MODEL").hexdigest()
    assert digest.reads == [model]


def test_entries_for_deleted_files_are_dropped_when_the_memo_is_read(model: Path, tmp_path: Path) -> None:
    """Otherwise the memo grows for the life of the machine."""
    # Given
    digest = _CountingDigest()
    memo = tmp_path / "digests.json"
    doomed = tmp_path / "gone.safetensors"
    doomed.write_bytes(b"TEMPORARY")
    cache = ModelDigestCache(digest, memo)
    cache(model)
    cache(doomed)

    # When
    doomed.unlink()
    ModelDigestCache(digest, memo)(model)

    # Then
    assert str(doomed) not in memo.read_text(encoding="utf-8")
    assert str(model.resolve()) in memo.read_text(encoding="utf-8")


def test_the_memo_lands_in_the_shared_cli_cache_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    # When
    resolved = cache_path()

    # Then
    assert resolved.parent == tmp_path / "cache" / "comfy-cli"
