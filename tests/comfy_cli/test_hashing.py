from pathlib import Path

from blake3 import blake3

from comfy_cli import hashing

_CHUNK_SIZE = 1024 * 1024


def test_blake3_file_matches_the_abc_known_answer(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")

    # When
    digest = hashing.blake3_file(path)

    # Then
    assert digest == "blake3:6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"


def test_blake3_file_matches_one_shot_hashing_across_multiple_chunks(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "multi-chunk.bin"
    path.write_bytes((b"comfy" * ((_CHUNK_SIZE * 2) // 5 + 1)) + b"tail")

    # When
    digest = hashing.blake3_file(path)

    # Then
    expected = f"blake3:{blake3(path.read_bytes()).hexdigest()}"
    assert digest == expected


def test_file_digests_reads_once_for_both_hashes_and_each_block(tmp_path: Path) -> None:
    # Given
    import hashlib

    path = tmp_path / "multi-chunk.bin"
    data = (b"comfy" * ((_CHUNK_SIZE * 2) // 5 + 1)) + b"tail"
    path.write_bytes(data)

    # When
    digests = hashing.file_digests(path)

    # Then
    assert digests.blake3 == hashing.blake3_file(path) == f"blake3:{blake3(data).hexdigest()}"
    assert digests.sha256 == hashlib.sha256(data).hexdigest()
    assert digests.size == len(data) and digests.block_size == _CHUNK_SIZE
    blocks = [data[i : i + _CHUNK_SIZE] for i in range(0, len(data), _CHUNK_SIZE)]
    assert len(blocks) == 3
    assert digests.blocks == tuple(blake3(b).digest() for b in blocks)


def test_file_digests_takes_a_block_size(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "small.bin"
    path.write_bytes(b"0123456789")

    # When
    digests = hashing.file_digests(path, block_size=4)

    # Then
    assert digests.block_size == 4 and digests.size == 10
    assert digests.blocks == tuple(blake3(b).digest() for b in (b"0123", b"4567", b"89"))
