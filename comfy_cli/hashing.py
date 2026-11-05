import hashlib
from pathlib import Path
from typing import Final, NamedTuple

from blake3 import blake3

_CHUNK_SIZE: Final = 1024 * 1024


def blake3_file(path: Path) -> str:
    h = blake3()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return f"blake3:{h.hexdigest()}"


class FileDigests(NamedTuple):
    """A file's hashes and size from one read, and the BLAKE3 digest of each block."""

    blake3: str
    sha256: str
    size: int
    block_size: int
    blocks: tuple[bytes, ...]


def file_digests(path: Path, block_size: int | None = None) -> FileDigests:
    """Hash the file once: ``blake3`` (``"blake3:<hex>"``), ``sha256`` (hex) and each block's BLAKE3 digest.

    BLAKE3 is the hash Comfy Cloud stores a model under; SHA-256 is the one a
    person can check on any machine (``shasum -a 256``, ``certutil``). The
    block digests let a later read of the same path be checked block by block,
    before each block is used.
    """
    size_of_block = block_size or _CHUNK_SIZE
    b, s, size, blocks = blake3(), hashlib.sha256(), 0, []
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(size_of_block), b""):
            b.update(chunk)
            s.update(chunk)
            size += len(chunk)
            blocks.append(blake3(chunk).digest())
    return FileDigests(f"blake3:{b.hexdigest()}", s.hexdigest(), size, size_of_block, tuple(blocks))
