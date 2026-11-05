import hashlib
from pathlib import Path
from typing import Final

from blake3 import blake3

_CHUNK_SIZE: Final = 1024 * 1024


def blake3_file(path: Path) -> str:
    h = blake3()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return f"blake3:{h.hexdigest()}"


def blake3_and_sha256_file(path: Path) -> tuple[str, str]:
    """``("blake3:<hex>", "<sha256 hex>")`` for the file, read once.

    BLAKE3 is the hash Comfy Cloud stores a model under; SHA-256 is the one a
    person can check on any machine (``shasum -a 256``, ``certutil``).
    """
    b, s = blake3(), hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            b.update(chunk)
            s.update(chunk)
    return f"blake3:{b.hexdigest()}", s.hexdigest()
