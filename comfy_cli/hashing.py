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
