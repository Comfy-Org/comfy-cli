"""On-disk memo of local model digests for ``comfy build``.

``prepare_push`` re-hashes every local model on every run *by design*: that read
is how content drift is detected, and how a ``blobId`` minted for older bytes is
dropped before it can be published. What it has no need to do is re-read tens of
gigabytes to conclude that nothing moved, which is what even a no-op re-push
costs. This narrows the read to files whose identity stamp actually changed,
without narrowing the check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

from comfy_cli.command.build_push import ModelDigest
from comfy_cli.file_utils import atomic_write_text

_CACHE_FILENAME: Final = "build-model-digests.json"


def cache_path() -> Path:
    """Where the memo lives. XDG-respecting, like every other comfy-cli cache."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / _CACHE_FILENAME


def _memoized(entry: Any, status: os.stat_result) -> str | None:
    """The recorded digest when *entry* still describes *status*, else ``None``."""
    if not isinstance(entry, dict):
        return None
    digest = entry.get("sha256")
    if not isinstance(digest, str) or not digest:
        return None
    if entry.get("mtimeNs") != status.st_mtime_ns or entry.get("sizeBytes") != status.st_size:
        return None
    return digest


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError):
        # ValueError covers JSONDecodeError *and* the UnicodeDecodeError a
        # non-UTF-8 (corrupt) file raises. The memo is regenerable either way.
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str)}


class ModelDigestCache:
    """A ``ModelDigest`` that reads a file only when its identity stamp moved.

    A hit requires the resolved path, ``st_mtime_ns`` **and** ``st_size`` to all
    still match what was hashed; anything else — including an edit that happens
    to preserve the size — misses and reads the file again. That asymmetry is
    the whole design: a miss costs one hash, while a wrong hit would leave a
    stale ``sha256`` in the spec, keep the ``blobId`` minted for the old bytes,
    and publish a release built from content nobody uploaded.

    Each miss is persisted as it happens rather than at the end of the command,
    for the same reason `push` checkpoints its blob ids: work already paid for
    should survive an interrupt.
    """

    def __init__(self, digest: ModelDigest, path: Path | None = None) -> None:
        self._digest = digest
        self._path = cache_path() if path is None else path
        stored = _read(self._path)
        # An entry whose file is gone can never be hit again. Pruning on read is
        # what keeps a long-lived memo bounded by what the machine still holds —
        # and it has to be written back here, because a run that is all hits
        # never reaches `_save` and would carry the dead weight forever.
        self._entries = {key: value for key, value in stored.items() if Path(key).exists()}
        if len(self._entries) != len(stored):
            self._save()

    def __call__(self, path: Path) -> str:
        status = path.stat()
        key = str(path.resolve())
        memoized = _memoized(self._entries.get(key), status)
        if memoized is not None:
            return memoized
        digest = self._digest(path)
        self._entries[key] = {"mtimeNs": status.st_mtime_ns, "sizeBytes": status.st_size, "sha256": digest}
        self._save()
        return digest

    def _save(self) -> None:
        try:
            atomic_write_text(self._path, json.dumps(self._entries), fsync=False)
        except OSError:
            # A read-only cache dir must slow the next push down, never fail it.
            pass
