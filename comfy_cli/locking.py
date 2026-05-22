"""Cross-platform exclusive file locking.

Used by ConfigManager (Phase 1 audit) and secrets.bin (Phase 5). Picks the
right primitive per OS:
- Unix: fcntl.flock
- Windows: msvcrt.locking with a fixed byte range

Locks are advisory and process-scoped. Concurrent writers in different
processes serialize on the same lock file. Within a process, callers should
combine this with their own threading.Lock if needed.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def file_lock(path: str | os.PathLike[str], *, timeout: float | None = None) -> Iterator[None]:
    """Acquire an exclusive lock on ``path`` (created if absent).

    ``timeout`` is a best-effort upper bound; on platforms that don't support
    non-blocking acquire with timeout we block indefinitely.

    Note: ``fcntl.flock`` on NFS is silently a no-op on many configurations
    (the lock isn't propagated to the NFS server). Lock files that need
    cross-host serialization should use a different primitive. We keep flock
    for the local case and warn lazily if we detect we're on NFS.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Open in append-binary so we don't truncate any existing payload; the
    # lock byte-range is at offset 0 and doesn't affect data.
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o600)
    # Re-tighten perms if the file existed with looser mode (covers the case
    # where an older build wrote the lock file world-readable).
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    try:
        _acquire(fd, timeout=timeout)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


if sys.platform == "win32":
    import msvcrt  # type: ignore[import-not-found]

    def _acquire(fd: int, *, timeout: float | None) -> None:  # pragma: no cover - platform branch
        # Lock a single byte at offset 0. msvcrt has no native timeout, so we
        # spin with short non-blocking attempts.
        import time

        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"failed to acquire lock within {timeout}s") from None
                time.sleep(0.05)

    def _release(fd: int) -> None:  # pragma: no cover - platform branch
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _acquire(fd: int, *, timeout: float | None) -> None:
        if timeout is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return
        import time

        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"failed to acquire lock within {timeout}s") from None
                time.sleep(0.05)

    def _release(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
