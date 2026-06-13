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
import threading
from collections.abc import Iterator
from pathlib import Path

# Per-thread reentrancy bookkeeping. ``fcntl.flock`` locks are tied to the open
# file *description*, so a single thread that opens the same lock file twice
# (different fds) and tries to take LOCK_EX on both would block on itself —
# a self-deadlock. That happens for real in the OAuth refresh path:
# ``ensure_fresh_session`` holds the secrets lock and then calls
# ``auth_store.get_cloud_session`` / ``save_cloud_session``, which each lock the
# same file. We make ``file_lock`` reentrant within one thread so those nested
# acquisitions are cheap no-ops and only the outermost frame touches the OS
# lock. Cross-process serialization is unchanged.
_reentrant = threading.local()


def _held_depths() -> dict[str, int]:
    depths = getattr(_reentrant, "depths", None)
    if depths is None:
        depths = {}
        _reentrant.depths = depths
    return depths


def _lock_key(p: Path) -> str:
    # realpath collapses symlinks / .. so two spellings of the same file share
    # one reentrancy counter. Works even when the file doesn't exist yet.
    return os.path.realpath(str(p))


@contextlib.contextmanager
def file_lock(path: str | os.PathLike[str], *, timeout: float | None = None) -> Iterator[None]:
    """Acquire an exclusive lock on ``path`` (created if absent).

    ``timeout`` is a best-effort upper bound; on platforms that don't support
    non-blocking acquire with timeout we block indefinitely.

    Reentrant within a single thread: nesting the same path returns
    immediately without re-acquiring the OS lock (and the lock is only released
    when the outermost frame exits).

    Note: ``fcntl.flock`` on NFS is silently a no-op on many configurations
    (the lock isn't propagated to the NFS server). Lock files that need
    cross-host serialization should use a different primitive. We keep flock
    for the local case and warn lazily if we detect we're on NFS.
    """
    p = Path(path)
    key = _lock_key(p)
    depths = _held_depths()
    if depths.get(key, 0) > 0:
        # Already held by this thread — reentrant no-op.
        depths[key] += 1
        try:
            yield
        finally:
            depths[key] -= 1
            if depths[key] == 0:
                del depths[key]
        return
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Open in append-binary so we don't truncate any existing payload; the
    # lock byte-range is at offset 0 and doesn't affect data.
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o600)
    # Re-tighten perms if the file existed with looser mode (covers the case
    # where an older build wrote the lock file world-readable). ``os.fchmod`` is
    # POSIX-only — it doesn't exist on Windows (which uses ACLs, not mode bits),
    # so guard on the attribute to avoid an AttributeError there.
    if hasattr(os, "fchmod"):
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
    try:
        _acquire(fd, timeout=timeout)
        depths[key] = 1
        try:
            yield
        finally:
            depths.pop(key, None)
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
                # Lock/unlock are relative to the file position; seek to 0 so
                # _acquire and _release always target the same byte (mirrors
                # _release) even if the fd position ever drifts.
                os.lseek(fd, 0, os.SEEK_SET)
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
