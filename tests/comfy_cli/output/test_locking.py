"""File-lock primitive: serialization and timeout."""

import threading
import time
from pathlib import Path

from comfy_cli.locking import file_lock


def test_serializes_concurrent_acquires(tmp_path: Path):
    lock_file = tmp_path / "test.lock"
    order: list[str] = []
    barrier = threading.Barrier(2)

    def worker(name: str, hold_s: float) -> None:
        barrier.wait()
        with file_lock(lock_file):
            order.append(f"{name}-enter")
            time.sleep(hold_s)
            order.append(f"{name}-exit")

    t1 = threading.Thread(target=worker, args=("a", 0.10))
    t2 = threading.Thread(target=worker, args=("b", 0.05))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Each worker's enter and exit must be contiguous — no interleaving.
    assert len(order) == 4
    assert order[0].endswith("-enter")
    assert order[1].endswith("-exit")
    assert order[2].endswith("-enter")
    assert order[3].endswith("-exit")
    assert order[0][0] == order[1][0]
    assert order[2][0] == order[3][0]


def test_creates_parent_dir(tmp_path: Path):
    lock_file = tmp_path / "nested" / "subdir" / "thing.lock"
    with file_lock(lock_file):
        pass
    assert lock_file.exists()


def test_reentrant_in_same_process_is_ok(tmp_path: Path):
    # flock is per-fd, not per-process; entering twice from the same process
    # should still work as long as we use a fresh fd each time.
    lock_file = tmp_path / "rt.lock"
    with file_lock(lock_file):
        pass
    with file_lock(lock_file):
        pass


def test_nested_same_path_does_not_self_deadlock(tmp_path: Path):
    # The OAuth refresh path holds the secrets lock and then calls store
    # helpers that lock the SAME file. With per-fd flock that would block on
    # itself; the lock is reentrant within a thread, so this must not hang.
    lock_file = tmp_path / "nested.lock"
    reached_inner = False
    with file_lock(lock_file):
        with file_lock(lock_file):  # nested — different fd, same path
            reached_inner = True
    assert reached_inner


def test_nested_different_spelling_same_file_is_reentrant(tmp_path: Path):
    # Two spellings of the same file (".." in the path) must share one
    # reentrancy counter so the inner acquire is recognized as nested.
    target = tmp_path / "sub" / "x.lock"
    target.parent.mkdir(parents=True, exist_ok=True)
    alias = tmp_path / "sub" / ".." / "sub" / "x.lock"
    with file_lock(target):
        with file_lock(alias):
            pass


def test_lock_released_after_nested_block_exits(tmp_path: Path):
    # After a nested acquire fully unwinds, the OS lock must be free again so a
    # second thread can take it (i.e. reentrancy didn't leak the held state).
    lock_file = tmp_path / "release.lock"
    with file_lock(lock_file):
        with file_lock(lock_file):
            pass

    acquired = threading.Event()

    def worker():
        with file_lock(lock_file, timeout=2.0):
            acquired.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=3.0)
    assert acquired.is_set()
