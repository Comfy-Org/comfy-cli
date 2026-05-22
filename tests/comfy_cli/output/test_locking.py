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
