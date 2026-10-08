"""Real-process tests for ``_is_pid_alive`` / ``_is_watcher_alive``.

Deliberately unmocked: the point is the probe's behaviour against a real OS
process on the real platform. The still-alive assertion is what catches the
Windows regression this probe was rewritten for — ``os.kill(pid, 0)`` routes
through ``GenerateConsoleCtrlEvent`` there and, on Python <= 3.13.1, falls
through to ``TerminateProcess``, so the old implementation *killed* the process
it was asked to inspect.
"""

from __future__ import annotations

import subprocess
import sys

import psutil
import pytest

from comfy_cli.command.jobs import _is_pid_alive, _is_watcher_alive
from comfy_cli.jobs_state import JobState

# Long enough that the probe can never race the child's natural exit.
_SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def _reap(p: subprocess.Popen) -> None:
    """Tear down a test child, tolerating one a destructive probe already killed."""
    try:
        p.terminate()
    except ProcessLookupError:
        pass
    p.wait()


def test_live_child_is_alive_and_survives_the_probe():
    """A running child reads as alive — and is still running afterwards."""
    p = subprocess.Popen(_SLEEPER)
    try:
        assert _is_pid_alive(p.pid) is True
        # The probe must be non-destructive: this is the assertion that fails
        # against the old os.kill(pid, 0) implementation on Windows. A bounded
        # wait, not `poll()` — Windows' TerminateProcess returns *before* the
        # child is gone, so a zero-timeout check would still read `None` and
        # let the regression through.
        with pytest.raises(subprocess.TimeoutExpired):
            p.wait(timeout=0.5)
    finally:
        _reap(p)


def test_dead_child_is_not_alive():
    """A terminated *and reaped* child reads as dead."""
    p = subprocess.Popen(_SLEEPER)
    p.terminate()
    # Reap before probing: on POSIX an unreaped zombie still exists as a pid.
    p.wait()

    assert _is_pid_alive(p.pid) is False


def test_non_positive_pids_are_rejected():
    """pid 0 and negative pids are never treated as live watchers.

    psutil.pid_exists(0) is True on POSIX (pid 0 is the swapper/kernel
    process), and negative values are process *groups* — neither is ever a
    watcher pid, so the guard short-circuits both.
    """
    assert _is_pid_alive(0) is False
    assert _is_pid_alive(-1) is False


def test_out_of_range_pid_reads_as_dead():
    """A garbage pid from a corrupt state file is dead, not an exception.

    ``watcher_pid`` survives a tolerant JSON load with no range validation, and
    one unreadable record must not abort the whole ``jobs ls`` scan.
    """
    assert _is_pid_alive(2**64) is False


def _state(pid: int | None, create_time: float | None) -> JobState:
    return JobState(
        prompt_id="p",
        client_id="c",
        workflow="/tmp/w.json",
        where="local",
        watcher_pid=pid,
        watcher_pid_create_time=create_time,
    )


def test_watcher_with_matching_start_time_is_alive():
    p = subprocess.Popen(_SLEEPER)
    try:
        assert _is_watcher_alive(_state(p.pid, psutil.Process(p.pid).create_time())) is True
    finally:
        _reap(p)


def test_recycled_pid_is_not_the_watcher():
    """A live process whose start time doesn't match is a stranger, not us."""
    p = subprocess.Popen(_SLEEPER)
    try:
        stale = psutil.Process(p.pid).create_time() - 86400.0
        assert _is_pid_alive(p.pid) is True
        assert _is_watcher_alive(_state(p.pid, stale)) is False
    finally:
        _reap(p)


def test_watcher_without_recorded_start_time_falls_back_to_liveness():
    """Pre-existing state files carry no start time — they get the old answer."""
    p = subprocess.Popen(_SLEEPER)
    try:
        assert _is_watcher_alive(_state(p.pid, None)) is True
    finally:
        _reap(p)

    assert _is_watcher_alive(_state(p.pid, None)) is False


def test_watcher_without_pid_is_not_alive():
    assert _is_watcher_alive(_state(None, None)) is False
    assert _is_watcher_alive(_state(0, None)) is False
