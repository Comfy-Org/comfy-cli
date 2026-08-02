"""Real-process tests for ``_is_pid_alive``.

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

from comfy_cli.command.jobs import _is_pid_alive

# Long enough that the probe can never race the child's natural exit.
_SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_live_child_is_alive_and_survives_the_probe():
    """A running child reads as alive — and is still running afterwards."""
    p = subprocess.Popen(_SLEEPER)
    try:
        assert _is_pid_alive(p.pid) is True
        # The probe must be non-destructive: this is the assertion that fails
        # against the old os.kill(pid, 0) implementation on Windows.
        assert p.poll() is None
    finally:
        p.terminate()
        p.wait()


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
