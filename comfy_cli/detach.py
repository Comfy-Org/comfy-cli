"""Spawn a child process that must outlive the command that started it.

Background workers (``comfy model download --background``, the ``comfy run``
job watcher, the templates cache refresher) are started detached so they
survive the parent returning and the terminal closing:

* POSIX: ``start_new_session=True`` (``setsid``), unchanged.
* Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` — ``start_new_session``
  is POSIX-only and CPython silently ignores it there — plus
  ``CREATE_BREAKAWAY_FROM_JOB``.

The breakaway flag matters when comfy-cli itself runs inside a kill-on-close
Job Object (the local Comfy agent wraps every invocation in one). Without it the
detached child is still a member of the parent's job and is killed the moment
the job handle closes — i.e. as soon as the command returns — so a background
download sat at ``starting`` forever. ``CreateProcess`` refuses the flag with
``ERROR_ACCESS_DENIED`` when the enclosing job does not set
``JOB_OBJECT_LIMIT_BREAKAWAY_OK``; we then retry once without it, which is
exactly the pre-breakaway behaviour (and outside any job the flag is a no-op).

Leaf module: imports nothing from ``comfy_cli``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

# Not exposed on POSIX builds of the stdlib (and only on 3.7+ Windows builds).
_CREATE_BREAKAWAY_FROM_JOB_FALLBACK = 0x01000000
_ERROR_ACCESS_DENIED = 5


def _windows_detach_flags() -> tuple[int, int]:
    """Return ``(base_flags, breakaway_flag)`` for a detached Windows child."""
    detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", _CREATE_BREAKAWAY_FROM_JOB_FALLBACK)
    return detached | new_group, breakaway


def _is_breakaway_denied(exc: OSError) -> bool:
    """True when ``CreateProcess`` rejected the breakaway flag (job forbids it)."""
    return getattr(exc, "winerror", None) == _ERROR_ACCESS_DENIED or isinstance(exc, PermissionError)


def popen_detached(args: Any, **kwargs: Any) -> subprocess.Popen:
    """``subprocess.Popen`` a child detached from this process's session/job.

    Accepts the usual ``Popen`` keyword arguments (stdio, ``cwd``, ``env``,
    ``close_fds`` …); callers must not pass ``creationflags`` or
    ``start_new_session`` themselves. Exceptions from ``Popen`` propagate, so
    callers keep their existing failure handling.
    """
    if sys.platform != "win32":
        return subprocess.Popen(args, start_new_session=True, **kwargs)

    base, breakaway = _windows_detach_flags()
    try:
        return subprocess.Popen(args, creationflags=base | breakaway, **kwargs)
    except OSError as exc:
        if not _is_breakaway_denied(exc):
            raise
    # The enclosing Job Object does not allow breakaway: detach as before and
    # accept that the child shares the parent's job.
    return subprocess.Popen(args, creationflags=base, **kwargs)
