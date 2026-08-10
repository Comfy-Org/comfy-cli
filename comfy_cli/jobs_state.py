"""On-disk state for in-flight workflow runs.

When ``comfy run`` submits a workflow (the default, non-blocking path), the
prompt's lifecycle state lives in ``<state-dir>/jobs/<prompt_id>.json``. A
detached watcher subprocess updates the file as the job progresses; any
agent or shell session can ``cat`` it to find the current status, outputs,
or error — no second API call needed.

State-file contract (the same shape across local and cloud):

    {
      "prompt_id": "...",
      "client_id": "...",
      "workflow": "/abs/path/to/x.json",
      "where": "local" | "cloud",
      "host": "127.0.0.1" | null,
      "port": 8188 | null,
      "base_url": "https://..." | null,
      "submitted_at": "<iso8601>",
      "updated_at": "<iso8601>",
      "completed_at": "<iso8601>" | null,
      "status": "queued" | "running" | "completed" | "error" | "cancelled",
      "outputs": [<url>, ...],
      "error": {"code": "...", "message": "...", "details": {...}} | null,
      "watcher_pid": <int> | null,
      "watcher_pid_create_time": <float epoch seconds> | null,
      "record": {<full final cloud history record>} | null,
      "item_map": {<item>: {"nodes": [...], "save_node": "...", "prefix": "..."}} | null
    }

``record`` is the node-keyed history record stashed when a cloud job reaches
a terminal state; ``item_map`` maps blueprint foreach items to the node ids
they produced (written at submit by ``comfy run``). Both are null for older
files and local runs — readers must tolerate their absence.

Terminal states (``completed``, ``error``, ``cancelled``) mean the file
won't change further; agents can stop polling.
"""

from __future__ import annotations

import json
import os
import re
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli import constants, locking
from comfy_cli.file_utils import atomic_write_text
from comfy_cli.utils import get_os

TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})

# Cloud's /api/jobs status enum (ingest ``toFilterStatus``: pending,
# in_progress, completed, failed, cancelled) -> the CLI's published jobs
# vocabulary (``comfy_cli/schemas/jobs.json``). Legacy raw-jobstate spellings
# from the deprecated /api/job/<id>/status endpoint are kept as cheap defense;
# `canceled` has never been observed from ingest but is one typo-of-vocabulary
# away.
#
# Lives here rather than in ``command/jobs.py`` so both ``command/jobs.py`` and
# ``command/job_watcher.py`` can share one copy — ``job_watcher`` imports
# ``command.jobs``, so a map owned by ``jobs`` could only be shared by importing
# it the wrong way round.
#
# Read-only (like ``TERMINAL_STATUSES`` above) because the two consumers now
# bind this object by identity rather than copying it: an in-place ``.update()``
# / ``.pop()`` / ``monkeypatch.setitem`` used to be contained to one module and
# would now rewrite terminal classification in the watcher and the exit code of
# ``jobs watch`` process-wide.
CLOUD_STATUS_ALIASES = types.MappingProxyType(
    {
        "pending": "pending",
        "in_progress": "running",
        "completed": "completed",
        "failed": "error",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        # legacy raw jobstate vocabulary (deprecated endpoint), kept defensively:
        "success": "completed",
        "error": "error",
        "non_retryable_error": "error",
        # `retryable_error` is terminal-as-error here for the same reason
        # ``output/glyphs.py`` already renders it ✗: the CLI is not told when
        # (or whether) a retry happens, so the alternative is not "wait for the
        # retry" but "sit on a status nothing recognizes" — `jobs watch` spins
        # to `cloud_timeout`, `jobs ls` drops the state file's `error_code`, and
        # the watcher's stall guard invents a terminal verdict after 300s
        # anyway. Reporting the error the server reported is the honest one.
        "retryable_error": "error",
        "lost": "error",
        "executing": "running",
    }
)


def state_dir() -> Path:
    """Return ``<config-root>/jobs`` and ensure it exists with safe mode."""
    base = Path(constants.DEFAULT_CONFIG[get_os()]) / "jobs"
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base


_SAFE_PROMPT_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def state_path(prompt_id: str) -> Path:
    """Canonical path for one prompt's state file."""
    safe = prompt_id.replace("/", "_").replace("\\", "_")
    if not _SAFE_PROMPT_ID.match(safe):
        raise ValueError(f"unsafe prompt_id: {prompt_id!r}")
    return state_dir() / f"{safe}.json"


@dataclass
class JobState:
    prompt_id: str
    client_id: str | None
    workflow: str
    where: str
    host: str | None = None
    port: int | None = None
    base_url: str | None = None
    submitted_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    status: str = "queued"
    outputs: list[Any] = field(default_factory=list)
    error: dict[str, Any] | None = None
    watcher_pid: int | None = None
    # Watcher process start time, recorded next to its pid so a recycled pid
    # can't pass for the original watcher. Null on files written before this
    # field existed (and when psutil couldn't read it).
    watcher_pid_create_time: float | None = None
    # Full final cloud history record (node-keyed outputs), stashed at terminal.
    record: dict[str, Any] | None = None
    # foreach item -> {"nodes": [...], "save_node": ..., "prefix": ...} map,
    # written at submit time by `comfy run` for composed workflows.
    item_map: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(state: JobState) -> Path | None:
    """Atomically write a state file. Returns the path (or ``None`` if the
    write was skipped because ``prompt_id`` wasn't a sane string).

    The string-check is defensive against tests that mock WorkflowExecution
    and let MagicMock prompt_ids slip through. Real users always have
    real prompt_ids (UUIDs from the server), so this is a no-op in
    practice.
    """
    if not isinstance(state.prompt_id, str) or not state.prompt_id.strip():
        return None
    state.updated_at = _now_iso()
    if state.is_terminal and state.completed_at is None:
        state.completed_at = state.updated_at
    path = state_path(state.prompt_id)
    # Lock per-file so a watcher and a foreground update can't tear each
    # other's writes.
    with locking.file_lock(path.with_suffix(".lock")):
        # fsync=True: durability against power loss before the atomic rename.
        atomic_write_text(path, json.dumps(state.to_dict(), indent=2, default=str), fsync=True)
    return path


def read(prompt_id: str) -> JobState | None:
    """Read a state file. Returns None if the file doesn't exist."""
    path = state_path(prompt_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Tolerant load: drop unknown keys, fill defaults for missing ones.
    known = {f.name for f in JobState.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    try:
        return JobState(**filtered)
    except TypeError:
        # Required fields missing (e.g. truncated/legacy file) — treat as absent.
        return None


def stamp_watcher_identity(state: JobState) -> None:
    """Record the calling process as this record's watcher: pid + create_time
    (both, or pid reuse defeats ``_is_watcher_alive``'s identity check).

    Used by the detached watcher subprocess and by foreground ``--wait`` runs
    alike — whichever process is actively finalizing the record stamps itself,
    so the stale-watcher reap in ``jobs ls`` can finalize the record if that
    process dies without running its handlers (killed from outside).
    """
    state.watcher_pid = os.getpid()
    try:
        import psutil

        state.watcher_pid_create_time = psutil.Process().create_time()
    except Exception:  # noqa: BLE001 — best effort; None just means liveness-only
        state.watcher_pid_create_time = None


@contextmanager
def locked(prompt_id: str) -> Iterator[JobState | None]:
    """Hold ``prompt_id``'s per-file lock and yield the record as it is on disk
    *right now* (``None`` if there is no readable record).

    Every writer of a job record — this module's ``write``, the watcher, the
    foreground run, the reap in ``jobs ls`` — takes this same lock, so a
    read-modify-write performed inside this block cannot interleave with
    another process's write. ``write`` re-takes the lock, which is reentrant
    within a thread, so calling it from inside the block is fine.

    Yields ``None`` (without failing) when the id isn't one we can even name a
    file for: callers are all best-effort bookkeeping paths that must not take
    a command down over a state file.
    """
    try:
        path = state_path(prompt_id)
    except (ValueError, OSError, AttributeError, TypeError):
        yield None
        return
    with locking.file_lock(path.with_suffix(".lock")):
        yield read(prompt_id)


def clear_watcher_identity(state: JobState) -> bool:
    """Un-stamp this process as ``state``'s watcher and persist that, for a
    process that is about to exit while deliberately leaving the record
    NON-terminal (e.g. ``run --wait`` timing out on a job that is still running
    server-side). Returns True if the on-disk record was rewritten.

    Without this the record keeps a pid that is about to be dead, which is
    exactly what the stale-watcher reap in ``jobs ls`` finalizes — it would
    flip a healthy job to ``error``/``watcher_crashed``.

    The read-modify-write runs under the record's lock against a *re-read*, not
    against the caller's in-memory snapshot, which may be many minutes stale:
    a concurrent ``comfy jobs cancel`` (or a watcher) can have made the record
    terminal in the meantime, and blindly writing back the snapshot would walk
    that verdict backwards to a non-terminal ``running`` that is then neither
    terminal nor reapable — a permanent phantom. A record that is already
    terminal, or that some other process has since stamped, is left untouched.
    """
    if not isinstance(state.prompt_id, str) or not state.prompt_id.strip():
        return False
    rewrote = False
    try:
        with locked(state.prompt_id) as on_disk:
            if on_disk is not None and not on_disk.is_terminal and on_disk.watcher_pid == os.getpid():
                on_disk.watcher_pid = None
                on_disk.watcher_pid_create_time = None
                rewrote = write(on_disk) is not None
    except (OSError, ValueError):
        # Same tolerance every other state-file write gets: bookkeeping must
        # never fail an otherwise-handled exit path.
        return False
    # Keep the caller's snapshot in step so a later write can't re-stamp us.
    state.watcher_pid = None
    state.watcher_pid_create_time = None
    return rewrote


def new(
    prompt_id: str,
    *,
    client_id: str | None,
    workflow: str,
    where: str,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
) -> JobState:
    """Build a fresh JobState in ``queued`` status. Call ``write()`` to persist."""
    now = _now_iso()
    return JobState(
        prompt_id=prompt_id,
        client_id=client_id,
        workflow=workflow,
        where=where,
        host=host,
        port=port,
        base_url=base_url,
        submitted_at=now,
        updated_at=now,
        status="queued",
    )
