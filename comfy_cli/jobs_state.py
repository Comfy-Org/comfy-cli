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
import secrets as _secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli import constants, locking
from comfy_cli.utils import get_os

TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})


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
        tmp = path.with_suffix(f".{os.getpid()}.{_secrets.token_hex(4)}.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, default=str), encoding="utf-8")
        # fsync for durability before atomic rename
        try:
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        os.replace(tmp, path)
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
