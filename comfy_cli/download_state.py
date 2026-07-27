"""On-disk state for backgrounded model downloads.

``comfy model download --background`` resolves the download in the foreground
(CivitAI/HF metadata, token config, filename, destination check) and then
detaches a worker that performs only the byte transfer. The worker's progress
lives in ``<workspace>/.comfy-downloads/<id>.json``; its stdout/stderr are
appended to ``<workspace>/.comfy-downloads/<id>.log``. Any agent or shell
session can poll the file — or ``comfy model download-status <id>`` — without a
second network call.

State-file contract (``download-state/1``)::

    {
      "schema": "download-state/1",
      "id": "<12 hex chars>",
      "pid": <int> | null,          # worker pid, written by the worker itself
      "url": "https://...",         # RESOLVED download url
      "dest": "/abs/path/model.safetensors",
      "total_bytes": <int> | null,  # null until response headers are read
      "completed_bytes": <int>,
      "status": "starting" | "downloading" | "completed" | "failed" | "cancelled",
      "error": "<friendly message>" | null,
      "started_at": "<iso8601>",
      "updated_at": "<iso8601>",
      "downloader": "httpx" | "aria2",
      "needs_civitai_auth": <bool>,
      "needs_hf_auth": <bool>
    }

No auth tokens or headers are ever persisted — the worker re-derives them from
config the same way the foreground ``download()`` does. ``needs_civitai_auth`` /
``needs_hf_auth`` only record *which* credential the resolved URL wants.

Terminal statuses (``completed``, ``failed``, ``cancelled``) mean the file won't
change further; agents can stop polling.
"""

from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_SCHEMA = "download-state/1"

STATE_DIRNAME = ".comfy-downloads"

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset({"starting", "downloading"})

# The worker rewrites the state file at most this often while bytes stream in.
# Terminal transitions always write, regardless of the throttle.
PROGRESS_THROTTLE_S = 1.0

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def state_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.comfy-downloads`` and ensure it exists."""
    base = Path(workspace) / STATE_DIRNAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def state_path(workspace: Path, download_id: str) -> Path:
    if not _SAFE_ID.match(download_id or ""):
        raise ValueError(f"unsafe download id: {download_id!r}")
    return state_dir(workspace) / f"{download_id}.json"


def log_path(workspace: Path, download_id: str) -> Path:
    if not _SAFE_ID.match(download_id or ""):
        raise ValueError(f"unsafe download id: {download_id!r}")
    return state_dir(workspace) / f"{download_id}.log"


@dataclass
class DownloadState:
    id: str
    url: str
    dest: str
    schema: str = STATE_SCHEMA
    pid: int | None = None
    total_bytes: int | None = None
    completed_bytes: int = 0
    status: str = "starting"
    error: str | None = None
    started_at: str = ""
    updated_at: str = ""
    downloader: str = "httpx"
    needs_civitai_auth: bool = False
    needs_hf_auth: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new(
    *,
    url: str,
    dest: str,
    downloader: str = "httpx",
    needs_civitai_auth: bool = False,
    needs_hf_auth: bool = False,
    download_id: str | None = None,
) -> DownloadState:
    """Build a fresh state in ``starting``. Call :func:`write` to persist."""
    now = _now_iso()
    return DownloadState(
        id=download_id or new_id(),
        url=url,
        dest=dest,
        downloader=downloader,
        needs_civitai_auth=needs_civitai_auth,
        needs_hf_auth=needs_hf_auth,
        started_at=now,
        updated_at=now,
    )


def write(workspace: Path, state: DownloadState) -> Path:
    """Atomically persist ``state`` under ``workspace``'s state directory."""
    return write_path(state_path(workspace, state.id), state)


def write_path(path: Path, state: DownloadState) -> Path:
    """Atomically persist ``state`` at ``path`` (write a tmp file, ``os.replace``).

    The worker addresses its state file by path rather than by workspace: it is
    handed ``--state <file>`` and must not have to re-resolve a workspace that
    the foreground already resolved.
    """
    state.updated_at = _now_iso()
    path = Path(path)
    tmp = path.with_suffix(f".{os.getpid()}.{_secrets.token_hex(4)}.tmp")
    payload = json.dumps(state.to_dict(), indent=2, default=str)
    try:
        tmp.write_text(payload, encoding="utf-8")
        try:
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        os.replace(tmp, path)
    except OSError:
        # Never let a bookkeeping failure kill an in-flight transfer.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def read(workspace: Path, download_id: str) -> DownloadState | None:
    """Read one state file. Returns None when it's absent or unreadable."""
    try:
        path = state_path(workspace, download_id)
    except ValueError:
        return None
    return read_path(path)


def read_path(path: Path) -> DownloadState | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    known = set(DownloadState.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    try:
        return DownloadState(**filtered)
    except TypeError:
        # Truncated/legacy file missing a required field — treat as absent.
        return None


def list_all(workspace: Path) -> list[DownloadState]:
    """Every readable state file, newest ``started_at`` first."""
    base = Path(workspace) / STATE_DIRNAME
    if not base.is_dir():
        return []
    states = [s for s in (read_path(p) for p in sorted(base.glob("*.json"))) if s is not None]
    states.sort(key=lambda s: (s.started_at or "", s.id), reverse=True)
    return states


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------


def _dest_size(dest: str) -> int | None:
    try:
        return os.stat(dest).st_size
    except OSError:
        return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def elapsed_seconds(state: DownloadState) -> float:
    """Wall-clock seconds the download has been (or was) running."""
    started = _parse_iso(state.started_at)
    if started is None:
        return 0.0
    end = _parse_iso(state.updated_at) if state.is_terminal else None
    if end is None:
        end = datetime.now(timezone.utc)
    return max(0.0, (end - started).total_seconds())


def reconcile(state: DownloadState, *, pid_alive=None) -> DownloadState:
    """Return a copy of ``state`` corrected against reality on disk.

    A worker that was SIGKILLed (or whose machine rebooted) never gets to write
    a terminal status, so a state file claiming ``downloading`` is only
    trustworthy while its pid is alive. Two corrections happen here:

    * ``completed_bytes`` prefers a live ``stat(dest)`` over the last value the
      worker managed to persist — the file on disk is the ground truth.
    * an active status whose worker is gone becomes ``completed`` when the file
      reached the known total, and ``failed`` ("worker died") otherwise.

    ``pid_alive`` is injectable for tests; it defaults to the same liveness
    helper the launch/stop machinery uses.
    """
    if pid_alive is None:
        from comfy_cli.utils import is_running as pid_alive  # noqa: N813

    fresh = DownloadState(**state.to_dict())

    size = _dest_size(fresh.dest)
    if size is not None and fresh.status != "cancelled":
        fresh.completed_bytes = size

    if fresh.status not in ACTIVE_STATUSES:
        return fresh

    if fresh.pid is not None and pid_alive(fresh.pid):
        return fresh

    total = fresh.total_bytes
    if total is not None and fresh.completed_bytes >= total and size is not None:
        fresh.status = "completed"
        fresh.error = None
    else:
        fresh.status = "failed"
        fresh.error = fresh.error or "worker died before the download finished"
    return fresh


def percent(state: DownloadState) -> float | None:
    """Completion percentage, or None while the total is unknown."""
    total = state.total_bytes
    if not total or total <= 0:
        return None
    return round(min(100.0, state.completed_bytes / total * 100.0), 2)


def status_payload(state: DownloadState) -> dict[str, Any]:
    """The ``download-status`` / ``downloads`` envelope row for one download."""
    return {
        "id": state.id,
        "status": state.status,
        "completed_bytes": state.completed_bytes,
        "total_bytes": state.total_bytes,
        "percent": percent(state),
        "elapsed_seconds": round(elapsed_seconds(state), 1),
        "dest": state.dest,
        "error": state.error,
    }


# ---------------------------------------------------------------------------
# process control
# ---------------------------------------------------------------------------


def kill_worker(pid: int | None) -> bool:
    """Terminate a worker and everything it spawned. Best effort.

    POSIX workers are detached with ``start_new_session=True``, so the worker is
    its own process-group leader and one ``killpg`` reaches the whole tree.
    Windows workers get ``CREATE_NEW_PROCESS_GROUP``; there is no ``killpg``, so
    fall back to killing the process and its children directly.
    """
    if not pid or pid <= 0:
        return False
    if sys.platform != "win32":
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            pass  # fall through to the per-process path below

    try:
        import psutil

        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except Exception:  # noqa: BLE001 — best effort
                pass
        proc.kill()
        return True
    except Exception:  # noqa: BLE001 — already gone, or not ours to signal
        return False
