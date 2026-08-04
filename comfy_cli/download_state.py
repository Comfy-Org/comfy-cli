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
      "pid_create_time": <float> | null,  # worker process start time (identity)
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

``pid`` is only ever written by the worker itself, together with
``pid_create_time`` — the pair identifies the process, so a recycled pid can
never be mistaken for (or signalled as) a live worker. Until the worker's first
write both are null; :func:`reconcile` gives a fresh ``starting`` record
:data:`STARTUP_GRACE_S` to get that far before declaring it dead.

No auth tokens or headers are ever persisted — the worker re-derives them from
config the same way the foreground ``download()`` does. ``needs_civitai_auth`` /
``needs_hf_auth`` only record *which* credential the resolved URL wants. The
state directory is created ``0700`` and each file ``0600`` regardless: a
resolved url can still embed a presigned/SAS query token, and only the owner
should be able to read — or tamper with — these records.

Cancellation is signalled out-of-band by an empty ``<id>.cancel`` sentinel next
to the state file. A sentinel can't be clobbered by a state write that raced it,
so a worker that comes up (or ticks) after ``download-cancel`` always observes
the request; see :func:`request_cancel`.

Terminal statuses (``completed``, ``failed``, ``cancelled``) mean the file won't
change further; agents can stop polling.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets as _secrets
import sys
import time
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

# The state dir can hold presigned urls; keep it owner-only.
STATE_DIR_MODE = 0o700
STATE_FILE_MODE = 0o600

# How long a `starting` record with no pid yet is trusted before reconcile
# calls it dead. This only has to cover interpreter startup for the worker.
STARTUP_GRACE_S = 60.0

# Slack when matching a recorded process start time against the live one. Both
# come from the same psutil source so they agree exactly in practice.
PID_CREATE_TIME_TOLERANCE_S = 1.0

# Every worker's argv contains this; used to identify a worker when its start
# time was never recorded.
WORKER_ARGV_MARKER = "_download-worker"

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def state_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.comfy-downloads`` and ensure it exists, owner-only."""
    base = Path(workspace) / STATE_DIRNAME
    base.mkdir(parents=True, exist_ok=True, mode=STATE_DIR_MODE)
    if sys.platform != "win32":
        # mkdir's mode is masked by the umask, and the directory may predate
        # this code, so tighten it explicitly.
        with contextlib.suppress(OSError):
            base.chmod(STATE_DIR_MODE)
    return base


def state_path(workspace: Path, download_id: str) -> Path:
    if not _SAFE_ID.match(download_id or ""):
        raise ValueError(f"unsafe download id: {download_id!r}")
    return state_dir(workspace) / f"{download_id}.json"


def log_path(workspace: Path, download_id: str) -> Path:
    if not _SAFE_ID.match(download_id or ""):
        raise ValueError(f"unsafe download id: {download_id!r}")
    return state_dir(workspace) / f"{download_id}.log"


def cancel_path(workspace: Path, download_id: str) -> Path:
    if not _SAFE_ID.match(download_id or ""):
        raise ValueError(f"unsafe download id: {download_id!r}")
    return state_dir(workspace) / f"{download_id}.cancel"


def cancel_marker_for(state_file: Path) -> Path:
    """The cancel sentinel that pairs with a state file addressed by path.

    The worker is handed ``--state <file>`` and never re-resolves a workspace,
    so it derives the sentinel the same way.
    """
    state_file = Path(state_file)
    return state_file.with_name(f"{state_file.stem}.cancel")


def request_cancel(path: Path) -> bool:
    """Create the cancel sentinel at ``path``. Returns False if it couldn't be.

    Creating a separate file (rather than flipping a field in the state file)
    is what makes the request survive a worker write that raced it: the worker
    checks the sentinel before every state write, so once this returns the
    worker can only ever write ``cancelled``.
    """
    try:
        Path(path).touch(mode=STATE_FILE_MODE, exist_ok=True)
        return True
    except OSError:
        return False


@dataclass
class DownloadState:
    id: str
    url: str
    dest: str
    schema: str = STATE_SCHEMA
    pid: int | None = None
    pid_create_time: float | None = None
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
        if sys.platform != "win32":
            # write_text honours the umask, which may be group/world readable —
            # and the payload can contain a presigned url.
            with contextlib.suppress(OSError):
                tmp.chmod(STATE_FILE_MODE)
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


def _is_int(value: Any) -> bool:
    # bool is an int subclass; a `true` pid is corruption, not a pid.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


# Per-field validators. A file that fails any of them is corrupt or tampered
# with; it reads as *absent* rather than constructing a DownloadState that
# blows up later in kill_worker/reconcile and takes the whole command with it.
_FIELD_VALIDATORS: dict[str, Any] = {
    "id": lambda v: isinstance(v, str),
    "url": lambda v: isinstance(v, str),
    "dest": lambda v: isinstance(v, str),
    "schema": lambda v: isinstance(v, str),
    "pid": lambda v: v is None or _is_int(v),
    "pid_create_time": lambda v: v is None or _is_number(v),
    "total_bytes": lambda v: v is None or _is_int(v),
    "completed_bytes": _is_int,
    "status": lambda v: isinstance(v, str),
    "error": lambda v: v is None or isinstance(v, str),
    "started_at": lambda v: isinstance(v, str),
    "updated_at": lambda v: isinstance(v, str),
    "downloader": lambda v: isinstance(v, str),
    "needs_civitai_auth": lambda v: isinstance(v, bool),
    "needs_hf_auth": lambda v: isinstance(v, bool),
}


def read_path(path: Path) -> DownloadState | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    known = set(DownloadState.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    for key, value in filtered.items():
        validator = _FIELD_VALIDATORS.get(key)
        if validator is not None and not validator(value):
            return None
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


def process_create_time(pid: int | None) -> float | None:
    """The process start time for ``pid``, or None if it can't be determined."""
    if not pid or pid <= 0:
        return None
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 — gone, not ours, or unsupported platform
        return None


def is_worker_process(pid: int | None, create_time: float | None) -> bool:
    """True only when ``pid`` is *still the download worker* we recorded.

    Pids are recycled. Liveness alone is not proof: after a worker crashes and
    the OS hands its number to something else, a bare ``is_running`` check would
    keep a dead transfer pinned at ``downloading`` forever — and, worse, point
    ``download-cancel``'s ``killpg`` at an unrelated process group.

    The recorded start time is the discriminator: the worker writes it together
    with its own pid, so the pair either matches a live process or it doesn't.
    When no start time was recorded (psutil couldn't read it), fall back to
    matching the worker's argv — never to liveness alone.
    """
    if not pid or pid <= 0:
        return False
    try:
        import psutil

        proc = psutil.Process(pid)
        if create_time is not None:
            return abs(proc.create_time() - create_time) <= PID_CREATE_TIME_TOLERANCE_S
        return WORKER_ARGV_MARKER in " ".join(proc.cmdline() or [])
    except Exception:  # noqa: BLE001 — gone, or not ours to inspect
        return False


def worker_alive(state: DownloadState, *, pid_alive=None) -> bool:
    """True while ``state``'s worker is running and provably the same process.

    Deliberately laxer than :func:`kill_worker` in one case: if a record somehow
    carries a pid with no start time, this trusts liveness rather than falling
    back to the argv check. The two get different answers because they cost
    different things when wrong — mis-reporting a status is cheap and
    self-corrects on the next poll, while signalling a stranger's process group
    is not, so only the reporting side is allowed the benefit of the doubt.
    """
    if state.pid is None:
        return False
    if pid_alive is None:
        from comfy_cli.utils import is_running as pid_alive  # noqa: N813
    if not pid_alive(state.pid):
        return False
    if state.pid_create_time is None:
        return True
    return is_worker_process(state.pid, state.pid_create_time)


def reconcile(state: DownloadState, *, pid_alive=None) -> DownloadState:
    """Return a copy of ``state`` corrected against reality on disk.

    A worker that was SIGKILLed (or whose machine rebooted) never gets to write
    a terminal status, so a state file claiming ``downloading`` is only
    trustworthy while its worker is alive — and still *is* that worker, not a
    stranger who inherited its pid (see :func:`worker_alive`). Corrections:

    * ``completed_bytes`` prefers a live ``stat(dest)`` over the last value the
      worker managed to persist — when there *is* a file at ``dest``, it is taken
      as ground truth. The httpx downloader writes atomically (into a ``.part``
      sibling, renamed on completion), so for the case the submit path guarantees
      — a destination that was absent when the download started — that stat finds
      nothing mid-flight and the worker's own progress writes stand unmodified,
      which is what we want: a truncated file's length was never a meaningful
      progress reading, and the state file is the contract `download-status`
      reports from. Keep the stat: it is what turns "worker SIGKILLed after the
      rename but before it could write ``completed``" into ``completed`` rather
      than ``failed``. The stat cannot, however, tell a landed rename apart from
      a file something *else* dropped at ``dest`` after submission; such a file's
      size is then adopted as this download's progress.
    * a ``starting`` record that hasn't claimed a pid yet is left alone for
      :data:`STARTUP_GRACE_S`; that window is the worker's interpreter startup.
    * an active status whose worker is gone becomes ``completed`` when the file
      reached the known total, and ``failed`` ("worker died") otherwise.

    ``pid_alive`` is injectable for tests; it defaults to the same liveness
    helper the launch/stop machinery uses.
    """
    fresh = DownloadState(**state.to_dict())

    size = _dest_size(fresh.dest)
    if size is not None and fresh.status != "cancelled":
        fresh.completed_bytes = size

    if fresh.status not in ACTIVE_STATUSES:
        return fresh

    if fresh.pid is None and fresh.status == "starting" and elapsed_seconds(fresh) < STARTUP_GRACE_S:
        return fresh

    if worker_alive(fresh, pid_alive=pid_alive):
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


def kill_worker(pid: int | None, create_time: float | None = None, *, force: bool = False) -> bool:
    """Terminate a worker and everything it spawned. Best effort.

    Refuses to signal a pid that isn't provably still our worker — a recycled
    pid would otherwise get a ``killpg`` aimed at whatever unrelated process
    group now owns that number. See :func:`is_worker_process`.

    POSIX workers are detached with ``start_new_session=True``, so the worker is
    its own process-group leader and one ``killpg`` reaches the whole tree;
    ``force`` escalates SIGTERM to SIGKILL. Windows workers get
    ``CREATE_NEW_PROCESS_GROUP``; there is no ``killpg``, so fall back to
    killing the process and its children directly.
    """
    if not pid or pid <= 0:
        return False
    if not is_worker_process(pid, create_time):
        return False

    if sys.platform != "win32":
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
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


def stop_worker(state: DownloadState, *, grace_s: float = 5.0) -> bool:
    """Stop ``state``'s worker for good: SIGTERM, wait, then SIGKILL.

    Returns True once no verified worker is left running. A plain SIGTERM is not
    enough on its own — a worker wedged in a syscall (or one that has only just
    been spawned) can outlive it, and if it survives it goes on writing bytes
    and progress updates after the caller has already declared the download
    cancelled.
    """
    if state.pid is None:
        return True

    def gone() -> bool:
        return not is_worker_process(state.pid, state.pid_create_time)

    if gone():
        return True

    kill_worker(state.pid, state.pid_create_time)
    if _wait_for_exit(gone, grace_s):
        return True

    kill_worker(state.pid, state.pid_create_time, force=True)
    return _wait_for_exit(gone, 2.0)


def _wait_for_exit(gone, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if gone():
            return True
        time.sleep(0.05)
    return gone()
