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
      "kind": "background" | "foreground",
      "needs_civitai_auth": <bool>,
      "needs_hf_auth": <bool>
    }

``kind`` distinguishes a detached worker from a plain ``comfy model download``
running in the caller's own terminal. Both write records — a foreground transfer
that claimed no destination was invisible to every other invocation, so two of
them raced into the same file — but they are not interchangeable: a background
worker gets its own session (:func:`kill_worker` may ``killpg`` it), whereas a
foreground record's pid is the *user's CLI process*, sharing the terminal's
foreground process group. Signalling that group would kill the user's shell job,
so ``download-cancel`` refuses a live foreground record instead. The field is
additive within ``download-state/1``: readers drop unknown keys, and a record
written before it existed reads back as ``"background"``, which is what it was.
A record carrying an *unrecognized* ``kind`` is a different case — see
``_TOLERANT_FALLBACKS`` — and reads back as ``"foreground"``, the side that
refuses to be signalled.

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
import errno
import hashlib
import json
import os
import re
import secrets as _secrets
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

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

# How long a finished download's record is kept before :func:`prune` drops it.
PRUNE_MAX_AGE_S = 7 * 24 * 60 * 60

# Hard ceiling on retained *terminal* records, applied on top of the age rule so
# a burst of downloads can't leave the directory (and `downloads`' output) large
# for a week. Active records are never counted or evicted.
PRUNE_MAX_TERMINAL_RECORDS = 200

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


# Destination claims live in their own subdirectory of the (owner-only) state
# dir rather than next to the user's model files: a claim is our bookkeeping,
# not something a user should find in `models/loras`, and `list_all` globs
# `*.json` at the top level so the subdirectory stays invisible to every verb.
#
# **Ownership of a destination is HELD, not derived.** The process performing a
# transfer holds an exclusive OS lock (`flock` on POSIX, `msvcrt.locking` on
# Windows) on that destination's claim file for the whole transfer, and the lock
# — not the payload, and not the record the payload points at — is what proves
# the destination is taken. That inverts the previous design, in which ownership
# was re-derived on every access by resolving the payload's `download_id` to a
# state record and asking whether that record looked live. Deriving left three
# windows that holding closes:
#
# * a read-then-unlink race in the old conditional `release_claim`: the claim
#   could change hands between the compare and the unlink;
# * a record that reads terminal while its worker is demonstrably still running
#   (`download-cancel` writes `cancelled` even when `stop_worker` fails), and a
#   transient read failure on the record, both of which made a live claim look
#   stale;
# * a stale-clear that had to be retried, because clearing and re-creating were
#   two separate steps another submitter could interleave with.
#
# A SIGKILLed holder needs no sweeper and no grace wait: the kernel drops its
# lock when the process dies, so the next submitter acquires immediately.
#
# The payload is still written — the `download_id` it names is how a submitter
# words its refusal, and how a freshly spawned worker checks that the
# destination it was handed is still its own — but it is only ever rewritten
# **in place on the locked descriptor**. A temp-file-plus-rename publish would
# swap the inode the lock lives on, which is exactly the thing the lock cannot
# survive.
#
# Claim files are never unlinked on release. One is ~100 bytes keyed by the
# hash of a destination, so the directory is bounded by the number of distinct
# destinations a workspace has ever downloaded to, and leaving them in place
# sidesteps the unlink-under-lock pitfall entirely (an unlink makes every fd
# already opened on that path a lock on an orphan inode). :func:`prune` still
# removes the ones it can both lock and prove dead, on POSIX only.
#
# **Network filesystems are best effort.** `flock` over NFS/SMB is advisory at
# best and may be a local-only lock, so two machines sharing a mount can still
# both win. That is no worse than the pid-based liveness this replaces, which
# was meaningless across machines to begin with; cross-machine correctness is
# out of scope here, exactly as it was before.
CLAIMS_DIRNAME = "claims"

# Legacy staging suffix. The previous implementation published a claim by
# writing a private `.claim.<pid>.<id>.tmp` sibling and hard-linking it into
# place; nothing writes these any more, but a workspace last touched by that
# version can still hold one, so :func:`prune` keeps sweeping them.
CLAIM_TMP_SUFFIX = ".tmp"

# How old such a leftover must be before `prune` treats it as a crashed
# acquire's residue rather than an acquire in progress.
CLAIM_TMP_MAX_AGE_S = 60 * 60

# How many times :func:`lock_claim` re-opens after its re-stat guard rejects the
# file it just locked. A rejection means the claim was unlinked or replaced
# between our `open` and our lock, so the lock we hold is on an orphan inode and
# decides nothing. Bounded rather than unbounded: only a concurrent `prune`
# unlinks claims at all, so more than a couple of rounds means something else is
# churning the path and refusing is the safe direction.
CLAIM_LOCK_RESTAT_ATTEMPTS = 3

# Poll interval for `lock_claim`'s bounded blocking wait. The lock is only ever
# held across a ~100 byte read/rewrite by anyone who is not transferring, so the
# wait is short and a tight poll costs nothing.
CLAIM_LOCK_POLL_S = 0.02

# What a lock attempt raises when the lock is simply held elsewhere, as opposed
# to the filesystem being unable to lock at all. POSIX `flock` reports
# EWOULDBLOCK/EAGAIN; Windows `msvcrt.locking` reports EACCES for `LK_NBLCK` and
# EDEADLOCK when a blocking `LK_LOCK` gives up. Everything else (ENOLCK,
# ENOTSUP/EOPNOTSUPP on a filesystem with no locking) propagates, because it
# means no lock was taken and the caller has to degrade rather than conclude
# somebody else owns the destination.
_LOCK_CONTENDED_ERRNOS = frozenset(
    getattr(errno, name) for name in ("EACCES", "EAGAIN", "EWOULDBLOCK", "EDEADLK", "EDEADLOCK") if hasattr(errno, name)
)

# `os.open` needs O_BINARY on Windows or the descriptor does newline translation
# on a payload we byte-count. A no-op everywhere else.
_O_BINARY = getattr(os, "O_BINARY", 0)


def claims_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.comfy-downloads/claims`` and ensure it exists, owner-only."""
    base = state_dir(workspace) / CLAIMS_DIRNAME
    base.mkdir(parents=True, exist_ok=True, mode=STATE_DIR_MODE)
    if sys.platform != "win32":
        # Same reason as `state_dir`: mkdir's mode is masked by the umask and
        # the directory may predate this code.
        with contextlib.suppress(OSError):
            base.chmod(STATE_DIR_MODE)
    return base


def claim_filename(dest_key: str) -> str:
    """The claim file name for a destination key.

    Hashed rather than derived from the path: a destination is an arbitrary
    absolute path, and a claim named after one would hit the filesystem's name
    length limit and its separator rules. The key is expected to be already
    canonicalized by the caller (``models._dest_key``), so two spellings of one
    destination hash to one file.
    """
    return f"{hashlib.sha256(dest_key.encode('utf-8')).hexdigest()}.claim"


def claim_path(workspace: Path, dest_key: str) -> Path:
    """The claim path for ``dest_key`` under ``workspace``, creating ``claims/``."""
    return claims_dir(workspace) / claim_filename(dest_key)


def claim_marker_for(state_file: Path, dest_key: str) -> Path:
    """The claim that pairs with a state file addressed by path.

    The worker is handed ``--state <file>`` and never re-resolves a workspace
    (same reason as :func:`cancel_marker_for`), so it derives the claim from the
    state file's own directory. Creates nothing - the worker locks what is there.
    """
    return Path(state_file).parent / CLAIMS_DIRNAME / claim_filename(dest_key)


def _parse_claim_payload(raw: bytes | str) -> dict[str, Any] | None:
    """The claim payload in ``raw``, or None for every shape we cannot resolve.

    None for absent, empty, truncated mid-rewrite, corrupt, or carrying an id
    that could not name a state file. Every one of those is treated as *stale*
    by the caller, which is the safe direction now for a different reason than
    it used to be: the lock, not the payload, is what proves a destination is
    taken, so a payload nobody can read costs nothing but the id in an error
    message. A live holder is still refused on the strength of its lock.
    """
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    download_id = payload.get("download_id")
    if not isinstance(download_id, str) or not _SAFE_ID.match(download_id):
        return None
    return payload


def read_claim(path: Path) -> str | None:
    """The ``download_id`` recorded in the claim at ``path``, read *unlocked*.

    Advisory only, and there is exactly one legitimate caller: a submitter that
    could not take the lock and wants to name the holder in its refusal. It
    cannot be used to decide ownership — the payload it reads may be a rewrite
    in progress (the holder rewrites in place, so there is no atomic publish any
    more), and on Windows a locked byte range makes the read fail outright. Both
    come back None, which only costs the id in a message.

    Ownership questions go through :func:`lock_claim` and
    :meth:`ClaimLock.read_payload`, which read the same bytes under the lock.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    payload = _parse_claim_payload(raw)
    return payload["download_id"] if payload is not None else None


class ClaimLock:
    """An exclusive OS lock held on one destination's claim file.

    Held for as long as the holder owns the destination — for a worker, the
    whole transfer. Dropped by :meth:`release`, and by the kernel if the holder
    dies, which is what makes a SIGKILLed worker cost the next submitter nothing.

    The payload is read and rewritten **in place on the locked descriptor**.
    Never temp-file-plus-rename: a rename swaps the inode, and the lock lives on
    the inode, so the holder would be left guarding a file nobody can see.
    """

    def __init__(self, path: Path, fd: int) -> None:
        self._path = Path(path)
        self._fd: int | None = fd

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _descriptor(self) -> int:
        if self._fd is None:
            raise ValueError("this claim lock has already been released")
        return self._fd

    def read_payload(self) -> dict[str, Any] | None:
        """The claim payload, read under the lock. None when unreadable.

        Tolerant in exactly the way :func:`_parse_claim_payload` is, plus the
        read itself: a claim file we hold the lock on but cannot read tells us
        nothing about who owns the destination, and the lock has already
        answered that question.
        """
        fd = self._descriptor()
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                chunks.append(block)
        except OSError:
            return None
        return _parse_claim_payload(b"".join(chunks))

    def write_payload(self, download_id: str, dest: str) -> None:
        """Rewrite the payload in place. Raises ``OSError`` if it could not be.

        Same shape the claim has always carried, and still no url: a resolved
        download url can be presigned, and a claim needs an id and a path.
        """
        fd = self._descriptor()
        body = json.dumps(
            {"download_id": download_id, "dest": str(dest), "created_at": _now_iso()},
            indent=2,
        ).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view) :]

    def clear_payload(self) -> None:
        """Blank the payload, leaving the (lockable) file in place.

        An empty claim parses as no payload at all, which every reader treats as
        stale. Used where the old code unlinked: a submit whose worker never
        started has nothing to hand the destination to.
        """
        os.ftruncate(self._descriptor(), 0)

    def release(self) -> None:
        """Close the descriptor, which drops the lock. Idempotent.

        Deliberately **not** an unlink. Unlinking a claim while holding its lock
        leaves every descriptor already opened on that path guarding an orphan
        inode, so two submitters could each hold a "lock" on a different inode
        for one destination. Bounded growth is the cheaper problem, and
        :func:`prune` bounds it.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            os.close(fd)

    def __enter__(self) -> ClaimLock:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.release()
        return False


def _lock_fd_nb(fd: int) -> bool:
    """Take the exclusive lock on ``fd`` without blocking. False when held.

    Raises ``OSError`` for anything that is not contention — a filesystem with
    no locking is a degradation the caller has to handle, not a competitor.
    """
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in _LOCK_CONTENDED_ERRNOS:
            return False
        raise
    return True


def _is_same_file(fd: int, path: Path) -> bool:
    """True while ``fd`` still names the file at ``path``."""
    try:
        here = os.fstat(fd)
        there = os.stat(path)
    except OSError:
        return False
    return (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)


def lock_claim(path: Path, *, blocking_timeout: float = 0.0) -> ClaimLock | None:
    """Take the exclusive lock on the claim at ``path``, or None if it is held.

    ``O_CREAT`` without ``O_EXCL``: the claim file is a lock target that outlives
    every holder, so it usually already exists and creating it decides nothing.
    What decides is the lock.

    ``blocking_timeout`` bounds a non-blocking retry loop rather than issuing a
    blocking lock, so the wait can never outlive the caller's patience. Zero (the
    default) is a single attempt — the submit path wants an immediate answer,
    because a lock that is held *is* the answer. A freshly spawned worker passes
    a few seconds, because a competitor may hold the lock for the moment it takes
    to inspect the payload.

    **The re-stat guard.** Between our ``open`` and our lock, the file we opened
    can be unlinked (only :func:`prune` does this, and only on POSIX) and a
    different one created at the same path. We would then hold a real lock on an
    orphan inode while another process holds a real lock on the live one. So
    after locking, ``fstat`` the descriptor against ``stat`` of the path: a
    different (st_dev, st_ino) — or a path that is gone — means we lost the race,
    so close and retry, up to :data:`CLAIM_LOCK_RESTAT_ATTEMPTS` times. Windows
    skips the guard (inode identity there is not reliable) and pays for it by
    never unlinking a claim file at all, which removes the race instead.

    Raises ``OSError`` when the claim cannot be opened or the filesystem cannot
    lock. That is not a collision and must not be reported as one: the caller
    degrades to its advisory guard and says so.
    """
    path = Path(path)
    deadline = time.monotonic() + max(0.0, float(blocking_timeout))
    for _ in range(CLAIM_LOCK_RESTAT_ATTEMPTS):
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR | _O_BINARY, STATE_FILE_MODE)
        keep = False
        try:
            if sys.platform != "win32":
                # `os.open`'s mode is masked by the umask, exactly as
                # `write_path`'s is, and the file may predate this code.
                with contextlib.suppress(OSError):
                    os.fchmod(fd, STATE_FILE_MODE)
            while not _lock_fd_nb(fd):
                if time.monotonic() >= deadline:
                    return None
                time.sleep(CLAIM_LOCK_POLL_S)
            if sys.platform == "win32" or _is_same_file(fd, path):
                keep = True
                return ClaimLock(path, fd)
        finally:
            if not keep:
                with contextlib.suppress(OSError):
                    os.close(fd)
    # Every attempt raced a concurrent unlink. Reporting "held" rather than
    # "acquired" is the safe direction: the caller refuses instead of letting a
    # second transfer into a destination whose ownership we could not settle.
    return None


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
    kind: str = "background"
    needs_civitai_auth: bool = False
    needs_hf_auth: bool = False

    @property
    def is_foreground(self) -> bool:
        """True when this record's pid is a user CLI process, not a detached worker.

        The distinction is only ever load-bearing in the *refusing* direction
        (``download-cancel`` must not signal a foreground pid's process group),
        so an unrecognized value reads as ``foreground`` — the non-cancellable
        side. See :data:`_TOLERANT_FALLBACKS`. Only a record with no ``kind`` at
        all reads as ``background``, because that is what it provably was.
        """
        return self.kind == "foreground"

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

    Deliberately not ``file_utils.atomic_write_text`` — see the write-policy note
    in ``comfy_cli/file_utils.py`` (tier 2, secret-adjacent state).

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
    "kind": lambda v: v in ("background", "foreground"),
    "needs_civitai_auth": lambda v: isinstance(v, bool),
    "needs_hf_auth": lambda v: isinstance(v, bool),
}

# Fields whose validator failure replaces *the field* with the value below
# rather than rejecting the whole record.
#
# `kind` is in here because rejecting the record is the more dangerous outcome by
# far. A rejected record reads as absent to every caller, including the
# destination-claim scan — so one unrecognized `kind` on a *live* download would
# make its claim invisible and let a second writer into the same file, which is
# the exact corruption the claim exists to prevent. Keeping the record and
# distrusting one field is the smaller loss.
#
# The substitute is `"foreground"`, *not* the dataclass default. `kind` is the
# one field that gates a destructive action, so tolerance here has to fail
# closed: `download-cancel` refuses a live `foreground` record and sends the user
# to Ctrl-C, whereas a `background` one reaches `kill_worker` ->
# `os.killpg(os.getpgid(pid), ...)`. Guessing "background" for a value we could
# not parse would aim that killpg at a pid we have no reason to believe is a
# detached worker — and if it is in fact a foreground record whose `kind` was
# corrupted, at the user's own shell job. Refusing to cancel a record we cannot
# read is recoverable (the process is Ctrl-C-able, and the record reconciles once
# it exits); signalling the wrong process group is not.
#
# A record with no `kind` key at all is a different case and is *not* routed
# here: it falls through to the dataclass default, `"background"`, because every
# record written before this field existed really was a detached worker.
_TOLERANT_FALLBACKS: dict[str, Any] = {"kind": "foreground"}


def read_path(path: Path) -> DownloadState | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    known = set(DownloadState.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    for key, value in list(filtered.items()):
        validator = _FIELD_VALIDATORS.get(key)
        if validator is not None and not validator(value):
            if key in _TOLERANT_FALLBACKS:
                filtered[key] = _TOLERANT_FALLBACKS[key]
                continue
            return None
    try:
        return DownloadState(**filtered)
    except TypeError:
        # Truncated/legacy file missing a required field — treat as absent.
        return None


def delete(workspace: Path, download_id: str) -> bool:
    """Remove one state file. Returns False if it could not be removed.

    Used to *withdraw a claim*: both the background and the foreground submit
    paths write their record and then re-scan for a competitor, and the racer
    that loses that comparison has to take its record back off disk before it
    exits. Leaving it behind would be worse than never having written it — an
    abandoned ``downloading`` record with a pid that is about to disappear reads
    as a live claim until something reconciles it, so it would refuse every later
    submission to that destination for no reason.

    Absent is success: the caller's goal is "this claim is gone", and a record
    that was never written (or already swept) satisfies it.
    """
    try:
        state_path(workspace, download_id).unlink(missing_ok=True)
        return True
    except (OSError, ValueError):
        return False


def _sort_key(state: DownloadState) -> tuple[str, str]:
    """The newest-first ordering :func:`list_all` and :func:`prune` share.

    They must agree: ``prune`` evicts the records that fall off the end of the
    list ``downloads`` shows, so a divergent tiebreak would drop a record the
    user is still looking at while keeping one they aren't.
    """
    return (state.started_at or "", state.id)


def list_all(workspace: Path) -> list[DownloadState]:
    """Every readable state file, newest ``started_at`` first."""
    base = Path(workspace) / STATE_DIRNAME
    if not base.is_dir():
        return []
    states = [s for s in (read_path(p) for p in sorted(base.glob("*.json"))) if s is not None]
    states.sort(key=_sort_key, reverse=True)
    return states


def _has_partials(dest: str) -> bool:
    """True while a ``.part`` sibling of ``dest`` still holds bytes on disk.

    This is the same set of files ``download-cancel`` reclaims, so it is also
    the only handle a user has left on those bytes once a download has failed.
    """
    from comfy_cli import file_utils

    try:
        return bool(file_utils.partial_paths_for(Path(dest)))
    except (OSError, ValueError):
        # An unreadable parent or a nonsense dest tells us nothing about the
        # partial; assume there is one rather than deleting the record that
        # points at it.
        return True


def _remove_record(path: Path) -> bool:
    """Delete one state file and its companions. False if the record survived.

    The ``<id>.log`` and ``<id>.cancel`` siblings are unreachable once the
    record naming them is gone — no verb can look them up — so they go with it,
    otherwise pruning the records alone would leave the directory growing.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    for companion in (path.with_name(f"{path.stem}.log"), cancel_marker_for(path)):
        try:
            companion.unlink(missing_ok=True)
        except OSError:
            continue
    return True


def _sweep_claims(workspace: Path) -> None:
    """Drop claim files nothing owns any more, and legacy temp leftovers.

    A claim file is never unlinked on release (see :class:`ClaimLock`), so this
    is the only thing that bounds ``claims/`` below "every destination this
    workspace ever downloaded to". It is a nicety, not a correctness step: a
    left-behind claim costs ~100 bytes and refuses nobody, because ownership is
    the *lock*, which no longer exists once its holder is gone.

    Two conditions, both required, and the lock has to be held for the unlink:

    * we can take the lock — so no transfer owns the destination right now; and
    * the payload does not name a download that is still live. That second check
      is what keeps the sweep out of the **spawn gap**: between a submit's
      release and its worker's lock nobody holds the claim, and unlinking it
      there would make the worker come up, find an empty payload, and refuse the
      destination it was just handed.

    POSIX only. On Windows a claim file is never unlinked at all — inode
    identity is unreliable there, so :func:`lock_claim` cannot run its re-stat
    guard, and without that guard an unlink is how two holders end up locking
    two different inodes for one destination.

    Temp files are the previous implementation's private staging names, kept
    only so a workspace upgraded from it does not carry them forever. Best
    effort throughout, like the rest of :func:`prune`.
    """
    base = Path(workspace) / STATE_DIRNAME / CLAIMS_DIRNAME
    try:
        if not base.is_dir():
            return
        entries = sorted(base.iterdir())
    except OSError:
        return
    tmp_cutoff = time.time() - CLAIM_TMP_MAX_AGE_S
    for path in entries:
        if path.name.endswith(CLAIM_TMP_SUFFIX):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < tmp_cutoff:
                    path.unlink()
            continue
        if not path.name.endswith(".claim") or sys.platform == "win32":
            continue
        try:
            lock = lock_claim(path)
        except OSError:
            continue
        if lock is None:
            continue
        try:
            payload = lock.read_payload()
            download_id = payload["download_id"] if payload is not None else None
            if download_id is not None:
                record = read(workspace, download_id)
                if record is not None and (reconcile(record).status in ACTIVE_STATUSES or worker_alive(record)):
                    continue
            with contextlib.suppress(OSError):
                path.unlink()
        finally:
            lock.release()


def prune(workspace: Path) -> int:
    """Drop finished download records that are stale or over the retention cap.

    Nothing else ever removes a record, so ``<workspace>/.comfy-downloads``
    would otherwise grow for the life of the workspace — and ``list_all`` reads
    and JSON-parses every file in it on every ``comfy model downloads`` call.

    A record is removed when it is **terminal** (:data:`TERMINAL_STATUSES`) and
    either:

    * its ``updated_at`` is older than :data:`PRUNE_MAX_AGE_S` — except for a
      ``failed``/``cancelled`` record whose destination still has a ``.part``
      sibling. Those bytes are on disk and the record is the user's only handle
      for reclaiming them with ``download-cancel``, which is the same contract
      :func:`comfy_cli.file_utils.cleanup_partials` is written to; or
    * it falls outside the newest :data:`PRUNE_MAX_TERMINAL_RECORDS` terminal
      records. That ceiling is what makes the directory *bounded* rather than
      merely self-expiring, so unlike the age rule it applies unconditionally.

    An in-flight record (``starting``/``downloading``) is never touched at any
    age, and never counts toward — or is evicted by — the cap.

    Also sweeps ``claims/`` (see :func:`_sweep_claims`): a claim file is never
    unlinked on release — the lock lives on its inode — so without this the
    directory would grow to one small file per destination the workspace has
    ever downloaded to, the same unbounded growth the record rules above exist
    to prevent. Swept claims do not count toward the returned total, which stays
    "records removed".

    Every step is best effort, exactly like :func:`write_path`'s OSError
    handling: a read-only state directory, a permissions problem, or a file a
    concurrent prune already removed must never raise into a download. Returns
    the number of records actually removed.
    """
    # Suppressed here rather than inside the sweep: `_sweep_claims` guards its
    # own directory walk, but it then calls `read`, which resolves the state
    # directory (an `mkdir`) and only catches `ValueError`. That `OSError` would
    # escape `prune` and land in a download, which is exactly what the contract
    # above promises never happens.
    with contextlib.suppress(OSError):
        _sweep_claims(workspace)
    base = Path(workspace) / STATE_DIRNAME
    try:
        if not base.is_dir():
            return 0
        paths = sorted(base.glob("*.json"))
    except OSError:
        return 0

    terminal: list[tuple[DownloadState, Path]] = []
    for path in paths:
        state = read_path(path)
        if state is not None and state.status in TERMINAL_STATUSES:
            terminal.append((state, path))
    terminal.sort(key=lambda item: _sort_key(item[0]), reverse=True)

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=PRUNE_MAX_AGE_S)
    removed = 0
    for index, (state, path) in enumerate(terminal):
        if index < PRUNE_MAX_TERMINAL_RECORDS:
            updated = _parse_iso(state.updated_at)
            if updated is None or updated >= cutoff:
                continue
            if state.status in ("failed", "cancelled") and _has_partials(state.dest):
                continue
        if _remove_record(path):
            removed += 1
    return removed


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
    if not state.pid or state.pid <= 0:
        # Not just `is None`. The field validator accepts any non-bool int, so a
        # corrupt or tampered record can carry a negative pid — and
        # `psutil.Process(-1)` raises ValueError, which `utils.is_running` does
        # not catch. One bad state file would then traceback out of every command
        # that reconciles, including `model download` itself. `is_worker_process`
        # and `kill_worker` already screen the same way.
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
    """The ``download-status`` / ``downloads`` envelope row for one download.

    ``kind`` is reported because these rows are now the only place a foreground
    download is visible, and the two kinds are not interchangeable to a consumer:
    ``download-cancel`` refuses a live ``foreground`` row (its pid is a user CLI
    process sharing a terminal's process group, so signalling it would kill the
    surrounding shell job). Without the field the only way to learn that is to
    try the cancel and read the refusal.
    """
    return {
        "id": state.id,
        "status": state.status,
        "kind": state.kind,
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
