import functools
import json
import logging
import os
import pathlib
import stat
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable
from http import HTTPStatus

from pathspec import PathSpec

from comfy_cli import constants, ui
from comfy_cli._safe_exec import BinaryNotFoundError, resolve_required_binary
from comfy_cli.http import DEFAULT_HTTP_TIMEOUT, DOWNLOAD_TIMEOUT
from comfy_cli.output.sanitize import sanitize_value

logger = logging.getLogger(__name__)


def cache_dir() -> pathlib.Path:
    """comfy-cli's per-user cache root.

    ``COMFY_CACHE_DIR`` wins; else ``$XDG_CACHE_HOME/comfy-cli``; else ``~/.cache/comfy-cli``.

    Both env vars are resolved with ``os.path.expanduser`` (never raises, unlike
    ``pathlib.Path.expanduser``, when the home directory can't be determined —
    e.g. no ``HOME`` and no passwd entry, routine in a container) and made
    absolute, so the result doesn't depend on the calling process's cwd. That
    matters because a relative override would otherwise resolve differently in
    every process — including a background refresher that launches its child
    from inside this very directory.
    """
    explicit = os.environ.get("COMFY_CACHE_DIR", "").strip()
    if explicit:
        return pathlib.Path(os.path.abspath(os.path.expanduser(explicit)))
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or os.path.expanduser("~/.cache")
    return pathlib.Path(os.path.abspath(os.path.expanduser(base))) / "comfy-cli"


# ---------------------------------------------------------------------------
# Atomic writes — the write policy
# ---------------------------------------------------------------------------
#
# Every hand-rolled tmp+``os.replace`` writer in comfy-cli falls into one of
# four tiers. The tier decides whether the writer uses the shared helpers below
# or stays bespoke; it is decided by *what the file can contain*, never by
# convenience. The helpers deliberately take **no** ``mode=`` parameter —
# parameterizing permissions was considered and rejected, because a writer whose
# payload needs 0600 needs more than a mode (an ``O_EXCL`` open at that mode, a
# 0700 parent, an fsync) and is clearer written out in full at its call site.
#
# Tier 1 — secrets. ``comfy_cli/auth/store.py`` ``_write_all`` stays bespoke:
#   the tmp file is opened ``O_CREAT|O_EXCL`` with mode 0600 *at open*, so the
#   OAuth tokens are never briefly world-readable under a permissive umask; the
#   parent directory is forced to 0700; the write is fsynced.
#
# Tier 2 — secret-adjacent state. ``comfy_cli/download_state.py`` ``write_path``
#   stays bespoke: chmod 0600 + 0700 parent + fsync, because the resolved
#   download URL persisted in the state file can embed a presigned/SAS token,
#   which is bearer credential material even though the file itself is nominally
#   bookkeeping.
#
# Tier 3 — cross-process state. ``atomic_write_text(..., fsync=True)``:
#   umask mode, durable. A second process (a watcher, a later CLI invocation, an
#   agent shelling out) reads these, and losing one to power failure loses real
#   work. Members: ``jobs_state.write``, ``command/project.py``
#   ``_write_assets_lock``.
#
# Tier 4 — regenerable caches and manifests. ``fsync=False``: the content can be
#   recomputed or refetched, so paying a sync per write buys nothing. Callers may
#   additionally wrap the call best-effort (``except OSError: pass``) when a
#   read-only or full cache directory must not break the command. Members:
#   ``cql/loader``, ``command/outdated.py`` ``_save_cache``,
#   ``command/templates.py`` ``_persist_cache``, the skills manifest writers,
#   ``command/workflow.py``.
#
# Two invariants recorded here because they are the *reason* for a tier
# assignment, and a future change to either moves a writer between tiers:
#
# (a) ``JobState`` files are tier 3 (umask mode), not tier 2, because their
#     ``outputs`` are plain ``/view?filename=...`` URLs — they carry no embedded
#     credential, and the CLI never asks the cloud jobs API for ``?short_link=``
#     responses. If either changes (outputs start carrying signed URLs, or the
#     CLI starts requesting short links), jobs_state moves to tier 2 and needs
#     0600 like download_state.
#
# (b) No writer here fsyncs the parent directory *before* ``os.replace`` returns
#     durably — ``fsync=True`` syncs the file's contents and then the parent
#     directory, but on a power failure between the two an already-``replace``d
#     rename can still be lost. This is accepted: the failure mode is losing the
#     *newest* write, never observing a torn or half-written file. Contents are
#     never torn; renames may be lost.

# The process umask, captured once at import. os has no getter, so reading it
# means the classic set-and-restore dance; doing it here (single-threaded under
# the import lock) avoids re-running that dance per write, where it would leave
# a window in which any *other* thread's file lands at 0666/0777 — and where two
# overlapping probes can restore each other's zero and strand the umask at 0.
# Same reasoning, same idiom as `comfy_cli.command.transfer`.
_UMASK = os.umask(0)
os.umask(_UMASK)


def _apply_destination_mode(fd: int, path: pathlib.Path) -> None:
    """Give a mkstemp tmp file the permissions its destination should end up with.

    ``tempfile.mkstemp`` hardcodes the tmp file to 0600, and ``os.replace`` carries
    that mode onto the destination — so without this, writing through a tmp sibling
    would quietly strip the group/other read access a plain ``open(path, "w")``
    would have granted, turning a shared model file owner-only. Reuse the existing
    destination's permission bits when there is one, else fall back to the
    umask-derived default a fresh ``open()`` would have produced (0666 & ~umask).

    Only the 0o777 bits are copied: ``os.replace`` would otherwise carry a set-ID
    (or sticky) destination's bits onto freshly downloaded, network-controlled
    bytes, where an in-place write would have cleared them.

    Applied to the open descriptor rather than the tmp file's *name*: ``os.chmod``
    resolves a path and follows symlinks, so a name swap between ``mkstemp`` and
    the chmod would retarget the mode change at an arbitrary file (CWE-59) —
    reopening the exact race ``mkstemp``'s ``O_EXCL`` was chosen to close.

    Best-effort: a platform without POSIX permissions (Windows has no ``fchmod``)
    is left alone, matching the surrounding fsync handling.
    """
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:
        return
    try:
        dest_mode = stat.S_IMODE(os.stat(path).st_mode) & 0o777
    except OSError:
        dest_mode = 0o666 & ~_UMASK
    try:
        fchmod(fd, dest_mode)
    except OSError:
        pass


def atomic_write_text(path: pathlib.Path, content: str, *, fsync: bool = False) -> None:
    """Atomically write ``content`` to ``path`` via a sibling tmp file + ``os.replace``.

    The write goes to a uniquely-named tmp file in the same directory (so the rename
    stays on one filesystem and is atomic), which is then renamed over ``path``.
    A SIGINT or crash mid-write therefore never leaves a half-written or empty
    file at the destination — readers see either the old contents or the new.
    On any failure the tmp file is cleaned up and the exception re-raised.

    The tmp file is created with ``tempfile.mkstemp`` (``O_CREAT | O_EXCL``, no
    symlink following) in the destination directory, so concurrent writers never
    collide on a shared name and a pre-planted symlink can't redirect the write
    (CWE-377).

    Args:
        path: destination file. Parent directories are created if missing.
        content: text to write (UTF-8).
        fsync: if True, flush the tmp file's contents to disk before the rename
            and fsync the destination directory afterwards so both the data and
            the rename survive power loss, at the cost of a sync. Best-effort:
            a failing fsync is ignored, matching the prior per-site behavior.
    """
    _atomic_write(path, content.encode("utf-8"), fsync=fsync)


def atomic_write_bytes(path: pathlib.Path, data: bytes, *, fsync: bool = False) -> None:
    """Atomically write ``data`` to ``path`` — the bytes twin of :func:`atomic_write_text`.

    Identical semantics (same tmp-file creation, same permission handling, same
    cleanup-on-failure); it just skips the UTF-8 encode for a caller that already
    holds bytes. See :func:`atomic_write_text` for the full contract.
    """
    _atomic_write(path, data, fsync=fsync)


def _atomic_write(path: pathlib.Path, data: bytes, *, fsync: bool) -> None:
    """Shared implementation behind :func:`atomic_write_text` / :func:`atomic_write_bytes`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp gives a unique, O_EXCL, non-symlink-following fd opened O_RDWR in the
    # destination directory — same filesystem, so the os.replace below is atomic.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        try:
            file_obj = os.fdopen(fd, "wb")
        except BaseException:
            # fdopen only fails *before* taking ownership of the descriptor, so
            # this is the one place the raw fd still has to be closed by hand;
            # everywhere below, closing is the file object's job.
            os.close(fd)
            raise
        with file_obj as f:
            f.write(data)
            if fsync:
                f.flush()
                try:
                    os.fsync(f.fileno())  # O_RDWR fd, so fsync works on Windows too
                except OSError:
                    pass
            # Inside the `with`: the mode is applied through this descriptor, so
            # it has to happen before the file object closes it.
            _apply_destination_mode(f.fileno(), path)
        os.replace(tmp_name, path)
        if fsync:
            # Also fsync the parent directory so the rename itself is durable.
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
                finally:
                    os.close(dir_fd)
            except OSError:
                # e.g. Windows can't open a directory for fsync; best-effort.
                pass
    except BaseException:
        # BaseException (not just Exception) so a KeyboardInterrupt mid-write
        # still cleans up the tmp file, per the docstring.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class DownloadException(Exception):
    pass


# ``(completed_bytes, total_bytes_or_None)``. Called from inside the transfer
# loop, so implementations must be cheap and must never raise — see
# ``_report_progress``.
ProgressCallback = Callable[[int, int | None], None]


class DownloadCancelled(Exception):
    """Raised by a progress callback to abort an in-flight download.

    The one thing an observer is allowed to say that isn't advisory. The
    background-download worker polls its cancel sentinel from the progress
    callback and raises this to unwind out of the transfer; every other
    exception a callback raises is swallowed by :func:`_report_progress`.
    """


def _report_progress(callback: ProgressCallback | None, completed: int, total: int | None) -> None:
    """Invoke a progress callback, swallowing anything it raises.

    A misbehaving observer (a full disk while persisting state, say) must never
    abort an otherwise-healthy download. :class:`DownloadCancelled` is the sole
    exception: it means the caller wants the transfer stopped, not that
    reporting failed.
    """
    if callback is None:
        return
    try:
        callback(completed, total)
    except DownloadCancelled:
        raise
    except Exception:  # noqa: BLE001 — progress reporting is strictly advisory
        pass


def guess_status_code_reason(status_code: int, message: str) -> str:
    """Describe an HTTP failure for a human.

    Every branch but 401 returns a canned string. The 401 branch echoes the
    server's own JSON ``message`` back to the terminal, so that one value is
    attacker-chosen: it goes through :func:`sanitize_value` before it is
    interpolated. Sanitizing here rather than at each print site means every
    consumer of the reason — the ``comfy model download`` error line, the
    background-download state file, ``comfy node install`` — gets the same
    guarantee without having to remember. Markup escaping is deliberately NOT
    applied here: not every consumer renders through a markup-interpreting
    sink, and the escaping backslashes would be visible in the ones that don't.
    """
    if status_code == 401:

        def parse_json(input_data):
            try:
                # Check if the input is a byte string
                if isinstance(input_data, bytes):
                    # Decode the byte string to a regular string
                    input_data = input_data.decode("utf-8")

                # Parse the string as JSON
                return json.loads(input_data)

            except json.JSONDecodeError as e:
                # Handle JSON decoding error
                print(f"JSON decoding error: {e}")

        msg_json = parse_json(message)
        if msg_json is not None:
            if "message" in msg_json:
                server_message = sanitize_value(msg_json["message"])
                return f"Unauthorized download ({status_code}).\n{server_message}\nor you can set a CivitAI API token using `comfy model download --set-civitai-api-token` or via the `{constants.CIVITAI_API_TOKEN_ENV_KEY}` environment variable"
        return f"Unauthorized download ({status_code}), you might need to manually log into a browser to download this"
    elif status_code == 403:
        return f"Forbidden url ({status_code}), you might need to manually log into a browser to download this"
    elif status_code == 404:
        return "File not found on server (404)"
    return f"Unknown error occurred (status code: {status_code})"


def check_unauthorized(url: str, headers: dict | None = None) -> bool:
    """
    Perform a GET request to the given URL and check if the response status code is 401 (Unauthorized).

    Args:
        url (str): The URL to send the GET request to.
        headers (Optional[dict]): Optional headers to include in the request.

    Returns:
        bool: True if the response status code is 401, False otherwise.
    """
    # Imported lazily: requests costs ~30ms to import and this module is on
    # the import path of every CLI invocation.
    import requests

    try:
        with requests.get(
            url, headers=headers, allow_redirects=True, stream=True, timeout=DEFAULT_HTTP_TIMEOUT
        ) as response:
            return response.status_code == 401
    except requests.RequestException:
        # If there's an error making the request, we can't determine if it's unauthorized
        return False


def _poll_aria2_download(download, progress_callback: ProgressCallback | None = None) -> None:
    """Poll an aria2 download until completion, showing progress.

    ``progress_callback`` (optional) receives ``(completed_bytes, total_bytes)``
    on every poll, with ``total_bytes`` None until aria2 knows the size. It is
    fed from the same ``completed_length``/``total_length`` pair the progress bar
    uses, so a background worker sees exactly what the human would.

    If the callback raises :class:`DownloadCancelled` the daemon-side transfer is
    removed before the exception propagates: with aria2 the bytes move inside the
    aria2c process, not this one, so simply walking away would leave it happily
    finishing a download the user just cancelled.

    A :class:`KeyboardInterrupt` gets the same treatment, and needs it more.
    Ctrl-C is what `comfy model download` tells a user to press to stop a
    foreground transfer — and it lands here, in the poll loop's sleep, not in a
    progress callback, so `DownloadCancelled` never fires. Without this the
    interrupted CLI would tear down its destination *claim* on the way out while
    the aria2c daemon carried on writing to that same destination: the exact
    unguarded interleaving the claim exists to prevent.
    """
    import time

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        try:
            while True:
                try:
                    download.update()
                except Exception as e:
                    raise DownloadException(f"Lost connection to aria2 RPC server: {e}") from e

                total = download.total_length if download.total_length > 0 else None
                if total is not None:
                    progress.update(task, total=total, completed=download.completed_length)
                _report_progress(progress_callback, download.completed_length, total)

                if download.is_complete:
                    if total is not None:
                        progress.update(task, completed=total)
                        _report_progress(progress_callback, total, total)
                    break
                elif download.has_failed:
                    raise DownloadException(
                        f"aria2 download failed: {download.error_message} (code: {download.error_code})"
                    )
                elif download.is_removed:
                    raise DownloadException("aria2 download was removed before completion")

                time.sleep(0.5)
        except (DownloadCancelled, KeyboardInterrupt):
            _remove_aria2_download(download)
            raise


def _remove_aria2_download(download) -> None:
    """Stop and forget a daemon-side aria2 transfer. Best effort."""
    try:
        download.remove(force=True, files=True)
    except Exception:  # noqa: BLE001 — the daemon may already have dropped it
        pass


def _download_file_aria2(
    url: str,
    local_filepath: pathlib.Path,
    headers: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Download a file using aria2 RPC."""
    try:
        import aria2p
    except ImportError:
        raise DownloadException(
            "aria2p is required for aria2 downloads. Install it with: pip install aria2p\n"
            "You also need a running aria2c daemon. See: https://aria2.github.io/"
        ) from None

    server = os.environ.get(constants.ARIA2_SERVER_ENV_KEY)
    if not server:
        raise DownloadException(
            f"aria2 downloader selected but {constants.ARIA2_SERVER_ENV_KEY} environment variable is not set.\n"
            f"Set it to your aria2 RPC server URL, e.g.: export {constants.ARIA2_SERVER_ENV_KEY}=http://localhost:6800"
        )

    secret = os.environ.get(constants.ARIA2_SECRET_ENV_KEY, "")

    from urllib.parse import urlparse

    if "://" not in server:
        server = f"http://{server}"
    parsed = urlparse(server)
    if not parsed.hostname:
        raise DownloadException(f"Invalid aria2 server URL (cannot parse hostname): {server}")
    host = f"{parsed.scheme}://{parsed.hostname}"
    port = parsed.port or 6800

    try:
        api = aria2p.API(aria2p.Client(host=host, port=port, secret=secret))
    except Exception as e:
        raise DownloadException(f"Failed to connect to aria2 RPC server at {server}: {e}") from e

    options = {
        "dir": str(local_filepath.parent),
        "out": local_filepath.name,
    }

    if headers:
        options["header"] = [f"{k}: {v}" for k, v in headers.items()]

    try:
        download = api.add_uris([url], options=options)
    except Exception as e:
        raise DownloadException(f"Failed to add download to aria2: {e}") from e

    _poll_aria2_download(download, progress_callback)

    if not local_filepath.exists():
        raise DownloadException(f"aria2 download completed but file not found at expected path: {local_filepath}")


_VALID_DOWNLOADERS = {"httpx", "aria2"}

_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF = 2  # seconds multiplier
# HTTP statuses that typically indicate a transient server-side or rate-limit
# problem worth retrying with backoff. Auth/not-found/redirect statuses stay
# out of this set so they fail fast.
_RETRIABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class _TransientHTTPStatusError(Exception):
    """Retriable HTTP status returned by the server (e.g. 500/503/429)."""

    def __init__(self, status_code: int, reason: str):
        self.status_code = status_code
        self.reason = reason
        super().__init__(f"HTTP {status_code}: {reason}")


# Built on first use, not at import: httpx costs 11 ms warm to import, more on
# a cold cache, and most importers of this module only want ``atomic_write_*``.
@functools.cache
def _download_timeout():
    import httpx

    return httpx.Timeout(10.0, read=300.0)


@functools.cache
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    import httpx

    return (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError, httpx.ProxyError)


@functools.cache
def _retriable_exceptions() -> tuple[type[BaseException], ...]:
    return (*_transient_exceptions(), _TransientHTTPStatusError)


def _cleanup_partial(filepath: pathlib.Path) -> None:
    """Remove a partially downloaded file if it exists."""
    try:
        filepath.unlink(missing_ok=True)
    except OSError:
        pass


# The httpx downloader streams into a sibling of the destination named
# ``<dest name>.<mkstemp token>.part`` and renames it onto the destination only
# once the last byte has landed. The suffix is public in the sense that a killed
# transfer leaves one on disk, so `download-cancel` has to be able to find it —
# hence `partial_paths_for` below rather than an ad-hoc glob at the call site.
_PART_SUFFIX = ".part"
# ``tempfile.mkstemp`` fills the middle with exactly 8 characters from this
# alphabet (``tempfile._RandomNameSequence``). Matching its shape — not just the
# prefix and suffix — is what stops `partial_paths_for` from claiming an
# unrelated user file that happens to sit beside the model and end in ".part".
_MKSTEMP_TOKEN_LEN = 8
_MKSTEMP_TOKEN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
# A filesystem caps one path *component* (NAME_MAX: 255 bytes on Linux/ext4,
# APFS and NTFS the same in practice), and mkstemp builds `<prefix><8><suffix>`
# — so the prefix has to leave 8 + len(".part") bytes free or the create fails
# with ENAMETOOLONG where the old `open(dest, "wb")` succeeded. Destination
# names come from remote metadata (CivitAI `file["name"]`, HF path) and from
# `--filename`, none of which cap length, so this is reachable input rather
# than a hypothetical.
_NAME_MAX = 255
# What is left for the destination name once the separating ".", mkstemp's token
# and the ".part" suffix have taken their share.
_PART_STEM_MAX = _NAME_MAX - len(".") - _MKSTEMP_TOKEN_LEN - len(_PART_SUFFIX)


def _part_prefix(name: str) -> str:
    """The mkstemp ``prefix`` used for ``name``'s ``.part`` siblings.

    Normally just ``name + "."``. A destination name too long to also carry the
    token and suffix is truncated to fit — on a *byte* basis, since NAME_MAX
    counts bytes, but on a character boundary so the result stays valid UTF-8.
    The trailing ``"."`` is appended after the cut, so the prefix always ends in
    the separator :func:`partial_paths_for` slices on.

    Two destination names agreeing for that many bytes then share a temp
    namespace, which only affects which temps :func:`cleanup_partials` claims —
    a far smaller problem than being unable to name a temp at all.
    """
    encoded = name.encode("utf-8", "surrogatepass")
    if len(encoded) > _PART_STEM_MAX:
        name = encoded[:_PART_STEM_MAX].decode("utf-8", "ignore")
    return name + "."


def partial_paths_for(local_filepath: pathlib.Path) -> list[pathlib.Path]:
    """Every ``.part`` sibling this module would have created for ``local_filepath``.

    A transfer killed uncleanly (SIGKILL, OOM, power loss) never gets to run its
    own cleanup, so its ``.part`` file outlives it. This is how the cancel path
    finds those bytes; nothing else on disk is ever matched.
    """
    prefix = _part_prefix(local_filepath.name)
    try:
        entries = list(local_filepath.parent.iterdir())
    except OSError:
        return []

    matches = []
    for entry in entries:
        name = entry.name
        if not name.startswith(prefix) or not name.endswith(_PART_SUFFIX):
            continue
        token = name[len(prefix) : -len(_PART_SUFFIX)]
        if len(token) != _MKSTEMP_TOKEN_LEN or not set(token) <= _MKSTEMP_TOKEN_CHARS:
            continue
        matches.append(entry)
    return sorted(matches)


def cleanup_partials(local_filepath: pathlib.Path) -> int:
    """Best-effort removal of every ``.part`` sibling; returns how many went away.

    Used by `download-cancel`, which promises to reclaim the disk a dead worker
    was using. The destination itself is never touched here.
    """
    removed = 0
    for partial in partial_paths_for(local_filepath):
        try:
            partial.unlink()
        except OSError:
            continue
        removed += 1
    return removed


# The atomic-write helpers above stream through a sibling named
# ``<dest name>.<mkstemp token>.tmp``. Unlike ``.part``, nothing ever *asks* for
# one by name — they are pure in-flight scaffolding — so the only reason to name
# the suffix here is to sweep up the corpses a killed writer leaves behind.
_TMP_SUFFIX = ".tmp"
# An in-flight atomic write lives for milliseconds, so an hour puts a live write
# far outside the window under any normal clock — see the caveat in
# ``cleanup_stale_tmp_files`` for the abnormal ones — while still bounding how
# long a corpse survives in a long-running agent/CI environment.
_TMP_STALE_SECONDS = 3600


def cleanup_stale_tmp_files(
    directory: pathlib.Path,
    *,
    older_than_seconds: float = _TMP_STALE_SECONDS,
    stem_suffix: str = "",
) -> int:
    """Best-effort removal of stranded ``atomic_write_*`` temps in ``directory``.

    A writer killed uncleanly (SIGKILL, OOM, power loss) never runs its
    unlink-on-exception cleanup, so its ``<dest>.<mkstemp token>.tmp`` sibling
    outlives it — and mkstemp mints a fresh token per attempt, so nothing bounds
    how many a crash-prone process leaves.

    A candidate must be a *regular file* (the entry's own metadata decides, so a
    symlink shaped like a temp can't lend an unrelated target's mtime to the age
    test, and a dangling one is skipped rather than raised on), carry mkstemp's
    exact ``<stem>.<8 chars>.tmp`` shape, and have an mtime older than
    ``older_than_seconds``.

    Two honest limits on that, both worth knowing before adding a caller:

    * The shape proves nothing about *who* wrote the file. ``db.a1b2c3d4.tmp``
      matches it by coincidence, and this repo's own bespoke temps in
      ``auth/store.py`` / ``download_state.py``
      (``<name>.<pid>.<secrets.token_hex(4)>.tmp``) match it by construction.
      Pass ``stem_suffix`` to also require the destination stem to end in it
      (e.g. ``".json"`` in a directory whose only destinations are
      ``<id>.json``) and give the match some actual ownership evidence.
    * The age cutoff is a heuristic, not a lock. There is no coordination with
      the writer, so a forward clock step (NTP correction, VM/laptop resume), a
      lagging network-filesystem clock, or a writer stalled past the cutoff
      inside ``fsync`` can still make a live temp eligible — and unlinking it
      makes that writer's ``os.replace`` fail. An hour makes that vanishingly
      unlikely, not impossible; keep the default unless a caller can afford the
      loss.

    Returns how many files were removed. Never raises.
    """
    now = time.time()
    removed = 0
    try:
        # Streamed rather than materialized: the docstring's own premise is that
        # nothing bounds how many corpses a crash-loop leaves, and there is no
        # reason to hold the whole listing to delete entries one at a time.
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(_TMP_SUFFIX):
                    continue
                # ``<stem>.<token>`` — the stem is the destination file name, so
                # it must be a real name (``.``/``..`` are not) and, when the
                # caller says so, carry ``stem_suffix``; the token must have
                # mkstemp's exact shape. Together that keeps a hand-made
                # ``notes.tmp`` (no token segment) or a ``.lock``/``.part``
                # sibling (wrong suffix entirely) off the list.
                stem, sep, token = name[: -len(_TMP_SUFFIX)].rpartition(".")
                if not sep or not stem.strip(".") or not stem.endswith(stem_suffix):
                    continue
                if len(token) != _MKSTEMP_TOKEN_LEN or not set(token) <= _MKSTEMP_TOKEN_CHARS:
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                if now - st.st_mtime <= older_than_seconds:
                    continue
                try:
                    os.unlink(entry.path)
                except OSError:
                    continue
                # The count goes nowhere useful at the call sites (this is
                # hygiene, not a user-facing action), so a debug line is the
                # only way to attribute a file's disappearance after the fact.
                logger.debug("swept stranded atomic-write temp: %s", entry.path)
                removed += 1
    except OSError:
        # scandir failed to open, or died mid-iteration. Whatever we already
        # removed still counts.
        return removed
    return removed


def _friendly_network_error(exc: Exception) -> str:
    """Return a user-friendly description of a network error."""
    import httpx

    if isinstance(exc, _TransientHTTPStatusError):
        try:
            phrase = HTTPStatus(exc.status_code).phrase
            return f"the server returned HTTP {exc.status_code} {phrase}"
        except ValueError:
            return f"the server returned HTTP {exc.status_code}"
    if isinstance(exc, httpx.InvalidURL):
        return f"invalid URL ({exc})"
    if isinstance(exc, httpx.ReadTimeout):
        return "the server stopped sending data (read timeout)"
    if isinstance(exc, httpx.ConnectTimeout):
        return "could not connect to the server (connect timeout)"
    if isinstance(exc, httpx.TimeoutException):
        return f"the operation timed out ({type(exc).__name__})"
    if isinstance(exc, httpx.NetworkError):
        return f"a network error occurred ({type(exc).__name__}: {exc})"
    if isinstance(exc, httpx.ProtocolError):
        return f"a protocol error occurred ({type(exc).__name__}: {exc})"
    if isinstance(exc, httpx.ProxyError):
        return f"a proxy error occurred ({type(exc).__name__}: {exc})"
    return str(exc)


def _download_file_httpx(
    url: str,
    local_filepath: pathlib.Path,
    headers: dict | None = None,
    *,
    state: dict | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Download a file using httpx streaming. Raises on HTTP or network errors.

    The bytes stream into a uniquely-named ``.part`` sibling of ``local_filepath``
    and are renamed onto it with ``os.replace`` only after the last chunk lands, so
    the destination only ever transitions absent→complete (or old-complete→
    new-complete). A transfer killed uncleanly — SIGKILL, OOM, power loss, none of
    which run any Python cleanup — therefore leaves a ``.part`` file rather than a
    truncated model at the path ComfyUI is about to load from. Same-directory
    sibling keeps the rename on one filesystem, and therefore atomic, exactly as
    :func:`atomic_write_text` documents for text.

    ``progress_callback`` (optional) receives ``(completed_bytes, total_bytes)``
    as chunks land — ``total_bytes`` comes from Content-Length and is None when
    the server doesn't send one. It fires once with ``(0, total)`` before the
    first chunk so a caller learns the size as soon as the headers are read.

    If ``state`` is provided, ``state["file_opened"]`` is set to True immediately
    after the output file is opened for writing, and ``state["part_path"]`` holds
    that file's path. Callers use the flag to distinguish failures raised *before*
    any bytes were written (HTTP errors, ConnectError, etc.) from failures raised
    *after* writing started (mid-stream ReadTimeout) — it now guards the temp file
    rather than the destination, since the destination is no longer written
    through. Every failure path unlinks the temp before re-raising, with one
    deliberate exception: a :class:`KeyboardInterrupt` leaves it in place so
    :func:`download_file` can ask the user whether to keep the partial.
    """
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, headers=headers, timeout=_download_timeout()) as response:
        if response.status_code != 200:
            try:
                error_body = response.read()
            except _transient_exceptions():
                error_body = ""
            status_reason = guess_status_code_reason(response.status_code, error_body)
            if response.status_code in _RETRIABLE_STATUSES:
                raise _TransientHTTPStatusError(response.status_code, status_reason)
            raise DownloadException(f"Failed to download file.\n{status_reason}")

        content_length = response.headers.get("Content-Length")
        try:
            total = int(content_length) if content_length is not None else None
        except ValueError:
            # A broken server/proxy can send a non-numeric Content-Length. That is not
            # a reason to fail the transfer — treat it exactly like a missing header
            # (indeterminate progress) instead of letting ValueError escape the whole
            # download and end the command with a traceback.
            total = None
        if total is not None and total < 0:
            total = None
        if total is not None:
            description = f"Downloading {total // 1024 // 1024} MB"
        else:
            description = "Downloading..."

        # O_CREAT | O_EXCL and no symlink following, in the destination's own
        # directory: concurrent transfers never collide on a shared name, and a
        # pre-planted symlink can't redirect the write (CWE-377).
        fd, tmp_name = tempfile.mkstemp(
            dir=str(local_filepath.parent),
            prefix=_part_prefix(local_filepath.name),
            suffix=_PART_SUFFIX,
        )
        try:
            try:
                part_file = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only fails *before* taking ownership of the descriptor,
                # so this is the one place the raw fd still has to be closed by
                # hand; everywhere below, closing is the file object's job.
                os.close(fd)
                raise
            if state is not None:
                # Path first, flag second. `download_file`'s interrupt prompt is
                # gated on the flag but deletes via the path, so the reverse order
                # lets a KeyboardInterrupt landing in between ask the user about a
                # partial it then silently declines to remove.
                state["part_path"] = tmp_name
                state["file_opened"] = True
            with part_file as f:
                # Announce the size (and the zeroed counter) before the first chunk
                # so a background observer stops reporting `total_bytes: null` as
                # soon as the headers are in.
                _report_progress(progress_callback, 0, total)
                completed = 0
                for data in ui.show_progress(
                    response.iter_bytes(),
                    total,
                    description=description,
                ):
                    f.write(data)
                    completed += len(data)
                    _report_progress(progress_callback, completed, total)
                _apply_destination_mode(f.fileno(), local_filepath)
            os.replace(tmp_name, local_filepath)
        except KeyboardInterrupt:
            # The one failure that leaves the partial behind: download_file prompts
            # the user about it, and deleting it here would make "no" unanswerable.
            raise
        except BaseException:
            # BaseException, not Exception, so a DownloadCancelled (or anything
            # else a progress callback throws through the transfer loop) still
            # reclaims the temp file's disk.
            _cleanup_partial(pathlib.Path(tmp_name))
            raise


def download_file(
    url: str,
    local_filepath: pathlib.Path,
    headers: dict | None = None,
    downloader: str = "httpx",
    progress_callback: ProgressCallback | None = None,
):
    """Helper function to download a file.

    ``progress_callback`` (optional) is invoked with ``(completed_bytes,
    total_bytes)`` as the transfer advances; ``total_bytes`` is None until the
    size is known. When a retry discards a partial file the callback is reset to
    ``(0, None)`` first, so an observer never sees a counter run backwards from
    a stale high-water mark.

    **The two downloaders differ in how the destination is written, deliberately.**
    The httpx path is atomic: it streams into a ``.part`` sibling and renames onto
    ``local_filepath`` only once the transfer completes, so no kill can strand a
    truncated file where a complete model belongs, and each retry attempt gets its
    own temp rather than reusing a half-written destination. ``aria2`` is left
    writing straight to the destination on purpose — it owns its own ``.aria2``
    control file next to the output and resumes from it, and renaming the output
    from under aria2 would break that resume. So a killed aria2 transfer can still
    leave a partial at the destination; that is aria2's resume state, not corruption
    this module introduced, and `download-cancel` cleans up both shapes.
    """
    if downloader not in _VALID_DOWNLOADERS:
        raise DownloadException(
            f"Unknown downloader: {downloader!r}. Valid options: {', '.join(sorted(_VALID_DOWNLOADERS))}"
        )

    local_filepath.parent.mkdir(parents=True, exist_ok=True)

    if downloader == "aria2":
        return _download_file_aria2(url, local_filepath, headers, progress_callback)

    import httpx

    last_exc: Exception | None = None
    state: dict = {"file_opened": False, "part_path": None}

    for attempt in range(_DOWNLOAD_MAX_RETRIES):
        state["file_opened"] = False
        state["part_path"] = None
        try:
            _download_file_httpx(url, local_filepath, headers, state=state, progress_callback=progress_callback)
            return
        except _retriable_exceptions() as exc:
            last_exc = exc
            # The temp file this attempt was writing is already gone (the helper
            # unlinks it on the way out) and the destination was never touched, so
            # there is nothing here to delete — only an observer to correct: it
            # last heard a byte count for bytes that no longer exist anywhere.
            if state["file_opened"]:
                _report_progress(progress_callback, 0, None)
            if attempt < _DOWNLOAD_MAX_RETRIES - 1:
                wait = _DOWNLOAD_RETRY_BACKOFF * (attempt + 1)
                print(f"Download error (attempt {attempt + 1}/{_DOWNLOAD_MAX_RETRIES}): {_friendly_network_error(exc)}")
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            # Non-retriable httpx errors (e.g. UnsupportedProtocol, TooManyRedirects,
            # DecodingError, InvalidURL). Fail fast and convert to DownloadException
            # so callers only need to handle one error type.
            # InvalidURL inherits directly from Exception (not HTTPError), hence the
            # explicit inclusion.
            # No cleanup needed: the helper already removed its temp file, and the
            # destination is never written through.
            raise DownloadException(f"Download failed: {_friendly_network_error(exc)}") from exc
        except KeyboardInterrupt:
            # Only prompt if we actually started writing this attempt. If the
            # interrupt arrived during connection setup there is no partial file to
            # ask about. The helper deliberately leaves the interrupted `.part` on
            # disk for exactly this question; whichever way it is answered, the
            # destination is untouched — a pre-existing file there survives either
            # way, which is why only the temp is ever a candidate for deletion.
            if state["file_opened"]:
                delete_eh = ui.prompt_confirm_action("Download interrupted, cleanup files?", True)
                if delete_eh and state["part_path"]:
                    _cleanup_partial(pathlib.Path(state["part_path"]))
            raise

    raise DownloadException(
        f"Download failed after {_DOWNLOAD_MAX_RETRIES} attempts: "
        f"{_friendly_network_error(last_exc)}\n"
        f"Please try again later."
    ) from last_exc


def _load_comfyignore_spec(ignore_filename: str = ".comfyignore") -> PathSpec | None:
    if not os.path.exists(ignore_filename):
        return None
    try:
        with open(ignore_filename, encoding="utf-8") as ignore_file:
            patterns = [line.strip() for line in ignore_file if line.strip() and not line.lstrip().startswith("#")]
    except OSError:
        return None

    if not patterns:
        return None

    return PathSpec.from_lines("gitwildmatch", patterns)


def list_git_tracked_files(base_path: str | os.PathLike = ".") -> list[str]:
    """Git-tracked files under ``base_path``, or ``[]`` when git can't tell us.

    ``[]`` means "no git answer" and callers (see :func:`zip_files`) treat it as
    "not a git repository". An *absent* git has always produced that, so it still
    does. A git that was found and then **refused** must not: the caller would
    silently fall back to walking the whole directory, so the refusal is raised
    rather than flattened into the same empty list.
    """
    # Resolved outside the tolerant handler below so the two cases stay
    # distinguishable — ``BinaryNotFoundError`` subclasses ``FileNotFoundError``,
    # which that handler swallows.
    try:
        git_bin = resolve_required_binary("git")
    except BinaryNotFoundError as exc:
        if not exc.is_absent:
            raise
        return []

    try:
        result = subprocess.check_output(
            [git_bin, "-C", os.fspath(base_path), "ls-files"],
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    return [line for line in result.splitlines() if line.strip()]


def _normalize_path(path: str) -> str:
    rel_path = os.path.relpath(path, start=".")
    if rel_path == ".":
        return ""
    return rel_path.replace("\\", "/")


def _is_force_included(rel_path: str, include_prefixes: list[str]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in include_prefixes if prefix)


def zip_files(zip_filename, includes=None):
    """Zip git-tracked files respecting optional .comfyignore patterns.

    :raises BinaryNotFoundError: ``git`` was found but refused (see
        :func:`comfy_cli._safe_exec.resolve_required_binary`). The walk-everything
        fallback below is safe for "this isn't a git repo", but not for "we can't
        trust git": it would package untracked and gitignored files — ``.env``,
        keys, venvs — into an archive that ``comfy node publish`` uploads.
    """
    includes = includes or []
    include_prefixes: list[str] = [_normalize_path(os.path.normpath(include.lstrip("/"))) for include in includes]

    included_paths: set[str] = set()
    git_files: list[str] = []

    ignore_spec = _load_comfyignore_spec()

    def should_ignore(rel_path: str) -> bool:
        if not ignore_spec:
            return False
        if _is_force_included(rel_path, include_prefixes):
            return False
        return ignore_spec.match_file(rel_path)

    zip_target = os.fspath(zip_filename)
    zip_abs_path = os.path.abspath(zip_target)
    zip_basename = os.path.basename(zip_abs_path)

    git_files = list_git_tracked_files(".")
    if not git_files:
        print("Warning: Not in a git repository or git not installed. Zipping all files.")

    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zipf:
        if git_files:
            for file_path in git_files:
                if file_path == zip_basename:
                    continue

                rel_path = _normalize_path(file_path)
                if should_ignore(rel_path):
                    continue

                actual_path = os.path.normpath(file_path)
                if os.path.abspath(actual_path) == zip_abs_path:
                    continue
                if os.path.exists(actual_path):
                    arcname = rel_path or os.path.basename(actual_path)
                    zipf.write(actual_path, arcname)
                    included_paths.add(rel_path)
                else:
                    print(f"File not found. Not including in zip: {file_path}")
        else:
            for root, dirs, files in os.walk("."):
                if ".git" in dirs:
                    dirs.remove(".git")
                dirs[:] = [d for d in dirs if not should_ignore(_normalize_path(os.path.join(root, d)))]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = _normalize_path(file_path)
                    if (
                        os.path.abspath(file_path) == zip_abs_path
                        or rel_path in included_paths
                        or should_ignore(rel_path)
                    ):
                        continue
                    arcname = rel_path or file_path
                    zipf.write(file_path, arcname)
                    included_paths.add(rel_path)

        for include_dir in includes:
            include_dir = os.path.normpath(include_dir.lstrip("/"))
            rel_include = _normalize_path(include_dir)

            if os.path.isfile(include_dir):
                if not should_ignore(rel_include) and rel_include not in included_paths:
                    arcname = rel_include or include_dir
                    zipf.write(include_dir, arcname)
                    included_paths.add(rel_include)
                continue

            if not os.path.exists(include_dir):
                print(f"Warning: Included directory '{include_dir}' does not exist, creating empty directory")
                arcname = rel_include or include_dir
                if not arcname.endswith("/"):
                    arcname = arcname + "/"
                zipf.writestr(arcname, "")
                continue

            for root, dirs, files in os.walk(include_dir):
                dirs[:] = [d for d in dirs if not should_ignore(_normalize_path(os.path.join(root, d)))]
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = _normalize_path(file_path)
                    if (
                        os.path.abspath(file_path) == zip_abs_path
                        or rel_path in included_paths
                        or should_ignore(rel_path)
                    ):
                        continue
                    arcname = rel_path or file_path
                    zipf.write(file_path, arcname)
                    included_paths.add(rel_path)


def upload_file_to_signed_url(signed_url: str, file_path: str):
    import requests  # deferred; see check_unauthorized

    with open(file_path, "rb") as f:
        headers = {"Content-Type": "application/zip"}
        response = requests.put(signed_url, data=f, headers=headers, timeout=DOWNLOAD_TIMEOUT)

        if response.status_code == 200:
            print("Upload successful.")
        else:
            raise Exception(f"Upload failed with status code: {response.status_code}. Error: {response.text}")


def extract_package_as_zip(file_path: pathlib.Path, extract_path: pathlib.Path):
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted zip file to {extract_path}")
    except zipfile.BadZipFile:
        print("File is not a zip or is corrupted.")
