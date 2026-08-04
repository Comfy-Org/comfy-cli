"""Resolve CQL node-annotation data from Comfy-Org/comfy-complete.

The annotation files (``supported_nodes.yaml`` — pack membership + behavioral
labels; ``cloud_disable_config.yaml`` — which labels disable a node on cloud)
drive ``comfy nodes`` annotations (``pack``, ``labels``, ``cloud_disabled``).

They live in the public repo `Comfy-Org/comfy-complete` and change far more
often than comfy-cli ships releases, so they resolve through a **live-refresh +
fallback** chain rather than being pinned to the bundled snapshot:

    1. a TTL-fresh local cache (``~/.cache/comfy-cli/comfy-complete/``)
    2. a live fetch from the public repo (short bounded deadline, cached on
       success — and only when the caller opted into network I/O)
    3. the package-bundled snapshot under ``comfy_cli/cql/data/`` (offline-safe)

Three properties this module owes its callers, each learned the hard way:

**Never block a hot path for long.** ``comfy nodes`` reaches here implicitly on
every invocation. The two files are fetched *concurrently* on daemon threads
behind one hard wall-clock deadline (``_FETCH_DEADLINE``), so a hung DNS
resolver — which ``socket`` timeouts do *not* bound — cannot stall the CLI or
keep the process alive at exit. A failed attempt is negative-cached for
``_FAILURE_BACKOFF``, so a persistently offline machine pays the deadline once
an hour rather than on every command.

**Never cache what we can't parse.** A 200 with an HTML captive-portal body (or
a malformed upstream commit) used to be written to the cache and stamped fresh;
because ``engine.parse_supported_nodes`` degrades to "no annotations" rather
than raising, that silently zeroed every node's labels for the whole TTL and the
bundled fallback never got a chance. Bodies are parsed and shape-checked
*before* they reach the cache, so a bad body falls through to bundled instead.

**Never mix generations.** ``cloud_disabled`` is computed by matching labels
from ``supported_nodes.yaml`` against disable rules in
``cloud_disable_config.yaml``, so a fresh file paired with a stale one can
mis-classify a node. The pair is fetched, validated, and committed atomically:
either both files land or neither does.

Set ``COMFY_CLI_NO_REMOTE_REFRESH=1`` to skip the network entirely (cache →
bundled only) for airgapped or CI use.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from comfy_cli.http import plain_urlopen, read_capped

# Public source of truth. Both files are at the repo root on ``main``.
_BASE_URL = "https://raw.githubusercontent.com/Comfy-Org/comfy-complete/main"

_SUPPORTED_NODES = "supported_nodes.yaml"
_CLOUD_DISABLE = "cloud_disable_config.yaml"
_FILES = (_SUPPORTED_NODES, _CLOUD_DISABLE)

# Auto-refresh cadence for the implicit (hot-path) load. A manual
# ``comfy nodes refresh`` always forces a fetch regardless of this.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Per-connection socket timeout, and the hard ceiling on how long the *whole*
# background refresh may delay the caller. The deadline is what actually bounds
# the wait: ``urlopen(timeout=…)`` starts counting after the socket exists, so a
# black-holed DNS resolver sails straight past it.
_FETCH_TIMEOUT = 5.0
_FETCH_DEADLINE = 6.0

# How long a failed refresh suppresses further attempts. Without this an
# offline machine re-pays ``_FETCH_DEADLINE`` on every single ``comfy nodes``
# invocation, because a failure leaves the cache exactly as stale as it found it.
_FAILURE_BACKOFF = 60 * 60
_FAILURE_STAMP = ".refresh-failed"

# These are two small YAML documents (~32 KB bundled). The shared 64 MiB default
# is a ceiling for ``/object_info``-sized payloads; a much tighter cap here means
# a misbehaving or hostile upstream can't stream unbounded data into the memory
# of a routine introspection command.
_MAX_ANNOTATION_BYTES = 8 * 1024 * 1024

# Values that mean "no, don't disable the network". Everything else — ``1``,
# ``yes``, ``on``, or any other non-empty string — disables it. Compared
# case-insensitively so ``COMFY_CLI_NO_REMOTE_REFRESH=FALSE`` reads as the user
# plainly meant it, not as its opposite.
_FALSEY = {"", "0", "false", "no", "off"}


def network_disabled() -> bool:
    """True when ``COMFY_CLI_NO_REMOTE_REFRESH`` opts out of all network I/O."""
    return os.environ.get("COMFY_CLI_NO_REMOTE_REFRESH", "").strip().lower() not in _FALSEY


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "comfy-complete"


def bundled_bytes(filename: str) -> bytes | None:
    """The snapshot shipped inside the wheel, or ``None`` if package data is missing."""
    try:
        from importlib import resources

        return (resources.files("comfy_cli.cql.data") / filename).read_bytes()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Validation — what a body must look like before it is allowed near the cache
# ---------------------------------------------------------------------------


def _yaml_mapping(data: bytes) -> dict[str, Any] | None:
    try:
        import yaml

        parsed = yaml.safe_load(data)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_supported_nodes(data: bytes) -> bool:
    """A usable ``supported_nodes.yaml`` carries a non-empty ``node_packs`` list.

    Shape, not just syntax: an HTML error page fails the YAML parse, but a
    valid-YAML-yet-wrong document (an upstream file move leaving a stub, say)
    would parse fine and annotate nothing. ``node_packs`` is the key
    ``engine.parse_supported_nodes`` actually reads, so requiring it is the same
    question the consumer asks.
    """
    cfg = _yaml_mapping(data)
    return bool(cfg) and isinstance(cfg.get("node_packs"), list) and len(cfg["node_packs"]) > 0


def _valid_cloud_disable(data: bytes) -> bool:
    """A usable ``cloud_disable_config.yaml`` carries a ``disable_nodes`` mapping.

    Deliberately *not* requiring a non-empty rule list: "nothing is disabled on
    cloud" is a legitimate upstream state, whereas a missing ``disable_nodes``
    key means we're not looking at this file at all.
    """
    cfg = _yaml_mapping(data)
    return bool(cfg) and isinstance(cfg.get("disable_nodes"), dict)


_VALIDATORS: dict[str, Callable[[bytes], bool]] = {
    _SUPPORTED_NODES: _valid_supported_nodes,
    _CLOUD_DISABLE: _valid_cloud_disable,
}


class AnnotationFetchError(Exception):
    """A remote annotation fetch failed or returned an unusable body."""


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_one(filename: str) -> bytes:
    """Fetch and validate a single annotation file. Raises on anything unusable."""
    url = f"{_BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with plain_urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
        if resp.status != 200:
            raise AnnotationFetchError(f"{filename}: HTTP {resp.status}")
        data = read_capped(resp, url, max_bytes=_MAX_ANNOTATION_BYTES)
    if not _VALIDATORS[filename](data):
        raise AnnotationFetchError(f"{filename}: remote body is not a valid {filename} document")
    return data


def fetch_pair(*, deadline: float = _FETCH_DEADLINE) -> dict[str, bytes]:
    """Fetch **both** annotation files concurrently within one wall-clock budget.

    Returns ``{filename: bytes}`` with an entry for each of ``_FILES``, or
    raises :class:`AnnotationFetchError` if either half failed, returned an
    unusable body, or did not finish inside ``deadline``. All-or-nothing by
    design — see the module docstring on mixing generations.

    The workers are bare daemon threads rather than a
    ``ThreadPoolExecutor``: the executor registers an ``atexit`` hook that
    joins its workers, so a thread stuck in ``getaddrinfo`` would hold the
    interpreter open past the deadline we just spent effort enforcing.
    """
    results: dict[str, bytes] = {}
    errors: dict[str, str] = {}
    lock = threading.Lock()

    def work(filename: str) -> None:
        try:
            data = _fetch_one(filename)
        except Exception as e:  # noqa: BLE001 — every failure degrades the same way
            with lock:
                errors[filename] = str(e) or type(e).__name__
            return
        with lock:
            results[filename] = data

    threads = [threading.Thread(target=work, args=(f,), daemon=True, name=f"comfy-annot-{f}") for f in _FILES]
    for t in threads:
        t.start()
    end = time.monotonic() + deadline
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))

    with lock:
        if len(results) == len(_FILES):
            return dict(results)
        detail = "; ".join(errors.get(f, f"{f}: timed out after {deadline:.0f}s") for f in _FILES if f not in results)
    raise AnnotationFetchError(detail)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a same-directory temp file + ``os.replace``.

    A concurrent reader must never observe a half-written annotation file:
    a truncated YAML document parses to *something*, and
    ``engine.parse_supported_nodes`` would quietly annotate half the graph.
    Raises ``OSError`` — callers decide whether persisting is best-effort.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _persist_pair(fetched: dict[str, bytes]) -> tuple[bool, str | None]:
    """Commit a validated pair to the cache. Returns ``(persisted, error)``.

    Best-effort: a read-only cache dir or a full disk must not discard data we
    already hold, so a write failure is reported rather than raised. Reported
    separately from a fetch failure, because "downloaded fine, couldn't save it"
    and "couldn't download it" call for different user action.
    """
    cache_dir = _cache_dir()
    try:
        for filename, data in fetched.items():
            _write_atomic(cache_dir / filename, data)
    except OSError as e:
        return False, str(e)
    _clear_failure_stamp()
    return True, None


def _is_fresh(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    # A future mtime (clock skew, restored backup) yields a negative age; treat
    # it as stale so a cache entry can't pin itself "fresh" until wall-clock
    # time catches up.
    return 0 <= age < _CACHE_TTL_SECONDS


def _read_cached_pair(*, require_fresh: bool) -> dict[str, bytes] | None:
    """Both cached files if both are present, valid, and (optionally) fresh."""
    cache_dir = _cache_dir()
    out: dict[str, bytes] = {}
    for filename in _FILES:
        path = cache_dir / filename
        if require_fresh and not _is_fresh(path):
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        # A cache entry written by an older comfy-cli (before validation
        # existed) or corrupted on disk is treated as absent, so the bundled
        # snapshot can take over instead of silently blanking every annotation.
        if not _VALIDATORS[filename](data):
            return None
        out[filename] = data
    return out


def _failure_stamp() -> Path:
    return _cache_dir() / _FAILURE_STAMP


def _in_failure_backoff() -> bool:
    try:
        age = time.time() - _failure_stamp().stat().st_mtime
    except OSError:
        return False
    # A future mtime (clock skew) would otherwise suppress the network forever;
    # same guard as ``_is_fresh``, in the direction that fails open.
    return 0 <= age < _FAILURE_BACKOFF


def _mark_failure() -> None:
    stamp = _failure_stamp()
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()  # updates mtime when the stamp already exists
    except OSError:
        pass  # best-effort; worst case we retry sooner than intended


def _clear_failure_stamp() -> None:
    try:
        _failure_stamp().unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_annotation_bytes(*, allow_network: bool = True) -> tuple[bytes | None, bytes | None]:
    """Return ``(supported_nodes_yaml, cloud_disable_config_yaml)`` bytes.

    Resolution order: fresh cache → live fetch (only when ``allow_network`` and
    not suppressed) → stale cache → bundled snapshot. Either element is ``None``
    only if it cannot be resolved from any source at all, which in practice
    means the wheel's package data is missing.

    ``allow_network=False`` guarantees zero network I/O. Callers on a path the
    user expects to be offline — ``comfy nodes ls --input <dump>`` reads a local
    file and should not touch the wire because of an incidental annotation
    lookup — pass it.
    """
    fresh = _read_cached_pair(require_fresh=True)
    if fresh is not None:
        return fresh[_SUPPORTED_NODES], fresh[_CLOUD_DISABLE]

    if allow_network and not network_disabled() and not _in_failure_backoff():
        try:
            fetched = fetch_pair()
        except (AnnotationFetchError, OSError):
            _mark_failure()
        else:
            _persist_pair(fetched)
            return fetched[_SUPPORTED_NODES], fetched[_CLOUD_DISABLE]

    stale = _read_cached_pair(require_fresh=False)
    if stale is not None:
        return stale[_SUPPORTED_NODES], stale[_CLOUD_DISABLE]

    return bundled_bytes(_SUPPORTED_NODES), bundled_bytes(_CLOUD_DISABLE)


def refresh_annotations() -> list[dict]:
    """Force a re-fetch of the annotation pair into the cache.

    Returns one status dict per file: ``{name, source, bytes, path}`` where
    ``source`` is ``"remote"`` (freshly fetched), ``"bundled"`` (the fetch could
    not be used, so the package snapshot stands), or ``"unavailable"``. When the
    remote pair was fetched but could not be written to disk, ``source`` stays
    ``"remote"``, ``path`` is ``None``, and ``cache_error`` explains why — a
    caching problem is not a network problem and shouldn't be reported as one.
    """
    cache_dir = _cache_dir()
    error: str | None = None
    fetched: dict[str, bytes] | None = None

    if network_disabled():
        error = "remote refresh disabled (COMFY_CLI_NO_REMOTE_REFRESH)"
    else:
        try:
            fetched = fetch_pair()
        except (AnnotationFetchError, OSError) as e:
            error = str(e) or type(e).__name__
            # An explicit refresh is never itself gated on the backoff stamp,
            # but its failure is the best evidence we have that the implicit
            # hot path shouldn't keep paying the deadline either.
            _mark_failure()

    persisted, cache_error = _persist_pair(fetched) if fetched else (False, None)

    results: list[dict] = []
    for filename in _FILES:
        entry: dict = {"name": filename}
        if fetched is not None:
            entry.update(
                source="remote",
                bytes=len(fetched[filename]),
                path=str(cache_dir / filename) if persisted else None,
            )
            if cache_error:
                entry["cache_error"] = cache_error
        else:
            entry["error"] = error
            bundled = bundled_bytes(filename)
            if bundled is not None:
                entry.update(source="bundled", bytes=len(bundled), path=None)
            else:
                entry.update(source="unavailable", bytes=0, path=None)
        results.append(entry)
    return results
