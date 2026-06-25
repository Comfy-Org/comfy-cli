"""Resolve CQL node-annotation data from Comfy-Org/comfy-complete.

The annotation files (``supported_nodes.yaml`` — pack membership + behavioral
labels; ``cloud_disable_config.yaml`` — which labels disable a node on cloud)
drive ``comfy nodes`` annotations (``pack``, ``labels``, ``cloud_disabled``).

They live in the public repo `Comfy-Org/comfy-complete` and change far more
often than comfy-cli ships releases, so we resolve them with a **live-refresh +
fallback** strategy instead of pinning to the bundled snapshot:

    1. a TTL-fresh local cache (``~/.cache/comfy-cli/comfy-complete/``)
    2. a live fetch from the public repo (short timeout, cached on success)
    3. the package-bundled snapshot under ``comfy_cli/cql/data/`` (offline-safe)

This keeps the data current without a ``pip install -U`` while never breaking
offline / airgapped use. Set ``COMFY_CLI_NO_REMOTE_REFRESH=1`` to skip the
network entirely (cache → bundled only).
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

# Public source of truth. Both files are at the repo root on ``main``.
_BASE_URL = "https://raw.githubusercontent.com/Comfy-Org/comfy-complete/main"
_FILES = ("supported_nodes.yaml", "cloud_disable_config.yaml")

# Auto-refresh cadence for the implicit (hot-path) load. A manual
# ``comfy nodes refresh`` always forces a fetch regardless of this.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
_FETCH_TIMEOUT = 5.0


def _network_disabled() -> bool:
    return os.environ.get("COMFY_CLI_NO_REMOTE_REFRESH", "").strip() not in ("", "0", "false", "False")


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "comfy-complete"


def _bundled_bytes(filename: str) -> bytes | None:
    try:
        from importlib import resources

        return (resources.files("comfy_cli.cql.data") / filename).read_bytes()
    except Exception:
        return None


def _fetch(filename: str) -> bytes:
    url = f"{_BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:  # noqa: S310 — fixed https host
        if resp.status != 200:
            raise RuntimeError(f"annotation fetch failed: HTTP {resp.status}")
        return resp.read()


def _is_fresh(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS
    except OSError:
        return False


def _resolve_one(filename: str, *, refresh: bool) -> bytes | None:
    """Resolve one annotation file: cache(TTL) → fetch → stale cache → bundled."""
    cache = _cache_dir() / filename

    # Fresh cache wins outright (unless an explicit refresh was requested).
    if not refresh and cache.is_file() and _is_fresh(cache):
        try:
            return cache.read_bytes()
        except OSError:
            pass

    # Try the network when allowed and either forced or the cache is stale/missing.
    if not _network_disabled():
        try:
            data = _fetch(filename)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except OSError:
                pass  # cache write is best-effort
            return data
        except Exception:
            pass  # fall through to stale cache / bundled

    # Network unavailable or disabled — use whatever stale cache we have.
    if cache.is_file():
        try:
            return cache.read_bytes()
        except OSError:
            pass

    # Last resort: the snapshot shipped with the package.
    return _bundled_bytes(filename)


def load_annotation_bytes(*, refresh: bool = False) -> tuple[bytes | None, bytes | None]:
    """Return ``(supported_nodes_yaml, cloud_disable_config_yaml)`` bytes.

    Either element may be ``None`` if it cannot be resolved from any source.
    """
    sup = _resolve_one("supported_nodes.yaml", refresh=refresh)
    dis = _resolve_one("cloud_disable_config.yaml", refresh=refresh)
    return sup, dis


def refresh_annotations() -> list[dict]:
    """Force a re-fetch of every annotation file into the cache.

    Returns one status dict per file: ``{name, source, bytes, path}`` where
    ``source`` is ``"remote"`` (freshly fetched), ``"bundled"`` (network failed,
    used the package snapshot), or ``"unavailable"``.
    """
    cache_dir = _cache_dir()
    results: list[dict] = []
    for filename in _FILES:
        entry: dict = {"name": filename}
        if not _network_disabled():
            try:
                data = _fetch(filename)
                cache = cache_dir / filename
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
                entry.update(source="remote", bytes=len(data), path=str(cache))
                results.append(entry)
                continue
            except Exception as e:  # noqa: BLE001 — degrade to bundled, report why
                entry["error"] = str(e)
        else:
            entry["error"] = "remote refresh disabled (COMFY_CLI_NO_REMOTE_REFRESH)"

        bundled = _bundled_bytes(filename)
        if bundled is not None:
            entry.update(source="bundled", bytes=len(bundled), path=None)
        else:
            entry.update(source="unavailable", bytes=0, path=None)
        results.append(entry)
    return results
