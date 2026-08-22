"""Find, validate, and index the compiled comfy-knowledge bundle.

The bundle (``knowledge.json`` + optional ``manifest.json``) is produced by
``Comfy-Org/comfy-knowledge``. It reaches this process by one of three routes,
tried in order: an explicit ``COMFY_KNOWLEDGE_FILE``, the per-user cache, or a
fetch from ``COMFY_KNOWLEDGE_URL``. A missing or broken bundle is a normal
state: every entry point here returns ``None`` rather than raising, and nothing
is written to stdout or stderr.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli.cloud import get_base_url
from comfy_cli.cql.loader import _cache_dir
from comfy_cli.file_utils import atomic_write_bytes
from comfy_cli.http import assert_safe_url, authed_urlopen, plain_urlopen, read_capped

SCHEMA_VERSION = 1
ENV_FILE = "COMFY_KNOWLEDGE_FILE"
ENV_URL = "COMFY_KNOWLEDGE_URL"
ENV_TTL = "COMFY_KNOWLEDGE_TTL"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT_SECONDS = 10.0
MAX_BUNDLE_BYTES = 16 * 1024 * 1024

REASON_ENV_FILE = "COMFY_KNOWLEDGE_FILE is set but could not be loaded"
REASON_NO_URL = "no cache and COMFY_KNOWLEDGE_URL is not set"
REASON_FETCH_FAILED = "fetch failed and no cached bundle exists"


@dataclass(frozen=True)
class Bundle:
    version: str
    source: str  # "env" | "cache" | "fetch" | "stale-cache"
    stale: bool
    as_of: str  # ISO-8601 UTC, from the loaded file's mtime
    path: str
    models: dict[str, dict]
    capabilities: dict[str, dict]
    deprecations: dict[str, dict]  # keyed by id
    aliases: dict[str, str]  # lowercased alias or model id -> model id
    templates: dict[str, list[str]]  # template id -> model ids
    nodes: dict[str, list[str]]  # node class -> model ids


# One-element tuple so a memoized ``None`` is distinguishable from "not loaded yet".
_MEMO: tuple[Bundle | None] | None = None
_LAST_REASON: str | None = None


def _reset_for_testing() -> None:
    global _MEMO, _LAST_REASON
    _MEMO = None
    _LAST_REASON = None


def last_reason() -> str | None:
    """Why the most recent ``load_bundle`` returned ``None``; ``None`` after a hit."""
    return _LAST_REASON


def cache_paths() -> tuple[Path, Path]:
    base = _cache_dir() / "knowledge"
    return base / "knowledge.json", base / "manifest.json"


def ttl_seconds() -> float:
    raw = os.environ.get(ENV_TTL)
    if raw is None or not raw.strip():
        return DEFAULT_TTL_SECONDS
    try:
        ttl = float(raw)
    except ValueError:
        return DEFAULT_TTL_SECONDS
    return max(ttl, 0.0)


def load_bundle(*, force_fetch: bool = False) -> Bundle | None:
    """Return the indexed bundle, or ``None`` when no usable bundle exists.

    Memoized per process; ``force_fetch=True`` re-runs the load and skips the
    cache TTL gate so a fetch happens whenever ``COMFY_KNOWLEDGE_URL`` is set.
    """
    global _MEMO, _LAST_REASON
    if _MEMO is not None and not force_fetch:
        return _MEMO[0]
    bundle, reason = _load(force_fetch=force_fetch)
    _MEMO = (bundle,)
    _LAST_REASON = reason
    return bundle


def resolve(bundle: Bundle, query: str) -> dict | None:
    model_id = bundle.aliases.get(query.strip().lower())
    return bundle.models.get(model_id) if model_id is not None else None


def pick(bundle: Bundle, capability: str) -> dict | None:
    cap = bundle.capabilities.get(capability.strip().lower())
    if cap is None:
        return None
    raw_picks = cap.get("picks")
    picks = [p for p in raw_picks if isinstance(p, dict)] if isinstance(raw_picks, list) else []
    out = dict(cap)
    out["picks"] = sorted(picks, key=_rank_key)
    return out


def _rank_key(p: dict) -> tuple[int, float]:
    rank = p.get("rank")
    if isinstance(rank, int | float) and not isinstance(rank, bool):
        return (0, rank)
    return (1, 0.0)


# ---------------------------------------------------------------------------
# load order
# ---------------------------------------------------------------------------


def _load(*, force_fetch: bool) -> tuple[Bundle | None, str | None]:
    env_file = os.environ.get(ENV_FILE, "").strip()
    if env_file:
        path = Path(env_file)
        bundle = _load_file(path, path.parent / "manifest.json", source="env")
        return bundle, (None if bundle else REASON_ENV_FILE)

    knowledge_path, manifest_path = cache_paths()
    ttl = ttl_seconds()
    if not force_fetch and ttl > 0:
        try:
            age = time.time() - knowledge_path.stat().st_mtime
        except OSError:
            age = None
        if age is not None and 0 <= age < ttl:
            bundle = _load_file(knowledge_path, manifest_path, source="cache")
            if bundle is not None:
                return bundle, None

    url = os.environ.get(ENV_URL, "").strip()
    if url:
        bundle = _fetch(url, knowledge_path, manifest_path)
        if bundle is not None:
            return bundle, None

    bundle = _load_file(knowledge_path, manifest_path, source="stale-cache", stale=True)
    if bundle is not None:
        return bundle, None
    return None, (REASON_FETCH_FAILED if url else REASON_NO_URL)


def _load_file(path: Path, manifest_path: Path, *, source: str, stale: bool = False) -> Bundle | None:
    try:
        raw = path.read_bytes()
        mtime = path.stat().st_mtime
    except OSError:
        return None
    try:
        manifest = _parse_manifest(manifest_path.read_bytes())
    except OSError:
        manifest = None
    data = _parse(raw, manifest)
    if data is None:
        return None
    return _index(data, manifest, source=source, stale=stale, path=str(path), mtime=mtime)


def _fetch(url: str, knowledge_path: Path, manifest_path: Path) -> Bundle | None:
    try:
        assert_safe_url(url)
        raw = _http_get(url)
    except Exception:  # noqa: BLE001 — any failure is "fetch failed"; the caller falls back
        return None
    try:
        manifest_raw: bytes | None = _http_get(urllib.parse.urljoin(url, "manifest.json"))
    except Exception:  # noqa: BLE001 — a missing manifest only disables the sha check
        manifest_raw = None
    manifest = _parse_manifest(manifest_raw) if manifest_raw is not None else None
    data = _parse(raw, manifest)
    if data is None:
        return None
    try:
        atomic_write_bytes(knowledge_path, raw)
        if manifest_raw is not None:
            atomic_write_bytes(manifest_path, manifest_raw)
        else:
            # A leftover manifest from an earlier fetch would fail the sha check
            # against the new bytes and make the cache unreadable until the next fetch.
            manifest_path.unlink(missing_ok=True)
    except OSError:
        pass
    return _index(data, manifest, source="fetch", stale=False, path=str(knowledge_path), mtime=time.time())


def _http_get(url: str) -> bytes:
    """GET ``url`` and return the body; raises on any non-200 or transport error.

    Credentials go only to the cloud base URL; anything else is an anonymous fetch.
    """
    if url.startswith(get_base_url().rstrip("/") + "/"):
        from comfy_cli.target import resolve_target

        opened = authed_urlopen(url, resolve_target(where="cloud"), timeout=FETCH_TIMEOUT_SECONDS)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
        opened = plain_urlopen(req, timeout=FETCH_TIMEOUT_SECONDS)
    with opened as resp:
        if resp.status != 200:
            raise RuntimeError(f"knowledge fetch failed: HTTP {resp.status}")
        return read_capped(resp, url, max_bytes=MAX_BUNDLE_BYTES)


# ---------------------------------------------------------------------------
# validation + index
# ---------------------------------------------------------------------------


def _parse_manifest(raw: bytes) -> dict | None:
    try:
        manifest = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _parse(raw: bytes, manifest: dict | None) -> dict | None:
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        return None
    if manifest is not None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return None
        expected = _manifest_sha256(manifest)
        if expected is not None and expected != hashlib.sha256(raw).hexdigest():
            return None
    return data


def _manifest_sha256(manifest: dict) -> str | None:
    files = manifest.get("files")
    entry = files.get("knowledge.json") if isinstance(files, dict) else None
    sha = entry.get("sha256") if isinstance(entry, dict) else None
    return sha if isinstance(sha, str) else None


def _str_list(value: Any) -> list[str]:
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def _append_unique(index: dict[str, list[str]], key: str, model_id: str) -> None:
    ids = index.setdefault(key, [])
    if model_id not in ids:
        ids.append(model_id)


def _index(data: dict, manifest: dict | None, *, source: str, stale: bool, path: str, mtime: float) -> Bundle:
    models = {mid: row for mid, row in data["models"].items() if isinstance(mid, str) and isinstance(row, dict)}

    aliases: dict[str, str] = {}
    compiled = data.get("aliases")
    if isinstance(compiled, dict):
        for alias, mid in compiled.items():
            if isinstance(alias, str) and isinstance(mid, str):
                aliases.setdefault(alias.lower(), mid)
    for mid, row in models.items():
        aliases.setdefault(mid.lower(), mid)
        for alias in _str_list(row.get("aliases")):
            aliases.setdefault(alias.lower(), mid)

    caps_raw = data.get("capabilities")
    capabilities = (
        {cid: cap for cid, cap in caps_raw.items() if isinstance(cid, str) and isinstance(cap, dict)}
        if isinstance(caps_raw, dict)
        else {}
    )
    deps_raw = data.get("deprecations")
    deprecations = (
        {d["id"]: d for d in deps_raw if isinstance(d, dict) and isinstance(d.get("id"), str)}
        if isinstance(deps_raw, list)
        else {}
    )

    templates: dict[str, list[str]] = {}
    nodes: dict[str, list[str]] = {}
    for mid, row in models.items():
        resolves = row.get("resolves")
        if isinstance(resolves, dict):
            for template_id in _str_list(resolves.get("templates")):
                _append_unique(templates, template_id, mid)
            for node_class in _str_list(resolves.get("nodes")):
                _append_unique(nodes, node_class, mid)
        routing = row.get("routing")
        if isinstance(routing, list):
            for rule in routing:
                if isinstance(rule, dict) and isinstance(rule.get("use"), str):
                    _append_unique(templates, rule["use"], mid)
    for cap in capabilities.values():
        picks = cap.get("picks")
        if not isinstance(picks, list):
            continue
        for p in picks:
            if isinstance(p, dict) and isinstance(p.get("template"), str) and isinstance(p.get("model"), str):
                _append_unique(templates, p["template"], p["model"])

    version = manifest.get("version") if manifest is not None else None
    return Bundle(
        version=version if isinstance(version, str) else "unknown",
        source=source,
        stale=stale,
        as_of=datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        path=path,
        models=models,
        capabilities=capabilities,
        deprecations=deprecations,
        aliases=aliases,
        templates=templates,
        nodes=nodes,
    )
