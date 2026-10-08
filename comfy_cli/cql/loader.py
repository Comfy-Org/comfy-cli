"""Shape and load CQL ``object_info`` graphs.

This module contains two things:

- ``normalize`` — turn any supported input (a raw ``object_info`` dump, an
  API-format workflow, or an already-shaped CQL graph) into the uniform
  ``{"nodes": [...], "inputs": [...], "categories": [...]}`` dict the engine
  runs on. It is intentionally permissive: anything dict-shaped that looks
  like one of those formats is accepted.
- ``resilient_load_object_info`` — a cache + refresh-retry + stale-fallback
  wrapper over the engine's loaders (``comfy_cli.cql.engine._load_from_file``
  / ``_load_from_target``). It auto-caches every successful fetch per host,
  retries once after a token refresh on failure, and falls back to the cached
  dump (with a stderr warning) when the retry still fails.

The live network fetch and its security guards (loopback check, no-redirect
opener, byte cap, cloud HTTPS+auth) live in ``comfy_cli.cql.engine`` — this
module never opens a socket itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from comfy_cli.cql.errors import CQLRuntimeError
from comfy_cli.file_utils import atomic_write_text, cache_dir

# ---- normalization --------------------------------------------------------


def normalize(data: Any) -> dict[str, Any]:
    """Turn any supported input into ``{nodes, inputs, categories}``."""
    if not isinstance(data, dict):
        raise CQLRuntimeError("expected a JSON object at the top level")

    # Already CQL-shaped — trust it.
    if any(isinstance(data.get(k), list) for k in ("nodes", "inputs", "categories")):
        graph: dict[str, Any] = {
            "nodes": list(data.get("nodes") or []),
            "inputs": list(data.get("inputs") or []),
            "categories": list(data.get("categories") or []),
        }
        return graph

    if _looks_like_object_info(data):
        return _from_object_info(data)
    if _looks_like_api_workflow(data):
        return _from_api_workflow(data)

    raise CQLRuntimeError(
        "unrecognized graph shape",
        details={"keys_sample": sorted(list(data.keys()))[:10]},
    )


def _looks_like_object_info(data: dict[str, Any]) -> bool:
    # /object_info maps "ClassName" -> { "input": {...}, "category": "...",
    # "display_name": "...", "description": "...", "output": [...], ... }
    if not data:
        return False
    return any(isinstance(v, dict) and ("input" in v or "category" in v) for v in data.values())


def _looks_like_api_workflow(data: dict[str, Any]) -> bool:
    if not data:
        return False
    return any(isinstance(v, dict) and "class_type" in v for v in data.values())


def _from_object_info(data: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    categories: dict[str, int] = {}

    for class_name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        category = raw.get("category")
        node = {
            "name": class_name,
            "display_name": raw.get("display_name") or class_name,
            "category": category,
            "description": raw.get("description"),
            "output_node": bool(raw.get("output_node", False)),
            "output_types": list(raw.get("output") or []),
        }
        nodes.append(node)
        if category:
            categories[category] = categories.get(category, 0) + 1

        sections = raw.get("input") or {}
        if isinstance(sections, dict):
            for section, body in sections.items():  # "required" / "optional" / "hidden"
                if not isinstance(body, dict):
                    continue
                for input_name, spec in body.items():
                    inputs.append(_normalize_input(class_name, section, input_name, spec))

    return {
        "nodes": nodes,
        "inputs": inputs,
        "categories": [{"name": k, "node_count": v} for k, v in sorted(categories.items())],
    }


def _normalize_input(class_name: str, section: str, name: str, spec: Any) -> dict[str, Any]:
    type_name: Any = None
    options: dict[str, Any] = {}
    choices: list[Any] = []
    if isinstance(spec, list) and spec:
        type_name = spec[0]
        if isinstance(type_name, list):
            choices = list(type_name)
            type_name = "ENUM"
        if len(spec) > 1 and isinstance(spec[1], dict):
            options = dict(spec[1])
    elif isinstance(spec, str):
        type_name = spec
    return {
        "node": class_name,
        "section": section,
        "name": name,
        "type": type_name,
        "choices": choices,
        "options": options,
    }


def _from_api_workflow(data: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    node_ids = {str(k) for k in data}
    for nid, node in data.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        title = (node.get("_meta") or {}).get("title") if isinstance(node.get("_meta"), dict) else None
        nodes.append(
            {
                "id": nid,
                "name": class_type or "?",
                "class_type": class_type,
                "title": title,
                "category": None,
            }
        )
        raw_inputs = node.get("inputs") or {}
        if isinstance(raw_inputs, dict):
            for in_name, value in raw_inputs.items():
                ref = (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[1], int)
                    and not isinstance(value[1], bool)
                    and str(value[0]) in node_ids
                )
                inputs.append(
                    {
                        "node_id": nid,
                        "node": class_type,
                        "name": in_name,
                        "value": None if ref else value,
                        "ref_node": value[0] if ref else None,
                        "ref_slot": value[1] if ref else None,
                        "is_reference": ref,
                    }
                )
    return {"nodes": nodes, "inputs": inputs, "categories": []}


# ---------------------------------------------------------------------------
# Resilient object_info loading (cache + refresh-retry + stale fallback)
# ---------------------------------------------------------------------------
#
# The live ``/object_info`` fetch (``comfy nodes``, ``comfy workflow slots``,
# ``comfy validate``) intermittently returns HTTP 401 / ``cql_no_graph`` mid
# session when the cloud access token has gone stale. The session token DOES
# auto-refresh (see ``comfy_cli.cloud.oauth.ensure_fresh_session``), but the
# raw object_info path didn't leverage it, and there was no offline fallback.
#
# ``resilient_load_object_info`` wraps the engine's network fetch with:
#   1. a cache-first TTL gate: a cache entry younger than the TTL (default
#      10 minutes, ``COMFY_OBJECT_INFO_TTL`` seconds to override, ``0`` to
#      always fetch) is served without any network call,
#   2. auto-cache of every successful fetch (per host),
#   3. one refresh-and-retry on failure, and
#   4. a stale-cache fallback (with a clear stderr warning) when the retry
#      still fails — only raising the original error when no cache exists.
#
# An explicit ``--input <object_info.json>`` always wins and is never cached.


def _host_key_digest(host_key: str) -> str:
    """Short, filesystem-safe hash of the target identity.

    ``host_key`` is the resolved base URL (e.g. ``https://api.comfy.org`` or
    ``http://127.0.0.1:8188``) so local and cloud — and distinct cloud envs —
    each get their own cache file and never clobber one another.
    """
    return hashlib.sha256(host_key.encode("utf-8")).hexdigest()[:16]


def object_info_cache_path(host_key: str) -> Path:
    """Cache-file path for a given target identity."""
    return cache_dir() / f"object_info-{_host_key_digest(host_key)}.json"


def write_object_info_cache(host_key: str, data: dict[str, Any]) -> None:
    """Persist a freshly-fetched object_info dump. Best-effort; never raises.

    Written atomically (tmp + ``os.replace``) so a SIGINT mid-write can't leave
    a half-written file that later loads as garbage.
    """
    path = object_info_cache_path(host_key)
    try:
        atomic_write_text(path, json.dumps(data))
    except OSError:
        # A cache we can't write is not worth failing the command over.
        # atomic_write_text already cleaned up its own tmp file on failure.
        pass


def read_object_info_cache(host_key: str) -> dict[str, Any] | None:
    """Return the cached object_info dump for ``host_key``, or ``None``.

    Returns ``None`` on any problem (missing file, unreadable, corrupt JSON,
    wrong shape) — the caller treats "no usable cache" uniformly.
    """
    path = object_info_cache_path(host_key)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# Cache-first TTL policy. A cache entry younger than this is served without a
# network call; the entry's age is its file mtime (``write_object_info_cache``
# writes via tmp + ``os.replace``, so mtime == fetch time).
DEFAULT_OBJECT_INFO_TTL_SECONDS = 600.0
OBJECT_INFO_TTL_ENV = "COMFY_OBJECT_INFO_TTL"


def object_info_cache_ttl() -> float:
    """TTL (seconds) for the cache-first object_info gate.

    Reads ``COMFY_OBJECT_INFO_TTL``; unset/blank/unparseable values fall back
    to the 10-minute default. ``0`` (or any value <= 0) disables the
    cache-first gate entirely — every call fetches live, restoring the
    pre-TTL behavior (the stale-cache *failure* fallback still applies).
    """
    raw = os.environ.get(OBJECT_INFO_TTL_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_OBJECT_INFO_TTL_SECONDS
    try:
        ttl = float(raw)
    except ValueError:
        return DEFAULT_OBJECT_INFO_TTL_SECONDS
    return max(ttl, 0.0)


def read_fresh_object_info_cache(host_key: str, ttl: float) -> dict[str, Any] | None:
    """Return the cached dump for ``host_key`` iff it is younger than ``ttl``.

    Freshness is judged by the cache file's mtime. Returns ``None`` when the
    TTL gate is disabled (``ttl <= 0``), the file is missing/unreadable, the
    entry has expired, or the mtime is in the future (clock skew — treat as
    expired rather than trusting a timestamp we can't reason about).
    """
    if ttl <= 0:
        return None
    path = object_info_cache_path(host_key)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None
    if age < 0 or age >= ttl:
        return None
    return read_object_info_cache(host_key)


def _resolve_host_key(mode: str, host: str | None, port: int | None) -> str:
    """Resolve the cache key (the target base URL) without doing any I/O.

    Mirrors how the engine resolves its fetch target so the cache key matches
    the server actually queried. Falls back to a host:port string if the
    Target machinery is unavailable (e.g. unconfigured cloud).
    """
    try:
        from comfy_cli.target import resolve_target

        target = resolve_target(where=mode, host=host, port=port)
        return target.base_url
    except Exception:  # noqa: BLE001 — never let key resolution break the fetch
        return f"{mode}:{host}:{port}"


def resilient_load_object_info(
    *,
    mode: str = "local",
    host: str | None = None,
    port: int | None = None,
    input_path: str | None = None,
    _warn=None,
    on_stale=None,
) -> dict[str, Any]:
    """Fetch ``object_info`` cache-first, with refresh-retry + stale fallback.

    Resolution order:

    1. ``input_path`` — explicit offline dump always wins; never cached.
    2. Cache-first TTL gate: a per-host cache entry younger than the TTL
       (default 10 minutes; ``COMFY_OBJECT_INFO_TTL`` seconds to override,
       ``0`` to always fetch live) is returned with NO network call.
    3. Live fetch via the engine. On success, write the per-host cache.
    4. On failure: attempt ``ensure_fresh_session`` and retry the fetch ONCE.
       On success, write the cache.
    5. Still failing: fall back to the cached dump (if any, regardless of
       age) with a clear stderr WARNING that it may be stale.
    6. No cache: re-raise the original ``LoadError`` (callers map it to the
       ``cql_no_graph`` envelope with their existing hint).

    The cache key is the resolved target base URL, so local vs cloud — and
    distinct base URLs — never share an entry.

    ``_warn`` is an injectable sink for the stale-cache warning (defaults to
    stderr); tests pass their own to assert on it.
    """
    from comfy_cli.cql.engine import LoadError, _load_from_file, _load_from_target

    if input_path is not None:
        # Explicit dump wins and is intentionally not cached — the user is
        # already pinning a known-good file.
        return _load_from_file(input_path)

    # Offline default catalog: COMFY_OBJECT_INFO_FILE is honored exactly like an
    # explicit --input dump, so EVERY object_info consumer routed through this
    # loader — workflow edits, `nodes show`/`find`, `validate`, fragments —
    # resolves the node schema from a pre-warmed / baked file with no network
    # fetch and no cloud credential. A host (e.g. a server-side agent) sets it
    # once instead of threading --input through each command.
    env_dump = os.environ.get("COMFY_OBJECT_INFO_FILE")
    if env_dump:
        return _load_from_file(env_dump)

    host_key = _resolve_host_key(mode, host, port)

    # Cache-first TTL is CLOUD-only. The cloud catalog is stable and its remote
    # /object_info fetch is slow (multi-MB over the network), so a fresh cache hit
    # is a real win. Local is the opposite: the localhost fetch is cheap, and a
    # user installs custom nodes into their OWN server — serving a cached local
    # catalog would hide a just-added node for the whole TTL. So local always
    # fetches live. (The stale-cache *failure* fallback below still applies to
    # both: a cache is still written on a successful local fetch so a later
    # unreachable-server call can fall back to it.)
    if mode == "cloud":
        fresh = read_fresh_object_info_cache(host_key, object_info_cache_ttl())
        if fresh is not None:
            return fresh

    try:
        data = _load_from_target(mode=mode, host=host, port=port)
        write_object_info_cache(host_key, data)
        return data
    except LoadError as first_err:
        # (a) Best-effort token refresh, then retry the fetch exactly once.
        # Refresh only helps cloud auth, but it's cheap and a no-op locally.
        # ``force=True``: the fetch already failed (typically HTTP 401), and a
        # server 401 is authoritative — the access token is rejected even if
        # our local clock still thinks it is valid (skew / no recorded
        # expiry). A non-forced refresh would no-op in that case and the retry
        # would re-send the same dead token. Force-refresh spends the refresh
        # token so the retry carries a brand-new access token.
        try:
            from comfy_cli.credentials import get_session

            get_session(refresh=True, force=True)
        except Exception:  # noqa: BLE001 — refresh is best-effort
            pass

        try:
            data = _load_from_target(mode=mode, host=host, port=port)
            write_object_info_cache(host_key, data)
            return data
        except LoadError:
            # Retry failed too — fall through to the cache.
            pass

        # (b) Stale-cache fallback.
        cached = read_object_info_cache(host_key)
        if cached is not None:
            warn = _warn if _warn is not None else _default_warn
            warn(
                f"WARNING: could not refresh object_info from {host_key} "
                f"({first_err}); using a cached copy that may be stale. "
                f"Run the command again once the server/session is reachable."
            )
            if on_stale is not None:
                on_stale(host_key, str(first_err))
            return cached

        # (c) No cache — surface the original error untouched.
        raise first_err


def _default_warn(message: str) -> None:
    print(message, file=sys.stderr)
