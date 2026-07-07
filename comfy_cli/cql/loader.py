"""Build a CQL-shaped graph dict from sources.

Sources, in priority order:

1. A local file (``--input path``). May be:
   - A raw ``object_info`` JSON dump (the response from ``/object_info``).
   - An API-format workflow JSON.
   - An already-shaped CQL graph (``{"nodes": [...], "inputs": [...]}``).
2. A local ComfyUI server's ``/object_info`` endpoint (``--host`` / ``--port``).

The loader is intentionally permissive: anything dict-shaped that looks like
one of those formats is normalized into ``{"nodes": [...], "inputs": [...],
"categories": [...]}`` so the engine can run uniformly.

This module performs only local I/O. Network calls hit ``http://host:port``
and are short-circuited when no host is provided.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from comfy_cli.cql._net import is_loopback_host
from comfy_cli.cql.errors import CQLRuntimeError
from comfy_cli.http import NoRedirectHandler

# Cap raw bytes read from disk or the network. Real `object_info` dumps are a
# few MB; anything past 256 MiB is almost certainly a wrong path or a hostile
# server and would just OOM the CLI before json.loads even fails.
MAX_INPUT_BYTES = 256 * 1024 * 1024


_LOADER_OPENER = urllib.request.build_opener(NoRedirectHandler())


def load_graph(
    *,
    input_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    if input_path:
        return _load_from_file(input_path)
    if host and port:
        return _load_from_server(host, int(port), timeout=timeout)
    raise CQLRuntimeError(
        "no graph source available",
        details={"hint": "pass --input <path> or --host/--port pointing at a ComfyUI server"},
    )


def _load_from_file(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    try:
        size = p.stat().st_size
    except OSError as e:
        raise CQLRuntimeError(f"cannot stat {p}: {e}") from e
    if size > MAX_INPUT_BYTES:
        raise CQLRuntimeError(
            f"{p} is {size} bytes, exceeds MAX_INPUT_BYTES={MAX_INPUT_BYTES}",
            details={"hint": "shrink the input or raise MAX_INPUT_BYTES in cql.loader"},
        )
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise CQLRuntimeError(f"cannot read {p}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CQLRuntimeError(f"{p} is not valid JSON: {e}") from e
    return normalize(data)


def _load_from_server(host: str, port: int, *, timeout: float) -> dict[str, Any]:
    url = f"http://{host}:{port}/object_info"
    # Refuse anything that isn't a localhost-ish target — we don't want CQL
    # silently sending traffic to a remote box. (Cloud CQL goes through its
    # own path; this loader is local-only by design.)
    parsed = urllib.parse.urlsplit(url)
    hostname = (parsed.hostname or "").strip().lower()
    if not is_loopback_host(hostname):
        raise CQLRuntimeError(
            f"refusing non-loopback CQL server target: {host}",
            details={"hint": "pass --input <path> for remote object_info dumps"},
        )
    try:
        with _LOADER_OPENER.open(url, timeout=timeout) as resp:
            # Bounded read so a misbehaving server can't OOM us.
            raw = resp.read(MAX_INPUT_BYTES + 1)
            if len(raw) > MAX_INPUT_BYTES:
                raise CQLRuntimeError(
                    f"server response exceeds MAX_INPUT_BYTES={MAX_INPUT_BYTES}",
                    details={"host": host, "port": port},
                )
            data = json.loads(raw)
    except urllib.error.URLError as e:
        raise CQLRuntimeError(
            f"failed to reach {url}: {e.reason if hasattr(e, 'reason') else e}",
            details={"host": host, "port": port},
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise CQLRuntimeError(f"server returned invalid object_info: {e}") from e
    return normalize(data)


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
#   1. auto-cache of every successful fetch (per host),
#   2. one refresh-and-retry on failure, and
#   3. a stale-cache fallback (with a clear stderr warning) when the retry
#      still fails — only raising the original error when no cache exists.
#
# An explicit ``--input <object_info.json>`` always wins and is never cached.


def _cache_dir() -> Path:
    """Return the per-user cache directory for comfy-cli object_info dumps.

    Honors ``XDG_CACHE_HOME`` (Linux/freedesktop convention) and falls back to
    ``~/.cache/comfy-cli`` everywhere else. We deliberately use a plain cache
    dir rather than the config dir: this data is reconstructible and safe to
    delete at any time.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "comfy-cli"


def _host_key_digest(host_key: str) -> str:
    """Short, filesystem-safe hash of the target identity.

    ``host_key`` is the resolved base URL (e.g. ``https://api.comfy.org`` or
    ``http://127.0.0.1:8188``) so local and cloud — and distinct cloud envs —
    each get their own cache file and never clobber one another.
    """
    return hashlib.sha256(host_key.encode("utf-8")).hexdigest()[:16]


def object_info_cache_path(host_key: str) -> Path:
    """Cache-file path for a given target identity."""
    return _cache_dir() / f"object_info-{_host_key_digest(host_key)}.json"


def write_object_info_cache(host_key: str, data: dict[str, Any]) -> None:
    """Persist a freshly-fetched object_info dump. Best-effort; never raises.

    Written atomically (tmp + ``os.replace``) so a SIGINT mid-write can't leave
    a half-written file that later loads as garbage.
    """
    path = object_info_cache_path(host_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # A cache we can't write is not worth failing the command over.
        try:
            tmp.unlink()  # type: ignore[possibly-undefined]
        except (OSError, NameError, UnboundLocalError):
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


def _resolve_host_key(mode: str, host: str, port: int) -> str:
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
    host: str = "127.0.0.1",
    port: int = 8188,
    input_path: str | None = None,
    _warn=None,
    on_stale=None,
) -> dict[str, Any]:
    """Fetch ``object_info`` with cache + refresh-retry + stale fallback.

    Resolution order:

    1. ``input_path`` — explicit offline dump always wins; never cached.
    2. Live fetch via the engine. On success, write the per-host cache.
    3. On failure: attempt ``ensure_fresh_session`` and retry the fetch ONCE.
       On success, write the cache.
    4. Still failing: fall back to the cached dump (if any) with a clear
       stderr WARNING that it may be stale.
    5. No cache: re-raise the original ``LoadError`` (callers map it to the
       ``cql_no_graph`` envelope with their existing hint).

    ``_warn`` is an injectable sink for the stale-cache warning (defaults to
    stderr); tests pass their own to assert on it.
    """
    from comfy_cli.cql.engine import LoadError, _load_from_file, _load_from_target

    if input_path is not None:
        # Explicit dump wins and is intentionally not cached — the user is
        # already pinning a known-good file.
        return _load_from_file(input_path)

    host_key = _resolve_host_key(mode, host, port)

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
