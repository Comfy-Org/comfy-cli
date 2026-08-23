"""Find, validate, and index the compiled comfy-knowledge bundle.

The bundle (``knowledge.json`` + optional ``manifest.json``) is produced by
``Comfy-Org/comfy-knowledge``. It reaches this process by one of three routes,
tried in order: an explicit ``COMFY_KNOWLEDGE_FILE``, the per-user cache, or a
fetch from ``COMFY_KNOWLEDGE_URL``. A missing or broken bundle is a normal
state: every entry point here returns ``None`` rather than raising, and nothing
is written to stdout or stderr. :func:`attach` is how discovery commands add a
capped ``knowledge`` block to their payload; it is fail-open for the same reason.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
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

MAX_MODELS = 3
MAX_MODELS_BRIEF = 20
MAX_PICKS = 8
MAX_LIST_ITEMS = 8
MAX_BLOCK_BYTES = 8192
MAX_QUERY_CHARS = 200  # CLI text is unbounded; the clip bounds the prefix walk and the nudge echo

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
    # _normalize(alias) -> id; a key two ids would share is left out, so only the exact path decides it.
    normalized_aliases: dict[str, str] = field(default_factory=dict)
    normalized_capabilities: dict[str, str] = field(default_factory=dict)


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
    return max(ttl, 0.0) if math.isfinite(ttl) else DEFAULT_TTL_SECONDS


def load_bundle(*, force_fetch: bool = False, cache_only: bool = False) -> Bundle | None:
    """Return the indexed bundle, or ``None`` when no usable bundle exists.

    Memoized per process; ``force_fetch=True`` re-runs the load and skips the
    cache TTL gate so a fetch happens whenever ``COMFY_KNOWLEDGE_URL`` is set.
    ``cache_only=True`` never touches the network: env file, then any cache
    (fresh or stale), else ``None``. The memo is shared, except that a
    cache-only miss is not memoized so a later full load can still fetch.
    """
    global _MEMO, _LAST_REASON
    if _MEMO is not None and not force_fetch:
        return _MEMO[0]
    bundle, reason = _load(force_fetch=force_fetch, cache_only=cache_only)
    if bundle is not None or not cache_only:
        _MEMO = (bundle,)
        _LAST_REASON = reason
    return bundle


def _normalize(s: str) -> str:
    """Spelling-variant key: "Hailuo 3", "hailuo-03" and "HAILUO 03" all become "hailuo3"."""
    s = re.sub(r"(?<!\d)0+(?=\d)", "", s.lower())
    return re.sub(r"[^a-z0-9]", "", s)


def _normalized_map(keys: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, target in keys.items():
        norm = _normalize(key)
        if not norm:
            continue
        if out.setdefault(norm, target) != target:
            ambiguous.add(norm)
    for norm in ambiguous:
        del out[norm]
    return out


def resolve_id(bundle: Bundle, query: str) -> str | None:
    """Model id for an alias or id; ``None`` unless that id has a row."""
    q = query.strip().lower()
    model_id = bundle.aliases.get(q)
    if model_id is None:
        model_id = bundle.normalized_aliases.get(_normalize(q))
    return model_id if model_id in bundle.models else None


def resolve(bundle: Bundle, query: str) -> dict | None:
    model_id = resolve_id(bundle, query)
    return bundle.models[model_id] if model_id is not None else None


def pick(bundle: Bundle, capability: str) -> dict | None:
    cap_id = capability.strip().lower()
    if cap_id not in bundle.capabilities:
        cap_id = bundle.normalized_capabilities.get(_normalize(cap_id), cap_id)
    cap = bundle.capabilities.get(cap_id)
    if cap is None:
        return None
    raw_picks = cap.get("picks")
    picks = [p for p in raw_picks if isinstance(p, dict)] if isinstance(raw_picks, list) else []
    out = dict(cap)
    out["picks"] = sorted(picks, key=_rank_key)
    return out


def pick_rank(p: dict) -> int | float | None:
    """The pick's numeric ``rank``; ``None`` when missing or not a number."""
    rank = p.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int | float) or not math.isfinite(rank):
        return None
    return rank


def _rank_key(p: dict) -> tuple[int, float]:
    rank = pick_rank(p)
    return (0, rank) if rank is not None else (1, 0.0)


# ---------------------------------------------------------------------------
# load order
# ---------------------------------------------------------------------------


def _load(*, force_fetch: bool, cache_only: bool = False) -> tuple[Bundle | None, str | None]:
    env_file = os.environ.get(ENV_FILE, "").strip()
    if env_file:
        path = Path(env_file)
        bundle = _load_file(path, path.parent / "manifest.json", source="env")
        return bundle, (None if bundle else REASON_ENV_FILE)

    knowledge_path, manifest_path = cache_paths()
    fresh = _cache_is_fresh(knowledge_path)
    if fresh and not force_fetch:
        bundle = _load_file(knowledge_path, manifest_path, source="cache")
        if bundle is not None:
            return bundle, None

    url = os.environ.get(ENV_URL, "").strip()
    if url and not cache_only:
        bundle = _fetch(url, knowledge_path, manifest_path)
        if bundle is not None:
            return bundle, None

    source = "cache" if fresh else "stale-cache"
    bundle = _load_file(knowledge_path, manifest_path, source=source, stale=not fresh)
    if bundle is not None:
        return bundle, None
    return None, (REASON_FETCH_FAILED if url else REASON_NO_URL)


def _cache_is_fresh(path: Path) -> bool:
    ttl = ttl_seconds()
    if ttl <= 0:
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return 0 <= age < ttl


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
        # The old manifest's sha256 would reject the new bytes if the manifest
        # write below fails or a reader lands between the two writes; an absent
        # manifest only skips the check.
        manifest_path.unlink(missing_ok=True)
        atomic_write_bytes(knowledge_path, raw)
        if manifest_raw is not None:
            atomic_write_bytes(manifest_path, manifest_raw)
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
        # plain_urlopen follows redirects; the final hop must stay https too.
        assert_safe_url(resp.url)
        if resp.status != 200:
            raise RuntimeError(f"knowledge fetch failed: HTTP {resp.status}")
        return read_capped(resp, url, max_bytes=MAX_BUNDLE_BYTES)


# ---------------------------------------------------------------------------
# validation + index
# ---------------------------------------------------------------------------


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant: {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _loads(raw: bytes) -> Any:
    """``json.loads``, refusing the values that cannot round-trip back out as JSON.

    ``NaN``/``Infinity`` are literals json accepts, and an overflowing exponent
    like ``1e400`` becomes ``inf``. Either one re-emitted into an envelope is a
    bare ``NaN``/``Infinity`` token that a strict consumer refuses, so a bundle
    carrying one is rejected whole.
    """
    return json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)


def _parse_manifest(raw: bytes) -> dict | None:
    try:
        manifest = _loads(raw)
    except (ValueError, RecursionError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _parse(raw: bytes, manifest: dict | None) -> dict | None:
    try:
        data = _loads(raw)
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

    # Exact ids first so no alias can shadow a model's own id; then the compiled
    # alias map, which wins over row-level aliases.
    aliases: dict[str, str] = {}
    for mid in models:
        aliases.setdefault(mid.lower(), mid)
    compiled = data.get("aliases")
    if isinstance(compiled, dict):
        for alias, mid in compiled.items():
            if isinstance(alias, str) and isinstance(mid, str):
                aliases.setdefault(alias.lower(), mid)
    for mid, row in models.items():
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

    capability_keys: dict[str, str] = {cid: cid for cid in capabilities}
    for cid, cap in capabilities.items():
        for alias in _str_list(cap.get("aliases")):
            if alias:
                capability_keys.setdefault(alias, cid)

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
        normalized_aliases=_normalized_map(aliases),
        normalized_capabilities=_normalized_map(capability_keys),
    )


# ---------------------------------------------------------------------------
# envelope enrichment
# ---------------------------------------------------------------------------


def _model_keys(q: str) -> list[str]:
    """Keys to try for a model query, most specific first.

    'kling-lipsync' -> ['klinglipsync', 'kling']; 'Hailuo 3' -> ['hailuo3', 'hailuo'].
    A family alias ('kling-i2v', 'flux-kontext-max') falls back to its family row.
    """
    parts = [p for p in re.split(r"[\s_\-/]+", q.strip().lower()) if p]
    return [_normalize("".join(parts[:i])) for i in range(len(parts), 0, -1)]


def _lookup(bundle: Bundle, queries: Iterable[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """-> ([(model_id, matched_key), ...], [capability_id, ...]), de-duplicated, input order."""
    models: list[tuple[str, str]] = []
    seen: set[str] = set()
    caps: list[str] = []
    for q in queries:
        exact = q.strip().lower()
        matched = exact
        mid = bundle.aliases.get(exact)
        if mid is None:
            for prefix in _model_keys(q):
                mid = bundle.normalized_aliases.get(prefix)
                if mid is not None:
                    matched = prefix
                    break
        if mid is not None and mid not in seen:
            seen.add(mid)
            models.append((mid, matched))
        cid = exact if exact in bundle.capabilities else bundle.normalized_capabilities.get(_normalize(q))
        if cid is not None and cid not in caps:
            caps.append(cid)
    return models, caps


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _texts(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [item[key] for item in value if isinstance(item, dict) and isinstance(item.get(key), str)]
    return out[:MAX_LIST_ITEMS]


def _model_entry(bundle: Bundle, model_id: str, row: dict, *, matched_on: str, brief: bool) -> dict:
    dep = bundle.deprecations.get(model_id)
    dep = dep if isinstance(dep, dict) else {}
    entry: dict[str, Any] = {
        "id": model_id,
        "matched_on": matched_on,
        "status": _text(row.get("status")),
        "tier": _text(row.get("tier")),
        "route": _text(row.get("route")),
    }
    superseded_by = dep.get("superseded_by") or row.get("superseded_by")
    if isinstance(superseded_by, str) and superseded_by:
        entry["superseded_by"] = superseded_by
    if isinstance(dep.get("deprecated_on"), str) and dep["deprecated_on"]:
        entry["deprecated_on"] = dep["deprecated_on"]
    if best_for := _str_list(row.get("best_for"))[:MAX_LIST_ITEMS]:
        entry["best_for"] = best_for
    if brief:
        return entry
    for key in ("not_for", "never_confuse_with"):
        if values := _str_list(row.get(key))[:MAX_LIST_ITEMS]:
            entry[key] = values
    routing_raw = row.get("routing")
    routing = (
        [{"when": r.get("when"), "use": r.get("use")} for r in routing_raw if isinstance(r, dict)][:MAX_LIST_ITEMS]
        if isinstance(routing_raw, list)
        else []
    )
    if routing:
        entry["routing"] = routing
    for key, text_key in (("pitfalls", "text"), ("corrections", "claim"), ("warnings", "text")):
        if texts := _texts(row.get(key), text_key):
            entry[key] = texts
    if isinstance(row.get("as_of"), str) and row["as_of"]:
        entry["as_of"] = row["as_of"]
    return entry


def _pick_entries(bundle: Bundle, capability_id: str, *, catalog_templates: Collection[str] | None) -> list[dict]:
    cap = pick(bundle, capability_id)
    if cap is None:
        return []
    out: list[dict] = []
    for p in cap["picks"]:
        template = p.get("template") if isinstance(p.get("template"), str) else None
        if catalog_templates is not None and template is not None and template not in catalog_templates:
            continue
        model_id = p.get("model")
        row = bundle.models.get(model_id, {}) if isinstance(model_id, str) else {}
        dep = bundle.deprecations.get(model_id, {}) if isinstance(model_id, str) else {}
        out.append(
            {
                "capability": capability_id,
                "rank": pick_rank(p),
                "model": _text(model_id),
                "route": _text(p.get("route")),
                "template": template,
                "caveat": _text(p.get("caveat")),
                "status": _text(row.get("status")),
                "superseded_by": _text(dep.get("superseded_by") or row.get("superseded_by")),
            }
        )
        if len(out) >= MAX_PICKS:
            break
    return out


def _resolves_locally(
    row: dict, *, catalog_templates: Collection[str] | None, catalog_nodes: Collection[str] | None
) -> bool:
    """Version-skew filter: a row whose every known id is absent from the live catalog is dropped."""
    resolves = row.get("resolves")
    if not isinstance(resolves, dict):
        return True
    for catalog, key in ((catalog_templates, "templates"), (catalog_nodes, "nodes")):
        if catalog is None:
            continue
        ids = _str_list(resolves.get(key))
        if ids and not any(i in catalog for i in ids):
            return False
    return True


def _block_bytes(block: dict) -> int:
    """Size of the block as the renderer will actually emit it (UTF-8, unescaped)."""
    return len(json.dumps(block, ensure_ascii=False).encode())


def _fit(block: dict) -> None:
    """Drop whole entries (models first, then picks) until the block fits MAX_BLOCK_BYTES."""
    while True:
        caps: list[str] = []
        for p in block["picks"]:
            if p["capability"] not in caps:
                caps.append(p["capability"])
        block["hit_ids"] = [m["id"] for m in block["models"]] + [f"cap:{c}" for c in caps]
        if _block_bytes(block) <= MAX_BLOCK_BYTES:
            return
        if block["models"]:
            block["models"].pop()
        elif block["picks"]:
            block["picks"].pop()
        else:
            return


def attach(
    payload: dict,
    *,
    queries: Iterable[str] = (),
    templates: Iterable[str] = (),
    nodes: Iterable[str] = (),
    catalog_templates: Collection[str] | None = None,
    catalog_nodes: Collection[str] | None = None,
    brief: bool = False,
    thin: bool = False,
) -> None:
    """Append a capped ``knowledge`` block to ``payload`` for what the command was asked about.

    ``queries`` are resolved as model aliases or capability names; ``templates``
    and ``nodes`` go through the reverse index. ``catalog_*`` are the ids the
    command actually loaded, used to drop rows and picks that do not resolve
    locally. ``thin`` marks a command whose own result was empty, which is the
    only case that earns ``zero_hit``; a query that resolved to nothing earns a
    nudge on its own. Fail-open: any exception leaves ``payload`` exactly as it
    was.
    """
    try:
        bundle = load_bundle(cache_only=True)
        if bundle is None:
            return
        query_list = [q.strip()[:MAX_QUERY_CHARS] for q in queries if isinstance(q, str) and q.strip()]
        model_hits, cap_hits = _lookup(bundle, query_list)
        query_resolved = bool(model_hits or cap_hits)
        seen = {mid for mid, _ in model_hits}
        for ids, index in ((templates, bundle.templates), (nodes, bundle.nodes)):
            for ident in ids:
                for mid in index.get(ident, ()):
                    if mid not in seen:
                        seen.add(mid)
                        model_hits.append((mid, ident))

        entries: list[dict] = []
        for mid, matched_on in model_hits:
            row = bundle.models.get(mid)
            if not isinstance(row, dict):
                continue
            if not _resolves_locally(row, catalog_templates=catalog_templates, catalog_nodes=catalog_nodes):
                continue
            entries.append(_model_entry(bundle, mid, row, matched_on=matched_on, brief=brief))
        entries.sort(key=lambda e: e.get("tier") != "law")
        entries = entries[: MAX_MODELS_BRIEF if brief else MAX_MODELS]

        picks: list[dict] = []
        for cid in cap_hits:
            picks.extend(_pick_entries(bundle, cid, catalog_templates=catalog_templates))
        picks = picks[:MAX_PICKS]

        block: dict[str, Any] = {
            "bundle_version": bundle.version,
            "stale": bundle.stale,
            "as_of": bundle.as_of,
            "models": entries,
            "picks": picks,
            "hit_ids": [],
            "zero_hit": False,
        }
        had_hits = bool(entries or picks)
        _fit(block)
        if not block["models"] and not block["picks"]:
            if had_hits or not (thin and query_list):
                return
            block["zero_hit"] = True
            head = f"no curated knowledge for {query_list[0]!r}"
            block["nudge"] = f"{head}; covered capabilities: {', '.join(sorted(bundle.capabilities))}"
            if _block_bytes(block) > MAX_BLOCK_BYTES:
                block["nudge"] = head
        elif query_list and not query_resolved:
            # A browse that returns rows the reverse index enriched, while the
            # query itself matched neither a model nor a capability. Not a
            # zero_hit: that feeds the miss log and means an empty block.
            head = f"no curated knowledge for {query_list[0]!r}"
            block["nudge"] = f"{head}; covered capabilities: {', '.join(sorted(bundle.capabilities))}"
            if _block_bytes(block) > MAX_BLOCK_BYTES:
                block["nudge"] = head
            if _block_bytes(block) > MAX_BLOCK_BYTES:
                del block["nudge"]
        payload["knowledge"] = block
    except Exception:  # noqa: BLE001 — knowledge is additive; the payload ships unchanged on any failure
        return
