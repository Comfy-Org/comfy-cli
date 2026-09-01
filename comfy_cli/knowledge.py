"""Find, validate, and index the compiled comfy-knowledge bundle.

The bundle (``knowledge.json`` + optional ``manifest.json``) is produced by
the curated knowledge bundle repo. It reaches this process by one of three routes,
tried in order: an explicit ``COMFY_KNOWLEDGE_FILE``, the per-user cache, or a
fetch from ``COMFY_KNOWLEDGE_URL``, which defaults to the knowledge channel
under the cloud base URL. A missing or broken bundle is a normal
state: every entry point here returns ``None`` rather than raising, and nothing
is written to stdout or stderr. :func:`attach` is how discovery commands add a
capped ``knowledge`` block to their payload; it is fail-open for the same reason.
Setting ``COMFY_KNOWLEDGE_DISABLE`` turns that enrichment off without
disturbing the explicit ``comfy knowledge`` verbs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comfy_cli.cloud import get_base_url
from comfy_cli.file_utils import atomic_write_bytes, cache_dir
from comfy_cli.http import assert_safe_url, authed_urlopen, plain_urlopen, read_capped

SCHEMA_VERSION = 1
ENV_FILE = "COMFY_KNOWLEDGE_FILE"
ENV_URL = "COMFY_KNOWLEDGE_URL"
ENV_TTL = "COMFY_KNOWLEDGE_TTL"
ENV_DISABLE = "COMFY_KNOWLEDGE_DISABLE"
DEFAULT_URL_PATH = "/api/knowledge/knowledge.json"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT_SECONDS = 10.0
MAX_BUNDLE_BYTES = 16 * 1024 * 1024

MAX_MODELS = 3
MAX_MODELS_BRIEF = 20
MAX_PICKS = 8
MAX_LIST_ITEMS = 8
# best_for items per pick. Bounds item count, not bytes: with today's bundle, 1
# keeps every capability's `knowledge pick` envelope under the 4096 bytes the
# cloud agent admits verbatim (2, or not_for alongside, sends image-edit over),
# but a bundle recompile with longer best_for text isn't guaranteed to stay
# under that.
MAX_PICK_BEST_FOR = 1
MAX_BLOCK_BYTES = 8192
MAX_QUERY_CHARS = 200  # CLI text is unbounded; the clip bounds the lookup key and the nudge echo
MAX_VERSION_CHARS = 64

UNAVAILABLE_LOCALLY = "the templates or nodes this row resolves to are absent from this install"

# Grammatical filler dropped before subset matching. Deliberately generic English:
# naming a capability, model or gallery tag here would be a second copy of the
# bundle's vocabulary, which then drifts every time comfy-knowledge is recompiled.
_FILLER = frozenset(
    "a an the of to for from with and or in on at by as is are be do it which where while "
    "my me some this that into make create generate".split()
)
# A description counts as match context only up to its first sentence end or
# negator, whichever comes first: what follows says what the row is *not*, or
# how it routes. "e.g." and "No.1" are not ends.
_CONTEXT_END = re.compile(
    r"(?<=[a-z]{2}[.;])\s|\b(?:no|not|never|neither|nor|without|except)\b(?![.\d])", re.IGNORECASE
)
# A one-word key must be at least this long to match on its own. Guards against a
# future alias like "3d" dragging a capability into every query mentioning it.
MIN_SINGLE_TOKEN_CHARS = 4

REASON_ENV_FILE = "COMFY_KNOWLEDGE_FILE is set but could not be loaded"
REASON_SIGNED_OUT = (
    "fetch from the cloud knowledge channel failed and no cached bundle exists; "
    "the usual cause is being signed out (run `comfy cloud login`)"
)
REASON_FETCH_FAILED = "fetch from COMFY_KNOWLEDGE_URL failed and no cached bundle exists"


@dataclass(frozen=True)
class Bundle:
    version: str
    source: str  # "env" | "cache" | "fetch" | "stale-cache"
    # Not an age signal. True only on the "stale-cache" path: the TTL expired
    # and no fetch replaced it. A caller that passes COMFY_KNOWLEDGE_FILE takes
    # the "env" branch, where this is always False no matter how old the file
    # is, so a consumer reading it as freshness learns nothing.
    stale: bool
    # When this machine got the file, from its mtime. Says nothing about how
    # old the content is: a bundle fetched at pod start reads as new whatever
    # it holds. Read compiled_at for the content's date.
    as_of: str
    # The content's date, from the manifest's compiled_at. None for a bundle
    # compiled before the key existed, or one built outside a git checkout.
    compiled_at: str | None
    path: str
    models: dict[str, dict]
    capabilities: dict[str, dict]
    deprecations: dict[str, dict]  # keyed by id
    aliases: dict[str, str]  # lowercased alias or model id -> model id
    templates: dict[str, list[str]]  # template id -> model ids
    nodes: dict[str, list[str]]  # node class -> model ids
    model_capabilities: dict[str, list[str]]  # model id -> capability ids that rank it
    # _normalize(alias) -> id. A key two ids would share is left out, so only the
    # exact path decides it; a real id always keeps its own key.
    normalized_aliases: dict[str, str] = field(default_factory=dict)
    normalized_capabilities: dict[str, str] = field(default_factory=dict)
    # (word set, normalized key, capability id) for every capability id and alias,
    # so an intent phrase reaches the row its own wording names. See :func:`_resolve_tokens`.
    capability_tokens: tuple[tuple[frozenset[str], str, str], ...] = ()
    capability_context: dict[str, frozenset[str]] = field(default_factory=dict)


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
    base = cache_dir() / "knowledge"
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


def default_url() -> str:
    """``knowledge.json`` under the cloud base URL, derived on every call.

    Never stored: :func:`_http_get` attaches credentials only under
    ``get_base_url()``, which resolves per invocation, so a remembered
    production URL would silently fetch unauthenticated for anyone pointed at
    another environment.
    """
    return get_base_url().rstrip("/") + DEFAULT_URL_PATH


def bundle_url() -> str:
    """``COMFY_KNOWLEDGE_URL`` when set, else :func:`default_url`."""
    return os.environ.get(ENV_URL, "").strip() or default_url()


def load_bundle(*, force_fetch: bool = False, cache_only: bool = False) -> Bundle | None:
    """Return the indexed bundle, or ``None`` when no usable bundle exists.

    Memoized per process; ``force_fetch=True`` re-runs the load and skips the
    cache TTL gate so a fetch happens unless ``COMFY_KNOWLEDGE_FILE`` is set.
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


def _stem(word: str) -> str:
    """Suffix strip, repeated until nothing matches, so "editing", "edits" and "edit" agree.

    Only inflections: "-al" is left alone so "musical" never lands on the alias "Music".
    """
    stripped = True
    while stripped:
        stripped = False
        for suffix in ("ing", "ed", "s", "e"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                stripped = True
                break
    return word


_FILLER_STEMS = frozenset(map(_stem, _FILLER))


def _split(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _tokens(s: str) -> frozenset[str]:
    """Stemmed word set of a key or description, with grammatical filler dropped."""
    return frozenset(w for w in map(_stem, _split(s)) if w not in _FILLER_STEMS)


def _short_key(key: str) -> bool:
    """A one-word key (filler aside) under MIN_SINGLE_TOKEN_CHARS never matches on wording."""
    raw = {w for w in _split(key) if _stem(w) not in _FILLER_STEMS}
    return len(raw) == 1 and len(next(iter(raw))) < MIN_SINGLE_TOKEN_CHARS


def _query_tokens(s: str) -> frozenset[str]:
    """:func:`_tokens` plus each adjacent pair joined, so "lip sync" reaches the key ``lipsync``."""
    words = _split(s)
    return _tokens(s) | frozenset(_stem(a + b) for a, b in zip(words, words[1:]))


def _normalized_map(keys: dict[str, str], *, ids: Collection[str] = ()) -> dict[str, str]:
    """``_normalize(key) -> target``, with every spelling two targets share left out.

    Ids in ``ids`` are resolved after the aliases and override them, so no alias
    can delete a real id. Two ids sharing one key cancel each other the same way
    two aliases do: the tie stays untied, and only the exact spelling resolves it.
    """
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

    owners: dict[str, str] = {}
    contested: set[str] = set()
    for identifier in ids:
        norm = _normalize(identifier)
        if not norm:
            continue
        if owners.setdefault(norm, identifier) != identifier:
            contested.add(norm)
    for norm, identifier in owners.items():
        if norm in contested:
            out.pop(norm, None)
        else:
            out[norm] = identifier
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
    """Ranked picks for a capability, keyed exactly, then by spelling, then by wording.

    The wording pass is the same :func:`_resolve_tokens` enrichment uses, so a
    phrased request ("upscale this video") resolves here as well as it does when
    it rides along on a block.
    """
    cap_id = capability.strip().lower()
    if cap_id not in bundle.capabilities:
        cap_id = bundle.normalized_capabilities.get(_normalize(cap_id)) or _resolve_tokens(bundle, capability) or cap_id
    cap = bundle.capabilities.get(cap_id)
    if cap is None:
        return None
    raw_picks = cap.get("picks")
    picks = [p for p in raw_picks if isinstance(p, dict)] if isinstance(raw_picks, list) else []
    out = dict(cap)
    # The key resolved through spelling and wording is the answer; a row that
    # omits its own ``id`` must not send the caller back to the raw phrase.
    out["id"] = cap_id
    out["picks"] = [_with_fits(bundle, p) for p in sorted(picks, key=_rank_key)]
    return out


def _with_fits(bundle: Bundle, p: dict) -> dict:
    """The pick plus its model row's ``fits`` block, copied so the bundle stays untouched."""
    model = p.get("model")
    row = bundle.models.get(model) if isinstance(model, str) else None
    fits = row.get("fits") if row is not None else None
    if not isinstance(fits, dict):
        return p
    return {**p, "fits": copy.deepcopy(fits)}


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

    explicit_url = os.environ.get(ENV_URL, "").strip()
    if not cache_only:
        bundle = _fetch(explicit_url or default_url(), knowledge_path, manifest_path)
        if bundle is not None:
            return bundle, None

    source = "cache" if fresh else "stale-cache"
    bundle = _load_file(knowledge_path, manifest_path, source=source, stale=not fresh)
    if bundle is not None:
        return bundle, None
    return None, (REASON_FETCH_FAILED if explicit_url else REASON_SIGNED_OUT)


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

        # Best-effort background fetch: a spurious refresh failure must not log the user out.
        target = resolve_target(where="cloud", allow_clear=False)
        opened = authed_urlopen(url, target, timeout=FETCH_TIMEOUT_SECONDS)
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

    model_capabilities: dict[str, list[str]] = {}
    for cid, cap in capabilities.items():
        cap_picks = cap.get("picks")
        if not isinstance(cap_picks, list):
            continue
        for p in cap_picks:
            if isinstance(p, dict) and isinstance(p.get("model"), str):
                _append_unique(model_capabilities, p["model"], cid)

    capability_keys: dict[str, str] = {cid: cid for cid in capabilities}
    for cid, cap in capabilities.items():
        for alias in _str_list(cap.get("aliases")):
            if alias:
                capability_keys.setdefault(alias, cid)

    capability_tokens = tuple(
        (words, _normalize(key), cid)
        for key, cid in capability_keys.items()
        if (words := _tokens(key)) and not _short_key(key)
    )
    key_words: dict[str, set[str]] = defaultdict(set)
    for words, _, cid in capability_tokens:
        key_words[cid] |= words
    capability_context = {
        cid: _tokens(_CONTEXT_END.split(desc, maxsplit=1)[0]) - key_words[cid]
        for cid, cap in capabilities.items()
        if isinstance(desc := cap.get("description"), str)
    }

    version = manifest.get("version") if manifest is not None else None
    compiled_at = manifest.get("compiled_at") if manifest is not None else None
    return Bundle(
        version=version[:MAX_VERSION_CHARS] if isinstance(version, str) else "unknown",
        source=source,
        stale=stale,
        as_of=datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        compiled_at=(compiled_at if isinstance(compiled_at, str) and len(compiled_at) <= MAX_VERSION_CHARS else None),
        path=path,
        models=models,
        capabilities=capabilities,
        deprecations=deprecations,
        aliases=aliases,
        templates=templates,
        nodes=nodes,
        model_capabilities=model_capabilities,
        normalized_aliases=_normalized_map(aliases, ids=models),
        normalized_capabilities=_normalized_map(capability_keys, ids=capabilities),
        capability_tokens=capability_tokens,
        capability_context=capability_context,
    )


# ---------------------------------------------------------------------------
# envelope enrichment
# ---------------------------------------------------------------------------


def _lookup(bundle: Bundle, queries: Iterable[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """-> ([(model_id, matched_key), ...], [capability_id, ...]), de-duplicated, input order.

    Keyed only, through the same :func:`resolve_id` the ``comfy knowledge resolve``
    verb uses, so a string the verb calls unknown never gets a row attached here.
    """
    models: list[tuple[str, str]] = []
    seen: set[str] = set()
    caps: list[str] = []
    for q in queries:
        exact = q.strip().lower()
        mid = resolve_id(bundle, q)
        if mid is not None and mid not in seen:
            seen.add(mid)
            models.append((mid, exact))
        cid = exact if exact in bundle.capabilities else bundle.normalized_capabilities.get(_normalize(q))
        if cid is None:
            cid = _resolve_tokens(bundle, q)
        if cid is not None and cid not in caps:
            caps.append(cid)
    return models, caps


def _resolve_tokens(bundle: Bundle, query: str) -> str | None:
    """Capability whose id or alias is worded inside ``query``; ``None`` if none is.

    Callers ask in sentences ("image to 3d model mesh") while the bundle is keyed
    on tags and ids ("Image to Model", "image-to-3d"), and :func:`_normalize`
    only strips punctuation, so the exact path misses everything phrased.

    Each key scores by the share of its words the query contains. A full match
    always counts. A partial one counts only when the query shares more words
    with the capability's description than the key words it lacks, so "poster
    with readable text" reaches ``text-in-image`` while "4K video generation"
    reaches nothing. Among capabilities that score the same, the key spelled
    out literally in the query wins ("text to image" over ``text-in-image``),
    then the one whose description the query echoes.

    Ambiguity resolves to nothing rather than to a guess, matching
    :func:`_normalized_map`: a tie between two capabilities is not an answer.
    """
    words = _query_tokens(query)
    if not words:
        return None
    literal = _normalize(query)
    scored: dict[str, tuple[float, int, int, int]] = {}
    for key_words, key_norm, cid in bundle.capability_tokens:
        hit = key_words & words
        if not hit:
            continue
        context = len(bundle.capability_context.get(cid, frozenset()) & words)
        missing = len(key_words) - len(hit)
        if missing and context <= missing:
            continue
        score = (len(hit) / len(key_words), len(hit), len(key_norm) if key_norm in literal else 0, context)
        if score > scored.get(cid, (0.0, 0, 0, 0)):
            scored[cid] = score
    if not scored:
        return None
    best = max(scored.values())
    winners = [cid for cid, score in scored.items() if score == best]
    return winners[0] if len(winners) == 1 else None


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
        [{"when": _text(r.get("when")), "use": _text(r.get("use"))} for r in routing_raw if isinstance(r, dict)][
            :MAX_LIST_ITEMS
        ]
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


def pick_entry(bundle: Bundle, p: dict) -> dict:
    """One pick as emitted, plus the status and routing opinion of its model row.

    ``best_for`` is the head of the row's list, omitted when the row has none.
    The pick's ``caveat`` is written for one capability; the row's ``best_for``
    says what the model is for across all of them, which is what a routing
    decision reads.
    """
    model_id = _text(p.get("model"))
    row = (bundle.models.get(model_id) or {}) if model_id is not None else {}
    dep = (bundle.deprecations.get(model_id) or {}) if model_id is not None else {}
    entry = {
        "rank": pick_rank(p),
        "model": model_id,
        "route": _text(p.get("route")),
        "template": _text(p.get("template")),
        "caveat": _text(p.get("caveat")),
        "status": _text(row.get("status")),
        "superseded_by": _text(dep.get("superseded_by")) or _text(row.get("superseded_by")),
    }
    if best_for := [x for x in _str_list(row.get("best_for")) if x][:MAX_PICK_BEST_FOR]:
        entry["best_for"] = best_for
    return entry


def _pick_entries(bundle: Bundle, capability_id: str, *, catalog_templates: Collection[str] | None) -> list[dict]:
    cap = pick(bundle, capability_id)
    if cap is None:
        return []
    out: list[dict] = []
    for p in cap["picks"]:
        entry = {"capability": capability_id, **pick_entry(bundle, p)}
        template = entry["template"]
        if catalog_templates is not None and template is not None and template not in catalog_templates:
            entry["available_locally"] = False
            entry["unavailable_reason"] = UNAVAILABLE_LOCALLY
        out.append(entry)
        if len(out) >= MAX_PICKS:
            break
    return out


def _resolves_locally(
    row: dict, *, catalog_templates: Collection[str] | None, catalog_nodes: Collection[str] | None
) -> bool:
    """Version-skew check: False only when the row names ids for the catalogs on
    hand and not one of them is there. Either catalog resolving the row is enough.

    A False row is annotated, never dropped — a curated answer the install cannot
    run today is still the answer, and hiding it reads as "nothing is curated".
    """
    resolves = row.get("resolves")
    if not isinstance(resolves, dict):
        return True
    checked = False
    for catalog, key in ((catalog_templates, "templates"), (catalog_nodes, "nodes")):
        if catalog is None:
            continue
        ids = _str_list(resolves.get(key))
        if not ids:
            continue
        checked = True
        if any(i in catalog for i in ids):
            return True
    return not checked


def _block_bytes(block: dict) -> int:
    """Size of the block as the renderer will actually emit it (UTF-8, unescaped)."""
    return len(json.dumps(block, ensure_ascii=False).encode())


def _nudge_text(block: dict, head: str) -> str:
    """Point at ``capabilities_available`` rather than repeating the id list."""
    if "capabilities_available" not in block:
        return head
    return f"{head}; see capabilities_available"


def _hit_ids(block: dict) -> list[str]:
    """Model ids then ``cap:<id>`` for the capabilities the picks rank, in block order."""
    caps: list[str] = []
    for p in block["picks"]:
        if p["capability"] not in caps:
            caps.append(p["capability"])
    return [m["id"] for m in block["models"]] + [f"cap:{c}" for c in caps]


def _fit(block: dict) -> None:
    """Drop whole entries until the block fits MAX_BLOCK_BYTES.

    Read alongside :func:`_hit_ids`, which names what survives each pass.

    Models first, then picks, and ``capabilities_available`` last: it is the
    only thing that tells an agent which search terms reach a ranked table, so
    it outlives the answers to the current query.
    """
    while True:
        block["hit_ids"] = _hit_ids(block)
        if _block_bytes(block) <= MAX_BLOCK_BYTES:
            return
        if block["models"]:
            block["models"].pop()
        elif block["picks"]:
            block["picks"].pop()
        elif "capabilities_available" in block:
            del block["capabilities_available"]
        else:
            return


def _capabilities_for(bundle: Bundle, entries: list[dict]) -> list[str]:
    """Capability ids ranking any entry the local catalog cannot resolve, in block order."""
    out: list[str] = []
    for entry in entries:
        if entry.get("available_locally") is not False:
            continue
        for cid in bundle.model_capabilities.get(entry["id"], ()):
            if cid not in out:
                out.append(cid)
    return out


def _set_nudge(block: dict, query: str, *, shed_vocabulary: bool = False) -> None:
    """One line naming the miss and pointing at the ids the bundle does cover.

    The pointer goes first if the block is over the cap, then the whole nudge —
    a block that ships without its nudge still beats one nothing reads.
    ``shed_vocabulary`` drops ``capabilities_available`` ahead of the nudge, for
    an otherwise empty block where the nudge is the only thing left to read.
    """
    head = f"no curated knowledge for {query!r}"
    block["nudge"] = _nudge_text(block, head)
    if _block_bytes(block) > MAX_BLOCK_BYTES:
        block["nudge"] = head
    if shed_vocabulary and _block_bytes(block) > MAX_BLOCK_BYTES:
        block.pop("capabilities_available", None)
    if _block_bytes(block) > MAX_BLOCK_BYTES:
        del block["nudge"]


def log_query(
    command: str,
    queries: list[str],
    *,
    hit_ids: list,
    zero_hit: bool,
    bundle: Bundle,
    uncurated: list[str] | None = None,
) -> None:
    """Feed the curation miss log. Consent-gated and best-effort, like every other event.

    This is the one place a search term ships verbatim, clipped to
    ``MAX_QUERY_CHARS``: what people asked for that the bundle does not cover is
    the whole reason the event exists. The generic ``track_command`` kwarg dump
    gets no search term, which is why ``capability`` sits in its redaction set.

    ``hit_ids`` names capabilities ``cap:<id>`` and models by bare id, the same
    way :func:`attach` fills the block's own ``hit_ids``.

    ``uncurated`` is the subset of ``queries`` the bundle keys nothing for, the
    same list :func:`attach` puts on the block. The verbs that resolve a single
    term omit it rather than send it empty, since ``hit_ids`` already says so.
    """
    try:
        from comfy_cli import tracking

        props = {
            "command": command,
            "queries": queries,
            "hit_ids": hit_ids,
            "zero_hit": zero_hit,
            "bundle_version": bundle.version,
            "compiled_at": bundle.compiled_at,
        }
        if uncurated is not None:
            props["uncurated_queries"] = uncurated
        tracking.track_event("knowledge_query", props)
    except Exception:  # noqa: BLE001 — telemetry never decides whether a payload ships
        return


def attach(
    payload: dict,
    *,
    command: str = "",
    queries: Iterable[str] = (),
    models: Iterable[str] = (),
    templates: Iterable[str] = (),
    nodes: Iterable[str] = (),
    catalog_templates: Collection[str] | None = None,
    catalog_nodes: Collection[str] | None = None,
    brief: bool = False,
    thin: bool = False,
    qualified: bool = True,
) -> None:
    """Append a capped ``knowledge`` block to ``payload`` for what the command was asked about.

    ``queries`` is the subject the caller was asked about, resolved as model
    aliases or capability names. ``models`` are aliases the command listed on its
    own, resolved the same way but never treated as the subject, so a listing's
    rows cannot satisfy the nudge or enter the miss log as a typed query.
    ``templates`` and ``nodes`` go through the reverse index. ``catalog_*`` are the ids the
    command actually loaded; a row or pick they do not resolve is marked
    ``available_locally: false`` rather than hidden, and a row marked that way
    pulls in the ranked picks for its capabilities so the caller still sees
    something runnable. ``thin`` marks a command
    whose own result was empty, which is the only case that earns ``zero_hit``;
    a query that resolved to nothing earns a nudge on its own.

    Three fields carry the miss, and every one of them also reaches
    :func:`log_query`, so a consumer holding only the envelope recovers what the
    consent-gated event was told. ``uncurated_queries`` names the terms the
    bundle keys nothing for, on any block, including one whose rows arrived
    through the reverse index while the typed term matched nothing. ``hit_ids:
    []`` says nothing curated came back at all. ``zero_hit`` says the command's
    own result was empty too, which separates a gap the caller felt from one it
    never noticed.

    A query nothing curated answers therefore still attaches, carrying no rows.
    A rowless block naming ids in ``hit_ids`` and no ``uncurated_queries`` is
    knowledge that exists and did not fit, shed by the byte ceiling, rather than
    a gap in the bundle.

    ``qualified=False`` says the call named no subject — an unfiltered listing,
    whose rows are the whole catalog rather than an answer to anything. Nothing
    is attached there: a curated row picked out of 3655 listed nodes reads as the
    answer to a question nobody asked. Fail-open: any exception leaves ``payload``
    exactly as it was.

    ``COMFY_KNOWLEDGE_DISABLE`` suppresses the block entirely. A cached bundle
    keeps being read once it exists, stale or not, and the default URL means a
    signed-in install fetches one without being asked, so this is the only off
    switch. Following ``DO_NOT_TRACK``, any value but empty or ``"0"`` disables.
    """
    try:
        disable = os.environ.get(ENV_DISABLE, "")
        if disable and disable != "0":
            return
        if not qualified:
            return
        bundle = load_bundle(cache_only=True)
        if bundle is None:
            return
        query_list = [q.strip()[:MAX_QUERY_CHARS] for q in queries if isinstance(q, str) and q.strip()]
        model_hits, cap_hits = _lookup(bundle, query_list)
        # Per term, because one term landing says nothing about the ones beside
        # it: `templates ls --tag lipsync --name-sub FLF2V` resolves the tag and
        # misses the name, and a curator ranking gaps needs to see FLF2V.
        uncurated = [q for q in query_list if _lookup(bundle, [q]) == ([], [])]
        query_resolved = bool(model_hits or cap_hits)
        listed_hits, listed_caps = _lookup(bundle, [m for m in models if isinstance(m, str) and m.strip()])
        seen = {mid for mid, _ in model_hits}
        for mid, matched_on in listed_hits:
            if mid not in seen:
                seen.add(mid)
                model_hits.append((mid, matched_on))
        cap_hits.extend(cid for cid in listed_caps if cid not in cap_hits)
        reverse_answers: set[str] = set()
        for ids, index in ((templates, bundle.templates), (nodes, bundle.nodes)):
            for ident in ids:
                mids = index.get(ident, ())
                if mids:
                    reverse_answers.add(_normalize(ident))
                for mid in mids:
                    if mid not in seen:
                        seen.add(mid)
                        model_hits.append((mid, ident))

        entries: list[dict] = []
        for mid, matched_on in model_hits:
            row = bundle.models.get(mid)
            if not isinstance(row, dict):
                continue
            entry = _model_entry(bundle, mid, row, matched_on=matched_on, brief=brief)
            if not _resolves_locally(row, catalog_templates=catalog_templates, catalog_nodes=catalog_nodes):
                entry["available_locally"] = False
                entry["unavailable_reason"] = UNAVAILABLE_LOCALLY
            entries.append(entry)
        entries.sort(key=lambda e: (e.get("tier") != "law", e.get("available_locally") is False))
        entries = entries[: MAX_MODELS_BRIEF if brief else MAX_MODELS]
        if uncurated:
            # A node class or template id passed as both a query and a reverse-index
            # key is a term the bundle does cover. Read before the cap and the
            # model-id dedupe, because a term the bundle answers is not a gap
            # whether or not its row won a place in the block. Normalized on both
            # sides, since node search matches a class name loosely and the term
            # here is whatever the caller typed.
            uncurated = [q for q in uncurated if _normalize(q) not in reverse_answers]
        query_resolved = len(uncurated) < len(query_list)

        picks: list[dict] = []
        for cid in cap_hits:
            picks.extend(_pick_entries(bundle, cid, catalog_templates=catalog_templates))
        if not picks:
            # A row matched by model name rather than by capability, which this
            # install cannot run, would otherwise ship as a dead end: curated,
            # unavailable, and next to nothing the caller could reach for. Its
            # capabilities carry the ranked answer, so borrow them.
            for cid in _capabilities_for(bundle, entries):
                picks.extend(_pick_entries(bundle, cid, catalog_templates=catalog_templates))
        picks = picks[:MAX_PICKS]

        block: dict[str, Any] = {
            "bundle_version": bundle.version,
            "stale": bundle.stale,
            "as_of": bundle.as_of,
            "compiled_at": bundle.compiled_at,
            "models": entries,
            "picks": picks,
            "capabilities_available": sorted(bundle.capabilities),
            "hit_ids": [],
            "zero_hit": False,
        }
        had_hits = bool(entries or picks)
        if uncurated:
            # Inside the block before _fit, so the ceiling sheds a curated row
            # ahead of the record of the gap. A row runs to hundreds of bytes and
            # answers one query; this is every term that missed, and the event it
            # has to agree with sheds nothing.
            block["uncurated_queries"] = uncurated
        matched_ids = _hit_ids(block)
        _fit(block)
        if _block_bytes(block) > MAX_BLOCK_BYTES:
            # Terms long enough to overrun on their own, after _fit shed all it
            # can. The record is the last thing left to give up.
            block.pop("uncurated_queries", None)
        attached = True
        if not block["models"] and not block["picks"]:
            if had_hits or not (thin and query_list):
                attached = False
            else:
                block["zero_hit"] = True
                _set_nudge(block, query_list[0], shed_vocabulary=True)
        elif query_list and not query_resolved:
            # A browse that returns rows the reverse index enriched, while the
            # query itself matched neither a model nor a capability. Not a
            # zero_hit: that feeds the miss log and means an empty block.
            _set_nudge(block, query_list[0])
        if not attached and query_list:
            # The miss reaches log_query, and a consumer with no telemetry
            # client of its own has only the envelope to read it from. Carrying
            # no rows is what keeps an empty block off the caller's context.
            marker = {
                "bundle_version": bundle.version,
                "stale": bundle.stale,
                "as_of": bundle.as_of,
                "compiled_at": bundle.compiled_at,
                # What matched before _fit, not what survived it. Rows shed to
                # make the ceiling are curated knowledge that exists, and filing
                # them as a gap would put a covered term in the curation inbox.
                "hit_ids": matched_ids,
                "zero_hit": block["zero_hit"],
            }
            if uncurated:
                marker["uncurated_queries"] = uncurated
            if _block_bytes(marker) <= MAX_BLOCK_BYTES:
                block = marker
                attached = True
        if query_list:
            log_query(
                command,
                query_list,
                hit_ids=block.get("hit_ids", []),
                zero_hit=block.get("zero_hit", False),
                uncurated=uncurated or None,
                bundle=bundle,
            )
        if attached:
            payload["knowledge"] = block
    except Exception:  # noqa: BLE001 — knowledge is additive; the payload ships unchanged on any failure
        return


def refresh_if_stale() -> None:
    """Re-fetch the bundle when the cache has expired.

    For warm paths only — commands already doing slower work, or holding a
    process open long enough for a fetch to land. :func:`attach` never calls
    this: a discovery turn must not wait on the network.
    """
    try:
        if os.environ.get(ENV_FILE, "").strip():
            return
        knowledge_path, _ = cache_paths()
        if _cache_is_fresh(knowledge_path):
            return
        load_bundle(force_fetch=True)
    except Exception:  # noqa: BLE001 — a refresh that fails leaves the cache exactly as it was
        return
