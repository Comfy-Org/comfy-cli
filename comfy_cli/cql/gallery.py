"""Pure-Python CQL gallery-search engine.

The companion to :mod:`comfy_cli.cql.engine` (the node-graph engine over
``object_info``). Where that engine answers *node* questions, this one answers
*workflow-template gallery* questions: it loads the curated gallery index from
``Comfy-Org/workflow_templates``, flattens the nested (category → templates)
shape into queryable rows, and evaluates the gallery-search predicates.

Port of ``github.com/Comfy-Org/cql/nodegraph`` gallery_search predicates.

This module is pure value-in, value-out: it loads/flattens/filters and returns
plain dicts. It does no rendering and knows nothing about Typer or error codes —
the CLI shell in ``command/templates.py`` wraps it (exactly as ``command/nodes``
wraps the node engine).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

GALLERY_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/index.json"

# Auto-refresh cadence for the implicit load. `templates refresh` / `--refresh`
# always force a fetch regardless. Mirrors comfy_cli.cql.annotations_source so
# both public-repo data sources stay fresh without a `pip install -U`.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

# Where each template's workflow JSON lives. The gallery index lists each
# template by ``name``; the workflow is at ``templates/<name>.json``.
TEMPLATE_WORKFLOW_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}.json"


class GalleryError(RuntimeError):
    """A gallery fetch/parse failed (HTTP error, bad JSON, etc.)."""


# ---------------------------------------------------------------------------
# Loading + caching
# ---------------------------------------------------------------------------


def cache_path() -> Path:
    """Where the gallery index lives on disk. XDG-respecting."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "gallery" / "index.json"


def fetch_gallery(url: str = GALLERY_URL, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        if resp.status != 200:
            raise GalleryError(f"gallery fetch failed: HTTP {resp.status}")
        return resp.read()


def _is_fresh(path: Path) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS
    except OSError:
        return False


def load_gallery(explicit_path: str | None, *, refresh: bool = False) -> list[dict[str, Any]]:
    """Resolve the gallery index. Precedence: explicit path > fresh cache > fetch.

    Resolution (when no explicit path): a TTL-fresh cache wins outright; an
    expired/missing cache triggers a fetch (cached on success); and if that
    fetch fails the stale cache is reused so the command degrades gracefully
    offline instead of erroring. ``refresh=True`` always forces a fetch.

    Returns the raw decoded JSON (a list of category dicts). Filtering is a
    separate step (:func:`flatten_templates` + :func:`matches`).
    """
    if explicit_path:
        return json.loads(Path(explicit_path).read_bytes())

    cache = cache_path()
    if not refresh and cache.is_file() and _is_fresh(cache):
        return json.loads(cache.read_bytes())

    try:
        data = fetch_gallery()
    except (GalleryError, OSError) as e:
        # Fetch failed — fall back to whatever cache we have (even if stale).
        if cache.is_file():
            return json.loads(cache.read_bytes())
        raise GalleryError(f"gallery unavailable and no cache present: {e}") from e

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return json.loads(data)


def fetch_template_workflow(name: str, *, timeout: float = 15.0) -> bytes:
    """Pull a single template's workflow JSON from the canonical GitHub raw URL."""
    url = TEMPLATE_WORKFLOW_URL.format(name=urllib.parse.quote(name, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https host
        if resp.status != 200:
            raise GalleryError(f"template workflow fetch failed: HTTP {resp.status}")
        return resp.read()


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def flatten_templates(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk the nested (category → templates) shape and flatten to a list.

    Each row gets a few extras: ``category_title``, ``group_category``, and
    ``output_type`` (from the parent category's ``type`` — the per-template
    ``mediaType`` is actually the thumbnail format and is misleading).
    Providers from ``logos[].provider`` are flattened to a flat string list
    that tolerates the scalar-or-array variance in real data.
    """
    rows: list[dict[str, Any]] = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        output_type = cat.get("type") or ""
        for t in cat.get("templates", []) or []:
            if not isinstance(t, dict):
                continue
            rows.append(
                {
                    "name": t.get("name") or "",
                    "title": (t.get("title") or "").strip(),
                    "description": t.get("description") or "",
                    "output_type": output_type,
                    "category_title": cat.get("title") or "",
                    "group_category": cat.get("category") or "",
                    "tags": list(t.get("tags") or []),
                    "models": list(t.get("models") or []),
                    "providers": flatten_providers(t.get("logos") or []),
                    "date": t.get("date") or "",
                    "open_source": bool(t.get("openSource", False)),
                    "usage": int(t.get("usage") or 0),
                    "media_subtype": t.get("mediaSubtype") or "",
                    "io": t.get("io") or {},
                }
            )
    return rows


def flatten_providers(logos: list[Any]) -> list[str]:
    """``logos[].provider`` may be a string or a list-of-strings. Coalesce."""
    out: list[str] = []
    seen: set[str] = set()
    for logo in logos:
        if not isinstance(logo, dict):
            continue
        prov = logo.get("provider")
        if isinstance(prov, str):
            if prov and prov not in seen:
                seen.add(prov)
                out.append(prov)
        elif isinstance(prov, list):
            for p in prov:
                if isinstance(p, str) and p and p not in seen:
                    seen.add(p)
                    out.append(p)
    return out


# ---------------------------------------------------------------------------
# Predicates — port of gallery_search.go
# ---------------------------------------------------------------------------


def matches(
    row: dict[str, Any],
    *,
    type_: str | None = None,
    category: str | None = None,
    tag: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    name_sub: str | None = None,
) -> bool:
    if type_ and (row.get("output_type") or "").lower() != type_.lower():
        return False
    if category and (row.get("category_title") or "").lower() != category.lower():
        return False
    if tag and not any((t or "").lower() == tag.lower() for t in row.get("tags") or []):
        return False
    if model and not any(model.lower() in (m or "").lower() for m in row.get("models") or []):
        return False
    if provider and not any(provider.lower() in (p or "").lower() for p in row.get("providers") or []):
        return False
    if name_sub and name_sub.lower() not in (row.get("name") or "").lower():
        return False
    return True


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    type_: str | None = None,
    category: str | None = None,
    tag: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    name_sub: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the gallery-search predicates to a flattened row list."""
    return [
        r
        for r in rows
        if matches(
            r,
            type_=type_,
            category=category,
            tag=tag,
            model=model,
            provider=provider,
            name_sub=name_sub,
        )
    ]
