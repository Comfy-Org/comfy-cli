"""``comfy templates`` — workflow-template gallery introspection.

Mirrors the shape of ``comfy nodes`` but queries the curated
**workflow-template gallery** from ``Comfy-Org/workflow_templates``
(the same content that drives comfy.org/workflows). Three primitives:

    comfy templates ls   [--type T] [--category PAT] [--tag T] [--model M]
                         [--provider P] [--name SUB] [--limit N]
    comfy templates show <name>
    comfy templates refresh                            # re-fetch index.json

The gallery file ``templates/index.json`` is cached under
``~/.cache/comfy-cli/gallery/index.json``. The CLI side here parses the
index in Python (no WASM needed); for the full CQL grammar over templates
use the flag-based filters for browsing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.file_utils import atomic_write_bytes
from comfy_cli.http import ResponseTooLarge, plain_urlopen, read_capped
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Browse the Comfy workflow-template gallery.")

GALLERY_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/index.json"

# How long a cached gallery index stays fresh before ``_load_gallery`` transparently
# re-fetches it. 24h (not the spec's 7 days): the gallery updates weekly-ish and the
# fetch is one small JSON file, so a tighter TTL keeps agents off a frozen catalog
# cheaply. A network-down machine still lists from the stale cache (fetch failure
# falls back), and ``comfy templates refresh`` remains the manual force-refresh.
GALLERY_TTL_SECONDS = 24 * 60 * 60

# Everything a gallery load can throw. ``_fetch_gallery`` raises ``RuntimeError``
# on a non-200 status (which ``urlopen`` doesn't already turn into an
# ``HTTPError``), the fetch itself raises ``URLError``/``OSError``, decoding a
# 200-with-garbage body raises ``JSONDecodeError``, and an over-cap body raises
# ``ResponseTooLarge`` — all of which must route through the same stale-cache
# fallback / command-level error, never an uncaught traceback.
_GALLERY_LOAD_ERRORS = (
    urllib.error.URLError,
    OSError,
    RuntimeError,
    json.JSONDecodeError,
    ResponseTooLarge,
)


# ---------------------------------------------------------------------------
# Gallery loading + caching
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    """Where the gallery index lives on disk. XDG-respecting."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "gallery" / "index.json"


def _fetch_gallery(url: str = GALLERY_URL, timeout: float = 15.0) -> bytes:
    """Pull the gallery index from GitHub raw. Bounded read — this is a body
    from a host we don't control, so it goes through the shared cap rather than
    letting the remote decide how much memory we buffer."""
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with plain_urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"gallery fetch failed: HTTP {resp.status}")
        return read_capped(resp, url)


def _parse_gallery(data: bytes) -> list[Any]:
    """Decode a gallery index body, rejecting anything that isn't a category list.

    ``json.loads`` alone accepts ``"null"``, ``"7"`` and ``{"error": …}`` — all
    valid JSON, none of them a gallery. ``_flatten_templates`` would then walk a
    non-list and raise something unrelated to the actual problem, or silently
    produce zero rows. Checking the shape here means every caller's
    ``_GALLERY_LOAD_ERRORS`` handler catches it as the load failure it is;
    ``RuntimeError`` is used because that tuple already routes it.
    """
    parsed = json.loads(data)
    if not isinstance(parsed, list):
        raise RuntimeError(f"gallery index is not a list of categories (got {type(parsed).__name__})")
    return parsed


def _load_gallery(
    explicit_path: str | None,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the gallery index. Precedence: explicit --gallery > cache > fetch.

    Returns the raw decoded JSON (a list of category dicts). The CLI does
    its own filtering on top.

    A cache older than ``GALLERY_TTL_SECONDS`` is transparently re-fetched so
    ``templates ls/show`` never serves a frozen catalog forever. If that refresh
    fails (offline / GitHub down), we fall back to the stale cache with a
    non-fatal renderer warning rather than erroring out.
    """
    if explicit_path:
        return _parse_gallery(Path(explicit_path).read_bytes())

    cache = _cache_path()
    have_cache = cache.exists()

    if not refresh and have_cache and not _cache_is_stale(cache):
        try:
            return _parse_gallery(cache.read_bytes())
        except _GALLERY_LOAD_ERRORS:
            # A cache poisoned by an older build (which wrote before validating)
            # is treated as a miss rather than raising for the rest of the TTL —
            # fall through and re-fetch so the CLI self-heals.
            have_cache = False

    # A TTL-expired cache is refreshed transparently, so a fetch failure here
    # must NOT break `templates ls` — fall back to the stale cache with a
    # non-fatal warning. An explicit `--refresh` (or a genuinely absent cache),
    # by contrast, surfaces the fetch error so the user learns it failed.
    ttl_auto_refresh = have_cache and not refresh
    try:
        data = _fetch_gallery()
        # Validate BEFORE we touch the cache: a 200 with a non-JSON body
        # (rate-limit HTML, captive portal, truncated response) must never
        # clobber the last-known-good cache with garbage.
        parsed = _parse_gallery(data)
    except _GALLERY_LOAD_ERRORS as e:
        if ttl_auto_refresh:
            # The stale cache is our fallback — but a concurrent `refresh` may
            # have removed it or left it corrupt mid-write. If reading it back
            # also fails, surface the original fetch error, not the read error.
            try:
                stale = _parse_gallery(cache.read_bytes())
            except _GALLERY_LOAD_ERRORS:
                raise e
            get_renderer().warn(
                f"gallery refresh failed ({e}); using cached index (last updated {_cache_age_str(cache)} ago)",
                hint="run `comfy templates refresh` once back online to update it",
            )
            return stale
        raise
    _persist_cache(cache, data)
    return parsed


def _persist_cache(cache: Path, data: bytes) -> str | None:
    """Persist a freshly fetched index to the cache, atomically and best-effort.

    * Atomic — write to a temp file in the same directory then ``os.replace``
      it into place, so a concurrent ``templates`` reader never observes a
      half-written index (which would parse-fail as ``gallery_load_failed``).
    * Best-effort — a read-only cache dir (e.g. a gallery baked into a
      container image) or a full disk must not break the command once we
      already hold valid data, so a write failure is returned rather than
      raised. ``_load_gallery`` ignores it (it has the data); ``templates
      refresh`` reports it, because caching is the whole job of that command.

    Returns ``None`` on success, or the error text when the write failed.
    """
    try:
        # Tier 4 (regenerable cache) per the write policy in
        # comfy_cli/file_utils.py — fsync=False, wrapped best-effort below.
        # The previous inline `tempfile.mkstemp` left the cache at mode 0600
        # (mkstemp hardcodes that and `os.replace` carries it onto the
        # destination); the helper instead reuses the existing cache file's
        # mode when there is one, else falls back to the umask-derived default
        # for a fresh file — either way intentionally relaxed from 0600: the
        # payload is the public template-gallery index, not user data.
        atomic_write_bytes(cache, data, fsync=False)
    except OSError as e:
        # Couldn't persist (read-only dir, disk full, …). We still have valid
        # data in hand, so proceed without caching rather than failing the run.
        return str(e)
    return None


def _cache_is_stale(cache: Path) -> bool:
    """True when the cache file is older than ``GALLERY_TTL_SECONDS``."""
    try:
        age = time.time() - cache.stat().st_mtime
    except OSError:
        # Can't stat it → treat as stale so we attempt a refresh.
        return True
    # A future mtime (clock skew, or a restored/tampered file) yields a
    # negative age; treat it as stale so the cache can't be pinned "fresh"
    # indefinitely until wall-clock time catches up.
    return age < 0 or age > GALLERY_TTL_SECONDS


def _cache_age_str(cache: Path) -> str:
    """Human-friendly age of the cache file for the stale-fallback warning."""
    try:
        age = max(0.0, time.time() - cache.stat().st_mtime)
    except OSError:
        return "unknown time"
    hours = age / 3600.0
    if hours >= 24:
        return f"{hours / 24:.1f}d"
    if hours >= 1:
        return f"{hours:.1f}h"
    return f"{age / 60:.0f}m"


def _as_str_list(value: Any) -> list[str]:
    """Coerce a gallery field to a list of strings, tolerating the scalar-or-junk
    variance in real data. A bare string is treated as a single-element list (not
    split into characters); a non-list/non-string scalar becomes an empty list.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _flatten_templates(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "requires_custom_nodes": _as_str_list(t.get("requiresCustomNodes")),
                    "providers": _flatten_providers(t.get("logos") or []),
                    "date": t.get("date") or "",
                    "open_source": bool(t.get("openSource", False)),
                    "usage": int(t.get("usage") or 0),
                    "media_subtype": t.get("mediaSubtype") or "",
                    "io": t.get("io") or {},
                }
            )
    return rows


def _flatten_providers(logos: list[Any]) -> list[str]:
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
# Filters — Python equivalents of nodegraph/gallery_search.go predicates
# ---------------------------------------------------------------------------


def _matches(
    row: dict[str, Any],
    *,
    type_: str | None,
    category: str | None,
    tag: str | None,
    model: str | None,
    provider: str | None,
    name_sub: str | None,
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


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _ls_via_query(
    renderer,
    query: str,
    gallery_path: str | None,
    refresh: bool,
    limit: int | None,
) -> None:
    """CQL grammar queries over the template gallery are not available.
    Emit an actionable error pointing the user at the flag-based filters instead.
    """
    renderer.error(
        code="cql_query_invalid",
        message="CQL grammar queries are not available. Use flag-based filtering instead.",
        hint="comfy templates ls --type image --tag API --model Flux",
    )
    raise typer.Exit(code=1)


@app.command(
    "ls",
    help="List gallery templates. Filter by type/category/tag/model/provider/name.",
)
@tracking.track_command("templates")
def ls_cmd(
    type_: Annotated[
        str | None,
        typer.Option("--type", help="Output kind: image, video, audio, 3d."),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Exact category title (e.g. 'Image', 'Video')."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Tag (case-insensitive exact match, e.g. 'API')."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name substring (e.g. 'Flux')."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider substring (e.g. 'Kling', 'Black Forest Labs')."),
    ] = None,
    name_sub: Annotated[
        str | None,
        typer.Option("--name", help="Substring match on template name."),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            show_default=False,
            hidden=True,
            help="Removed — there is no CQL grammar over templates. Use the flag filters above.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(show_default=False, help="Cap output to N rows."),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option(
            "--gallery",
            show_default=False,
            help="Path to a local templates/index.json (skips the cache + fetch).",
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch index.json from GitHub before listing."),
    ] = False,
):
    renderer = get_renderer()

    # CQL grammar path — routes through WASM with the gallery loaded.
    if query is not None:
        return _ls_via_query(renderer, query, gallery_path, refresh, limit)

    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except _GALLERY_LOAD_ERRORS as e:
        renderer.error(
            code="gallery_load_failed",
            message=str(e),
            hint="check your network, or pass --gallery <path> to a local index.json",
        )
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    total = len(rows)
    rows = [
        r
        for r in rows
        if _matches(
            r,
            type_=type_,
            category=category,
            tag=tag,
            model=model,
            provider=provider,
            name_sub=name_sub,
        )
    ]
    matched = len(rows)
    if limit is not None:
        rows = rows[: max(0, limit)]

    payload = {
        "total_in_gallery": total,
        "matched": matched,
        "shown": len(rows),
        "filters": {
            "type": type_,
            "category": category,
            "tag": tag,
            "model": model,
            "provider": provider,
            "name": name_sub,
        },
        "rows": [
            {
                "name": r["name"],
                "title": r["title"],
                "output_type": r["output_type"],
                "category_title": r["category_title"],
                "tags": r["tags"],
                "models": r["models"],
                "providers": r["providers"],
                "description": r["description"][:120],
            }
            for r in rows
        ],
    }

    if renderer.is_pretty():
        from rich.table import Table

        if not rows:
            rprint("[dim]0 templates matched.[/dim]")
        else:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("type", style="dim")
            tbl.add_column("title")
            tbl.add_column("tags", style="dim")
            for r in rows:
                tbl.add_row(
                    r["name"],
                    r["output_type"],
                    r["title"] or "(untitled)",
                    ", ".join(r["tags"]),
                )
            renderer.console().print(tbl)
            tail = f" (of {matched} matched, {total} in gallery)" if (matched != len(rows) or matched != total) else ""
            rprint(f"[dim]{len(rows)} template(s){tail}[/dim]")
    renderer.emit(payload, command="templates ls")


@app.command(
    "show",
    help="Show full details for a single template by name.",
)
@tracking.track_command("templates")
def show_cmd(
    name: Annotated[str, typer.Argument(help="Template name (e.g. 'image_flux2').")],
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch from GitHub before showing."),
    ] = False,
):
    renderer = get_renderer()
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except _GALLERY_LOAD_ERRORS as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r}",
            hint="try `comfy templates ls --name <substring>` to search",
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        rprint(f"[bold]{match['name']}[/bold]")
        if match["title"]:
            rprint(f"  [dim]{match['title']}[/dim]")
        rprint(f"  type:        {match['output_type']}")
        rprint(f"  category:    {match['category_title']} ({match['group_category']})")
        if match["tags"]:
            rprint(f"  tags:        {', '.join(match['tags'])}")
        if match["models"]:
            rprint(f"  models:      {', '.join(match['models'])}")
        if match["providers"]:
            rprint(f"  providers:   {', '.join(match['providers'])}")
        if match["date"]:
            rprint(f"  date:        {match['date']}")
        if match["description"]:
            rprint("")
            rprint(match["description"])
    renderer.emit({"template": match}, command="templates show")


@app.command("refresh", help="Re-download templates/index.json into the local cache.")
@tracking.track_command("templates")
def refresh_cmd():
    renderer = get_renderer()
    try:
        data = _fetch_gallery()
        # Validate BEFORE the cache is touched, same rule as `_load_gallery`.
        parsed = _parse_gallery(data)
    except _GALLERY_LOAD_ERRORS as e:
        # Same family `_load_gallery` routes: a non-200 (`RuntimeError`), a
        # non-JSON or wrong-shaped body, and an over-cap body
        # (`ResponseTooLarge`) all surface as this envelope, never a traceback.
        renderer.error(code="gallery_fetch_failed", message=str(e))
        raise typer.Exit(code=1) from e
    cache = _cache_path()
    # Persist exactly the way `_load_gallery` does — validated above, written
    # atomically, and never a bare `write_bytes`. This command used to clobber
    # the cache with whatever came back: a 200 carrying rate-limit HTML was
    # written verbatim, reported success, and then failed every subsequent
    # `templates ls` for the full TTL, with the stale-cache fallback unable to
    # help because the cache *was* the garbage.
    cache_error = _persist_cache(cache, data)
    if cache_error is not None:
        # Unlike the implicit TTL refresh, caching is the entire job here, so a
        # write failure is the command failing — not a detail to swallow.
        renderer.error(
            code="gallery_cache_write_failed",
            message=f"fetched the gallery but could not cache it: {cache_error}",
            hint=f"check permissions and free space on {cache.parent}",
        )
        raise typer.Exit(code=1)
    payload = {"path": str(cache), "bytes": len(data), "categories": len(parsed)}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] cached gallery to {cache} ({len(data)} bytes, {len(parsed)} categories)")
    renderer.emit(payload, command="templates refresh")


# Where the per-template workflow JSONs live on GitHub. The gallery index lists
# each template by ``name``; the corresponding workflow is at
# ``Comfy-Org/workflow_templates/templates/<name>.json``.
_TEMPLATE_WORKFLOW_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}.json"


def _workflow_node_count(workflow: Any) -> int | None:
    """Count the nodes in a workflow, in either serialization format.

    Gallery templates ship in the *frontend* shape — ``{id, revision, nodes,
    links, …}`` — so a plain ``len(workflow)`` counts those wrapper keys and
    reports ~10 for every template regardless of its real size
    (``api_seedance2_0_r2v``, a 3-node workflow, read as 10). The frontend shape
    is identified by its ``nodes`` list; the API shape is a bare
    ``{node_id: {class_type, …}}`` map, where the key count *is* the node count.

    Subgraph interiors are deliberately not counted: this number exists so a
    caller can confirm the workflow it just fetched looks like the workflow it
    asked for, and that's the canvas-level count.
    """
    if not isinstance(workflow, dict):
        return None
    nodes = workflow.get("nodes")
    if isinstance(nodes, list):
        return len(nodes)
    return len(workflow)


def _fetch_template_workflow(name: str, *, timeout: float = 15.0) -> bytes:
    """Pull a single template's workflow JSON from the canonical GitHub raw URL."""
    url = _TEMPLATE_WORKFLOW_URL.format(name=urllib.parse.quote(name, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with plain_urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"template workflow fetch failed: HTTP {resp.status}")
        return read_capped(resp, url)


@app.command(
    "fetch",
    help=(
        "Fetch a template's workflow JSON from the curated gallery. "
        "Verifies the name against the gallery index first, then pulls "
        "templates/<name>.json from Comfy-Org/workflow_templates."
    ),
)
@tracking.track_command("templates")
def fetch_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    out: Annotated[
        str | None,
        typer.Option("--out", "-o", show_default=False, help="Write to this file instead of stdout."),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index from GitHub before resolving."),
    ] = False,
):
    renderer = get_renderer()

    # `--out ""` is not a writable path — treat it as "no file requested" so the
    # write branch and the envelope ride-along agree. Left divergent, an empty
    # string is falsy (nothing written to disk) yet not None (workflow omitted
    # from the envelope), and the fetched workflow vanishes entirely.
    out = out or None

    # Resolve against the gallery index first so we surface "no such template"
    # with the same close_matches affordance the rest of the CLI uses, instead
    # of letting the user hit a raw GitHub 404.
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except _GALLERY_LOAD_ERRORS as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        # Build a small list of close matches so the agent can self-correct.
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    try:
        body = _fetch_template_workflow(name)
    # ``RuntimeError`` (a 2xx-but-not-200 status) and ``ResponseTooLarge`` (an
    # over-cap body) are the same class of upstream misbehaviour as a URLError
    # and belong in the same envelope, not in an uncaught traceback.
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, RuntimeError, ResponseTooLarge) as e:
        status = getattr(e, "code", None)
        renderer.error(
            code="template_fetch_failed",
            message=f"failed to fetch workflow for {name!r}: {e}",
            hint=(
                "the gallery index references a template whose workflow JSON "
                "is missing upstream — report at "
                "https://github.com/Comfy-Org/workflow_templates/issues"
                if status == 404
                else "check network connectivity"
            ),
            details={"status": status} if status else None,
        )
        raise typer.Exit(code=1) from e

    # Parse so we (a) validate it's well-formed JSON and (b) can report the
    # node count in the envelope without re-reading.
    try:
        wf = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"upstream returned non-JSON for {name!r}: {e}",
            hint="report at https://github.com/Comfy-Org/workflow_templates/issues",
        )
        raise typer.Exit(code=1) from e

    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        target_repr = str(out_path)
    else:
        # In JSON mode, the renderer's emit() is the only thing on stdout — the
        # raw workflow goes into the envelope under data.workflow. In pretty
        # mode we print it to stdout so the user can pipe it.
        if renderer.is_pretty():
            import sys

            sys.stdout.write(body.decode("utf-8"))
            sys.stdout.write("\n")
        target_repr = "stdout"

    payload = {
        "name": name,
        "title": match["title"],
        "output_type": match["output_type"],
        "out": target_repr,
        "bytes": len(body),
        "node_count": _workflow_node_count(wf),
    }
    if not out:
        # No file was written, so the JSON envelope is the only place the caller
        # can get the workflow — emit() owns stdout in JSON mode, so without this
        # the fetch would produce nothing but metadata.
        payload["workflow"] = wf
    if renderer.is_pretty() and out:
        rprint(f"[green]✓[/green] wrote {len(body):,} bytes ({payload['node_count']} nodes) to {target_repr}")
    renderer.emit(payload, command="templates fetch")


# ---------------------------------------------------------------------------
# templates check — per-template runnable/missing/api-required/unknown verdict
# ---------------------------------------------------------------------------


def _template_workflow_cache_path(name: str) -> Path:
    """Where a fetched per-template workflow JSON is cached. Same base-dir logic
    as :func:`_cache_path`, one file per template under ``gallery/templates``.

    ``name`` comes from the (untrusted) gallery index, so it's URL-encoded the
    same way :func:`_fetch_template_workflow` encodes it for the fetch URL —
    ``quote(..., safe="")`` turns any ``/`` or ``..`` segment into ``%2F``/``..``
    with no path separators, keeping the file strictly inside ``gallery/templates``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    safe_name = urllib.parse.quote(name, safe="")
    return Path(base) / "comfy-cli" / "gallery" / "templates" / f"{safe_name}.json"


def _iter_workflow_nodes(wf: dict[str, Any]):
    """Yield every node dict in a UI-format workflow: top-level ``nodes`` first,
    then the nodes inside each subgraph *definition*.

    The subgraph walk is mandatory — modern templates (e.g. ``image_z_image_turbo``)
    carry their model references only inside subgraph definitions, so a top-level
    walk alone would report them as having zero model requirements.
    """
    if not isinstance(wf, dict):
        return
    top = wf.get("nodes")
    if isinstance(top, list):
        for node in top:
            if isinstance(node, dict):
                yield node
    definitions = wf.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    if isinstance(subgraphs, list):
        for sg in subgraphs:
            if not isinstance(sg, dict):
                continue
            sg_nodes = sg.get("nodes")
            if isinstance(sg_nodes, list):
                for node in sg_nodes:
                    if isinstance(node, dict):
                        yield node


def _collect_model_requirements(wf: dict[str, Any]) -> list[dict[str, str]]:
    """Gather every ``node["properties"]["models"]`` entry across top-level and
    subgraph nodes. Each entry is normalized to ``{name, directory, url}`` and
    deduped by ``(directory, name)`` preserving first-seen order.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for node in _iter_workflow_nodes(wf):
        props = node.get("properties")
        if not isinstance(props, dict):
            continue
        models = props.get("models")
        if not isinstance(models, list):
            continue
        for m in models:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "")
            directory = str(m.get("directory") or "")
            if not name:
                # A model ref with no filename isn't matchable against a folder
                # listing — keeping it would always report a phantom empty-name
                # "missing" model and wrongly flip the verdict to missing-models.
                continue
            key = (directory, name)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "directory": directory, "url": str(m.get("url") or "")})
    return out


def _collect_node_class_types(wf: dict[str, Any]) -> list[str]:
    """Distinct ``node["type"]`` values across top-level + subgraph nodes.

    For a regular node this is the ComfyUI class name (``CheckpointLoaderSimple``);
    for a subgraph *instance* it's the definition UUID (never loader-ish, never in
    object_info), which is why the loader/api heuristics simply skip those.
    """
    seen: set[str] = set()
    out: list[str] = []
    for node in _iter_workflow_nodes(wf):
        t = node.get("type")
        if isinstance(t, str) and t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _basename(path: str) -> str:
    """Last path segment of a folder-relative listing entry. Model folder
    listings return paths that may include subdirectories (``sdxl/model.safetensors``)
    and always use ``/`` on the wire; also tolerate a stray backslash defensively.
    """
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _list_local_folder(target, folder: str) -> list[str] | None:
    """List file paths in ``/models/<folder>`` on the local server.

    Returns the folder-relative names, or ``None`` if the folder 404s (a
    custom-node folder like ``SEEDVR2`` that ComfyUI doesn't register). Connection
    errors (server not running) and other HTTP errors propagate to the caller.
    """
    # Reuse the exact target/URL plumbing `comfy models list-folder` uses.
    from comfy_cli.command.models.search import _http_get_json, _models_path_parts

    url = target.url(*_models_path_parts(target), folder)
    try:
        data = _http_get_json(url, target)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    names: list[str] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                n = entry.get("name", "")
            elif isinstance(entry, str):
                n = entry
            else:
                n = ""
            if n:
                names.append(n)
    return names


def _template_is_api_by_index(name: str, tags: list[str]) -> bool:
    """Index-only API heuristic: an ``api_*`` template name, or an ``API`` tag."""
    if name.startswith("api_"):
        return True
    return any(str(t).strip().lower() == "api" for t in tags)


def _compute_verdict(*, api_dependent: bool, missing: list, required_count: int, node_types: list[str]) -> str:
    """Verdict precedence: api-required > missing-models > runnable > unknown."""
    if api_dependent:
        return "api-required"
    if missing:
        return "missing-models"
    if required_count > 0:
        # Every referenced model resolved on disk.
        return "runnable"
    # Zero model references: runnable only if nothing looks like it *should* load
    # a model; otherwise we genuinely can't tell (loader with no declared models).
    loaderish = any("loader" in (t or "").lower() for t in node_types)
    return "unknown" if loaderish else "runnable"


@app.command(
    "check",
    help=(
        "Report whether a gallery template is runnable on THIS install: which of "
        "its models are present vs missing locally, whether it needs partner-API "
        "access, and any custom nodes it declares. Resolves the name against the "
        "gallery, fetches the workflow, and intersects its model refs with the "
        "local server's model folders."
    ),
)
@tracking.track_command("templates")
def check_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index AND the template workflow before checking."),
    ] = False,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()

    # 1. Resolve the name against the gallery index (same affordance as `fetch`).
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    # 2. Fetch (or read from cache) the per-template workflow JSON.
    cache_path = _template_workflow_cache_path(name)
    body: bytes | None = None
    if not refresh and cache_path.exists():
        try:
            body = cache_path.read_bytes()
        except OSError:
            body = None
    if body is None:
        try:
            body = _fetch_template_workflow(name)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            status = getattr(e, "code", None)
            renderer.error(
                code="template_fetch_failed",
                message=f"failed to fetch workflow for {name!r}: {e}",
                hint=(
                    "the gallery index references a template whose workflow JSON "
                    "is missing upstream — report at "
                    "https://github.com/Comfy-Org/workflow_templates/issues"
                    if status == 404
                    else "check network connectivity"
                ),
                details={"status": status} if status else None,
            )
            raise typer.Exit(code=1) from e
        # Write atomically: a truncated file (interrupted write / full disk) would
        # otherwise be trusted by the read path above on the next non-refresh run
        # and fail `template_workflow_invalid_json` until the user passed --refresh.
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.parent / f"{cache_path.name}.{os.getpid()}.tmp"
            try:
                tmp_path.write_bytes(body)
                os.replace(tmp_path, cache_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)
                raise
        except OSError:
            pass  # a non-writable cache dir must not fail the check

    try:
        wf = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"template workflow for {name!r} is not valid JSON: {e}",
            hint="re-run with --refresh to re-fetch, or report upstream",
        )
        raise typer.Exit(code=1) from e
    if not isinstance(wf, dict):
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"template workflow for {name!r} is not a JSON object",
            hint="re-run with --refresh to re-fetch, or report upstream",
        )
        raise typer.Exit(code=1)

    # 3. Model requirements (top-level + subgraph walk) and node class types.
    required = _collect_model_requirements(wf)
    node_types = _collect_node_class_types(wf)

    # 4. API dependence. Tier (a) is the index heuristic; tier (b) upgrades it with
    #    object_info when a local server is reachable (best-effort — a down server
    #    just leaves the index answer in place).
    api_dependent = _template_is_api_by_index(name, match.get("tags") or [])
    api_source = "index"
    api_nodes: list[str] = []
    try:
        from comfy_cli.cql.engine import Graph

        graph = Graph.load(mode="local")
        api_nodes = [cls for cls in node_types if (m := graph.node(cls)) is not None and m.is_api_node]
        # Only attribute the signal to object_info when it actually found API nodes;
        # an empty scan must leave `source` reflecting the index heuristic, not claim
        # object_info as the source of an api-required verdict it didn't produce.
        if api_nodes:
            api_source = "object_info"
            api_dependent = True
    except Exception:
        # Server down / object_info unavailable → index heuristic alone decides.
        # Discard any partial object_info result so `source` and `api_nodes` agree.
        api_nodes = []
        api_source = "index"

    # 5. Installed intersection: list each distinct folder once on the local server
    #    and match required files by basename.
    warnings: list[str] = []
    present: list[str] = []
    missing: list[dict[str, str]] = []
    distinct_dirs = list(dict.fromkeys(req["directory"] for req in required))
    if required:
        target = resolve_target(where="local")
        listings: dict[str, list[str] | None] = {}
        try:
            for directory in distinct_dirs:
                if not directory or ".." in directory or "/" in directory or "\\" in directory:
                    # Not addressable as a `/models/<folder>` segment — treat as absent.
                    listings[directory] = None
                    warnings.append(
                        f"model directory {directory!r} isn't a valid model folder — its files are reported missing"
                    )
                    continue
                folder_files = _list_local_folder(target, directory)
                listings[directory] = folder_files
                if folder_files is None:
                    warnings.append(
                        f"model folder {directory!r} not found on the local server "
                        f"(custom-node folder?) — its files are reported missing"
                    )
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
            renderer.error(
                code="server_not_running",
                message=f"local ComfyUI server is unreachable, cannot check installed models: {e}",
                hint="run `comfy launch` to start a local server",
            )
            raise typer.Exit(code=1) from e

        for req in required:
            listing = listings.get(req["directory"])
            # Normalize BOTH sides: a model ref may itself carry a subfolder
            # (e.g. ``SDXL/model.safetensors``, as ComfyUI loader widgets emit),
            # so compare basenames on the required side too.
            req_base = _basename(req["name"])
            if listing and any(_basename(entry) == req_base for entry in listing):
                present.append(req["name"])
            else:
                missing.append(dict(req))

    # 6. Custom nodes are report-only in v1 (surfaced verbatim, not verified).
    custom_nodes_required = list(match.get("requires_custom_nodes") or [])

    # 7. Verdict.
    verdict = _compute_verdict(
        api_dependent=api_dependent,
        missing=missing,
        required_count=len(required),
        node_types=node_types,
    )

    # 8. Emit.
    payload = {
        "name": name,
        "title": match["title"],
        "verdict": verdict,
        "models": {"required": len(required), "present": present, "missing": missing},
        "api": {"dependent": api_dependent, "source": api_source, "api_nodes": api_nodes},
        "custom_nodes_required": custom_nodes_required,
        "warnings": warnings,
    }

    if renderer.is_pretty():
        # `name`, node-class names, model names/urls and warnings are all untrusted
        # (they come from the gallery index / workflow JSON), so escape them before
        # Rich interprets them — a stray `[foo]` would otherwise raise MarkupError.
        from rich.markup import escape

        mark = "[bold green]✓[/bold green]" if verdict == "runnable" else "[bold red]✗[/bold red]"
        rprint(f"{mark} {escape(name)} — [bold]{verdict}[/bold]")
        if api_dependent:
            extra = f": {escape(', '.join(api_nodes))}" if api_nodes else ""
            rprint(f"  [yellow]needs partner-API access[/yellow] (via {api_source}){extra}")
        if custom_nodes_required:
            rprint(f"  custom nodes: {escape(', '.join(custom_nodes_required))} [dim](not verified)[/dim]")
        if missing:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("missing model")
            tbl.add_column("directory", style="dim")
            tbl.add_column("download url")
            for m in missing:
                tbl.add_row(escape(m["name"]), escape(m["directory"]), escape(m["url"]) or "[dim](none)[/dim]")
            renderer.console().print(tbl)
        elif required:
            rprint(f"  [dim]{len(present)}/{len(required)} model(s) present[/dim]")
        for w in warnings:
            rprint(f"  [yellow]⚠[/yellow] {escape(w)}")
    renderer.emit(payload, command="templates check")


# ---------------------------------------------------------------------------
# run-template — fetch → fill params → spend-gate → run via the run path
# ---------------------------------------------------------------------------


def _parse_param_value(raw: str) -> Any:
    """Parse a ``--param`` value as JSON; fall back to the literal string.

    Mirrors ``comfy workflow set-slot`` semantics so `--param seed=42` writes
    an int and `--param prompt="a cat"` writes a string.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _workflow_node_types(workflow: Any) -> set[str]:
    """Collect node class names from a workflow in either format.

    Frontend format: ``nodes[].type`` plus every ``definitions.subgraphs[].nodes[].type``
    (gallery templates routinely hide partner nodes inside UUID subgraphs).
    API format: ``values()[].class_type``.
    """
    types: set[str] = set()
    if not isinstance(workflow, dict):
        return types
    if isinstance(workflow.get("nodes"), list):
        node_lists = [workflow.get("nodes") or []]
        subgraphs = (workflow.get("definitions") or {}).get("subgraphs") or []
        for sg in subgraphs:
            if isinstance(sg, dict):
                node_lists.append(sg.get("nodes") or [])
        for nodes in node_lists:
            for node in nodes:
                if isinstance(node, dict) and isinstance(node.get("type"), str):
                    types.add(node["type"])
        return types
    for node in workflow.values():
        if isinstance(node, dict) and isinstance(node.get("class_type"), str):
            types.add(node["class_type"])
    return types


def _detect_paid_nodes(workflow: Any, object_info: dict) -> list[str]:
    """Sorted node class names in ``workflow`` that are partner-API (paid) nodes.

    Same signals as ``comfy_cli.command.run``'s partner detection — the
    authoritative ``api_node: true`` flag with a ``partner/`` category-prefix
    fallback — but format-agnostic so it works on the frontend-format JSON
    gallery templates ship as (run's detector only reads API format).
    """
    from comfy_cli.command.run import PARTNER_NODE_CATEGORY_PREFIXES

    out: list[str] = []
    for ct in _workflow_node_types(workflow):
        info = object_info.get(ct) or {}
        if not isinstance(info, dict):
            continue
        if info.get("api_node") is True:
            out.append(ct)
            continue
        category = info.get("category")
        if isinstance(category, str) and category.startswith(PARTNER_NODE_CATEGORY_PREFIXES):
            out.append(ct)
    return sorted(out)


def _gallery_paid_signals(row: dict[str, Any]) -> list[str]:
    """Gallery-index evidence that a template runs partner/API nodes.

    Belt-and-suspenders for the spend gate: when the local server's
    object_info is unavailable (fail-open fetch) node detection can miss, but
    the curated gallery still marks paid templates with the ``API`` tag and
    provider logos. Returns human-readable signal strings, empty = no signal.
    """
    signals: list[str] = []
    for tag in row.get("tags") or []:
        if isinstance(tag, str) and tag.lower() == "api":
            signals.append("tag:API")
    for prov in row.get("providers") or []:
        if isinstance(prov, str) and prov:
            signals.append(f"provider:{prov}")
    return signals


def _enforce_spend_gate(
    renderer,
    *,
    name: str,
    workflow: Any,
    row: dict[str, Any],
    object_info: dict,
    allow_spend: bool,
) -> None:
    """Consent interlock before submitting a template that spends Comfy credits.

    Returns None when the run may proceed (no paid signals, --allow-spend, or
    an interactive yes); raises typer.Exit(1) otherwise. Behavior is the
    BE-4113 gate moved verbatim out of run_template_cmd.
    """
    import sys

    from rich.markup import escape

    paid_nodes = _detect_paid_nodes(workflow, object_info)
    gallery_signals = _gallery_paid_signals(row)
    if (paid_nodes or gallery_signals) and not allow_spend:
        evidence = {
            "template": name,
            "partner_nodes": paid_nodes,
            "gallery_signals": gallery_signals,
        }
        if renderer.is_pretty() and sys.stdin and sys.stdin.isatty():
            rprint(
                f"[yellow]⚠ Template [bold]{escape(name)}[/bold] uses partner-API nodes that spend Comfy credits.[/yellow]"
            )
            if paid_nodes:
                rprint(f"  [dim]nodes:[/dim] {escape(', '.join(paid_nodes))}")
            if gallery_signals:
                rprint(f"  [dim]gallery:[/dim] {escape(', '.join(gallery_signals))}")
            if not typer.confirm("Run anyway and spend credits?", default=False):
                renderer.error(
                    code="spend_consent_required",
                    message="declined — template not submitted, no credits spent",
                    details=evidence,
                )
                raise typer.Exit(code=1)
        else:
            renderer.error(
                code="spend_consent_required",
                message=(
                    f"template {name!r} uses partner-API (paid) nodes; "
                    "re-run with --allow-spend to consent to spending Comfy credits"
                ),
                hint="paid nodes only run with explicit consent; OSS templates run without this flag",
                details=evidence,
            )
            raise typer.Exit(code=1)


def _resolve_param_addresses(
    renderer,
    overrides: dict[str, Any],
    slots: list[dict],
) -> dict[str, Any]:
    """Map ``--param KEY=VALUE`` keys onto slot addresses.

    KEY may be a full slot address (``6.text``, ``62/34.text``) or a bare slot
    name (``prompt``) when exactly one slot carries that name. Ambiguous or
    unknown keys error with the candidate list so agents can self-correct.
    """
    addresses = {s.get("address") for s in slots if isinstance(s, dict)}
    by_name: dict[str, list[str]] = {}
    for s in slots:
        if not isinstance(s, dict):
            continue
        n = s.get("name")
        if isinstance(n, str) and n:
            by_name.setdefault(n.lower(), []).append(str(s.get("address")))

    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in addresses:
            resolved[key] = value
            continue
        candidates = by_name.get(key.lower(), [])
        if len(candidates) == 1:
            resolved[candidates[0]] = value
            continue
        if len(candidates) > 1:
            renderer.error(
                code="workflow_slot_invalid",
                message=f"--param key {key!r} is ambiguous: {len(candidates)} slots share that name",
                hint="use the full slot address instead: " + ", ".join(f"{a}={key}" for a in candidates[:5]),
                details={"key": key, "candidates": candidates},
            )
            raise typer.Exit(code=1)
        sample = sorted(a for a in addresses if a)[:10]
        renderer.error(
            code="workflow_slot_invalid",
            message=f"--param key {key!r} matches no slot in this template",
            hint=(
                "valid addresses include: " + ", ".join(sample)
                if sample
                else "this template exposes no tweakable slots"
            ),
            details={"key": key, "available": sample},
        )
        raise typer.Exit(code=1)
    return resolved


@tracking.track_command("templates")
def run_template_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    params: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            "-p",
            metavar="KEY=VALUE",
            show_default=False,
            help=(
                "Fill a parameterized input before running (repeatable). KEY is a slot "
                "address (`6.text`) or a unique slot name (`prompt`); VALUE parses as "
                "JSON with string fallback. List slots with `comfy templates fetch "
                "<name> -o wf.json && comfy workflow slots wf.json`."
            ),
        ),
    ] = None,
    allow_spend: Annotated[
        bool,
        typer.Option(
            "--allow-spend",
            help=(
                "Consent to running partner-API (paid) nodes that spend Comfy credits. "
                "Required for API templates when not confirming interactively."
            ),
        ),
    ] = False,
    async_: Annotated[
        bool,
        typer.Option(
            "--async",
            show_default=False,
            help="Submit and return immediately instead of waiting for completion.",
        ),
    ] = False,
    host: Annotated[
        str | None,
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(help="Per-event timeout in seconds (same semantics as `comfy run --timeout`)."),
    ] = 120,
    verbose: Annotated[
        bool,
        typer.Option(help="Verbose execution output."),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="COMFY_API_KEY",
            help="Comfy API key for partner-API nodes (prefer the COMFY_API_KEY env var).",
        ),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index from GitHub before resolving."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Stream NDJSON run events to stdout (same dialect as `comfy run --json`).",
        ),
    ] = False,
):
    """Fetch a gallery template, fill its parameterized inputs, and run it on local ComfyUI.

    OSS templates need their referenced models installed locally first
    (`comfy model download`); missing models surface through the normal run
    validation errors. Templates that embed partner-API nodes spend Comfy
    credits and are gated behind --allow-spend / an interactive confirmation.
    """
    import tempfile

    from comfy_cli.command import run as run_module
    from comfy_cli.env_checker import check_comfy_server_running

    renderer = get_renderer()
    if json_output:
        renderer.force_stream()

    # -- Parse --param pairs up front so syntax errors fail before any I/O.
    overrides: dict[str, Any] = {}
    for raw in params or []:
        if "=" not in raw:
            renderer.error(
                code="workflow_slot_invalid",
                message=f"Expected `--param KEY=VALUE`, got {raw!r}",
                hint='example: --param 6.text="a cat" or --param prompt="a cat"',
            )
            raise typer.Exit(code=1)
        key, _, val = raw.partition("=")
        overrides[key.strip()] = _parse_param_value(val)

    # -- Resolve the template against the gallery index (close-matches on miss).
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    # The full `_GALLERY_LOAD_ERRORS` family, matching every other `_load_gallery`
    # call site — a non-200 or an over-cap body escapes here just as a URLError does.
    except _GALLERY_LOAD_ERRORS as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    row = next((r for r in rows if r["name"] == name), None)
    if row is None:
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    # -- Fetch + parse the template's workflow JSON.
    try:
        body = _fetch_template_workflow(name)
    # ``RuntimeError`` (a 2xx-but-not-200 status) and ``ResponseTooLarge`` (an
    # over-cap body) are the same class of upstream misbehaviour as a URLError
    # and belong in the same envelope, not in an uncaught traceback.
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, RuntimeError, ResponseTooLarge) as e:
        status = getattr(e, "code", None)
        renderer.error(
            code="template_fetch_failed",
            message=f"failed to fetch workflow for {name!r}: {e}",
            hint=(
                "the gallery index references a template whose workflow JSON "
                "is missing upstream — report at "
                "https://github.com/Comfy-Org/workflow_templates/issues"
                if status == 404
                else "check network connectivity"
            ),
            details={"status": status} if status else None,
        )
        raise typer.Exit(code=1) from e
    try:
        workflow = json.loads(body)
    except json.JSONDecodeError as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"upstream returned non-JSON for {name!r}: {e}",
            hint="report at https://github.com/Comfy-Org/workflow_templates/issues",
        )
        raise typer.Exit(code=1) from e

    # -- Resolve host/port through the shared resolver, exactly like `comfy
    # run`'s local branch (cmdline.py). This validates the host (rejecting
    # URL-injection characters), brackets IPv6 literals, and honors
    # config.background — behavior the old hand-rolled block lacked.
    from comfy_cli.host_port import parse_host_port_arg, report_usage_error, resolve_host_port

    # ``report_usage_error``: a rejected ``--host``/``--port`` raises
    # ``typer.BadParameter``, which click renders as a stderr usage panel and
    # exit 2 — with nothing on stdout. Emit the terminating envelope first so
    # JSON/NDJSON consumers still get a parseable final line (exit stays 2).
    with report_usage_error(renderer):
        if host:
            host, parsed_port = parse_host_port_arg(host)
            # ``port is None``, not ``not port``: a typed ``--port`` always wins over
            # one embedded in ``--host h:p``, including ``--port 0``, which
            # ``resolve_host_port`` then rejects as out of range instead of silently
            # running against the embedded port.
            if port is None and parsed_port is not None:
                port = parsed_port
        host, port = resolve_host_port(host, port)

    if not check_comfy_server_running(port, host, timeout=timeout):
        renderer.error(
            code="server_not_running",
            message=f"ComfyUI not running on specified address ({host}:{port})",
            hint="run: comfy launch",
            details={"host": host, "port": port},
        )
        raise typer.Exit(code=1)

    # object_info powers both slot filling and paid-node detection. Fail-open
    # ({}) keeps template runs working against bare servers — the gallery-index
    # signals below still gate paid templates in that case.
    object_info = run_module._fetch_object_info(host, port)

    # -- Fill parameterized inputs via the CQL slot engine.
    if overrides:
        if not isinstance(workflow.get("nodes"), list):
            renderer.error(
                code="workflow_slot_invalid",
                message=f"template {name!r} is not a frontend-format workflow; --param is not supported for it",
                hint="run it without --param, or fetch + edit it directly",
            )
            raise typer.Exit(code=1)
        if not object_info:
            renderer.error(
                code="object_info_unavailable",
                message="could not fetch /object_info from the server; --param needs the node catalog to fill slots",
                hint="check the ComfyUI server logs, or run without --param",
            )
            raise typer.Exit(code=1)
        from comfy_cli.cql.engine import Graph

        graph = Graph.from_object_info(object_info)
        graph._try_default_annotations()
        try:
            schema = graph.get_template_schema(name, workflow)
        except (ValueError, KeyError) as e:
            renderer.error(code="workflow_slot_invalid", message=f"Could not extract slots: {e}")
            raise typer.Exit(code=1) from e
        resolved = _resolve_param_addresses(renderer, overrides, schema.get("slots") or [])
        try:
            workflow, warnings = graph.apply_slots(workflow, resolved)
        except ValueError as e:
            renderer.error(
                code="workflow_slot_invalid",
                message=str(e),
                hint="fetch the template and run `comfy workflow slots <file>` to see valid addresses + types",
            )
            raise typer.Exit(code=1) from e
        if renderer.is_pretty():
            rprint(f"[bold green]✓[/bold green] filled {len(resolved)} parameter(s)")
            for addr in resolved:
                rprint(f"  [dim]·[/dim] {addr}")
            for w in warnings:
                rprint(f"  [yellow]warning:[/yellow] {w}")

    # -- Spend gate (BE-4113): partner-API nodes spend Comfy credits. Require
    # explicit consent before submitting anything that would burn them.
    _enforce_spend_gate(
        renderer,
        name=name,
        workflow=workflow,
        row=row,
        object_info=object_info,
        allow_spend=allow_spend,
    )

    # -- Hand off to the existing run path (UI→API conversion, partner
    # credential injection, preflight validation, execution, jobs state).
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    fd, tmp_path = tempfile.mkstemp(prefix=f"comfy_template_{safe}_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(workflow, f)
        run_module.execute(
            tmp_path,
            host,
            port,
            wait=not async_,
            verbose=verbose,
            timeout=timeout,
            api_key=api_key,
            # run-template's own spend gate (above) has already consented (or
            # found no paid nodes), so forward consent to avoid a second gate in
            # execute() (BE-4326). run-template's gate is strictly stronger — it
            # also inspects gallery signals — and has already run.
            allow_spend=True,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
