"""``comfy node deps`` — read-only per-pack Python dependency report.

For every installed custom node pack (or just the ones named on the command
line), reports the pack's *declared* Python requirements — from its
``requirements.txt`` and/or its ``pyproject.toml`` ``[project] dependencies``
— alongside the version actually installed in the workspace venv and a
computed status per requirement.

``--registry <node-id>`` extends the same diff to a pack that is **not yet
installed**: the registry's published dependency list for that pack's latest
version, diffed against the very same ``pip list`` map, so conflict risk can be
assessed *before* installing anything.

Read-only by construction: it never installs anything and never shells out to
``cm-cli``. The only subprocess it runs is a single ``<workspace python> -m pip
list --format=json`` for the whole report (not once per pack). The only network
call is the registry's side-effect-free ``GET /nodes/{id}``, made solely for
``--registry`` ids — never ``install_node``, whose endpoint records an
installation and fires an analytics event server-side on every call.

Statuses:

- ``satisfied``   — installed, and the declared specifier accepts that version
  (a bare name with no specifier is satisfied by any installed version);
- ``mismatch``    — installed, but the specifier rejects that version (both
  versions are reported);
- ``missing``     — not present in the workspace venv at all;
- ``unparseable`` — a pip option (``--extra-index-url``, ``-r other.txt``), a
  bare VCS/URL line, or anything else :class:`packaging.requirements.Requirement`
  rejects. Kept in the report rather than dropped silently;
- ``unknown``     — the comparison could not be made: either the ``pip list``
  probe failed (accompanied by an ``installed_versions_unavailable`` warning on
  the report) or the installed dist records a version that isn't PEP 440.

Known caveat: pyproject dependencies are read through the shared registry
parser (:func:`comfy_cli.command.pack_scan.read_pyproject`), which drops the
``comfyui-frontend-package`` pin from ``[project] dependencies``. Such a pin is
a core-ComfyUI concern rather than a pack dependency, so it will not appear in
this report even though it is declared.

Known caveat (``--registry``): the registry exposes dependency metadata only on
a node's *latest* published version — there is no read-only endpoint for a
pinned version's dependencies — so a ``--registry`` row always describes the
latest version, whatever version an eventual install would pick.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from comfy_cli.command.pack_scan import iter_pack_dirs, read_pyproject

PIP_LIST_TIMEOUT_SECONDS = 60

# Same comment rules as ``uv.py::parse_req_file``: a ``#`` at line start or
# preceded by whitespace begins a comment. VCS URL fragments like
# ``#subdirectory=pkg`` / ``#egg=foo`` survive because they are not preceded by
# whitespace.
_INLINE_COMMENT_RE = re.compile(r"(^|\s+)#.*$")

# Every status a requirement row can carry; the per-pack summary always reports
# all of them so consumers can index without a KeyError.
STATUSES = ("satisfied", "mismatch", "missing", "unparseable", "unknown")

SOURCE_REQUIREMENTS_TXT = "requirements.txt"
SOURCE_PYPROJECT = "pyproject.toml"
# Not a file: the registry's published metadata for a not-yet-installed pack.
SOURCE_REGISTRY = "registry"

# Namespaced away from ``outdated``'s own ``pack:<id>`` entries, which share the
# cache file but hold a bare version string rather than this dict.
REGISTRY_CACHE_PREFIX = "deps-registry:"

NO_REGISTRY_DEPENDENCY_METADATA = "registry did not return dependency metadata for this pack"

# A registry node id is an opaque server-side slug that ``RegistryAPI.get_node``
# interpolates unescaped into ``GET {base_url}/nodes/{id}``. Every id in the
# public registry matches this charset, so nothing legitimate is rejected — but
# ``/``, ``?`` and ``#`` are URL-significant: ``--registry <pack>/install``
# would build the exact URL of ``install_node`` (itself a GET on
# ``/nodes/{id}/install``, which records an installation server-side), and
# ``?``/``#`` inject a query string or truncate the path. A read-only report
# must not be steerable into those, so ids are validated before any request.
_REGISTRY_NODE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

# A registry failure message is server-controlled text — on a captive-portal or
# misbehaving-proxy network it is a whole HTML page — and gets copied into the
# single-line JSON envelope consumers parse. Clamp it first.
MAX_REGISTRY_ERROR_CHARS = 300


# ---------------------------------------------------------------------------
# Installed versions
# ---------------------------------------------------------------------------


def pip_list(python: str) -> tuple[dict[str, str] | None, str | None]:
    """Return ``(canonical_name -> version, error)`` for the workspace venv.

    Runs ``<python> -m pip list --format=json`` exactly once. On any failure
    the map is ``None`` and the second element carries a human-readable reason
    — a report that can't see the venv is still a useful report.
    """
    from packaging.utils import canonicalize_name

    try:
        proc = subprocess.run(
            [python, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=PIP_LIST_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        return None, f"`pip list` exited {e.returncode}" + (f": {detail[-1]}" if detail else "")
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"could not run `pip list`: {e}"

    try:
        rows = json.loads(proc.stdout)
    except ValueError as e:
        return None, f"could not parse `pip list --format=json` output: {e}"
    if not isinstance(rows, list):
        return None, "`pip list --format=json` did not return a list"

    versions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, version = row.get("name"), row.get("version")
        if isinstance(name, str) and name:
            versions[canonicalize_name(name)] = str(version) if version is not None else ""
    return versions, None


# ---------------------------------------------------------------------------
# Declared requirements
# ---------------------------------------------------------------------------


def _requirements_txt_lines(path: Path) -> tuple[list[str], str | None]:
    """Return ``(lines, error)`` for a pack ``requirements.txt``.

    Mirrors ``uv.py::parse_req_file``'s comment/blank handling, but keeps pip
    option lines (``--extra-index-url``, ``-r base.txt``) instead of splitting
    them off — this report surfaces them as ``unparseable`` rather than
    dropping them.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [], f"could not read {path}: {e}"

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _INLINE_COMMENT_RE.sub("", raw_line).strip()
        if line:
            lines.append(line)
    return lines, None


def _pyproject_lines(path: Path) -> list[str]:
    cfg = read_pyproject(str(path))
    if cfg is None:
        return []
    deps = getattr(cfg.project, "dependencies", None) or []
    return [d.strip() for d in deps if isinstance(d, str) and d.strip()]


def _classify(raw: str, installed_versions: dict[str, str] | None) -> dict[str, Any]:
    """Turn one declared requirement line into a report row."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    row: dict[str, Any] = {
        "raw": raw,
        "name": None,
        "specifier": None,
        "installed": None,
        "status": "unparseable",
    }

    # A leading ``-`` is a pip option (``-r``, ``-e``, ``--extra-index-url``),
    # never a requirement — `Requirement` would either reject it or, worse,
    # misread it. Short-circuit so the reason is unambiguous.
    if raw.startswith("-"):
        return row

    try:
        req = Requirement(raw)
    except InvalidRequirement:
        # Bare VCS/URL lines (``git+https://…``, ``https://…/x.whl``) land here.
        return row

    row["name"] = req.name
    row["specifier"] = str(req.specifier)
    if req.marker is not None:
        # Markers are NOT evaluated: the relevant environment is the workspace
        # venv's, not this CLI process's. Surfaced so a consumer can tell why a
        # platform-specific pin reads as `missing`.
        row["marker"] = str(req.marker)

    if installed_versions is None:
        row["status"] = "unknown"
        return row

    installed = installed_versions.get(canonicalize_name(req.name))
    row["installed"] = installed
    if installed is None:
        row["status"] = "missing"
    elif not req.specifier:
        # A bare name is satisfied by any installed version.
        row["status"] = "satisfied"
    else:
        from packaging.version import InvalidVersion

        try:
            contains = req.specifier.contains(installed, prereleases=True)
        except InvalidVersion:
            # A dist whose recorded version isn't PEP 440 (rare, but a broken
            # `.dist-info` will do it) can't be compared. Say so rather than
            # letting the exception abort the whole report.
            row["status"] = "unknown"
            return row
        row["status"] = "satisfied" if contains else "mismatch"
    return row


def _pack_report(pack_dir: Path, workspace: Path, installed_versions: dict[str, str] | None) -> tuple[dict, list[str]]:
    """Build one pack's report row. Returns ``(row, warnings)``."""
    warnings: list[str] = []
    requirement_files: list[str] = []
    # ``raw`` line -> row, so a line declared in BOTH files is reported once
    # (requirements.txt wins the ``source`` attribution, being read first).
    rows: dict[str, dict[str, Any]] = {}

    req_txt = pack_dir / "requirements.txt"
    if req_txt.is_file():
        requirement_files.append(SOURCE_REQUIREMENTS_TXT)
        lines, err = _requirements_txt_lines(req_txt)
        if err:
            warnings.append(err)
        for line in lines:
            if line not in rows:
                rows[line] = {**_classify(line, installed_versions), "source": SOURCE_REQUIREMENTS_TXT}

    pyproject = pack_dir / "pyproject.toml"
    if pyproject.is_file():
        lines = _pyproject_lines(pyproject)
        # Only advertise pyproject as a requirement file when it actually
        # declares dependencies — most packs ship one purely for registry
        # metadata.
        if lines:
            requirement_files.append(SOURCE_PYPROJECT)
        for line in lines:
            if line not in rows:
                rows[line] = {**_classify(line, installed_versions), "source": SOURCE_PYPROJECT}

    requirements = list(rows.values())
    summary = {status: 0 for status in STATUSES}
    for row in requirements:
        summary[row["status"]] += 1

    try:
        rel_path = str(pack_dir.relative_to(workspace))
    except ValueError:
        rel_path = str(pack_dir)

    return (
        {
            "pack": pack_dir.name,
            "path": rel_path,
            "status": "installed",
            "requirement_files": requirement_files,
            "requirements": requirements,
            "summary": summary,
        },
        warnings,
    )


# ---------------------------------------------------------------------------
# Registry candidates (not-yet-installed packs)
# ---------------------------------------------------------------------------


def _truncate(message: str, limit: int = MAX_REGISTRY_ERROR_CHARS) -> str:
    """Clamp server-controlled error text before it enters the JSON envelope.

    Whitespace is collapsed first so a multi-line HTML error page cannot break
    the single-line JSON output consumers read.
    """
    text = " ".join(message.split())
    return text if len(text) <= limit else text[:limit] + "… (truncated)"


def _registry_cache_key(registry_api: Any, node_id: str) -> str:
    """Cache key for one registry lookup.

    Includes the registry base URL because ``RegistryAPI`` selects localhost,
    staging or production from ``ENVIRONMENT``: metadata fetched from a dev or
    staging registry must not be served to a later production run, where it
    could hide a real dependency conflict. The id is lower-cased because the
    registry resolves ids case-insensitively (``GET /nodes/comfyui-lcm`` 302s to
    ``/nodes/ComfyUI-LCM``), so two spellings share one entry rather than
    duplicating both the request and the stored value.
    """
    base = getattr(registry_api, "base_url", "") or ""
    return f"{REGISTRY_CACHE_PREFIX}{base}:{node_id.lower()}"


def _normalize_dependencies(dependencies: Any) -> tuple[list[str] | None, bool]:
    """Return ``(declared, dropped_any)`` for a raw registry dependency list.

    ``map_node_version`` defaults a missing ``dependencies`` to ``[]``, so an
    empty list is indistinguishable from "the field was absent" — we must not
    claim the pack declares zero dependencies. Both collapse to ``None``, as
    does anything that isn't a list of strings (iterating a stray bare string
    would yield one row per character). ``dropped_any`` reports whether a
    non-string entry was filtered out, so a caller holding an otherwise usable
    list can warn that the row is partial rather than silently under-reporting.

    Applied to the cached value as well as the network one: a cache entry
    written by another comfy-cli version (or hand-edited) can hold ``[null]``,
    which would reach ``_classify`` and raise ``AttributeError`` on
    ``raw.startswith`` — aborting the whole report, the opposite of this
    module's degrade-to-a-warning design.
    """
    if not isinstance(dependencies, list):
        return None, False
    kept = [d.strip() for d in dependencies if isinstance(d, str) and d.strip()]
    dropped = any(not (isinstance(d, str) and d.strip()) for d in dependencies)
    return (kept or None), dropped


def _registry_error(node_id: str, exc: Exception) -> dict[str, str]:
    """Map a ``get_node`` failure to a ``{"code", "message"}`` warning.

    A 404 is permanent — the id is misspelled or was never published — and must
    not be reported as ``registry_unavailable``, whose hint tells the caller to
    check network access and retry with ``--refresh``. An agent following that
    hint against a 404 retries forever against an id that will never resolve.
    """
    if getattr(exc, "status_code", None) == 404:
        return {
            "code": "registry_node_not_found",
            "message": f"registry has no node '{node_id}' (HTTP 404) — check the id, or the pack may be unpublished",
        }
    return {
        "code": "registry_unavailable",
        "message": f"could not fetch registry metadata for '{node_id}': {_truncate(str(exc))}",
    }


def _registry_declared(
    node_id: str,
    cache: dict[str, Any],
    refresh: bool,
    registry_api: Any,
) -> tuple[list[str] | None, str | None, dict[str, str] | None, bool]:
    """Return ``(declared, version, error, partial)`` for a registry node id.

    ``declared`` is the latest version's published dependency list, or ``None``
    when the registry gave us nothing usable. ``error`` is a ``{"code",
    "message"}`` warning. ``partial`` marks an otherwise usable list that lost malformed
    entries. Results (including "the registry published no dependencies") are
    cached for an hour under the same file ``comfy outdated`` uses, so repeated
    agent calls stay cheap; ``refresh`` bypasses the read.
    """
    from comfy_cli.command.outdated import _cache_get, _cache_set

    key = _registry_cache_key(registry_api, node_id)
    if not refresh:
        cached = _cache_get(cache, key)
        if isinstance(cached, dict):
            # Normalized on the way out too — see ``_normalize_dependencies``.
            declared, _ = _normalize_dependencies(cached.get("dependencies"))
            version = cached.get("version")
            return declared, (version if isinstance(version, str) else None), None, False

    try:
        # get_node, NOT install_node: the install endpoint records an
        # installation + fires an analytics event server-side on every call, so
        # a pre-install *report* must never touch it.
        node = registry_api.get_node(node_id)
    except Exception as e:  # noqa: BLE001 - registry unreachable → a per-entry warning, not a failed command
        return None, None, _registry_error(node_id, e), False

    latest = getattr(node, "latest_version", None)
    version = getattr(latest, "version", None)
    declared, partial = _normalize_dependencies(getattr(latest, "dependencies", None))

    _cache_set(cache, key, {"version": version if isinstance(version, str) else None, "dependencies": declared})
    return declared, (version if isinstance(version, str) else None), None, partial


def _registry_report(
    node_id: str,
    installed_versions: dict[str, str] | None,
    cache: dict[str, Any],
    refresh: bool,
    registry_api: Any,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Build one registry-candidate row. Returns ``(row, warnings)``."""
    declared, version, error, partial = _registry_declared(node_id, cache, refresh, registry_api)

    warnings: list[dict[str, str]] = []
    row: dict[str, Any] = {
        "pack": node_id,
        "path": None,
        "status": "registry",
        "registry": True,
        "version": version,
        "declared": declared,
        "requirement_files": [],
        "requirements": [],
        "summary": dict.fromkeys(STATUSES, 0),
    }

    if error is not None:
        row["warning"] = error["message"]
        warnings.append(error)
        return row, warnings

    if declared is None:
        row["warning"] = NO_REGISTRY_DEPENDENCY_METADATA
        warnings.append(
            {
                "code": "registry_no_dependency_metadata",
                "message": f"{NO_REGISTRY_DEPENDENCY_METADATA}: '{node_id}'",
            }
        )
        return row, warnings

    if partial:
        # The list is usable but lossy: a dropped entry that would have
        # conflicted is invisible, so the row must not read as complete metadata.
        message = f"registry returned malformed dependency entries for '{node_id}'; this row is incomplete"
        row["warning"] = message
        warnings.append({"code": "registry_partial_dependency_metadata", "message": message})

    # Same per-requirement diff, against the same single ``pip list`` map.
    rows: dict[str, dict[str, Any]] = {}
    for line in declared:
        if line not in rows:
            rows[line] = {**_classify(line, installed_versions), "source": SOURCE_REGISTRY}
    row["requirements"] = list(rows.values())
    row["requirement_files"] = [SOURCE_REGISTRY]
    for req in row["requirements"]:
        row["summary"][req["status"]] += 1
    return row, warnings


def _not_installed_row(name: str) -> dict[str, Any]:
    return {
        "pack": name,
        "path": None,
        "status": "not_installed",
        "requirement_files": [],
        "requirements": [],
        "summary": dict.fromkeys(STATUSES, 0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    comfy_path: str | None,
    pack_names: list[str] | None = None,
    *,
    python: str | None = None,
    registry_ids: list[str] | None = None,
    refresh: bool = False,
    registry_api: Any | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Build the per-pack dependency report.

    Returns ``(report, warnings)``. ``report`` is ``None`` when *comfy_path*
    resolves to no usable workspace — the caller turns that into a
    ``not_in_workspace`` error envelope. Each warning is
    ``{"code": ..., "message": ...}``.

    *registry_ids* names not-yet-installed registry packs to include as extra
    rows (``status: "registry"``). Inject *registry_api* (anything with
    ``get_node``) to unit-test without the network.
    """
    from comfy_cli.resolve_python import resolve_workspace_python

    warnings: list[dict[str, str]] = []
    if not comfy_path or not os.path.isdir(comfy_path):
        return None, warnings

    workspace = Path(comfy_path)
    python = python or resolve_workspace_python(comfy_path)

    installed_versions, pip_error = pip_list(python)
    if installed_versions is None:
        warnings.append(
            {
                "code": "installed_versions_unavailable",
                "message": (
                    f"could not read installed packages from {python}"
                    + (f" ({pip_error})" if pip_error else "")
                    + " — every parseable requirement is reported with status 'unknown'"
                ),
            }
        )

    # Validate, then dedupe (order-preserving). Dedupe is case-insensitive
    # because the registry resolves ids that way, so `--registry Some-Pack
    # --registry some-pack` is one pack, not two rows and two network calls;
    # the first spelling seen is the one reported. Every rejected id is
    # surfaced as a warning rather than dropped silently — a caller whose only
    # `--registry` value was rejected must not read an empty report as "no
    # conflicts".
    wanted_registry: list[str] = []
    seen_registry: set[str] = set()
    for raw_id in registry_ids or []:
        candidate = (raw_id or "").strip()
        if not _REGISTRY_NODE_ID_RE.fullmatch(candidate):
            warnings.append(
                {
                    "code": "registry_invalid_node_id",
                    "message": (
                        f"ignored --registry value {candidate!r}: a registry node id must be non-empty and "
                        "match [A-Za-z0-9._-]"
                    ),
                }
            )
            continue
        if candidate.lower() in seen_registry:
            continue
        seen_registry.add(candidate.lower())
        wanted_registry.append(candidate)

    pack_dirs = iter_pack_dirs(workspace / "custom_nodes")
    packs: list[dict[str, Any]] = []
    if pack_names:
        by_lower = {p.name.lower(): p for p in pack_dirs}
        for requested in pack_names:
            match = by_lower.get(requested.strip().lower())
            if match is None:
                packs.append(_not_installed_row(requested))
                continue
            row, pack_warnings = _pack_report(match, workspace, installed_versions)
            packs.append(row)
            warnings.extend({"code": "pack_read_error", "message": w} for w in pack_warnings)
    elif not registry_ids:
        # Bare `comfy node deps` reports the whole workspace. A `--registry`-only
        # invocation is a targeted pre-install question, so it does NOT also dump
        # every installed pack; name packs positionally to get both. Keyed on the
        # *requested* ids, not the surviving ones: if every `--registry` value
        # was rejected above, the answer is that targeted question's warning, not
        # a surprise dump of every installed pack.
        for pack_dir in pack_dirs:
            row, pack_warnings = _pack_report(pack_dir, workspace, installed_versions)
            packs.append(row)
            warnings.extend({"code": "pack_read_error", "message": w} for w in pack_warnings)

    if wanted_registry:
        from comfy_cli.command.outdated import _cache_get, _load_cache, _save_cache

        # Shares ``comfy outdated``'s 1h cache file under a distinct key prefix.
        cache = _load_cache()
        api = registry_api
        if api is None:
            from comfy_cli.registry import RegistryAPI

            api = RegistryAPI()
        touched: set[str] = set()
        for node_id in wanted_registry:
            key = _registry_cache_key(api, node_id)
            before = cache.get(key)
            row, registry_warnings = _registry_report(node_id, installed_versions, cache, refresh, api)
            packs.append(row)
            warnings.extend(registry_warnings)
            # Identity, not membership: a lookup that failed (or was served from
            # cache) leaves any pre-existing entry untouched, and an *expired*
            # one of those must stay prunable below rather than being carried
            # forward as though we had just written it.
            if cache.get(key) is not before:
                touched.add(key)

        # `node deps` is a second writer of a file `comfy outdated` already
        # owns, so it must not blind-write the dict it read minutes ago: re-read
        # and merge only our own keys, or a concurrent `outdated` run's writes
        # are silently discarded. (`_save_cache` renames into place, so a reader
        # never sees a half-written file — but this is still last-writer-wins on
        # a genuinely simultaneous write, which for a rebuildable 1h cache costs
        # only a re-fetch.)
        fresh = _load_cache()
        for key in touched:
            fresh[key] = cache[key]
        # Prune expired registry entries while we hold the file. `_cache_get`
        # only checks the TTL at read time and `_save_cache` rewrites every key,
        # so without this these never leave: unlike `outdated`'s keys, this
        # prefix's key space is arbitrary caller-supplied ids holding whole
        # dependency lists, which a loop over many ids would grow without bound.
        for key in [k for k in fresh if k.startswith(REGISTRY_CACHE_PREFIX) and k not in touched]:
            if _cache_get(fresh, key) is None:
                del fresh[key]
        _save_cache(fresh)

    compiled = workspace / "requirements.compiled"
    compiled_present = compiled.is_file()
    report = {
        "workspace": str(workspace),
        "python": python,
        "compiled_lock": {
            "present": compiled_present,
            # Presence + path only in v1 — the lock is never parsed or diffed.
            "path": str(compiled) if compiled_present else None,
        },
        "packs": packs,
    }
    return report, warnings


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "satisfied": "[green]satisfied[/green]",
    "mismatch": "[bold red]mismatch[/bold red]",
    "missing": "[bold yellow]missing[/bold yellow]",
    "unparseable": "[dim]unparseable[/dim]",
    "unknown": "[dim]unknown[/dim]",
}


def _render_pretty(renderer, report: dict[str, Any]) -> None:
    from rich.markup import escape
    from rich.table import Table

    console = renderer.console()
    for pack in report["packs"]:
        title = escape(pack["pack"])
        if pack.get("registry"):
            version = pack.get("version")
            title += " [dim](registry, latest " + (escape(version) if version else "unknown") + ")[/dim]"
        if pack["status"] == "not_installed":
            renderer.print(f"[bold]{title}[/bold] — [yellow]not installed[/yellow]")
            continue
        if not pack["requirements"]:
            reason = pack.get("warning") or "no declared requirements"
            renderer.print(f"[bold]{title}[/bold] — [dim]{escape(reason)}[/dim]")
            continue

        tbl = Table(show_header=True, header_style="bold", title=title, title_justify="left")
        tbl.add_column("requirement")
        tbl.add_column("source")
        tbl.add_column("installed")
        tbl.add_column("status")
        for req in pack["requirements"]:
            # Values come from pack-controlled files; escape so a spec like
            # ``foo[/]`` can't raise a rich MarkupError and crash the report.
            installed = escape(req["installed"]) if req["installed"] is not None else "[dim]-[/dim]"
            tbl.add_row(
                escape(req["raw"]),
                escape(req["source"]),
                installed,
                _STATUS_STYLE.get(req["status"], escape(req["status"])),
            )
        console.print(tbl)

    totals = dict.fromkeys(STATUSES, 0)
    for pack in report["packs"]:
        for status, count in pack["summary"].items():
            totals[status] = totals.get(status, 0) + count
    renderer.print(
        "[bold]totals[/bold]: "
        + ", ".join(f"{count} {status}" for status, count in totals.items() if count)
        + (" — nothing declared" if not any(totals.values()) else "")
    )
    if report["compiled_lock"]["present"]:
        renderer.print(f"[dim]compiled lock: {escape(report['compiled_lock']['path'])}[/dim]")


def execute(
    renderer,
    comfy_path: str | None,
    pack_names: list[str] | None = None,
    *,
    registry_ids: list[str] | None = None,
    refresh: bool = False,
) -> None:
    """Entry point wired from ``comfy node deps``."""
    from rich.markup import escape

    report, warnings = build_report(comfy_path, pack_names, registry_ids=registry_ids, refresh=refresh)
    if report is None:
        import typer

        renderer.error(
            code="not_in_workspace",
            message=(
                "ComfyUI workspace not found"
                + (f" at {comfy_path!r}" if comfy_path else "")
                + ". Run 'comfy install', run 'comfy' from a ComfyUI directory, or pass '--workspace'."
            ),
            hint="run: comfy install   (or pass --workspace /path/to/ComfyUI)",
            command="node deps",
        )
        # Mirrors the `comfy which` convention: renderer.error records the code,
        # typer.Exit is what actually makes the process exit non-zero.
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        _render_pretty(renderer, report)
    for warning in warnings:
        renderer.warn(escape(warning["message"]))
    # Warnings ride along in the payload too: a JSON consumer reads only stdout.
    report["warnings"] = warnings
    renderer.emit(report, command="node deps")
