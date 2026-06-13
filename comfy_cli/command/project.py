"""``comfy project`` — the project/1 convention: init and status.

A project is a directory with a ``comfy.yaml`` marker (``schema: project/1``)
and five conventional dirs: ``assets/ fragments/ blueprints/ outputs/
.comfy/`` (machine-owned). The convention is the contract — ``init`` lays it
down, ``status`` is the queryable state agents read instead of hand-written
manifests: blueprints, assets (joined against the push lock), the run
journal, and layout warnings.

Discovery/journaling logic lives in the pure :mod:`comfy_cli.project`; this
module is only the Typer/renderer surface (template: ``auth/command.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.command.transfer import _upload_file
from comfy_cli.output import get_renderer, rprint
from comfy_cli.project import (
    ASSETS_LOCK_SCHEMA,
    CONVENTIONAL_DIRS,
    PROJECT_MARKER,
    PROJECT_SCHEMA,
    Project,
    find_project,
    read_assets_lock,
    read_journal,
    unknown_dirs,
)
from comfy_cli.target import resolve_target

app = typer.Typer(
    no_args_is_help=True,
    help="Project conventions: init and status.",
)

# `comfy assets` lives in this module too: assets are a project/1 concept
# (the lock is project state under .comfy/), not a generic transfer.
assets_app = typer.Typer(
    no_args_is_help=True,
    help="Project assets: push assets/ to the run target over its HTTP upload API.",
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file through SHA-256 in fixed-size chunks so large assets don't
    get loaded fully into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_asset_files(assets_dir: Path):
    """Yield (path, relative-name) for real files under ``assets/``, skipping
    dotfiles, symlinks, and anything whose resolved path escapes ``assets/``
    (a symlink could otherwise leak files from outside the project)."""
    assets_root = assets_dir.resolve()
    for path in sorted(assets_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(assets_root):
            continue
        rel = path.relative_to(assets_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        yield path, rel.as_posix()


@assets_app.callback()
def _assets_callback():
    """Project assets — push and track.

    (Also keeps Typer from collapsing the single-command group, so the
    command stays ``comfy assets push``, not a bare ``comfy assets``.)
    """


# The marker `comfy project init` writes — deliberately literal and minimal.
# The where default is resolved at init time (flag, else auto-detect) so a
# local-only machine never gets a project that routes every command to cloud.
MARKER_TEMPLATE = "schema: project/1\ndefaults:\n  where: {where}\n"


@app.command("init", help="Initialize the project/1 convention here: comfy.yaml marker + conventional dirs.")
@tracking.track_command("project")
def init_cmd(
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            help="Default backend for this project (local|cloud). Omitted: auto-detect "
            "(cloud if cloud credentials are configured, else local).",
        ),
    ] = None,
):
    renderer = get_renderer()
    cwd = Path.cwd().resolve()

    from comfy_cli import where as where_module

    try:
        # project_value=None: a project can't govern its own init; resolve from
        # flag → env → config → auto-detect, same chain every routed command uses.
        decision = where_module.resolve(flag=where, project_value=None)
    except ValueError as e:
        renderer.error(code="invalid_argument", message=str(e))
        raise typer.Exit(code=1) from e
    default_where = decision.target.value

    existing = find_project(cwd)
    if existing is not None:
        renderer.error(
            code="project_already_exists",
            message=f"This directory is already governed by the project at {existing.root}.",
            hint="use the existing project (edit its comfy.yaml), or init outside it",
            details={"root": str(existing.root)},
        )
        raise typer.Exit(code=1)

    (cwd / PROJECT_MARKER).write_text(MARKER_TEMPLATE.format(where=default_where), encoding="utf-8")
    created = [PROJECT_MARKER]
    for d in CONVENTIONAL_DIRS:
        (cwd / d).mkdir(exist_ok=True)
        created.append(f"{d}/")

    if renderer.is_pretty():
        rprint(f"[bold green]Initialized project[/bold green] at {cwd}")
        for name in created:
            rprint(f"  [dim]created[/dim] {name}")
    renderer.emit(
        {"root": str(cwd), "created": created, "action": "init", "where_default": default_where},
        command="project init",
        changed=True,
    )


@app.command("status", help="Show the governing project: defaults, blueprints, assets vs lock, recent runs.")
@tracking.track_command("project")
def status_cmd():
    renderer = get_renderer()
    project = find_project()
    if project is None:
        renderer.error(
            code="project_not_found",
            message="No comfy.yaml project (schema project/1) governs this directory.",
            hint="run: comfy project init",
        )
        raise typer.Exit(code=1)

    data = {
        "root": str(project.root),
        "schema": PROJECT_SCHEMA,
        "defaults": _defaults(project),
        "blueprints": _blueprint_names(project),
        "assets": _asset_entries(project),
        "recent_runs": read_journal(project, limit=20),
        "warnings": [f"unknown top-level directory: {d}" for d in unknown_dirs(project)],
        "action": "status",
    }
    if renderer.is_pretty():
        _render_status_pretty(data)
    renderer.emit(data, command="project status")


# ---------------------------------------------------------------------------
# comfy assets push
# ---------------------------------------------------------------------------


@assets_app.command("push", help="Upload changed files under assets/ to the run target; record them in the lock.")
@tracking.track_command("assets")
def assets_push_cmd(
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            show_default=False,
            help="Push target (local|cloud). Omitted: the usual chain — env, project default, config, auto.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-push every file, even when the lock says it is already current."),
    ] = False,
):
    """Sync ``assets/`` to the resolved target's input directory.

    Uploads go through the server's ``/upload/image`` HTTP API only — the CLI
    never writes into a ComfyUI install's folders. Each pushed file is
    recorded in ``.comfy/assets.lock.json`` with its sha256 and the
    server-returned name; a file whose lock entry matches (same sha256 AND
    same target) is skipped unless ``--force``.
    """
    renderer = get_renderer()
    project = find_project()
    if project is None:
        renderer.error(
            code="project_not_found",
            message="No comfy.yaml project (schema project/1) governs this directory.",
            hint="run: comfy project init",
        )
        raise typer.Exit(code=1)

    try:
        target = resolve_target(where=where)
    except ValueError as e:
        renderer.error(code="invalid_argument", message=str(e))
        raise typer.Exit(code=1) from e
    where_kind = target.kind

    assets_dir = project.root / "assets"
    lock_path = project.root / ".comfy" / "assets.lock.json"
    lock_assets = read_assets_lock(project)

    pushed: list[dict] = []
    # Assets verified already-current on this target (lock hit). Listed in the
    # envelope so a script can assert its files are on the server from the
    # push result alone (pushed ∪ current) — a truthful `pushed: []` after a
    # peer already pushed otherwise reads as failure.
    current: list[str] = []
    skipped = 0
    for path, name in _iter_asset_files(assets_dir) if assets_dir.is_dir() else []:
        sha = _sha256_file(path)
        locked = lock_assets.get(name)
        if not force and isinstance(locked, dict) and locked.get("sha256") == sha and locked.get("where") == where_kind:
            skipped += 1
            current.append(name)
            continue
        try:
            result = _upload_file(path, target, overwrite=True)
        except urllib.error.HTTPError as e:
            renderer.error(
                code="upload_failed",
                message=f"Failed to push {name}: HTTP {e.code}",
                hint="check the server is reachable (and `comfy cloud login` for cloud)",
                details={"status": e.code, "name": name, "where": where_kind},
            )
            raise typer.Exit(code=1) from e
        # `cloud_name` = the name on the push target (the server-returned
        # name), whatever the target kind — the key project status joins on.
        cloud_name = result.get("name", path.name)
        size = path.stat().st_size
        lock_assets[name] = {
            "sha256": sha,
            "cloud_name": cloud_name,
            "size": size,
            "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "where": where_kind,
        }
        pushed.append({"name": name, "sha256": sha, "cloud_name": cloud_name, "size": size})

    if pushed:
        _write_assets_lock(lock_path, lock_assets)

    if renderer.is_pretty():
        for entry in pushed:
            rprint(f"[green]✓[/green] pushed {entry['name']} → {entry['cloud_name']}")
        rprint(f"[dim]{len(pushed)} pushed, {skipped} skipped ({where_kind}) → {lock_path}[/dim]")
    renderer.emit(
        {
            "pushed": pushed,
            "current": current,
            "skipped": skipped,
            "lock": str(lock_path),
            "where": where_kind,
        },
        command="assets push",
        changed=bool(pushed),
    )


def _write_assets_lock(path: Path, assets: dict) -> None:
    """Atomically rewrite the lock (tmp + fsync + rename, the jobs_state
    pattern) so a crash mid-push can't leave a torn JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": ASSETS_LOCK_SCHEMA, "assets": assets}
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    fd = os.open(str(tmp), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _defaults(project: Project) -> dict:
    defaults = project.config.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _blueprint_names(project: Project) -> list[str]:
    bp_dir = project.root / "blueprints"
    if not bp_dir.is_dir():
        return []
    return sorted(p.name for p in bp_dir.glob("*.yaml") if p.is_file())


def _asset_entries(project: Project) -> list[dict]:
    """One entry per file under ``assets/`` (recursive, dotfiles skipped),
    joined against the push lock: ``pushed`` = name present in the lock,
    ``stale`` = pushed but the on-disk sha256 no longer matches it."""
    assets_dir = project.root / "assets"
    if not assets_dir.is_dir():
        return []
    lock = read_assets_lock(project)
    entries: list[dict] = []
    for path, name in _iter_asset_files(assets_dir):
        sha = _sha256_file(path)
        locked = lock.get(name)
        pushed = isinstance(locked, dict)
        stale = pushed and locked.get("sha256") != sha
        entries.append(
            {
                "name": name,
                "sha256": sha,
                "size": path.stat().st_size,
                "pushed": pushed,
                "stale": stale,
            }
        )
    return entries


def _render_status_pretty(data: dict) -> None:
    from rich.table import Table

    renderer = get_renderer()
    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style="dim", no_wrap=True)
    tbl.add_column(overflow="fold")
    tbl.add_row("root", data["root"])
    tbl.add_row("schema", data["schema"])
    if data["defaults"]:
        tbl.add_row("defaults", ", ".join(f"{k}={v}" for k, v in data["defaults"].items()))
    tbl.add_row("blueprints", "\n".join(data["blueprints"]) or "[dim]none[/dim]")
    if data["assets"]:
        lines = []
        for a in data["assets"]:
            flag = "stale" if a["stale"] else ("pushed" if a["pushed"] else "not pushed")
            lines.append(f"{a['name']}  [dim]({a['size']} B, {flag})[/dim]")
        tbl.add_row("assets", "\n".join(lines))
    else:
        tbl.add_row("assets", "[dim]none[/dim]")
    if data["recent_runs"]:
        runs = [f"{r.get('ts', '?')}  {r.get('cmd', '?')}" for r in data["recent_runs"][-5:]]
        tbl.add_row("recent runs", "\n".join(runs))
    for w in data["warnings"]:
        tbl.add_row("warning", f"[yellow]{w}[/yellow]")
    renderer.console().print(tbl)
