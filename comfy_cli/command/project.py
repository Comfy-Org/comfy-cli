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
from pathlib import Path

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint
from comfy_cli.project import (
    CONVENTIONAL_DIRS,
    PROJECT_MARKER,
    PROJECT_SCHEMA,
    Project,
    find_project,
    read_journal,
    unknown_dirs,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Project conventions: init and status.",
)

# The marker `comfy project init` writes — deliberately literal and minimal.
MARKER_CONTENT = "schema: project/1\ndefaults:\n  where: cloud\n"

ASSETS_LOCK_SCHEMA = "assets-lock/1"


@app.command("init", help="Initialize the project/1 convention here: comfy.yaml marker + conventional dirs.")
@tracking.track_command("project")
def init_cmd():
    renderer = get_renderer()
    cwd = Path.cwd().resolve()

    existing = find_project(cwd)
    if existing is not None:
        renderer.error(
            code="project_already_exists",
            message=f"This directory is already governed by the project at {existing.root}.",
            hint="use the existing project (edit its comfy.yaml), or init outside it",
            details={"root": str(existing.root)},
        )
        raise typer.Exit(code=1)

    (cwd / PROJECT_MARKER).write_text(MARKER_CONTENT, encoding="utf-8")
    created = [PROJECT_MARKER]
    for d in CONVENTIONAL_DIRS:
        (cwd / d).mkdir(exist_ok=True)
        created.append(f"{d}/")

    if renderer.is_pretty():
        rprint(f"[bold green]Initialized project[/bold green] at {cwd}")
        for name in created:
            rprint(f"  [dim]created[/dim] {name}")
    renderer.emit(
        {"root": str(cwd), "created": created, "action": "init"},
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


def _read_assets_lock(project: Project) -> dict:
    """The ``assets-lock/1`` map ``{name: {sha256, cloud_name, …}}`` written
    by ``comfy assets push``; ``{}`` when absent or malformed."""
    path = project.root / ".comfy" / "assets.lock.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if isinstance(parsed, dict) and parsed.get("schema") == ASSETS_LOCK_SCHEMA and isinstance(parsed.get("assets"), dict):
        return parsed["assets"]
    return {}


def _asset_entries(project: Project) -> list[dict]:
    """One entry per file under ``assets/`` (recursive, dotfiles skipped),
    joined against the push lock: ``pushed`` = name present in the lock,
    ``stale`` = pushed but the on-disk sha256 no longer matches it."""
    assets_dir = project.root / "assets"
    if not assets_dir.is_dir():
        return []
    lock = _read_assets_lock(project)
    entries: list[dict] = []
    for path in sorted(assets_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(assets_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        name = rel.as_posix()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
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
