"""``comfy skills`` — install the agent skills so Claude / Cursor / Aider can drive comfy directly.

The unlock: instead of running an MCP server, this command teaches every
agent on the machine how to call ``comfy`` natively. One file per skill,
three targets, zero protocol.

Bundled skills (5 total) — see ``comfy skills list`` for descriptions:

  - ``comfy``           — the consolidated driver skill (command surface,
                          output contract, routing, discovery, execution,
                          image, video, audio, cloud, edit, condition, pipeline)
  - ``comfy-fragments`` — typed reusable workflow fragments + YAML blueprint composition
  - ``comfy-debug``     — debugging when workflows fail or jobs hang
  - ``comfy-relay``     — what to put in chat while driving the CLI
  - ``comfy-director``  — narrative multi-shot video production (screenplay,
                          continuity, audio design, conform discipline)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint
from comfy_cli.skills import (
    BUNDLED_SKILLS,
    TargetKind,
    _compute_skill_state,
    bundled_skill_names,
    load_skill_source,
    plan_install,
    read_manifest,
    skill_content,
)
from comfy_cli.skills import (
    install as _install,
)
from comfy_cli.skills import (
    prune_retired as _prune_retired,
)
from comfy_cli.skills import (
    uninstall as _uninstall,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Install the comfy agent skills — Claude, Cursor, Aider, and any AGENTS.md-aware tool.",
)


def _print_skill_panel(title: str, body) -> None:
    """Helper: render any ``skill <verb>`` body inside the canonical
    branded panel so install / uninstall / list / status all share chrome."""
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import branded_panel

    get_renderer().console().print(branded_panel(body, title=title, version=ConfigManager().get_cli_version()))


def _kinds(targets: list[str] | None) -> list[TargetKind] | None:
    if not targets:
        return None
    valid: list[TargetKind] = []
    for t in targets:
        if t not in ("claude-code", "cursor", "agents-md"):
            raise typer.BadParameter(f"unknown target {t!r}; pick claude-code, cursor, or agents-md")
        valid.append(t)  # type: ignore[arg-type]
    return valid


def _validate_skills(skills: list[str] | None) -> list[str] | None:
    if not skills:
        return None
    import os

    renderer = get_renderer()
    known = bundled_skill_names()
    for s in skills:
        p = Path(s).expanduser()
        looks_like_path = os.sep in s or s.startswith((".", "~")) or p.exists()
        if looks_like_path:
            # Path-based token: validate it eagerly so we fail fast before any writes.
            try:
                load_skill_source(s)
            except ValueError as e:
                renderer.error(
                    code="skill_invalid",
                    message=str(e),
                    hint="a skill is a directory named after the skill containing SKILL.md with `name:` and `description:` frontmatter; check with `comfy skills validate <path>`",
                )
                raise typer.Exit(code=1) from e
        elif s not in known:
            raise typer.BadParameter(f"unknown skill {s!r}; choices: {', '.join(known)}")
    return skills


def _scope(scope: str) -> Literal["user", "project"]:
    if scope not in ("user", "project"):
        raise typer.BadParameter("scope must be 'user' or 'project'")
    return scope  # type: ignore[return-value]


_PROJECT_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "AGENTS.md")


def _ensure_project_root(cwd: Path) -> None:
    """Refuse `--scope project` when cwd doesn't look like a project root."""
    for marker in _PROJECT_MARKERS:
        if (cwd / marker).exists():
            return
    raise typer.BadParameter(
        f"--scope project from {cwd} doesn't look like a project root "
        f"(no {', '.join(_PROJECT_MARKERS)}). cd into your project first, "
        "or run `touch AGENTS.md` to mark this directory."
    )


@app.command("install", help="Write the comfy skills into Claude Code, Cursor, and AGENTS.md.")
@tracking.track_command("skill")
def install_cmd(
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            help="user → ~/.claude/ ~/.cursor/ ~/AGENTS.md (default); project → ./.claude/ ./.cursor/ ./AGENTS.md.",
        ),
    ] = "user",
    target: Annotated[
        list[str] | None,
        typer.Option(
            "--target",
            help="Install only the named target(s). Repeatable. Default: all three.",
        ),
    ] = None,
    skill: Annotated[
        list[str] | None,
        typer.Option(
            "--skill",
            help="Install only the named skill(s). Repeatable. Default: all 4 bundled skills (see `comfy skills list`).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be written without touching anything."),
    ] = False,
):
    renderer = get_renderer()
    s = _scope(scope)
    cwd = Path.cwd()
    if s == "project":
        _ensure_project_root(cwd)
    kinds = _kinds(target)
    skills = _validate_skills(skill)
    # Prune skills we've since retired so old machines converge on every install.
    # Surface only the ones that actually changed — `absent` is a non-event.
    prune_results = [
        r for r in _prune_retired(scope=s, targets=kinds, dry_run=dry_run, project_root=cwd) if r.action != "absent"
    ]
    results = prune_results + _install(scope=s, targets=kinds, skills=skills, dry_run=dry_run, project_root=cwd)

    if renderer.is_pretty():
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        title_word = "would install" if dry_run else "installed"
        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        tbl.add_column("Skill", style="bold cyan", no_wrap=True)
        tbl.add_column("Target", style="bold white", no_wrap=True)
        tbl.add_column("Action", no_wrap=True)
        tbl.add_column("Path", style="dim", overflow="fold")
        for r in results:
            action_style = {
                "wrote": "bold green",
                "would_write": "yellow",
                "skipped": "red",
                "absent": "dim",
                "removed": "yellow",
                "would_remove": "yellow",
            }.get(r.action, "white")
            tbl.add_row(r.skill, r.kind, f"[{action_style}]{r.action}[/{action_style}]", str(r.path))

        header = Text(f"{title_word} · {s} scope", style="dim")
        body = Group(header, Text(""), tbl)
        if not dry_run and any(r.action == "wrote" for r in results):
            body = Group(
                body,
                Text(""),
                Text(
                    "Done. Restart Claude Code / Cursor to pick up the new skills.",
                    style="bold green",
                ),
            )
        _print_skill_panel("skill install", body)
    renderer.emit(
        {
            "scope": s,
            "dry_run": dry_run,
            "results": [r.to_dict() for r in results],
        },
        command="skill install",
        changed=not dry_run,
    )


@app.command("uninstall", help="Remove the comfy skills from Claude Code, Cursor, and AGENTS.md.")
@tracking.track_command("skill")
def uninstall_cmd(
    scope: Annotated[str, typer.Option("--scope")] = "user",
    target: Annotated[list[str] | None, typer.Option("--target")] = None,
    skill: Annotated[
        list[str] | None,
        typer.Option("--skill", help="Uninstall only the named skill(s). Default: all bundled."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    renderer = get_renderer()
    s = _scope(scope)
    kinds = _kinds(target)
    skills = _validate_skills(skill)
    results = _uninstall(scope=s, targets=kinds, skills=skills, dry_run=dry_run, project_root=Path.cwd())

    if renderer.is_pretty():
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        tbl.add_column("Skill", style="bold cyan", no_wrap=True)
        tbl.add_column("Target", style="bold white", no_wrap=True)
        tbl.add_column("Action", no_wrap=True)
        tbl.add_column("Path", style="dim", overflow="fold")
        for r in results:
            style = {
                "removed": "yellow",
                "would_remove": "yellow",
                "absent": "dim",
                "skipped": "red",
            }.get(r.action, "white")
            tbl.add_row(r.skill, r.kind, f"[{style}]{r.action}[/{style}]", str(r.path))

        header = Text(f"{s} scope", style="dim")
        _print_skill_panel("skill uninstall", Group(header, Text(""), tbl))
    renderer.emit(
        {
            "scope": s,
            "dry_run": dry_run,
            "results": [r.to_dict() for r in results],
        },
        command="skill uninstall",
        changed=not dry_run,
    )


_LIST_HELP = "List the bundled skills (" + ", ".join(name for name, _ in BUNDLED_SKILLS) + ")."


@app.command("list", help=_LIST_HELP)
@tracking.track_command("skill")
def list_cmd():
    renderer = get_renderer()
    rows = []
    for name, _subdir in BUNDLED_SKILLS:
        # Pull the description out of the SKILL.md frontmatter, if any.
        text = skill_content(name)
        desc = ""
        if text.startswith("---\n"):
            _, _, rest = text.partition("---\n")
            front, _, _ = rest.partition("---\n")
            for line in front.splitlines():
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
                    break
        rows.append({"name": name, "description": desc})

    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        tbl.add_column("Skill", style="bold cyan", no_wrap=True)
        tbl.add_column("Description", style="white", overflow="fold")
        for r in rows:
            tbl.add_row(r["name"], r["description"])
        _print_skill_panel("skill list", tbl)
    renderer.emit({"skills": rows}, command="skill list")


@app.command("show", help="Print a bundled SKILL.md to stdout (default: comfy).")
@tracking.track_command("skill")
def show_cmd(
    name: Annotated[
        str,
        typer.Argument(help="Which bundled skill to print (see `comfy skills list` for all 4)."),
    ] = "comfy",
):
    renderer = get_renderer()
    try:
        content = skill_content(name)
    except ValueError as e:
        renderer.error(code="unknown_skill", message=str(e), hint="run `comfy skills list`")
        raise typer.Exit(code=1) from e
    if renderer.is_pretty():
        from rich.markdown import Markdown

        renderer.console().print(Markdown(content))
        return
    renderer.emit({"name": name, "content": content}, command="skill show")


@app.command("status", help="Show which skills are installed across Claude Code / Cursor / AGENTS.md.")
@tracking.track_command("skill")
def status_cmd(
    scope: Annotated[str, typer.Option("--scope")] = "user",
):
    renderer = get_renderer()
    s = _scope(scope)
    plans = plan_install(scope=s, project_root=Path.cwd())
    manifest = read_manifest()
    rows = []
    for p in plans:
        state = _compute_skill_state(p.path, p.skill, manifest)
        rows.append(
            {
                "skill": p.skill,
                "kind": p.kind,
                "scope": p.scope,
                "path": str(p.path),
                "installed": p.exists,
                "state": state,
            }
        )
    if renderer.is_pretty():
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        tbl.add_column("Skill", style="bold cyan", no_wrap=True)
        tbl.add_column("Target", style="bold white", no_wrap=True)
        tbl.add_column("State", no_wrap=True)
        tbl.add_column("Path", style="dim", overflow="fold")
        _STATE_STYLES = {
            "current": "[bold green]current[/bold green]",
            "stale": "[yellow]stale[/yellow]",
            "modified": "[bold yellow]modified[/bold yellow]",
            "missing": "[dim]missing[/dim]",
            "unmanaged": "[cyan]unmanaged[/cyan]",
        }
        for r in rows:
            badge = _STATE_STYLES.get(r["state"], r["state"])
            tbl.add_row(r["skill"], r["kind"], badge, r["path"])
        header = Text(f"{s} scope", style="dim")
        _print_skill_panel("skill status", Group(header, Text(""), tbl))
    renderer.emit({"scope": s, "targets": rows}, command="skill status")


@app.command("validate", help="Validate a skill directory or SKILL.md against the format contract.")
@tracking.track_command("skill")
def validate_cmd(
    path: Annotated[str, typer.Argument(help="Skill dir or SKILL.md path.")],
):
    renderer = get_renderer()
    try:
        src = load_skill_source(path)
    except ValueError as e:
        renderer.error(
            code="skill_invalid",
            message=str(e),
            hint="a skill is a directory named after the skill containing SKILL.md with `name:` and `description:` frontmatter",
        )
        raise typer.Exit(code=1) from e
    payload = {"valid": True, "name": src.name, "bundled": src.bundled, "path": path}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] {src.name} is a valid skill")
    renderer.emit(payload, command="skills validate")
