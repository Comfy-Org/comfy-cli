"""``comfy skills`` — install the agent skills so Claude / Cursor / Aider can drive comfy directly.

The unlock: instead of running an MCP server, this command teaches every
agent on the machine how to call ``comfy`` natively. One file per skill,
three targets, zero protocol.

Bundled skills (6 total) — see ``comfy skills list`` for descriptions:

  - ``comfy``           — the consolidated driver skill (command surface,
                          output contract, routing, discovery, execution,
                          image, video, audio, cloud, edit, condition, pipeline)
  - ``comfy-debug``     — debugging when workflows fail or jobs hang
  - ``comfy-relay``     — what to put in chat while driving the CLI
  - ``comfy-director``  — narrative multi-shot video production (screenplay,
                          continuity, audio design, conform discipline)
  - ``comfy-build``     — building a custom ComfyUI environment on the developer
                          platform, versioned with this CLI release
  - ``comfy-deploy``    — running a Build release as a serverless deployment:
                          compute choice, worker bounds, submitting workflows,
                          and what keeps costing money

Reference skills (``REFERENCE_SKILLS``) are the other half: ``show`` resolves
them and no default install writes them, so a parent skill can cite one by name
and load its depth only on the tasks that need it. A bare name passed to
``install`` is redirected to ``show``; an explicit path, or a direct
``install()`` call, still writes one, and ``uninstall <name>`` removes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from comfy_cli import knowledge, tracking
from comfy_cli.output import get_renderer, rprint
from comfy_cli.skills import (
    BUNDLED_SKILLS,
    REFERENCE_SKILLS,
    TargetKind,
    _compute_skill_state,
    _looks_like_path,
    default_skill_names,
    frontmatter_description,
    load_skill_source,
    plan_install,
    read_manifest,
    readable_skill_names,
    reference_skill_names,
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


def _validate_skills(skills: list[str] | None, *, refuse_reference: bool = True) -> list[str] | None:
    """Validate ``--skill`` tokens for install or uninstall.

    ``refuse_reference`` is the asymmetry between the two verbs. A bare
    reference name passed to ``install`` is nearly always the right intent with
    the wrong verb, so it is redirected to ``show`` rather than written. The
    same name passed to ``uninstall`` is unambiguous — it can only mean a file
    already on disk — so it is accepted. Refusing there would strand anything
    an explicit install had put down, which is the one case where naming it is
    the only way to remove it.
    """
    if not skills:
        return None
    renderer = get_renderer()
    known = default_skill_names() if refuse_reference else readable_skill_names()
    for s in skills:
        if _looks_like_path(s):
            # Path-based token: validate it eagerly so we fail fast before any writes.
            # A path is an explicit request, including for a reference skill, so it
            # is honoured rather than redirected — see the branch below.
            try:
                load_skill_source(s)
            except ValueError as e:
                renderer.error(
                    code="skill_invalid",
                    message=str(e),
                    hint="a skill is a directory named after the skill containing SKILL.md with `name:` and `description:` frontmatter; check with `comfy skills validate <path>`",
                )
                raise typer.Exit(code=1) from e
        elif refuse_reference and s in reference_skill_names():
            # A nudge, not a boundary: the reach is for the material, not for a
            # file that would then sit in every agent's context on every task.
            # Anyone who does want it installed says so by path or via `install()`.
            raise typer.BadParameter(
                f"{s!r} is a reference skill, which is read on demand rather than installed; "
                f"run `comfy skills show {s}` to read it"
            )
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
            help=f"Install only the named skill(s). Repeatable. Default: all {len(default_skill_names())} "
            "(see `comfy skills list`).",
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
    if not dry_run:
        # The skills being written are what reads the bundle; this is the one
        # command whose whole job is setting that up, so it pays for the fetch.
        knowledge.refresh_if_stale()

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
        # One line per skipped target: every remaining skip is a per-path OSError,
        # so two targets of one skill fail with two different reasons and the table
        # has nowhere to put either.
        for r in results:
            if r.action == "skipped" and r.reason:
                body = Group(body, Text(f"{r.skill} ({r.kind}) skipped: {r.reason}", style="yellow"))
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
        typer.Option(
            "--skill",
            help="Uninstall only the named skill(s), including a reference skill something "
            "installed explicitly. Default: all bundled.",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    renderer = get_renderer()
    s = _scope(scope)
    kinds = _kinds(target)
    skills = _validate_skills(skill, refuse_reference=False)
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


_LIST_HELP = (
    "List the skills `comfy skills install` writes ("
    + ", ".join(default_skill_names())
    + "), plus the reference skills `comfy skills show` reads on demand."
)


def _skill_rows(catalog: tuple[tuple[str, str], ...]) -> list[dict]:
    return [{"name": name, "description": frontmatter_description(skill_content(name))} for name, _ in catalog]


def _skill_table(rows: list[dict], *, heading: str):
    from rich.table import Table

    tbl = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        pad_edge=False,
        expand=True,
        title=heading,
        title_justify="left",
    )
    tbl.add_column("Skill", style="bold cyan", no_wrap=True)
    tbl.add_column("Description", style="white", overflow="fold")
    for r in rows:
        tbl.add_row(r["name"], r["description"])
    return tbl


@app.command("list", help=_LIST_HELP)
@tracking.track_command("skill")
def list_cmd():
    renderer = get_renderer()
    rows = _skill_rows(BUNDLED_SKILLS)
    reference_rows = _skill_rows(REFERENCE_SKILLS)

    if renderer.is_pretty():
        from rich.console import Group

        _print_skill_panel(
            "skill list",
            Group(
                _skill_table(rows, heading="Installed by `comfy skills install`"),
                "",
                _skill_table(reference_rows, heading="Read on demand with `comfy skills show <name>`"),
            ),
        )
    renderer.emit({"skills": rows, "reference_skills": reference_rows}, command="skill list")


@app.command("show", help="Print a bundled or reference SKILL.md to stdout (default: comfy).")
@tracking.track_command("skill")
def show_cmd(
    name: Annotated[
        str,
        typer.Argument(
            help="Which skill to print — any installed skill, or a reference skill a parent cites "
            "(see `comfy skills list` for both)."
        ),
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
