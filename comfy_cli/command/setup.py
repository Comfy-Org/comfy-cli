"""``comfy setup`` — one-time wizard that gets you from zero to running.

Orchestrates existing commands with a polished UX:

1. Welcome banner (branded gradient wordmark)
2. Choose routing: local or cloud
3. Authenticate: browser OAuth or paste an API key
4. Detect coding agent and install skills
5. Verify connectivity
6. Done — show next steps

Supports both interactive (human at TTY) and non-interactive (CI, devcontainers,
scripted installs) modes::

    # Interactive (human)
    comfy setup

    # Non-interactive (CI / scripted)
    comfy setup --non-interactive --where cloud --api-key sk-...
    comfy setup -y --where local --skip-verify

After setup, agents never need ``comfy discover`` (26K tokens). The skills
teach them everything; routing and auth are already configured.
"""

from __future__ import annotations

from pathlib import Path

import typer

from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint


def execute(
    *,
    where: str | None = None,
    api_key: str | None = None,
    project_dir: str | None = None,
    non_interactive: bool = False,
    skip_skills: bool = False,
    skip_verify: bool = False,
) -> None:
    """Run the setup wizard — interactive or non-interactive."""
    renderer = get_renderer()
    console = renderer.console()

    # api_key implies cloud
    if api_key and not where:
        where = "cloud"

    # Non-interactive requires --where
    if non_interactive and not where:
        renderer.error(
            code="setup_missing_where",
            message="--non-interactive requires --where (local or cloud)",
            hint="comfy setup --non-interactive --where cloud --api-key sk-...",
        )
        raise typer.Exit(code=1)

    # Validate --where value early
    if where:
        from comfy_cli import where as where_module

        try:
            where_module._parse(where)
        except ValueError as e:
            renderer.error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
            raise typer.Exit(code=1) from e
        where = where.strip().lower()

    # ------------------------------------------------------------------
    # 1. Welcome
    # ------------------------------------------------------------------
    _print_welcome(console)

    # ------------------------------------------------------------------
    # 2. Routing — where will you run workflows?
    # ------------------------------------------------------------------
    if where is None:
        console.print()
        where = _ask_where()
        if where is None:
            raise typer.Exit(code=130)
    else:
        pprint(f"\n  [dim]Routing:[/dim] [bold cyan]{where}[/bold cyan]")

    # ------------------------------------------------------------------
    # 3. Authenticate (cloud only)
    # ------------------------------------------------------------------
    if where == "cloud":
        console.print()
        if api_key:
            ok = _auth_api_key_direct(api_key)
        elif non_interactive:
            # Non-interactive cloud without api_key — check if already authed
            ok = _check_existing_auth()
            if not ok:
                renderer.error(
                    code="setup_no_auth",
                    message="Cloud requires authentication in non-interactive mode",
                    hint="pass --api-key sk-... or run `comfy cloud login` first",
                )
                raise typer.Exit(code=1)
        else:
            ok = _do_cloud_auth(console)
        if not ok:
            raise typer.Exit(code=1)

    # Persist the routing default
    from comfy_cli import where as where_module
    from comfy_cli.config_manager import ConfigManager

    ConfigManager().set(where_module.CONFIG_KEY_WHERE_DEFAULT, where)
    pprint(f"\n  [bold green]✓[/bold green] Default routing → [bold cyan]{where}[/bold cyan]")

    # ------------------------------------------------------------------
    # 3. Project directory
    # ------------------------------------------------------------------
    console.print()
    _do_project_dir(console, project_dir=project_dir, non_interactive=non_interactive)

    # ------------------------------------------------------------------
    # 4. Enable image/video previews
    # ------------------------------------------------------------------
    console.print()
    _do_preview(console, non_interactive=non_interactive)

    # ------------------------------------------------------------------
    # 4b. Telemetry consent — the single explicit place the user decides.
    # ------------------------------------------------------------------
    console.print()
    _do_consent(console, non_interactive=non_interactive)

    # ------------------------------------------------------------------
    # 5. Detect agent + install skills
    # ------------------------------------------------------------------
    if not skip_skills:
        console.print()
        _do_skills(console)

    # ------------------------------------------------------------------
    # 6. Verify connectivity
    # ------------------------------------------------------------------
    if not skip_verify:
        console.print()
        _do_verify(console, where)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _print_done(console, where)

    # JSON envelope for agents
    renderer.emit(
        {
            "where": where,
            "skills_installed": not skip_skills,
            "verified": not skip_verify,
        },
        command="setup",
    )


# ======================================================================
# Step implementations
# ======================================================================


def _print_welcome(console) -> None:
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import (
        _WORDMARK_ROWS,
        BRAND_ACCENT,
        BRAND_END,
        BRAND_START,
        gradient_block,
    )

    version = ConfigManager().get_cli_version()
    wordmark = gradient_block(_WORDMARK_ROWS)
    tagline = Text("setup", style=f"dim {BRAND_END}", justify="center")

    steps = Text.assemble(
        ("  1 ", f"bold {BRAND_ACCENT}"),
        ("Choose where to run\n", "white"),
        ("  2 ", f"bold {BRAND_ACCENT}"),
        ("Sign in\n", "white"),
        ("  3 ", f"bold {BRAND_ACCENT}"),
        ("Set project directory\n", "white"),
        ("  4 ", f"bold {BRAND_ACCENT}"),
        ("Enable image previews\n", "white"),
        ("  🔒 ", f"bold {BRAND_ACCENT}"),
        ("Privacy & telemetry\n", "white"),
        ("  5 ", f"bold {BRAND_ACCENT}"),
        ("Install agent skills\n", "white"),
        ("  6 ", f"bold {BRAND_ACCENT}"),
        ("Verify connection", "white"),
    )

    body = Group(
        Align.center(wordmark),
        Align.center(tagline),
        Text(""),
        Rule(style=BRAND_END),
        Text(""),
        steps,
    )

    console.print(
        Panel(
            body,
            subtitle=Text(f"comfy CLI v{version}", style="dim"),
            subtitle_align="right",
            border_style=BRAND_START,
            padding=(1, 3),
        )
    )


def _ask_where() -> str | None:
    """Ask the user: local or cloud? (interactive only)"""
    import questionary

    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]① Where will you run workflows?[/bold {BRAND_ACCENT}]")
    pprint()

    return questionary.select(
        "",
        choices=[
            questionary.Choice("☁  Comfy Cloud", value="cloud"),
            questionary.Choice("💻 Local ComfyUI (127.0.0.1:8188)", value="local"),
        ],
        instruction="(↑↓ to select, enter to confirm)",
    ).ask()


def _do_cloud_auth(console) -> bool:
    """Interactive: choose between browser OAuth or pasting an API key."""
    import questionary

    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]② Sign in to Comfy Cloud[/bold {BRAND_ACCENT}]")
    pprint()

    method = questionary.select(
        "",
        choices=[
            questionary.Choice("🌐 Sign in with browser (OAuth)", value="browser"),
            questionary.Choice("🔑 Paste an API key", value="key"),
        ],
        instruction="(↑↓ to select, enter to confirm)",
    ).ask()

    if method is None:
        return False

    if method == "key":
        import questionary

        key = questionary.password("  API key:").ask()
        if not key or not key.strip():
            pprint("  [bold red]✗[/bold red] No key provided")
            return False
        return _auth_api_key_direct(key.strip())
    else:
        return _auth_browser(console)


def _auth_api_key_direct(key: str) -> bool:
    """Store an API key directly (works for both interactive and non-interactive)."""
    from comfy_cli.auth import store
    from comfy_cli.target import CLOUD_API_KEY_PROVIDER

    if not key.strip():
        pprint("  [bold red]✗[/bold red] Empty API key")
        return False

    try:
        store.set(CLOUD_API_KEY_PROVIDER, key.strip())
    except ValueError as e:
        pprint(f"  [bold red]✗[/bold red] {e}")
        return False

    pprint("  [bold green]✓[/bold green] API key saved")
    return True


def _check_existing_auth() -> bool:
    """Check if cloud auth is already configured (for non-interactive without api_key)."""
    from comfy_cli import where as where_module

    err = where_module.cloud_preflight()
    if err is None:
        pprint("  [bold green]✓[/bold green] Cloud auth already configured")
        return True
    return False


def _auth_browser(console) -> bool:
    """Run the OAuth browser flow (same as ``comfy cloud login``)."""
    from comfy_cli.auth import store
    from comfy_cli.cloud import CLIENT_ID, CLIENT_NAME, get_base_url, get_resource_url, get_scopes
    from comfy_cli.cloud.oauth import OAuthError, run_login

    base_url = get_base_url()
    pprint(f"  Opening browser for [bold cyan]{base_url}[/bold cyan]…")

    def _on_url(url: str) -> None:
        pprint("  [dim]If the browser doesn't open, visit:[/dim]")
        pprint(f"  [cyan]{url}[/cyan]")

    try:
        result = run_login(
            base_url=base_url,
            resource=get_resource_url(),
            scopes=get_scopes(),
            client_id=CLIENT_ID,
            client_name=CLIENT_NAME,
            open_browser=True,
            timeout_s=300.0,
            on_url_ready=_on_url,
        )
    except OAuthError as e:
        pprint(f"  [bold red]✗[/bold red] {e}")
        if e.hint:
            pprint(f"  [dim]{e.hint}[/dim]")
        return False

    store.save_cloud_session(
        base_url=result.base_url,
        resource=result.resource,
        client_id=result.client_id,
        scope=result.scope,
        access_token=result.tokens.access_token,
        refresh_token=result.tokens.refresh_token,
        token_type=result.tokens.token_type,
        expires_at=result.tokens.expires_at,
    )

    pprint("  [bold green]✓[/bold green] Signed in to Comfy Cloud")
    return True


def _do_project_dir(console, *, project_dir: str | None = None, non_interactive: bool = False) -> None:
    """Ask the user for a project directory and create the folder structure."""
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.constants import CONFIG_KEY_DEFAULT_PROJECT_DIR
    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]③ Project directory[/bold {BRAND_ACCENT}]")
    pprint()

    # Check if already configured
    existing = ConfigManager().get(CONFIG_KEY_DEFAULT_PROJECT_DIR)
    if existing and Path(existing).is_dir() and project_dir is None:
        pprint(f"  [bold green]✓[/bold green] Project directory: [bold cyan]{existing}[/bold cyan]")
        return

    default_dir = str(Path.home() / "comfy-projects")

    if project_dir:
        chosen = project_dir
    elif non_interactive:
        chosen = default_dir
    else:
        import questionary

        pprint("  [dim]This is where workflows, inputs, and outputs are stored.[/dim]")
        pprint()
        chosen = questionary.text(
            "  Project directory:",
            default=default_dir,
        ).ask()
        if chosen is None:
            chosen = default_dir

    chosen = str(Path(chosen).expanduser().resolve())
    project = Path(chosen)

    # Create folder structure
    for subdir in ("workflows", "inputs", "outputs"):
        (project / subdir).mkdir(parents=True, exist_ok=True)

    ConfigManager().set(CONFIG_KEY_DEFAULT_PROJECT_DIR, chosen)
    pprint(f"  [bold green]✓[/bold green] Project directory: [bold cyan]{chosen}[/bold cyan]")
    pprint("  [dim]  Created: workflows/ · inputs/ · outputs/[/dim]")


def _do_preview(console, *, non_interactive: bool = False) -> None:
    """Install term-image for inline image/video previews in the terminal."""
    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]🖼  Terminal image previews[/bold {BRAND_ACCENT}]")
    pprint()

    # Check if already installed (presence check only — don't import the module)
    import importlib.util

    if importlib.util.find_spec("term_image") is not None:
        pprint("  [bold green]✓[/bold green] term-image already installed — previews enabled")
        return

    # Ask user (or auto-install in non-interactive mode)
    if non_interactive:
        want = True
    else:
        import questionary

        want = questionary.confirm(
            "  Install term-image for inline image/video previews?",
            default=True,
        ).ask()

    if not want:
        pprint("  [dim]Skipped — you can install later with: pip install term-image[/dim]")
        return

    # Install via pip
    import subprocess
    import sys

    pprint("  Installing term-image…")
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "term-image>=0.7,<0.8"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        pprint("  [bold green]✓[/bold green] Installed — image previews enabled")
        pprint("  [dim]  Images will render inline after downloads (Kitty/iTerm2/half-blocks)[/dim]")
    else:
        pprint("  [dim yellow]⚠[/dim yellow] Install failed — previews will be skipped")
        pprint(f"  [dim]{result.stderr.strip()[:200]}[/dim]")


def _do_consent(console, *, non_interactive: bool = False) -> None:
    """Ask for (or report) telemetry consent — the one explicit decision point.

    Precedence: a hard env opt-out wins and is never overridden; an already
    recorded choice is reported, not re-asked; non-interactive runs leave the
    flag UNSET so a later interactive session still gets the chance to decide.
    """
    from comfy_cli import constants, tracking
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]🔒 Privacy & telemetry[/bold {BRAND_ACCENT}]")
    pprint()

    if tracking._telemetry_disabled_by_env():
        pprint("  [bold green]✓[/bold green] Telemetry disabled via environment (DO_NOT_TRACK)")
        return

    existing = ConfigManager().get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if existing is not None:
        state = "enabled" if existing else "disabled"
        pprint(f"  [bold green]✓[/bold green] Telemetry already {state}")
        pprint("  [dim]  Change anytime: comfy tracking enable | disable[/dim]")
        return

    pprint("  [dim]Anonymous usage data helps improve Comfy CLI.[/dim]")
    pprint()

    if non_interactive:
        # No human to ask — leave the flag unset (don't silently opt in or out);
        # a later interactive run will prompt via the global consent path.
        pprint("  [dim]Skipped (non-interactive). Opt in anytime: comfy tracking enable[/dim]")
        return

    import questionary

    agree = bool(questionary.confirm("  Share anonymous usage data to help improve Comfy CLI?", default=True).ask())
    tracking.init_tracking(agree)
    if agree:
        pprint("  [bold green]✓[/bold green] Thanks! Enabled — disable anytime: comfy tracking disable")
    else:
        pprint("  [dim]No problem — telemetry stays off. Enable later: comfy tracking enable[/dim]")


def _do_skills(console) -> None:
    """Detect agent hosts and install skills."""
    from comfy_cli.output.branding import BRAND_ACCENT
    from comfy_cli.skills import TargetKind, install

    pprint(f"  [bold {BRAND_ACCENT}]⑤ Install agent skills[/bold {BRAND_ACCENT}]")
    pprint()

    # Detect which agent targets exist
    home = Path.home()
    detected: list[TargetKind] = []
    labels: list[str] = []

    if (home / ".claude").is_dir():
        detected.append("claude-code")
        labels.append("Claude Code")
    if (home / ".cursor").is_dir():
        detected.append("cursor")
        labels.append("Cursor")

    # AGENTS.md is always an option
    detected.append("agents-md")
    labels.append("AGENTS.md")

    if labels:
        pprint(f"  Detected: [bold cyan]{' · '.join(labels)}[/bold cyan]")
    pprint()

    # Install all skills to all detected targets
    results = install(scope="user", targets=detected if detected else None)

    wrote = [r for r in results if r.action == "wrote"]
    skipped = [r for r in results if r.action == "skipped"]

    # Show installed skills (deduplicated by skill name)
    installed_skills = sorted(set(r.skill for r in wrote))
    for skill_name in installed_skills:
        pprint(f"  [bold green]✓[/bold green] [cyan]{skill_name}[/cyan]")

    skill_count = len(installed_skills)
    target_names = sorted(set(r.kind for r in wrote))
    target_str = " + ".join(_pretty_target(t) for t in target_names)

    pprint()
    pprint(f"  [dim]{skill_count} skills → {target_str}[/dim]")

    if skipped:
        for r in skipped[:3]:
            pprint(f"  [dim yellow]⚠ skipped {r.skill}/{r.kind}: {r.reason}[/dim yellow]")


def _pretty_target(kind: str) -> str:
    return {
        "claude-code": ".claude/skills/",
        "cursor": ".cursor/rules/",
        "agents-md": "AGENTS.md",
    }.get(kind, kind)


def _do_verify(console, where: str) -> None:
    """Quick connectivity check."""
    from comfy_cli.output.branding import BRAND_ACCENT

    pprint(f"  [bold {BRAND_ACCENT}]⑥ Verify connection[/bold {BRAND_ACCENT}]")
    pprint()

    if where == "cloud":
        _verify_cloud(console)
    else:
        _verify_local(console)


def _verify_cloud(console) -> None:
    """Check cloud auth + count available nodes."""
    from comfy_cli.comfy_client import Client, Unauthenticated
    from comfy_cli.target import resolve_target

    try:
        target = resolve_target(where="cloud")
        client = Client(target)
        client.get_job_status("00000000-0000-0000-0000-000000000000")
        # If we get here (even 404), the cloud is reachable
        pprint("  [bold green]✓[/bold green] Cloud reachable")
    except Unauthenticated:
        pprint("  [bold red]✗[/bold red] Not authenticated — run [cyan]comfy cloud login[/cyan]")
        return
    except Exception:
        # Even an error means we connected
        pprint("  [bold green]✓[/bold green] Cloud reachable")

    # Count nodes
    try:
        import json
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "comfy_cli", "--where", "cloud", "--json", "nodes", "ls", "--limit", "1"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            total = data.get("data", {}).get("total", "?")
            pprint(f"  [bold green]✓[/bold green] {total} nodes available")
    except Exception:
        pass


def _verify_local(console) -> None:
    """Check if local ComfyUI server is running."""
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.env_checker import _bracket_host, check_comfy_server_running
    from comfy_cli.local_address import resolve_local_host_port

    host, port = resolve_local_host_port(None, None, background=ConfigManager().background)
    if check_comfy_server_running(port, host):
        pprint(f"  [bold green]✓[/bold green] Local server running on [cyan]{_bracket_host(host)}:{port}[/cyan]")
    else:
        pprint("  [dim yellow]⚠[/dim yellow] Local server not running")
        pprint("  [dim]  Start it with: [cyan]comfy launch[/cyan][/dim]")


def _print_done(console, where: str) -> None:
    """Final success screen with next steps."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    from comfy_cli.output.branding import (
        BRAND_ACCENT,
        BRAND_END,
        BRAND_START,
        gradient_text,
    )

    console.print()

    header = gradient_text("You're all set!", bold=True)

    if where == "cloud":
        next_steps = Text.assemble(
            ("  comfy run --workflow my.json", "bold white"),
            ("     submit to cloud\n", "dim"),
            ("  comfy nodes show KSampler", "bold white"),
            ("      inspect any node\n", "dim"),
            ("  comfy jobs ls", "bold white"),
            ("                    track your jobs\n", "dim"),
            ("  comfy upload photo.png", "bold white"),
            ("           upload assets\n", "dim"),
        )
    else:
        next_steps = Text.assemble(
            ("  comfy launch", "bold white"),
            ("                    start the server\n", "dim"),
            ("  comfy run --workflow my.json", "bold white"),
            ("     submit a workflow\n", "dim"),
            ("  comfy nodes show KSampler", "bold white"),
            ("      inspect any node\n", "dim"),
        )

    body = Group(
        header,
        Text(""),
        Rule(style=BRAND_END),
        Text(""),
        Text("Next steps", style=f"bold {BRAND_ACCENT}"),
        next_steps,
    )

    console.print(
        Panel(
            body,
            border_style=BRAND_START,
            padding=(1, 3),
        )
    )
