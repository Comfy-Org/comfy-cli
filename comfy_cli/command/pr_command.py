"""PR cache management commands.

This module provides CLI commands for managing the PR cache, including:
- Listing cached PR builds
- Cleaning specific or all cached builds
- Displaying cache information in a user-friendly format
"""

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from comfy_cli import tracking
from comfy_cli.pr_cache import PRCache

app = typer.Typer(help="Manage PR cache")
console = Console()


@app.command("list", help="List cached PR builds")
@tracking.track_command()
def list_cached() -> None:
    """List all cached PR builds."""
    cache = PRCache()
    cached_frontends = cache.list_cached_frontends()

    if not cached_frontends:
        rprint("[yellow]No cached PR builds found[/yellow]")
        return

    table = Table(title="Cached Frontend PR Builds")
    table.add_column("PR #", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Author", style="green")
    table.add_column("Age", style="yellow")
    table.add_column("Size (MB)", style="magenta")

    for info in cached_frontends:
        age = cache.get_cache_age(info.get("cached_at", ""))
        table.add_row(
            str(info.get("pr_number", "?")),
            info.get("pr_title", "Unknown")[:50],  # Truncate long titles
            info.get("user", "Unknown"),
            age,
            f"{info.get('size_mb', 0):.1f}",
        )

    console.print(table)

    # Show cache settings
    rprint(
        f"\n[dim]Cache settings: Max age: {cache.DEFAULT_MAX_CACHE_AGE_DAYS} days, "
        f"Max items: {cache.DEFAULT_MAX_CACHE_ITEMS}[/dim]"
    )


@app.command("clean", help="Clean PR cache")
@tracking.track_command()
def clean_cache(
    pr_number: int = typer.Argument(None, help="Specific PR number to clean (omit to clean all)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clean cached PR builds."""
    cache = PRCache()

    if pr_number:
        if not yes:
            confirm = typer.confirm(f"Remove cache for PR #{pr_number}?")
            if not confirm:
                rprint("[yellow]Cancelled[/yellow]")
                return
        cache.clean_frontend_cache(pr_number)
        rprint(f"[green]✓ Cleaned cache for PR #{pr_number}[/green]")
    else:
        if not yes:
            cached = cache.list_cached_frontends()
            if cached:
                rprint(f"[yellow]This will remove {len(cached)} cached PR build(s)[/yellow]")
                confirm = typer.confirm("Remove all cached PR builds?")
                if not confirm:
                    rprint("[yellow]Cancelled[/yellow]")
                    return
        cache.clean_frontend_cache()
        rprint("[green]✓ Cleaned all PR cache[/green]")
