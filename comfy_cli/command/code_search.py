"""Code search commands for searching ComfyUI codebases.

This module provides CLI commands for searching code across ComfyUI
repositories using the comfy-codesearch API powered by Sourcegraph.
"""

import json
from typing import Annotated

import requests
import typer
from rich.console import Console
from rich.text import Text

from comfy_cli import tracking

app = typer.Typer()
console = Console()

API_URL = "https://comfy-codesearch.vercel.app/api/search/code"
DEFAULT_COUNT = 20
REQUEST_TIMEOUT = 30


def _build_query(query: str, repo: str | None, count: int) -> str:
    parts = []
    if repo:
        # Support both "ComfyUI" and "Comfy-Org/ComfyUI" forms
        if "/" not in repo:
            repo = f"Comfy-Org/{repo}"
        parts.append(f"repo:{repo}")
    if count != DEFAULT_COUNT:
        parts.append(f"count:{count}")
    parts.append(query)
    return " ".join(parts)


def _fetch_results(query: str) -> dict:
    response = requests.get(API_URL, params={"query": query}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _format_results(data: dict) -> list[dict]:
    """Extract formatted results from the raw API response.

    The API returns raw Sourcegraph GraphQL data. This function normalises it
    into a flat list of file-match dicts with repository, file, branch, commit,
    and matches fields — the same shape the MCP server produces.
    """
    search = data.get("data", {}).get("search", {})
    raw_results = search.get("results", {}).get("results", [])

    formatted = []
    for result in raw_results:
        repo_info = result.get("repository") or {}
        repo_name = repo_info.get("name", "")
        clean_name = repo_name.removeprefix("github.com/")

        file_info = result.get("file") or {}
        file_path = file_info.get("path", "")

        if not clean_name or not file_path:
            continue

        default_branch = repo_info.get("defaultBranch") or {}
        branch_name = default_branch.get("displayName", "main")
        commit_hash = (default_branch.get("target") or {}).get("commit", {}).get("oid", "")
        ref = commit_hash or branch_name

        base_url = f"https://github.com/{clean_name}/blob/{ref}/{file_path}"
        matches = []
        for m in result.get("lineMatches") or []:
            line = m.get("lineNumber", 0) + 1
            preview = m.get("preview", "").rstrip()
            matches.append({"line": line, "preview": preview, "url": f"{base_url}#L{line}"})

        formatted.append(
            {
                "repository": clean_name,
                "file": file_path,
                "branch": branch_name,
                "commit": commit_hash,
                "matches": matches,
            }
        )

    return formatted


def _get_stats(data: dict) -> dict:
    search = data.get("data", {}).get("search", {})
    return {
        "approximate_count": search.get("stats", {}).get("approximateResultCount", "0"),
        "match_count": search.get("results", {}).get("matchCount", 0),
        "limit_hit": search.get("results", {}).get("limitHit", False),
    }


def _print_results(results: list[dict], stats: dict, json_output: bool) -> None:
    if json_output:
        console.print(json.dumps({"stats": stats, "results": results}, indent=2))
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for file_result in results:
        header = Text()
        header.append(file_result["repository"], style="bold cyan")
        header.append(" / ", style="dim")
        header.append(file_result["file"], style="bold")
        console.print(header)

        for match in file_result.get("matches", []):
            line_text = Text()
            line_text.append(f"  L{match['line']:>5}", style="green")
            line_text.append(f"  {match['preview']}")
            console.print(line_text)
            console.print(f"        [dim]{match['url']}[/dim]")

        console.print()

    limit_msg = " (limit hit — use --count to fetch more)" if stats.get("limit_hit") else ""
    console.print(
        f"[dim]{stats['approximate_count']} approximate results, {stats['match_count']} matches returned{limit_msg}[/dim]"
    )


@app.callback(invoke_without_command=True)
@tracking.track_command()
def code_search(
    query: Annotated[str, typer.Argument(help="Search query (supports Sourcegraph syntax)")],
    repo: Annotated[
        str | None,
        typer.Option("--repo", "-r", help="Filter by repository (e.g. ComfyUI, Comfy-Org/ComfyUI)"),
    ] = None,
    count: Annotated[
        int,
        typer.Option("--count", "-n", help="Maximum number of results"),
    ] = DEFAULT_COUNT,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output results as JSON"),
    ] = False,
):
    """Search code across ComfyUI repositories."""
    built_query = _build_query(query, repo, count)

    try:
        data = _fetch_results(built_query)
    except requests.ConnectionError:
        console.print("[bold red]Error: Could not connect to the code search service.[/bold red]")
        raise typer.Exit(code=1)
    except requests.Timeout:
        console.print("[bold red]Error: Request timed out.[/bold red]")
        raise typer.Exit(code=1)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        console.print(f"[bold red]Error: HTTP {status}[/bold red]")
        raise typer.Exit(code=1)

    results = _format_results(data)
    stats = _get_stats(data)
    _print_results(results, stats, json_output=json_output)
