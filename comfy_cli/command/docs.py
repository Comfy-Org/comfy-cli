"""Search and read documentation shipped with comfy-cli."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.text import Text

from comfy_cli import docs as docs_index
from comfy_cli.docs import DocsSearchPackError, DocsUnavailableError
from comfy_cli.output import get_renderer

app = typer.Typer(no_args_is_help=True, help="Search documentation shipped with comfy-cli.")


def _load_failure(renderer) -> None:
    renderer.error(
        code="docs_unavailable",
        message="the documentation shipped with comfy-cli could not be loaded",
        hint="reinstall comfy-cli to restore its documentation bundle",
    )
    raise typer.Exit(code=1)


def _pack_failure(renderer, error: DocsSearchPackError) -> None:
    if error.code == "docs_search_unavailable":
        renderer.error(code="docs_search_unavailable", message=str(error), hint=error.hint)
    else:
        renderer.error(code="docs_pack_incompatible", message=str(error), hint=error.hint)
    raise typer.Exit(code=1)


@app.command("search", help="Search bundled CLI and agent documentation.")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Words, command names, or flags to search for.")],
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=docs_index.MAX_RESULTS, help="Maximum number of sections to return."),
    ] = docs_index.DEFAULT_RESULTS,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Retrieval mode: auto, bm25, semantic, or hybrid."),
    ] = "auto",
) -> None:
    """Find relevant sections with local BM25 or optional LanceDB semantic search."""
    renderer = get_renderer()
    try:
        payload = docs_index.search(query, limit=limit, mode=mode.casefold())
    except DocsUnavailableError:
        _load_failure(renderer)
    except DocsSearchPackError as error:
        _pack_failure(renderer, error)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="query") from error

    if renderer.is_pretty():
        if payload["zero_hit"]:
            renderer.print(Text("No documentation matched.", style="yellow"))
            renderer.print(Text(payload["hint"], style="dim"))
        else:
            for result in payload["results"]:
                heading = " › ".join(result["headings"])
                renderer.print(Text(f"{result['id']}  {heading}", style="bold cyan"))
                renderer.print(Text(f"  {result['source']}:{result['source_line']}"))
                if result["excerpt"]:
                    renderer.print(Text(f"  {result['excerpt']}"))
    renderer.emit(payload, command="docs search")


@app.command("status", help="Show the documentation corpus and available retrieval modes.")
def status_cmd() -> None:
    """Report local docs pack compatibility without loading the model."""
    renderer = get_renderer()
    try:
        payload = docs_index.status()
    except DocsUnavailableError:
        _load_failure(renderer)

    if renderer.is_pretty():
        renderer.print(Text(f"Corpus: {payload['corpus_hash']}", style="bold"))
        renderer.print(Text(f"Search modes: {', '.join(payload['available_modes'])}"))
        if payload["pack_installed"]:
            renderer.print(Text(f"Search pack: {payload['pack_version']} (compatible={payload['pack_compatible']})"))
            if payload["model"]:
                renderer.print(Text(f"Model: {payload['model']}"))
        else:
            renderer.print(Text("Search pack: not installed"))
        if payload["unavailable_reason"]:
            renderer.print(Text(payload["unavailable_reason"], style="yellow"))
            renderer.print(Text(payload["hint"], style="dim"))
    renderer.emit(payload, command="docs status")


@app.command("show", help="Read a documentation section returned by `docs search`.")
def show_cmd(
    section_id: Annotated[str, typer.Argument(help="Section ID returned by `comfy docs search`.")],
    max_chars: Annotated[
        int,
        typer.Option(
            "--max-chars",
            min=1,
            max=docs_index.MAX_SECTION_CHARS,
            help="Maximum content characters to return.",
        ),
    ] = docs_index.DEFAULT_SECTION_CHARS,
    offset: Annotated[int, typer.Option("--offset", min=0, help="Zero-based character offset for long sections.")] = 0,
) -> None:
    """Print the complete section or one bounded page of a long section."""
    renderer = get_renderer()
    try:
        payload = docs_index.show(section_id, max_chars=max_chars, offset=offset)
    except DocsUnavailableError:
        _load_failure(renderer)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--offset") from error
    if payload is None:
        renderer.error(
            code="docs_not_found",
            message=f"documentation section {section_id!r} was not found",
            hint="search for the topic again with `comfy docs search`",
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        renderer.print(Text(" › ".join(payload["headings"]), style="bold cyan"))
        renderer.print(Text(f"{payload['source']}:{payload['source_line']}"))
        renderer.print(Text(payload["content"]))
        if payload["truncated"]:
            renderer.print(Text(f"Continue with --offset {payload['next_offset']}", style="dim"))
    renderer.emit(payload, command="docs show")
