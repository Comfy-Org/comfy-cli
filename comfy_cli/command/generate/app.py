"""``comfy generate`` — call ComfyUI cloud models from the CLI.

UX shape, modeled on fal-ai's genmedia but creative-user-first:

    comfy generate <model> [--<param> value]... [--download P] [--async]
    comfy generate list [--partner P] [--style S]
    comfy generate schema <model>
    comfy generate refresh
    comfy generate resume <model> <job_id> [--download P]

The first positional is either a reserved action (``list``/``schema``/
``refresh``/``resume``) or a model alias (``flux-pro``, ``ideogram-edit``, …).
Anything not in the reserved set falls through to the generate path.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import httpx
import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from comfy_cli import tracking, ui
from comfy_cli.command.generate import client, output, poll, schema, spec

_HELP = "Generate images via ComfyUI cloud models (Flux, Ideogram, DALL·E, Recraft, Stability, …)."

_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def register_with(parent: typer.Typer) -> None:
    """Wire the ``generate`` command into a Typer app. We register directly
    (rather than as a sub-app via ``add_typer``) so the first positional after
    ``generate`` can be a model alias — Click groups would treat that as a
    subcommand name and error."""

    @parent.command(name="generate", help=_HELP, context_settings=_CONTEXT_SETTINGS)
    @tracking.track_command()
    def _generate_entry(
        ctx: typer.Context,
        target: Annotated[
            str | None,
            typer.Argument(
                help="A model alias (e.g. flux-pro, ideogram-edit, dalle) or one of: list, schema, refresh, resume.",
            ),
        ] = None,
    ) -> None:
        if target is None or target in {"-h", "--help"}:
            _print_top_help()
            raise typer.Exit(code=0)
        if target == "list":
            return _list_models(list(ctx.args))
        if target == "schema":
            return _schema(list(ctx.args))
        if target == "refresh":
            return _refresh()
        if target == "resume":
            return _resume(list(ctx.args))
        _generate(target, list(ctx.args))


def _separate_meta_flags(extra_args: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    """Pull run-level flags out of the user's argv tail."""
    meta_names = {"download", "async", "json", "timeout", "api-key"}
    meta: dict[str, str | bool] = {}
    remaining: list[str] = []
    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if tok.startswith("--"):
            body = tok[2:]
            raw: str | None = None
            if "=" in body:
                body, raw = body.split("=", 1)
            if body in meta_names:
                if body in {"async", "json"}:
                    meta[body] = True if raw is None else raw.lower() not in {"false", "0", "no"}
                    i += 1
                    continue
                if raw is None:
                    if i + 1 >= len(extra_args):
                        raise schema.SchemaError(f"--{body}: missing value")
                    raw = extra_args[i + 1]
                    i += 2
                else:
                    i += 1
                meta[body] = raw
                continue
        remaining.append(tok)
        i += 1
    return remaining, meta


def _show_schema_help(endpoint: spec.Endpoint) -> None:
    """Print the schema-driven help block for a model."""
    flags = schema.flags_for(endpoint)
    alias = spec.preferred_alias(endpoint.id)
    name = alias or endpoint.id
    if alias:
        rprint(f"[bold]Model:[/bold] {alias}  [dim]({endpoint.id})[/dim]")
    else:
        rprint(f"[bold]Model:[/bold] {endpoint.id}")
    body = schema.help_text(endpoint, flags)
    rprint(body)
    rprint("")
    rprint("[dim]Example:[/dim]")
    rprint(f"  {schema.example_invocation(endpoint, flags, display_name=name)}")


def _spinner() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    )


def _emit_result(result: poll.PollResult, *, request_id: str, download: str | None, as_json: bool) -> None:
    if as_json:
        output.print_json(result.raw)
        return
    if result.status != "succeeded":
        rprint(f"[bold red]Job {result.status}: {result.error or 'unknown error'}[/bold red]")
        output.print_json(result.raw)
        raise typer.Exit(code=1)
    if download and result.image_urls:
        saved = output.save_urls(result.image_urls, download, request_id)
        output.print_urls(result.image_urls, request_id=request_id)
        output.print_saved(saved)
    else:
        output.print_urls(result.image_urls, request_id=request_id)
        if download and not result.image_urls:
            rprint("[yellow]--download requested but no image URLs found in response.[/yellow]")


def _generate(model: str, extra_args: list[str]) -> None:
    try:
        ep = spec.get_endpoint(model)
    except spec.SpecError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if any(a in {"--help", "-h"} for a in extra_args):
        _show_schema_help(ep)
        raise typer.Exit(code=0)

    try:
        remaining, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    flags = schema.flags_for(ep)
    try:
        values = schema.parse_args(flags, remaining)
    except schema.SchemaError as e:
        rprint(f"[bold red]{e}[/bold red]")
        name = spec.preferred_alias(ep.id) or ep.id
        rprint(f"[dim]Run `comfy generate schema {name}` for the full parameter list.[/dim]")
        raise typer.Exit(code=1)

    try:
        api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
    except client.ApiError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    timeout_raw = meta.get("timeout", "300")
    try:
        timeout = float(timeout_raw) if isinstance(timeout_raw, str) else 300.0
    except ValueError:
        rprint(f"[bold red]--timeout: expected number, got {timeout_raw!r}[/bold red]")
        raise typer.Exit(code=1)

    do_async = bool(meta.get("async", False))
    download = meta.get("download") if isinstance(meta.get("download"), str) else None
    as_json = bool(meta.get("json", False))

    request_id = str(uuid.uuid4())[:8]
    try:
        resp = client.send_request(ep, values, flags, api_key, timeout=timeout)
    except httpx.HTTPError as e:
        rprint(f"[bold red]Network error contacting {spec.base_url()}: {e}[/bold red]")
        raise typer.Exit(code=1) from e

    try:
        client.raise_for_status(resp)
    except client.ApiError as e:
        rprint(f"[bold red]API error {e.status}[/bold red]\n{e.body}")
        raise typer.Exit(code=1) from e

    if resp.headers.get("content-type", "").startswith("image/"):
        if download:
            saved = output.save_binary_response(resp, download, request_id)
            output.print_saved([saved])
        else:
            rprint("[yellow]Binary image response; nothing saved. Pass --download <path> to write it to disk.[/yellow]")
        return

    try:
        body = resp.json()
    except ValueError:
        rprint("[bold red]Unexpected non-JSON response.[/bold red]")
        rprint(resp.text[:500])
        raise typer.Exit(code=1)

    if ep.polling:
        job_id = str(body.get("id") or (body.get("data") or {}).get("task_id") or request_id)
        name = spec.preferred_alias(ep.id) or ep.id
        if do_async:
            if as_json:
                output.print_json(body)
            else:
                rprint(f"[bold green]Submitted:[/bold green] {name}")
                rprint(f"  job id: {job_id}")
                rprint(f"  resume: comfy generate resume {name} {job_id}")
            return

        poller = poll.get_poller(ep.polling)
        with _spinner() as prog:
            task = prog.add_task(f"Generating with {name} (job {job_id})", total=None)

            def _on_progress(p: float) -> None:
                prog.update(task, description=f"Generating ({p * 100:.0f}%)")

            result = poller(body, api_key=api_key, timeout=timeout, on_progress=_on_progress)
        _emit_result(result, request_id=job_id, download=download, as_json=as_json)
        return

    result = poll.sync_result_from_response(resp)
    _emit_result(result, request_id=request_id, download=download, as_json=as_json)


def _arg_value(args: list[str], *names: str) -> str | None:
    for i, tok in enumerate(args):
        for n in names:
            if tok == n and i + 1 < len(args):
                return args[i + 1]
            if tok.startswith(n + "="):
                return tok.split("=", 1)[1]
    return None


def _list_models(extra_args: list[str]) -> None:
    """`comfy generate list` — show available models with their short aliases."""
    partner = _arg_value(extra_args, "--partner", "-p")
    category = _arg_value(extra_args, "--category", "--style", "-c")
    query = _arg_value(extra_args, "--query", "-q")
    eps = spec.list_endpoints(partner=partner, category=category, query=query)
    if not eps:
        rprint("[yellow]No models match those filters.[/yellow]")
        raise typer.Exit(code=0)
    rows = [
        (
            spec.preferred_alias(e.id) or e.id,
            e.partner,
            e.category,
            "async" if e.polling else "sync",
            (e.summary[:60] + "…") if len(e.summary) > 61 else e.summary,
        )
        for e in eps
    ]
    ui.display_table(rows, ["Model", "Partner", "Style", "Mode", "Summary"], title="Comfy Generate — Models")
    rprint("\n[dim]Run `comfy generate schema <model>` to see parameters for a model.[/dim]")


def _schema(extra_args: list[str]) -> None:
    """`comfy generate schema <model>` — show params for a model (fal-style)."""
    if not extra_args or extra_args[0].startswith("-"):
        rprint("[bold red]Usage: comfy generate schema <model>[/bold red]")
        raise typer.Exit(code=1)
    try:
        ep = spec.get_endpoint(extra_args[0])
    except spec.SpecError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    _show_schema_help(ep)


def _refresh() -> None:
    url = spec.base_url() + "/openapi.yml"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as cli:
            r = cli.get(url, headers={"X-Comfy-Env": "comfy-cli", "User-Agent": "comfy-cli/api"})
            r.raise_for_status()
    except httpx.HTTPError as e:
        rprint(f"[bold red]Failed to fetch {url}: {e}[/bold red]")
        raise typer.Exit(code=1)
    path = spec.write_cache(r.text)
    rprint(f"[bold green]Refreshed model catalog at {path}[/bold green]")


def _resume(extra_args: list[str]) -> None:
    if len(extra_args) < 2:
        rprint("[bold red]Usage: comfy generate resume <model> <job_id> [--download PATH] [--json][/bold red]")
        raise typer.Exit(code=1)
    model, job_id = extra_args[0], extra_args[1]
    tail = extra_args[2:]
    try:
        ep = spec.get_endpoint(model)
    except spec.SpecError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    if not ep.polling:
        rprint(f"[bold red]{model} is a sync model; nothing to resume.[/bold red]")
        raise typer.Exit(code=1)
    try:
        _, meta = _separate_meta_flags(tail)
    except schema.SchemaError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    try:
        api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
    except client.ApiError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    timeout = float(meta.get("timeout") or 300.0) if isinstance(meta.get("timeout"), str) else 300.0
    download = meta.get("download") if isinstance(meta.get("download"), str) else None
    as_json = bool(meta.get("json", False))

    if ep.polling == "bfl":
        initial = {"polling_url": f"{spec.base_url()}/proxy/bfl/get_result?id={job_id}"}
    else:
        rprint(f"[bold red]Resume not implemented for partner {ep.partner}[/bold red]")
        raise typer.Exit(code=1)

    poller = poll.get_poller(ep.polling)
    with _spinner() as prog:
        task = prog.add_task(f"Resuming job {job_id}", total=None)

        def _on_progress(p: float) -> None:
            prog.update(task, description=f"Job {job_id} ({p * 100:.0f}%)")

        result = poller(initial, api_key=api_key, timeout=timeout, on_progress=_on_progress)
    _emit_result(result, request_id=job_id, download=download, as_json=as_json)


def _print_top_help() -> None:
    """Custom help that emphasizes the model-first UX over Typer's auto-help."""
    rprint("[bold]comfy generate[/bold] — call ComfyUI cloud models")
    rprint("")
    rprint("[bold]Usage:[/bold]")
    rprint("  comfy generate <model> [--<param> value]... [--download PATH] [--async] [--api-key KEY]")
    rprint("")
    rprint("[bold]Examples:[/bold]")
    rprint('  comfy generate flux-pro --prompt "a cat on the moon" --width 1024 --height 1024 --download cat.png')
    rprint(
        '  comfy generate ideogram-edit --image cat.png --mask m.png --prompt "add sunglasses" --rendering_speed TURBO'
    )
    rprint('  comfy generate dalle --prompt "a watercolor whale" --download whale.png')
    rprint("")
    rprint("[bold]Actions:[/bold]")
    rprint("  comfy generate list                    Browse available models")
    rprint("  comfy generate schema <model>          Show parameters for a model")
    rprint("  comfy generate refresh                 Refresh the model catalog")
    rprint("  comfy generate resume <model> <job>    Resume an async job")
    rprint("")
    rprint("[dim]Auth: set COMFY_API_KEY or pass --api-key. Get one at https://platform.comfy.org.[/dim]")
