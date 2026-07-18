"""``comfy generate`` — call ComfyUI partner nodes from the CLI.

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
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from comfy_cli import tracking, ui
from comfy_cli.command.generate import adapters, client, emit, output, poll, schema, spec, upload
from comfy_cli.output.renderer import get_renderer

_HELP = "Generate images via ComfyUI partner nodes (Flux, Ideogram, DALL·E, Recraft, Stability, …)."

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
    def _generate_entry(
        ctx: typer.Context,
        target: Annotated[
            str | None,
            typer.Argument(
                help="A model alias (e.g. flux-pro, ideogram-edit, dalle) "
                "or one of: list, schema, refresh, upload, resume.",
            ),
        ] = None,
    ) -> None:
        if target is None or target in {"-h", "--help"}:
            _print_top_help()
            raise typer.Exit(code=0)
        extra = list(ctx.args)
        if target == "list":
            tracking.track_event("generate:list")
            return _list_models(extra)
        if target == "schema":
            model_arg = extra[0] if extra and not extra[0].startswith("-") else None
            tracking.track_event("generate:schema", {"model": model_arg})
            return _schema(extra)
        if target == "refresh":
            tracking.track_event("generate:refresh")
            return _refresh()
        if target == "upload":
            tracking.track_event("generate:upload")
            return _upload(extra)
        if target == "resume":
            resume_model = extra[0] if extra and not extra[0].startswith("-") else None
            resume_job_id = extra[1] if len(extra) >= 2 and not extra[1].startswith("-") else None
            tracking.track_event(
                "generate:resume",
                {"model": resume_model, "job_id": resume_job_id},
            )
            return _resume(extra)
        _generate(target, extra)


def _separate_meta_flags(extra_args: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    """Pull run-level flags out of the user's argv tail."""
    meta_names = {"download", "async", "json", "timeout", "api-key", "emit-workflow", "output-prefix"}
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
        # Honor --download in JSON mode too. Previously this returned before
        # saving, so `--json --download` printed the URL but wrote no file,
        # forcing callers to curl the URL by hand. Save first, then surface the
        # local path alongside the raw response.
        if download and result.status == "succeeded" and result.image_urls:
            saved = output.save_urls(result.image_urls, download, request_id)
            output.print_json({"result": result.raw, "saved": [str(p) for p in saved]})
        else:
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
    # --help short-circuits before tracking — it's a help-display action, not an execution attempt.
    # If the model is unknown, fall through so the tracking path records the schema error.
    asks_help = any(a in {"--help", "-h"} for a in extra_args)
    if asks_help:
        try:
            help_ep = spec.get_endpoint(model)
        except spec.SpecError:
            help_ep = None
        if help_ep is not None:
            _show_schema_help(help_ep)
            raise typer.Exit(code=0)

    # generate:start fires at entry so every invocation has a paired start/end lifecycle.
    # Props are filled in progressively as model_alias / partner / async / has_download become known.
    gen_props: dict[str, object | None] = {
        "model": model,
        "model_alias": None,
        "async": None,
        "has_download": None,
        "partner": None,
    }
    tracking.track_event("generate:start", gen_props)

    def _track_error(error_kind: str, exc: BaseException) -> None:
        tracking.track_event(
            "generate:error",
            {**gen_props, "error_type": type(exc).__name__, "error_kind": error_kind},
        )

    try:
        try:
            ep = spec.get_endpoint(model)
        except spec.SpecError as e:
            rprint(f"[bold red]{e}[/bold red]")
            _track_error("schema", e)
            raise typer.Exit(code=1)

        gen_props["model_alias"] = spec.preferred_alias(ep.id)
        gen_props["partner"] = getattr(ep, "partner", None)

        try:
            remaining, meta = _separate_meta_flags(extra_args)
        except schema.SchemaError as e:
            rprint(f"[bold red]{e}[/bold red]")
            _track_error("schema", e)
            raise typer.Exit(code=1)

        do_async = bool(meta.get("async", False))
        download = meta.get("download") if isinstance(meta.get("download"), str) else None
        as_json = bool(meta.get("json", False))
        gen_props["async"] = do_async
        gen_props["has_download"] = bool(download)

        emit_path = meta.get("emit-workflow") if isinstance(meta.get("emit-workflow"), str) else None
        flags = schema.flags_for(ep)
        try:
            # In emit mode the partner node carries its own defaults, so don't
            # force every proxy-required flag — let the user override only what
            # they want.
            values = schema.parse_args(flags, remaining, require_all=not emit_path)
        except schema.SchemaError as e:
            rprint(f"[bold red]{e}[/bold red]")
            name = gen_props["model_alias"] or ep.id
            rprint(f"[dim]Run `comfy generate schema {name}` for the full parameter list.[/dim]")
            _track_error("schema", e)
            raise typer.Exit(code=1)

        if emit_path:
            # Emit a runnable workflow that drives the partner *node* and return
            # — no proxy call, no API key required. The artifact is the result.
            name = gen_props["model_alias"] or ep.id
            prefix = meta.get("output-prefix") if isinstance(meta.get("output-prefix"), str) else "generate"
            renderer = get_renderer()
            try:
                workflow = emit.write_workflow(name, values, Path(emit_path).expanduser(), output_prefix=prefix)
            except (emit.EmitError, OSError) as e:
                _track_error("emit", e)
                hint = (
                    "check destination path permissions and parent directory"
                    if isinstance(e, OSError)
                    else "check the model name and that all required inputs are provided"
                )
                renderer.error(
                    code="emit_workflow_failed",
                    message=str(e),
                    hint=hint,
                )
                raise typer.Exit(code=1) from e
            tracking.track_event("generate:emit", {**gen_props, "node_count": len(workflow)})
            if renderer.is_pretty():
                rprint(f"[bold green]Wrote workflow:[/bold green] {emit_path}")
                rprint(f"  run it: comfy run --workflow {emit_path}")
            renderer.emit(
                {"out": str(Path(emit_path).expanduser()), "model": name, "nodes": len(workflow)},
                command="generate emit-workflow",
            )
            return

        try:
            api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
        except client.ApiError as e:
            rprint(f"[bold red]{e}[/bold red]")
            _track_error("api", e)
            raise typer.Exit(code=1)

        timeout_raw = meta.get("timeout", "300")
        try:
            timeout = float(timeout_raw) if isinstance(timeout_raw, str) else 300.0
        except ValueError as e:
            rprint(f"[bold red]--timeout: expected number, got {timeout_raw!r}[/bold red]")
            _track_error("schema", e)
            raise typer.Exit(code=1)

        try:
            _apply_upload_transforms(values, flags, ep, api_key)
        except (client.ApiError, httpx.HTTPError) as e:
            rprint(f"[bold red]Upload failed: {e}[/bold red]")
            _track_error("upload", e)
            raise typer.Exit(code=1)

        request_id = str(uuid.uuid4())[:8]
        try:
            resp = client.send_request(ep, values, flags, api_key, timeout=timeout)
        except httpx.HTTPError as e:
            rprint(f"[bold red]Network error contacting {spec.base_url()}: {e}[/bold red]")
            _track_error("network", e)
            raise typer.Exit(code=1) from e

        try:
            client.raise_for_status(resp)
        except client.ApiError as e:
            rprint(f"[bold red]API error {e.status}[/bold red]\n{e.body}")
            _track_error("api", e)
            raise typer.Exit(code=1) from e

        if resp.headers.get("content-type", "").startswith("image/"):
            if download:
                saved = output.save_binary_response(resp, download, request_id)
                output.print_saved([saved])
            else:
                rprint(
                    "[yellow]Binary image response; nothing saved. Pass --download <path> to write it to disk.[/yellow]"
                )
            tracking.track_event("generate:success", gen_props)
            return

        try:
            body = resp.json()
        except ValueError as e:
            rprint("[bold red]Unexpected non-JSON response.[/bold red]")
            rprint(resp.text[:500])
            _track_error("non_json_response", e)
            raise typer.Exit(code=1)

        if ep.polling:
            job_id = poll.extract_job_id(ep.polling, body) or request_id
            name = gen_props["model_alias"] or ep.id
            if do_async:
                if as_json:
                    output.print_json(body)
                else:
                    rprint(f"[bold green]Submitted:[/bold green] {name}")
                    rprint(f"  job id: {job_id}")
                    rprint(f"  resume: comfy generate resume {name} {job_id}")
                # Submitted, not succeeded — the workflow runs on the partner side and completion is
                # observed server-side via partner_node:api_call_*. No generate:success pair here.
                tracking.track_event(
                    "generate:submitted",
                    {
                        "model": model,
                        "model_alias": gen_props["model_alias"],
                        "job_id": job_id,
                        "partner": gen_props["partner"],
                    },
                )
                return

            poller = poll.get_poller(ep.polling)
            with _spinner() as prog:
                task = prog.add_task(f"Generating with {name} (job {job_id})", total=None)

                def _on_progress(p: float) -> None:
                    prog.update(task, description=f"Generating ({p * 100:.0f}%)")

                try:
                    result = poller(
                        body,
                        api_key=api_key,
                        timeout=timeout,
                        on_progress=_on_progress,
                        create_path=ep.path,
                    )
                except (client.ApiError, httpx.HTTPError) as e:
                    _track_error("network" if isinstance(e, httpx.HTTPError) else "api", e)
                    raise typer.Exit(code=1) from e
            try:
                _emit_result(result, request_id=job_id, download=download, as_json=as_json)
                tracking.track_event("generate:success", gen_props)
            except typer.Exit as e:
                if (e.exit_code or 0) == 0:
                    tracking.track_event("generate:success", gen_props)
                else:
                    _track_error("api", e)
                raise
            return

        adapter = adapters.get(ep.id)
        if adapter is not None and adapter.decode_sync is not None:
            body = resp.json()
            if as_json:
                output.print_json(body)
                tracking.track_event("generate:success", gen_props)
                return
            if not download:
                rprint("[yellow]Image data returned inline. Pass --download <path> to save.[/yellow]")
                tracking.track_event("generate:success", gen_props)
                return
            saved = adapter.decode_sync(body, download, request_id)
            if saved:
                output.print_saved(saved)
            else:
                rprint("[yellow]No image data found in response.[/yellow]")
                output.print_json(body)
            tracking.track_event("generate:success", gen_props)
            return

        try:
            result = poll.sync_result_from_response(resp)
            _emit_result(result, request_id=request_id, download=download, as_json=as_json)
            tracking.track_event("generate:success", gen_props)
        except typer.Exit as e:
            if (e.exit_code or 0) == 0:
                tracking.track_event("generate:success", gen_props)
            else:
                _track_error("api", e)
            raise
    except typer.Exit:
        # Inline raise sites already emitted their lifecycle event.
        raise
    except Exception as e:
        # Safety net so an unexpected exception still pairs generate:start with a terminal generate:error.
        _track_error("unknown", e)
        raise


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
    clean, meta = _separate_meta_flags(extra_args)
    as_json = bool(meta.get("json", False))
    partner = _arg_value(clean, "--partner", "-p")
    category = _arg_value(clean, "--category", "--style", "-c")
    query = _arg_value(clean, "--query", "-q")
    eps = spec.list_endpoints(partner=partner, category=category, query=query)
    if as_json:
        models = [
            {
                "alias": spec.preferred_alias(e.id) or e.id,
                "id": e.id,
                "partner": e.partner,
                "category": e.category,
                "mode": "async" if e.polling else "sync",
                "summary": e.summary,
            }
            for e in eps
        ]
        output.print_json({"models": models, "count": len(models)})
        return
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
    clean, meta = _separate_meta_flags(extra_args)
    as_json = bool(meta.get("json", False))
    if not clean or clean[0].startswith("-"):
        if as_json:
            output.print_json({"error": "Usage: comfy generate schema <model>"})
            raise typer.Exit(code=1)
        rprint("[bold red]Usage: comfy generate schema <model>[/bold red]")
        raise typer.Exit(code=1)
    try:
        ep = spec.get_endpoint(clean[0])
    except spec.SpecError as e:
        if as_json:
            output.print_json({"error": str(e)})
            raise typer.Exit(code=1)
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    if as_json:
        flags = schema.flags_for(ep)
        output.print_json(
            {
                "model": spec.preferred_alias(ep.id) or ep.id,
                "id": ep.id,
                "params": [
                    {
                        "name": f.name,
                        "kind": f.kind,
                        "required": f.required,
                        "default": f.default,
                        "enum": f.enum,
                        "description": f.description,
                    }
                    for f in flags
                ],
            }
        )
        return
    _show_schema_help(ep)


def _fetch_spec(url: str) -> httpx.Response:
    with httpx.Client(timeout=30.0, follow_redirects=True) as cli:
        r = cli.get(url, headers={"Comfy-Env": "comfy-cli", "User-Agent": "comfy-cli/api"})
        r.raise_for_status()
        return r


def _refresh() -> None:
    base = spec.base_url()
    # The live spec is served at ``/openapi`` (no extension, JSON body). Older /
    # custom ``COMFY_API_BASE_URL`` deployments may still serve ``/openapi.yml``,
    # so fall back to it on a 404 to keep those working.
    primary, fallback = base + "/openapi", base + "/openapi.yml"
    fetched_from = primary
    try:
        try:
            r = _fetch_spec(primary)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            fetched_from = fallback
            r = _fetch_spec(fallback)
    except httpx.HTTPError as e:
        rprint(f"[bold red]Failed to fetch {fetched_from}: {e}[/bold red]")
        raise typer.Exit(code=1)

    # Validate before caching so a 200-with-garbage response never poisons the
    # ~/.comfy/openapi-cache.yml cache (used for CACHE_TTL_SECONDS by every
    # subsequent `comfy generate`).
    body = r.text
    try:
        spec.validate_spec_text(body)
    except spec.SpecError as e:
        rprint(f"[bold red]Refusing to cache spec from {fetched_from}: {e}[/bold red]")
        raise typer.Exit(code=1)

    path = spec.write_cache(body)
    rprint(f"[bold green]Refreshed model catalog at {path}[/bold green]")


def _upload(extra_args: list[str]) -> None:
    """`comfy generate upload <file-or-url> [--json] [--api-key K]`."""
    try:
        remaining, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    # `remaining` already excludes recognized --meta flags AND their values, so
    # `comfy generate upload --api-key KEY ./img.png` correctly resolves to "./img.png".
    if not remaining:
        rprint("[bold red]Usage: comfy generate upload <file-or-url> [--json][/bold red]")
        raise typer.Exit(code=1)
    target = remaining[0]
    try:
        api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
    except client.ApiError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    as_json = bool(meta.get("json", False))
    try:
        result = upload.upload_target(target, api_key)
    except (client.ApiError, httpx.HTTPError) as e:
        rprint(f"[bold red]Upload failed: {e}[/bold red]")
        raise typer.Exit(code=1)
    if as_json:
        output.print_json(
            {
                "url": result.url,
                "expires_at": result.expires_at,
                "existing_file": result.existing_file,
                "hint": "Pass this URL as the model's image/input_image field.",
            }
        )
        return
    rprint(f"[bold green]Uploaded:[/bold green] {result.url}")
    if result.expires_at:
        rprint(f"  expires: {result.expires_at}")
    if result.existing_file:
        rprint("  [dim](server already had a hash-match; no bytes transferred)[/dim]")


def _apply_upload_transforms(values: dict, flags: list[schema.FlagDef], endpoint: spec.Endpoint, api_key: str) -> None:
    """When the user supplies a local file path for a field that expects a
    base64 blob or a URL, transform it transparently.

    This only applies to JSON endpoints — multipart endpoints already stream
    file paths natively via httpx and don't need pre-uploading. Endpoints with
    a custom adapter handle their own asset shaping inside ``build_body``.
    """
    if adapters.get(endpoint.id) is not None:
        return
    if endpoint.request_content_type != "application/json":
        return
    flag_by_name = {f.name: f for f in flags}
    for name, value in list(values.items()):
        flag = flag_by_name.get(name)
        if flag is None or flag.upload_mode is None or not isinstance(value, str):
            continue
        if value.startswith(("http://", "https://", "data:")):
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        if flag.upload_mode == "base64":
            import base64 as _base64

            try:
                data = path.read_bytes()
            except OSError as e:
                raise client.ApiError(0, "", f"Unable to read file for --{name}: {path} ({e})") from e
            values[name] = _base64.b64encode(data).decode("ascii")
            rprint(f"[dim]base64-encoded {path.name} for --{name}[/dim]")
        elif flag.upload_mode == "url":
            rprint(f"[dim]uploading {path.name} for --{name}…[/dim]")
            result = upload.upload_path(path, api_key)
            values[name] = result.url


def _resume(extra_args: list[str]) -> None:
    if len(extra_args) < 2 or extra_args[0].startswith("-") or extra_args[1].startswith("-"):
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

    try:
        initial = poll.build_synthetic_initial(ep.polling, job_id, base_url=spec.base_url())
    except client.ApiError as e:
        rprint(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    poller = poll.get_poller(ep.polling)
    with _spinner() as prog:
        task = prog.add_task(f"Resuming job {job_id}", total=None)

        def _on_progress(p: float) -> None:
            prog.update(task, description=f"Job {job_id} ({p * 100:.0f}%)")

        result = poller(
            initial,
            api_key=api_key,
            timeout=timeout,
            on_progress=_on_progress,
            create_path=ep.path,
        )
    _emit_result(result, request_id=job_id, download=download, as_json=as_json)


def _print_top_help() -> None:
    """Custom help that emphasizes the model-first UX over Typer's auto-help."""
    rprint("[bold]comfy generate[/bold] — call ComfyUI partner nodes")
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
    rprint(
        '  comfy generate flux-pro --prompt "a fox" --emit-workflow flux.json   '
        "[dim]# write a runnable workflow instead of calling the proxy[/dim]"
    )
    rprint("")
    rprint("[bold]Actions:[/bold]")
    rprint("  comfy generate list                    Browse available models")
    rprint("  comfy generate schema <model>          Show parameters for a model")
    rprint("  comfy generate refresh                 Refresh the model catalog")
    rprint("  comfy generate upload <file-or-url>    Host a local file or remote URL and print its signed URL")
    rprint("  comfy generate resume <model> <job>    Resume an async job")
    rprint("")
    rprint(
        "[dim]Auth: run `comfy cloud login` (session outranks env var), set COMFY_API_KEY, or pass --api-key. Get one at https://platform.comfy.org.[/dim]"
    )
