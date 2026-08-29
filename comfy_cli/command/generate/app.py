"""``comfy generate`` — call ComfyUI partner nodes from the CLI.

UX shape, modeled on fal-ai's genmedia but creative-user-first:

    comfy generate <model> [--<param> value]... [--download P] [--async] [--yes]
    comfy generate list [--partner P] [--style S]
    comfy generate schema <model>
    comfy generate refresh
    comfy generate resume <model> <job_id> [--download P]
    comfy generate consent [show|always|ask]

The first positional is either a reserved action (``list``/``schema``/
``refresh``/``resume``/``consent``) or a model alias (``flux-pro``,
``ideogram-edit``, …). Anything not in the reserved set falls through to the
generate path.

Spend gate: a generation call spends Comfy credits, so the proxy call sits
behind a consent interlock (``_confirm_spend``) — an interactive TTY prompt,
bypassed by ``--yes`` or the persisted ``spend.auto_confirm`` config, and
fail-closed (error, no spend) when neither is present and no prompt is
possible (``--json`` or no TTY).
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import httpx
import typer

# NOT migrated to `comfy_cli.output.rprint` on purpose. `generate` carries its own
# `--json` flag (see `_emit_result` / `output.print_json`) and never emits a
# renderer envelope on its submit/sync/resume paths. Routing these calls through
# the shim would send the command's *primary result* (job ids, image URLs) to
# stderr whenever stdout isn't a TTY, leaving stdout empty -- so `comfy generate
# ... > out.txt` would write an empty file. Migrate only once `generate` emits
# envelopes via the renderer.
#
# That rationale covers *results* only. FAILURES go through `_fail` below, which
# emits an `envelope/1` error whenever the global renderer is in JSON /
# JSON-stream mode -- an envelope-consuming caller must never get exit 1 with a
# blank stdout.
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from comfy_cli import constants, knowledge, tracking, ui
from comfy_cli.command.generate import adapters, client, emit, output, poll, schema, spec, upload
from comfy_cli.config_manager import ConfigManager
from comfy_cli.output.renderer import Renderer, get_renderer
from comfy_cli.output.sanitize import sanitize_markup

_HELP = "Generate images via ComfyUI partner nodes (Flux, Ideogram, DALL·E, Recraft, Stability, …)."

_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

_TARGET_REQUIRED_MSG = (
    "`comfy generate` requires a partner model alias as its first argument "
    '(e.g. `comfy generate flux-pro --prompt "a cat on the moon"`); it is a cloud/partner '
    "verb that spends credits."
)
_TARGET_REQUIRED_HINT = (
    "Run `comfy generate list` to see model aliases. For local text-to-image, use `comfy run-template` instead."
)


def _fail(
    *,
    code: str,
    message: str,
    hint: str | None = None,
    details: Mapping[str, Any] | None = None,
    legacy_json: bool = False,
    pretty: str | None = None,
) -> None:
    r"""Report a `generate` failure through whichever channel the caller asked for.

    - Global `--json` / `--json-stream` (and any non-TTY stdout, which the
      renderer already resolves to JSON) → exactly one `envelope/1` error on
      stdout with a stable `code`. Without this an envelope-consuming caller
      (comfy-local-mcp, scripts, agents) sees exit 1 with nothing to read.
    - `legacy_json` → the command-local `--json` error object. The `error` key
      is byte-compatible with what those paths already emitted; `code` is added
      (additively — the `_consent` unknown-action and `_schema` usage/unknown-model
      paths previously emitted `error` alone) so a machine caller on the local
      flag gets the same stable identifier the envelope carries.
    - Otherwise → the historical rich-red line (plus a dim hint line where the
      path already printed one), so pretty/TTY output is unchanged.

    The default pretty line runs `message` (and `hint`) through `sanitize_markup`:
    several call sites interpolate server-controlled text (an `ApiError`'s body, a
    response preview, a partner's failure reason). Markup escaping alone is not
    enough for remote text — it stops a bracketed token from being read as a tag
    (which would swallow the text or raise `MarkupError`), but `\x1b` survives it,
    so a CSI/OSC sequence would reach the terminal and could clear the screen or
    repaint earlier lines to spoof CLI output. `sanitize_markup` strips the escape
    bytes *and* escapes markup; see `comfy_cli.output.sanitize` (#614), which this
    module must call explicitly because `generate` prints via a bare `rich.print`
    rather than through `Renderer`.

    A caller-supplied `pretty` is passed through verbatim: it is explicitly
    pre-formatted markup and is responsible for sanitizing its own interpolations
    (the two that carry remote text do).

    The JSON/NDJSON paths above deliberately do NOT sanitize: `json.dumps` encodes
    `\x1b` as a `\u` escape already, and stripping there would mutate the data
    agents parse.

    ``hint`` is only passed where pretty mode already printed that second line;
    everywhere else `renderer.error` falls back to the code's registered hint,
    which keeps pretty output identical while JSON callers still get navigation.

    Keyword-only on purpose: ``tests/comfy_cli/output/test_error_code_registry.py``
    scans for literal ``code="…"`` kwargs to pin every raised code against
    :mod:`comfy_cli.error_codes`, and a positional first argument would be
    invisible to it.
    """
    renderer = get_renderer()
    if renderer.is_json():
        renderer.error(code=code, message=message, hint=hint, details=details)
        return
    if legacy_json:
        output.print_json({"error": message, "code": code})
        return
    rprint(pretty if pretty is not None else f"[bold red]{sanitize_markup(message)}[/bold red]")
    if hint:
        rprint(f"[dim]{sanitize_markup(hint)}[/dim]")


def _transport_code(exc: BaseException) -> str:
    """`generate_network_error` for a transport failure, `generate_api_error` for
    an HTTP/API-level one — the two arrive together on most call sites.

    `httpx.HTTPStatusError` subclasses `HTTPError` but is *not* a transport
    failure: the request reached the server and came back non-2xx (e.g.
    `upload_remote_url`'s `raise_for_status` on a 404 source URL). Calling that
    a network error would tell automation to check connectivity and retry, which
    is exactly the wrong move for a 4xx.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return "generate_api_error"
    return "generate_network_error" if isinstance(exc, httpx.HTTPError) else "generate_api_error"


def _notice(markup: str, *, err: bool) -> None:
    """Print a human-facing Rich line to stdout, or to stderr when stdout is the
    machine channel."""
    if err:
        get_renderer().stderr_console().print(markup)
    else:
        rprint(markup)


def _track_kind(exc: BaseException) -> str:
    """Tracking bucket matching `_transport_code` — an HTTP-status failure is an
    API error, not a network one."""
    return "network" if _transport_code(exc) == "generate_network_error" else "api"


def _bail(
    track_error: Callable[[str, BaseException], None],
    exc: BaseException,
    *,
    code: str,
    message: str,
    kind: str,
    hint: str | None = None,
    details: Mapping[str, Any] | None = None,
    legacy_json: bool = False,
    pretty: str | None = None,
) -> NoReturn:
    """Report a failure, record its `generate:error`, and exit 1 — in that order.

    Every error branch in `_generate` owes the same three steps, and each one is
    load-bearing: skipping the tracking call orphans the `generate:start` fired
    at entry, and returning instead of raising falls through into the success
    path. Ten branches spelled that out identically; owning the order here makes
    it a property of the helper rather than of ten copies.

    `track_error` is a parameter rather than a module-level function because the
    tracker closes over `_generate`'s mutable `gen_props` — `model_alias`,
    `partner`, `async` and `has_download` are filled in as they become known, so
    the payload has to be read at raise time.

    `code` is keyword-only for the same reason `_fail`'s is:
    ``tests/comfy_cli/output/test_error_code_registry.py`` scans call sites for
    literal ``code="…"`` kwargs to pin every raised code against
    :mod:`comfy_cli.error_codes`, and a positional first argument would be
    invisible to it. `_fail`'s remaining keyword-only options (`hint`, `details`,
    `legacy_json`, `pretty`) are re-declared here rather than forwarded through a
    `**kwargs`, so a misspelled option is a type error at the call site instead of
    a `TypeError` raised inside `_fail` on an error-only path — which would
    replace the intended exit-1 envelope with a traceback.

    Tracking is best-effort, so it runs guarded: a telemetry failure (a malformed
    `enable_tracking` making `config_manager.get_bool` raise, say) must not cost
    the caller the exit this helper's `NoReturn` promises. Without the guard the
    exception escapes into `_generate`'s outer `except Exception`, which tracks
    again and re-raises — an unhandled traceback with no envelope on stdout.

    The exit is chained (`from exc`) so the original failure stays reachable at
    every site. Three of the collapsed sites chained explicitly; the rest raised
    from inside an `except` block, where `__context__` carried the cause
    implicitly. The explicit `from` earns its keep at the poll site, which reports
    from `if poll_error is not None:` after the spinner's `with` has exited — no
    exception is in flight there, so nothing would chain without it.
    """
    _fail(code=code, message=message, hint=hint, details=details, legacy_json=legacy_json, pretty=pretty)
    try:
        track_error(kind, exc)
    except Exception as track_exc:
        logging.warning(f"Failed to record generate:error: {track_exc}")
    raise typer.Exit(code=1) from exc


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
                "or one of: list, schema, refresh, upload, resume, consent.",
            ),
        ] = None,
    ) -> None:
        if target is None or target in {"-h", "--help"}:
            _print_top_help()
            raise typer.Exit(code=0)
        if target.startswith("-"):
            # `ignore_unknown_options` lets a flag token slide into the model-alias
            # positional, so `comfy generate --prompt=x` used to die deep inside
            # `spec.get_endpoint` as an "unknown model". Fail here instead, at the
            # earliest point, with an actionable message.
            _fail(
                code="generate_target_required",
                message=_TARGET_REQUIRED_MSG,
                hint=_TARGET_REQUIRED_HINT,
                details={"received": target},
            )
            raise typer.Exit(code=1)
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
        if target == "consent":
            consent_action = extra[0] if extra and not extra[0].startswith("-") else None
            tracking.track_event("generate:consent", {"action": consent_action})
            return _consent(extra)
        _generate(target, extra)


def _separate_meta_flags(extra_args: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    """Pull run-level flags out of the user's argv tail."""
    meta_names = {"download", "async", "json", "timeout", "api-key", "emit-workflow", "output-prefix", "yes"}
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
                if body in {"async", "json", "yes"}:
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
    renderer = get_renderer()
    # One failure path for every output mode, checked FIRST: a terminally
    # failed job is never a result. In JSON/NDJSON modes it is an ok=false
    # envelope with a registered code (the generate_result schema promises
    # exactly that); in pretty mode it is a red line plus the partner's raw
    # response. Either way the exit code is 1 — a consumer that trusts ``ok``
    # (or the exit code) must never read a failure as success.
    if result.status != "succeeded":
        message = f"Job {result.status}: {result.error or 'unknown error'}"
        if renderer.is_json():
            renderer.error(
                code="generate_job_failed",
                message=message,
                details={"status": result.status, "response": result.raw},
            )
        else:
            # `result.error` is the partner's own text. `sanitize_markup` (not a bare
            # `escape`) because it is remote: escaping alone stops a bracketed token
            # from being parsed as Rich markup but passes `\x1b` straight through,
            # letting the partner clear the screen or repaint earlier lines (#614).
            rprint(f"[bold red]{sanitize_markup(message)}[/bold red]")
            output.print_json(result.raw)
        raise typer.Exit(code=1)
    if renderer.is_json() or as_json:
        # Honor --download in machine modes too. Previously this returned
        # before saving, so `--json --download` printed the URL but wrote no
        # file, forcing callers to curl the URL by hand. Save first, then
        # surface the local path alongside the response.
        saved: list[str] = []
        if download and result.image_urls:
            saved = [str(p) for p in output.save_urls(result.image_urls, download, request_id)]
        if renderer.is_json():
            # JSON/NDJSON modes (the global ``--output json``, a redirected
            # stdout, or the tail ``--json``) get the envelope/1 contract every
            # other machine-readable command speaks: data.result wraps the
            # partner payload, data.saved lists --download artifacts.
            # Registered as COMMAND_SCHEMAS["comfy generate"] -> generate_result.json.
            data: dict[str, Any] = {"result": result.raw}
            if saved:
                data["saved"] = saved
            renderer.emit(data, ok=True, command="generate")
            return
        # Pretty mode with an explicit tail --json keeps the legacy raw blob.
        if saved:
            output.print_json({"result": result.raw, "saved": saved})
        else:
            output.print_json(result.raw)
        return
    if download and result.image_urls:
        saved = output.save_urls(result.image_urls, download, request_id)
        output.print_urls(result.image_urls, request_id=request_id)
        output.print_saved(saved)
    else:
        output.print_urls(result.image_urls, request_id=request_id)
        if download and not result.image_urls:
            rprint("[yellow]--download requested but no image URLs found in response.[/yellow]")


class SpendNotConfirmed(RuntimeError):
    """A credit-spending call lacked consent — declined at the prompt, or
    fail-closed because no prompt was possible and nothing pre-authorized it."""


def _spend_auto_confirmed() -> bool:
    """True only when the persisted ``spend.auto_confirm`` config is an
    affirmative boolean. A missing key or a garbage value never authorizes
    spending — the gate fails closed."""
    try:
        return bool(ConfigManager().get_bool(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM))
    except ValueError:
        return False


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:
        # stdin already closed (e.g. daemonized caller) — no prompt possible.
        return False


def _confirm_spend(*, model_name: str, assume_yes: bool, as_json: bool) -> None:
    """The money interlock: explicit consent before a credit-spending proxy call.

    - ``--yes`` or persisted ``spend.auto_confirm=true`` → proceed.
    - Interactive TTY → prompt, default No.
    - ``--json`` or no TTY with neither → fail closed: raise, spend nothing.
      Never hang on a prompt a machine caller can't answer.
    """
    if assume_yes or _spend_auto_confirmed():
        return
    if as_json or not _stdin_is_tty():
        # `--json`/no-TTY: no prompt is answerable, so fail closed.
        msg = (
            f"`comfy generate {model_name}` spends Comfy credits and no consent was given. "
            "Re-run with --yes, or persist consent with `comfy generate consent always`."
        )
        _fail(code="spend_consent_required", message=msg, legacy_json=as_json)
        raise SpendNotConfirmed(msg)
    # A TTY stdin but a machine stdout (`comfy generate … | jq`) is a real
    # combination, and the prompt is human I/O: written to a piped stdout it is
    # invisible to the person being asked — they see a silent hang — and it
    # splices human text ahead of whatever JSON the caller is parsing. Route the
    # notice and the prompt to stderr in that case; a TTY user reads them
    # exactly as before (same terminal) and stdout stays parseable. Pretty mode
    # is untouched.
    to_err = get_renderer().is_json()
    _notice(f"[bold]{model_name}[/bold] runs via the partner API and [bold]spends Comfy credits[/bold].", err=to_err)
    _notice(
        "[dim]Skip this prompt with --yes; persist always-proceed with `comfy generate consent always`.[/dim]",
        err=to_err,
    )
    if not typer.confirm("Proceed?", default=False, err=to_err):
        # Reachable with a TTY stdin but a redirected (JSON-mode) stdout, so the
        # decline still owes the caller an envelope rather than a bare line.
        _fail(
            code="spend_consent_required",
            message="Canceled — no credits were spent.",
            pretty="Canceled — no credits were spent.",
        )
        raise SpendNotConfirmed("user declined the spend confirmation prompt")


def _consent(extra_args: list[str]) -> None:
    """``comfy generate consent [show|always|ask]`` — inspect or persist the
    spend gate's always-proceed setting (``spend.auto_confirm`` in config.ini,
    the same store that backs the CLI's other persisted settings)."""
    try:
        clean, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        _fail(code="generate_bad_args", message=str(e))
        raise typer.Exit(code=1)
    as_json = bool(meta.get("json", False))
    action = clean[0] if clean and not clean[0].startswith("-") else "show"
    if action not in {"show", "always", "ask"}:
        msg = f"Unknown consent action {action!r}. Usage: comfy generate consent [show|always|ask]"
        _fail(code="generate_bad_args", message=msg, legacy_json=as_json)
        raise typer.Exit(code=1)
    if action == "always":
        ConfigManager().set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "true")
    elif action == "ask":
        ConfigManager().set(constants.CONFIG_KEY_SPEND_AUTO_CONFIRM, "false")
    auto = _spend_auto_confirmed()
    if as_json:
        output.print_json({"spend_auto_confirm": auto, "action": action})
        return
    if auto:
        rprint("[bold]spend.auto_confirm: true[/bold] — `comfy generate` spends credits without prompting.")
        rprint("[dim]Revert with `comfy generate consent ask`.[/dim]")
    else:
        rprint("[bold]spend.auto_confirm: false[/bold] — credit-spending calls prompt first (or need --yes).")
        rprint("[dim]Persist always-proceed with `comfy generate consent always`.[/dim]")


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
            _bail(
                _track_error, e, code="generate_unknown_model", message=str(e), kind="schema", details={"model": model}
            )

        gen_props["model_alias"] = spec.preferred_alias(ep.id)
        gen_props["partner"] = getattr(ep, "partner", None)

        try:
            remaining, meta = _separate_meta_flags(extra_args)
        except schema.SchemaError as e:
            _bail(_track_error, e, code="generate_bad_args", message=str(e), kind="schema")

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
            name = gen_props["model_alias"] or ep.id
            _bail(
                _track_error,
                e,
                code="generate_bad_args",
                message=str(e),
                kind="schema",
                hint=f"Run `comfy generate schema {name}` for the full parameter list.",
            )

        if emit_path:
            # Emit a runnable workflow that drives the partner *node* and return
            # — no proxy call, no API key required. The artifact is the result.
            name = gen_props["model_alias"] or ep.id
            prefix = meta.get("output-prefix") if isinstance(meta.get("output-prefix"), str) else "generate"
            renderer = get_renderer()
            try:
                workflow = emit.write_workflow(name, values, Path(emit_path).expanduser(), output_prefix=prefix)
            except emit.UnsupportedModelError as e:
                # Its own code: the remedy is "pick another model", which is
                # not what the umbrella `emit_workflow_failed` hint says, and
                # the supported set travels as data rather than prose.
                _track_error("emit", e)
                renderer.error(
                    code="emit_workflow_unsupported_model",
                    message=str(e),
                    hint=(
                        "choose a model whose `emit_supported` is true in `comfy --json generate list` "
                        "(see `details.supported`), or call the model through the proxy without --emit-workflow"
                    ),
                    details={"model": e.model, "supported": e.supported},
                )
                raise typer.Exit(code=1) from e
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

        # Spend gate — a proxy call spends Comfy credits, so consent comes
        # BEFORE any network side effect (auth refresh, asset uploads, the
        # generation request itself). Everything above this line is local:
        # spec lookup, arg parsing, emit-workflow.
        try:
            _confirm_spend(
                model_name=str(gen_props["model_alias"] or ep.id),
                assume_yes=bool(meta.get("yes", False)),
                as_json=as_json,
            )
        except SpendNotConfirmed as e:
            _track_error("consent", e)
            raise typer.Exit(code=1) from e

        try:
            api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
        except client.ApiError as e:
            _bail(_track_error, e, code="generate_api_error", message=str(e), kind="api")

        timeout_raw = meta.get("timeout", "300")
        try:
            timeout = float(timeout_raw) if isinstance(timeout_raw, str) else 300.0
        except ValueError as e:
            _bail(
                _track_error,
                e,
                code="generate_timeout_invalid",
                message=f"--timeout: expected number, got {timeout_raw!r}",
                kind="schema",
            )

        try:
            _apply_upload_transforms(values, flags, ep, api_key)
        except (client.ApiError, httpx.HTTPError) as e:
            _bail(_track_error, e, code=_transport_code(e), message=f"Upload failed: {e}", kind="upload")

        request_id = str(uuid.uuid4())[:8]
        try:
            resp = client.send_request(ep, values, flags, api_key, timeout=timeout)
        except httpx.HTTPError as e:
            _bail(
                _track_error,
                e,
                code="generate_network_error",
                message=f"Network error contacting {spec.base_url()}: {e}",
                kind="network",
            )

        try:
            client.raise_for_status(resp)
        except client.ApiError as e:
            _bail(
                _track_error,
                e,
                code="generate_api_error",
                message=f"API error {e.status}",
                kind="api",
                details={"status": e.status, "body": e.body},
                pretty=f"[bold red]API error {e.status}[/bold red]\n{sanitize_markup(e.body)}",
            )

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
            preview = resp.text[:500]
            _bail(
                _track_error,
                e,
                code="generate_api_error",
                message="Unexpected non-JSON response.",
                kind="non_json_response",
                details={"body_preview": preview},
                pretty=f"[bold red]Unexpected non-JSON response.[/bold red]\n{sanitize_markup(preview)}",
            )

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
            poll_error: client.ApiError | httpx.HTTPError | None = None
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
                    poll_error = e
            if poll_error is not None:
                # The spinner is transient, so without this the run ended with a
                # bare exit 1 and an empty screen in EVERY mode. Reported AFTER
                # the `with` exits: inside it a transient Progress is still
                # auto-refreshing on stdout, so on a TTY the envelope would come
                # out interleaved with spinner control codes.
                _bail(
                    _track_error,
                    poll_error,
                    code=_transport_code(poll_error),
                    message=f"Job {job_id} failed while polling: {poll_error}",
                    kind=_track_kind(poll_error),
                )
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


def _renderer_for(meta: dict[str, str | bool]) -> Renderer:
    """Resolve the renderer for a `generate` sub-action.

    ``generate`` runs with ``allow_extra_args``, so a trailing ``--json`` never
    reaches the global Typer callback that resolves output mode — it lands in
    ``meta`` instead. Honor both spellings: the global ``comfy --json generate
    list`` (already reflected in the renderer's mode) and the tail
    ``comfy generate list --json`` (upgrade a still-pretty renderer).
    """
    renderer = get_renderer()
    if meta.get("json"):
        renderer.force_json()
    return renderer


def _model_record(e: spec.Endpoint) -> dict[str, object]:
    """One structured catalog row. ``category`` is what the human table renders
    under its "Style" column; ``summary`` is the FULL text, not the ``…``-
    truncated form the table has to cut to fit its width."""
    return {
        "alias": spec.preferred_alias(e.id) or e.id,
        "id": e.id,
        "partner": e.partner,
        "category": e.category,
        "mode": "async" if e.polling else "sync",
        "summary": e.summary,
        # Whether `--emit-workflow` has a partner-node mapping for this model.
        # Most of the catalog is proxy-only; an agent that could not see this
        # asked for a workflow it could never get (`emit_workflow_failed`).
        "emit_supported": emit.is_supported(e.id),
    }


def _param_record(f: schema.FlagDef) -> dict[str, object]:
    """One structured parameter row for `generate schema`.

    ``kind`` is retained alongside ``type`` because it is the vocabulary
    ``comfy_cli.command.generate.schema`` uses internally and the name the
    pre-envelope ``--json`` payload shipped; they always carry the same value.
    """
    return {
        "name": f.name,
        "type": f.kind,
        "kind": f.kind,
        "required": f.required,
        "default": f.default,
        "enum": list(f.enum),
        "description": f.description,
        "item_type": f.item_kind,
        "upload_mode": f.upload_mode,
    }


def _list_models(extra_args: list[str]) -> None:
    """`comfy generate list` — show available models with their short aliases."""
    try:
        clean, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        # e.g. `comfy generate list --download` (meta flag with no value) — this
        # used to escape as an unhandled SchemaError traceback.
        _fail(code="generate_bad_args", message=str(e))
        raise typer.Exit(code=1)
    renderer = _renderer_for(meta)
    partner = _arg_value(clean, "--partner", "-p")
    category = _arg_value(clean, "--category", "--style", "-c")
    query = _arg_value(clean, "--query", "-q")
    # `list`-only, deliberately NOT in `_separate_meta_flags`' meta_names: that
    # set applies to every generate sub-action, and --select belongs to the
    # four heavy read commands only (V1-011).
    select_expr = _arg_value(clean, "--select")
    eps = spec.list_endpoints(partner=partner, category=category, query=query)
    payload = {
        "models": [_model_record(e) for e in eps],
        "count": len(eps),
        "filters": {"partner": partner, "category": category, "query": query},
    }
    if select_expr is not None:
        from comfy_cli.selector import emit_selected

        return emit_selected(renderer, payload, select_expr, command="generate list")
    knowledge.attach(
        payload,
        command="generate list",
        queries=[query] if query else [],
        models=[m["alias"] for m in payload["models"]],
        brief=True,
        thin=(not eps and bool(query)),
        qualified=any(payload["filters"].values()),
    )
    if renderer.is_pretty():
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
        return
    renderer.emit(payload, command="generate list")


def _schema(extra_args: list[str]) -> None:
    """`comfy generate schema <model>` — show params for a model (fal-style)."""
    try:
        clean, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        _fail(code="generate_bad_args", message=str(e))
        raise typer.Exit(code=1)
    renderer = _renderer_for(meta)
    if not clean or clean[0].startswith("-"):
        if renderer.is_pretty():
            rprint("[bold red]Usage: comfy generate schema <model>[/bold red]")
        else:
            renderer.error(
                code="generate_bad_args",
                message="Usage: comfy generate schema <model>",
                hint="pass a model alias, e.g. `comfy generate schema flux-pro` "
                "(run `comfy generate list` to see them)",
                command="generate schema",
            )
        raise typer.Exit(code=1)
    try:
        ep = spec.get_endpoint(clean[0])
    except spec.SpecError as e:
        if renderer.is_pretty():
            # The message embeds `clean[0]` verbatim, so `comfy generate schema
            # '[/bold]'` reached Rich as an unbalanced closing tag and died with a
            # MarkupError traceback and empty stdout — the exact failure this
            # module's `_fail` path exists to prevent.
            rprint(f"[bold red]{sanitize_markup(e)}[/bold red]")
        else:
            renderer.error(
                code="generate_unknown_model",
                message=str(e),
                hint="run `comfy generate list` to see the available model aliases",
                details={"requested": clean[0]},
                command="generate schema",
            )
        raise typer.Exit(code=1)
    if renderer.is_pretty():
        _show_schema_help(ep)
        return
    flags = schema.flags_for(ep)
    name = spec.preferred_alias(ep.id) or ep.id
    payload = {
        "model": name,
        "id": ep.id,
        "partner": ep.partner,
        "category": ep.category,
        "summary": ep.summary,
        "mode": "async" if ep.polling else "sync",
        "polling": ep.polling,
        "content_type": ep.request_content_type,
        "params": [_param_record(f) for f in flags],
        "example": schema.example_invocation(ep, flags, display_name=name),
    }
    knowledge.attach(payload, command="generate schema", queries=[clean[0], name])
    renderer.emit(payload, command="generate schema")


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
        _fail(code="generate_network_error", message=f"Failed to fetch {fetched_from}: {e}")
        raise typer.Exit(code=1)

    # Validate before caching so a 200-with-garbage response never poisons the
    # ~/.comfy/openapi-cache.yml cache (used for CACHE_TTL_SECONDS by every
    # subsequent `comfy generate`).
    body = r.text
    try:
        spec.validate_spec_text(body)
    except spec.SpecError as e:
        _fail(code="generate_spec_invalid", message=f"Refusing to cache spec from {fetched_from}: {e}")
        raise typer.Exit(code=1)

    path = spec.write_cache(body)
    rprint(f"[bold green]Refreshed model catalog at {path}[/bold green]")


def _upload(extra_args: list[str]) -> None:
    """`comfy generate upload <file-or-url> [--json] [--api-key K]`."""
    try:
        remaining, meta = _separate_meta_flags(extra_args)
    except schema.SchemaError as e:
        _fail(code="generate_bad_args", message=str(e))
        raise typer.Exit(code=1)
    # `remaining` already excludes recognized --meta flags AND their values, so
    # `comfy generate upload --api-key KEY ./img.png` correctly resolves to "./img.png".
    if not remaining:
        _fail(
            code="generate_bad_args",
            message="Usage: comfy generate upload <file-or-url> [--json]",
        )
        raise typer.Exit(code=1)
    target = remaining[0]
    try:
        api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
    except client.ApiError as e:
        _fail(code="generate_api_error", message=str(e))
        raise typer.Exit(code=1)
    as_json = bool(meta.get("json", False))
    try:
        result = upload.upload_target(target, api_key)
    except (client.ApiError, httpx.HTTPError) as e:
        _fail(code=_transport_code(e), message=f"Upload failed: {e}")
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
        _fail(
            code="generate_bad_args",
            message="Usage: comfy generate resume <model> <job_id> [--download PATH] [--json]",
        )
        raise typer.Exit(code=1)
    model, job_id = extra_args[0], extra_args[1]
    tail = extra_args[2:]
    try:
        ep = spec.get_endpoint(model)
    except spec.SpecError as e:
        _fail(code="generate_unknown_model", message=str(e), details={"model": model})
        raise typer.Exit(code=1)
    if not ep.polling:
        _fail(
            code="generate_bad_args", message=f"{model} is a sync model; nothing to resume.", details={"model": model}
        )
        raise typer.Exit(code=1)
    try:
        _, meta = _separate_meta_flags(tail)
    except schema.SchemaError as e:
        _fail(code="generate_bad_args", message=str(e))
        raise typer.Exit(code=1)
    try:
        api_key = client.resolve_api_key(meta.get("api-key") if isinstance(meta.get("api-key"), str) else None)
    except client.ApiError as e:
        _fail(code="generate_api_error", message=str(e))
        raise typer.Exit(code=1)
    timeout_raw = meta.get("timeout")
    try:
        # Same guard as the submit path: an unguarded `float()` here let
        # `generate resume <model> <job> --timeout nope` escape `main()` (which
        # only traps KeyboardInterrupt/typer.Exit/SystemExit) as a traceback —
        # exit 1 with a blank stdout, the exact failure this change removes.
        timeout = float(timeout_raw or 300.0) if isinstance(timeout_raw, str) else 300.0
    except ValueError:
        _fail(code="generate_timeout_invalid", message=f"--timeout: expected number, got {timeout_raw!r}")
        raise typer.Exit(code=1)
    download = meta.get("download") if isinstance(meta.get("download"), str) else None
    as_json = bool(meta.get("json", False))

    try:
        initial = poll.build_synthetic_initial(ep.polling, job_id, base_url=spec.base_url())
    except client.ApiError as e:
        _fail(code="generate_api_error", message=str(e))
        raise typer.Exit(code=1)

    poller = poll.get_poller(ep.polling)
    poll_error: client.ApiError | httpx.HTTPError | None = None
    with _spinner() as prog:
        task = prog.add_task(f"Resuming job {job_id}", total=None)

        def _on_progress(p: float) -> None:
            prog.update(task, description=f"Job {job_id} ({p * 100:.0f}%)")

        try:
            result = poller(
                initial,
                api_key=api_key,
                timeout=timeout,
                on_progress=_on_progress,
                create_path=ep.path,
            )
        except (client.ApiError, httpx.HTTPError) as e:
            poll_error = e
    if poll_error is not None:
        # Same transient-spinner trap as the submit path: an unhandled poll
        # failure here used to surface as a raw traceback, and reporting it
        # inside the `with` would interleave the envelope with the live spinner.
        _fail(code=_transport_code(poll_error), message=f"Job {job_id} failed while polling: {poll_error}")
        raise typer.Exit(code=1) from poll_error
    _emit_result(result, request_id=job_id, download=download, as_json=as_json)


def _print_top_help() -> None:
    """Custom help that emphasizes the model-first UX over Typer's auto-help."""
    rprint("[bold]comfy generate[/bold] — call ComfyUI partner nodes")
    rprint("")
    rprint("[bold]Usage:[/bold]")
    rprint("  comfy generate <model> [--<param> value]... [--download PATH] [--async] [--yes] [--api-key KEY]")
    rprint("")
    rprint("[bold]Examples:[/bold]")
    rprint('  comfy generate flux-pro --prompt "a cat on the moon" --width 1024 --height 1024 --download cat.png')
    rprint(
        '  comfy generate ideogram-edit --image cat.png --mask m.png --prompt "add sunglasses" --rendering_speed TURBO'
    )
    rprint('  comfy generate dalle --prompt "a watercolor whale" --download whale.png')
    rprint(
        '  comfy generate flux-2 --prompt "a fox" --emit-workflow flux.json   '
        "[dim]# write a runnable workflow instead of calling the proxy[/dim]"
    )
    rprint("")
    rprint("[bold]Actions:[/bold]")
    rprint("  comfy generate list                    Browse available models")
    rprint("  comfy generate schema <model>          Show parameters for a model")
    rprint("  comfy generate refresh                 Refresh the model catalog")
    rprint("  comfy generate upload <file-or-url>    Host a local file or remote URL and print its signed URL")
    rprint("  comfy generate resume <model> <job>    Resume an async job")
    rprint("  comfy generate consent [show|always|ask]  Inspect/persist the credit-spend confirmation")
    rprint("")
    rprint(
        "[dim]Generation spends Comfy credits: interactive runs confirm first; pass --yes (or "
        "`comfy generate consent always`) to skip, required for --json / non-TTY runs.[/dim]"
    )
    rprint(
        "[dim]Auth: run `comfy cloud login` (session outranks env var), set COMFY_API_KEY, or pass --api-key. Get one at https://platform.comfy.org.[/dim]"
    )
