"""``comfy run`` — submit workflows to local or cloud.

This module is the public surface (``execute``, ``execute_cloud``) plus a
re-export hub for the helpers used by tests via
``patch("comfy_cli.command.run.X")``. The implementation lives in the
sibling submodules; nothing else should import from those directly so the
patch surface stays stable.
"""

import json
import sys
import time
import uuid
from datetime import timedelta
from urllib import request  # noqa: F401 — patch target for tests (run.request.urlopen)

import typer
from rich.markup import escape as _rich_escape
from websocket import (  # noqa: F401 — patch target for tests (run.WebSocket)
    WebSocket,
    WebSocketException,
    WebSocketTimeoutException,
)

from comfy_cli import cancellation, execution_errors, jobs_state, tracking
from comfy_cli.caller import stream_is_tty

# Re-exports — names patched by tests live at this namespace.
from comfy_cli.command.run.credentials import _resolve_partner_credential as _resolve_partner_credential
from comfy_cli.command.run.execution import ExecutionProgress as ExecutionProgress
from comfy_cli.command.run.execution import WorkflowExecution as WorkflowExecution
from comfy_cli.command.run.execution import _safe_close as _safe_close
from comfy_cli.command.run.loader import _MAX_BODY_PREVIEW as _MAX_BODY_PREVIEW
from comfy_cli.command.run.loader import WorkflowLoadError as WorkflowLoadError
from comfy_cli.command.run.loader import _classify_api_workflow as _classify_api_workflow
from comfy_cli.command.run.loader import _load_workflow_file as _load_workflow_file
from comfy_cli.command.run.loader import _node_errors_to_list as _node_errors_to_list
from comfy_cli.command.run.loader import is_ui_workflow as is_ui_workflow
from comfy_cli.command.run.loader import pop_compose_meta as pop_compose_meta
from comfy_cli.command.run.preflight import PARTNER_NODE_CATEGORY_PREFIXES as PARTNER_NODE_CATEGORY_PREFIXES
from comfy_cli.command.run.preflight import _detect_partner_nodes as _detect_partner_nodes
from comfy_cli.command.run.preflight import _fetch_object_info as _fetch_object_info
from comfy_cli.command.run.preflight import _preflight_validate as _preflight_validate
from comfy_cli.command.run.preflight import _resolve_default_checkpoint_or_exit as _resolve_default_checkpoint_or_exit
from comfy_cli.command.run.preflight import fetch_object_info as fetch_object_info
from comfy_cli.command.run.watcher import _spawn_watcher as _spawn_watcher
from comfy_cli.command.run.watcher import _tail_state_file as _tail_state_file
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.output.sanitize import sanitize_markup
from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()

# Bounds on the `partner_nodes` telemetry property and on the partner-node names
# echoed in the missing-credential error. class_type strings are attacker- (or
# just accident-) controlled workflow JSON, so cap the list length and each name.
_TELEMETRY_NODE_LIST_CAP = 20
_TELEMETRY_NODE_NAME_MAX_LEN = 64


def _bounded_node_names(names: list[str]) -> tuple[list[str], int]:
    """Cap a partner-node name list in BOTH dimensions — element count and each
    name's length. Returns ``(shown, omitted)``.

    Applies to every payload built from these names: the telemetry property,
    the prose message, and the structured ``details`` of both the credential
    error and the spend gate. Capping only the prose would bound nothing —
    ``error_panel`` renders ``details`` as ``key=value`` rows directly beneath
    the message in pretty mode, and JSON mode serialises them. The exact total
    travels alongside as ``partner_node_count``, so nothing is lost: a consumer
    still learns how many nodes are involved, and the remedy is identical
    whichever ones they are.

    De-duplicates AFTER truncation, not before: ``_detect_partner_nodes``
    returns distinct class_types, but two sharing a 64-char prefix collapse to
    the same string once truncated, which would list one name twice.

    Consumes input until the cap is filled with DISTINCT truncated names rather
    than slicing the first 20 up front — otherwise a run of prefix-colliding
    names burns output slots and silently drops later, genuinely distinct ones.

    ``omitted`` counts only what the cap actually left behind, never what
    de-duplication collapsed: a caller appending "and N more" must not claim
    unlisted nodes that don't exist.
    """
    shown: list[str] = []
    consumed = 0
    for name in names:
        if len(shown) >= _TELEMETRY_NODE_LIST_CAP:
            break
        consumed += 1
        truncated = name[:_TELEMETRY_NODE_NAME_MAX_LEN]
        if truncated not in shown:
            shown.append(truncated)
    return shown, len(names) - consumed


def _stdin_is_interactive() -> bool:
    """True only when stdin is a live TTY.

    ``sys.stdin.isatty()`` assumes stdin is a live stream, but in detached /
    ``pythonw`` contexts ``sys.stdin`` can be ``None``, closed, or backed by a
    revoked file descriptor. Treat every such case as non-interactive so the
    spend gate falls through to the fail-closed machine-mode error instead of
    raising an uncontrolled exception (BE-4326). Delegates to the shared
    fail-safe probe so stdin and stdout are guarded identically.
    """
    return stream_is_tty(getattr(sys, "stdin", None))


def _spend_gate(renderer, partner_nodes: list[str], allow_spend: bool, *, details: dict) -> None:
    """Consent interlock for partner-API (paid) nodes (BE-4326).

    Mirrors the ``comfy run-template`` gate (BE-4113): a workflow that embeds
    partner-API nodes (Veo/Kling/BFL/Gemini/…) spends Comfy credits when it
    runs, so require explicit consent before submitting. A no-op when there are
    no partner nodes or ``--allow-spend`` was passed, so partner-free runs are
    byte-identical.

    Fires BEFORE any partner-credential resolution / cloud auth so a refusal
    never triggers a network OAuth refresh. Raises ``typer.Exit(1)`` when
    consent is withheld; returns normally when the run may proceed.
    """
    if not partner_nodes or allow_spend:
        return
    # Bound the names here rather than at each call site, so both `execute` and
    # `execute_cloud` are covered: this gate runs on the same untrusted
    # class_type strings and is the MORE commonly hit branch (it fires before
    # any credential resolution), so leaving it unbounded would let a
    # pathological graph flood the terminal and the JSON envelope anyway.
    shown, omitted = _bounded_node_names(partner_nodes)
    details = {**details, "partner_nodes": shown, "partner_node_count": len(partner_nodes)}
    if renderer.is_pretty() and _stdin_is_interactive():
        # Escape class_type names before interpolating into Rich markup: a name
        # containing markup like ``[bold]`` would otherwise be parsed as a tag
        # (MarkupError/StyleSyntaxError, or injected formatting).
        names = ", ".join(_rich_escape(n) for n in shown) + (f", and {omitted} more" if omitted > 0 else "")
        pprint(f"[yellow]⚠ This workflow uses partner-API nodes that spend Comfy credits: {names}.[/yellow]")
        if not typer.confirm("Run anyway and spend credits?", default=False):
            renderer.error(
                code="spend_consent_required",
                message="declined — workflow not submitted, no credits spent",
                details=details,
            )
            raise typer.Exit(code=1)
    else:
        renderer.error(
            code="spend_consent_required",
            message=(
                "workflow uses partner-API (paid) nodes; re-run with --allow-spend to consent to spending Comfy credits"
            ),
            hint="paid nodes only run with explicit consent; free (non-partner) workflows run without this flag",
            details=details,
        )
        raise typer.Exit(code=1)


# Mapping from the deleted legacy `comfy run --json` dialect (JsonEmitter,
# `{"event": …, "error": {"kind": …}}`) to the renderer dialect
# (`{"schema": "event/1", "type": …}` events + final `type: "envelope"` line).
#
#   legacy event       → renderer event (type)
#   ------------------   ------------------------------------------------------
#   converted          → converted        {node_count}
#   prompt_preview     → prompt_preview   {prompt}
#   queued             → queued           {prompt_id, client_id,
#                                          validation_warnings, nodes}
#   node_executing     → executing        {node, class_type, title, prompt_id}
#   node_cached        → execution_cached {node, class_type, title, prompt_id}
#   node_progress      → progress         {node, completed, total, prompt_id}
#   node_executed      → executed         {node, class_type, title, outputs,
#                                          prompt_id} (+ one `output` {url}
#                                          event per file output)
#   completed          → envelope ok=true {data.prompt_id, data.outputs,
#                                          data.cached_node_ids,
#                                          data.executed_node_ids, …}
#   failed             → envelope ok=false {error.code, error.message,
#                                           error.hint, error.details}
#
#   legacy error.kind            → registered error.code      exit
#   ---------------------------   -------------------------   ----
#   workflow_not_found           → workflow_not_found          1
#   workflow_invalid_json        → workflow_invalid_json       1
#   workflow_read_error          → workflow_read_error         1
#   workflow_format_invalid      → workflow_not_api_format     1
#   workflow_empty               → workflow_empty              1
#   conversion_error             → conversion_error            1
#   conversion_crash             → conversion_crash            1
#   connection_error (probe)     → server_not_running          1
#   connection_error (network)   → connection_error            1
#   object_info_unavailable      → object_info_unavailable     1
#   validation_error             → prompt_rejected             1
#   client_error                 → client_error                1
#   server_error                 → server_error                1
#   invalid_response             → invalid_response            1
#   timeout                      → ws_timeout                  1
#   connection_lost              → ws_disconnected             1
#   execution_interrupted        → cancelled                   130
#   execution_error              → execution_error             1
def execute(
    workflow: str,
    host,
    port,
    *,
    wait: bool = False,
    verbose: bool = False,
    local_paths: bool = False,
    timeout: int = 30,
    notify: bool = False,
    api_key: str | None = None,
    print_prompt: bool = False,
    preloaded: tuple[dict, str, bool, bool] | None = None,
    allow_spend: bool = False,
):
    # `0.0.0.0` is a wildcard bind, not a connect address. macOS / Windows
    # clients can't reach it; on Linux it happens to resolve to a loopback.
    # Substitute the canonical loopback so every downstream use (server
    # probe, /prompt POST, emitted /view URLs) is portable.
    if host == "0.0.0.0":
        host = "127.0.0.1"

    # Reject hosts with URL-special chars that could cause injection in
    # f"http://{host}:{port}/..." URL construction.
    _unsafe = frozenset("/@?#")
    if any(c in host for c in _unsafe):
        raise typer.BadParameter(f"invalid host: {host!r}")

    renderer = get_renderer()

    # `preloaded` short-circuits file loading: an in-memory API-format graph
    # (e.g. the `comfy run --prompt` injected default) is handed straight in as
    # (workflow_dict, display_name, is_ui, checkpoint_user_set). Everything
    # downstream is unchanged; `checkpoint_user_set` gates runtime checkpoint
    # resolution for the bundled default (skip it when the user pinned one).
    if preloaded is not None:
        raw_workflow, workflow_name, is_ui, checkpoint_user_set = preloaded
    else:
        checkpoint_user_set = False
        try:
            raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
        except WorkflowLoadError as e:
            renderer.error(code=e.code, message=str(e), hint=e.hint)
            raise typer.Exit(code=1) from e

    if not print_prompt and not check_comfy_server_running(port, host, timeout=timeout):
        renderer.error(
            code="server_not_running",
            message=f"ComfyUI not running on specified address ({host}:{port})",
            hint="run: comfy launch",
            details={"host": host, "port": port},
        )
        raise typer.Exit(code=1)

    compose_meta: dict | None = None
    if is_ui:
        if renderer.is_pretty():
            pprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        object_info = fetch_object_info(host, port, timeout)
        try:
            workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            renderer.error(
                code="conversion_error",
                message=f"Workflow conversion failed: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
                details={"exception_type": type(e).__name__},
            )
            raise typer.Exit(code=1) from e
        if not workflow:
            renderer.error(
                code="workflow_empty",
                message="Workflow conversion produced no executable nodes",
            )
            raise typer.Exit(code=1)
        renderer.event("converted", node_count=len(workflow))
    else:
        kind, validated = _classify_api_workflow(raw_workflow)
        if kind == "empty":
            renderer.error(
                code="workflow_empty",
                message="API workflow contains no nodes",
            )
            raise typer.Exit(code=1)
        if kind == "invalid":
            renderer.error(
                code="workflow_not_api_format",
                message="Specified workflow does not appear to be an API workflow json file",
                hint="use 'File > Export (API)' in the ComfyUI frontend",
            )
            raise typer.Exit(code=1)
        workflow = validated
        # Strip the compose/1 provenance block before preflight + submit; the
        # server would reject (or warn on) a top-level non-node key. Keep its
        # foreach item map to stash on the job state at submit time.
        compose_meta = pop_compose_meta(workflow)

    # Partner-API node preflight (below) and runtime checkpoint resolution both
    # need the server's object_info. `--print-prompt` is a documented
    # no-server-hit dry-run, so skip the fetch + resolution there and print the
    # graph as-is; the real submit flow resolves BEFORE the prompt_preview event
    # so the streamed audit trail advertises the graph we actually submit.
    object_info: dict = {}
    if not print_prompt:
        object_info = _fetch_object_info(host, port)

        # Runtime checkpoint resolution for the bundled `--prompt` default: swap
        # the pinned checkpoint for one the local server actually has (or
        # hard-error if it has none). Guarded to the bundled default graph and
        # skipped when the user pinned the checkpoint explicitly (honor it; let
        # preflight reject it).
        if preloaded is not None and workflow_name == "default_text2img" and not checkpoint_user_set:
            _resolve_default_checkpoint_or_exit(renderer, workflow, object_info, where="local")

    # Stream mode: emit the workflow graph so agents have a complete audit
    # trail of what the CLI is about to submit (no-op otherwise).
    renderer.event("prompt_preview", prompt=workflow)

    # --print-prompt: emit/print the workflow and exit without submitting. No
    # server hit (documented) — the graph is shown as-is, before any
    # server-dependent checkpoint resolution.
    if print_prompt:
        if renderer.is_pretty():
            print(json.dumps(workflow, indent=2, ensure_ascii=False))
        else:
            renderer.emit(
                {"workflow": workflow_name, "status": "preview", "prompt": workflow},
                command="run",
                where="local",
            )
        return

    partner_nodes = _detect_partner_nodes(workflow, object_info)
    # Spend gate (BE-4326): partner-API nodes spend Comfy credits. Require
    # explicit consent before resolving a credential or submitting. Fires
    # BEFORE _resolve_partner_credential() below so a refusal never triggers a
    # network OAuth refresh. Detection stays fail-open (object_info == {} → no
    # partner_nodes → no gate), same posture as run-template.
    _spend_gate(
        renderer,
        partner_nodes,
        allow_spend,
        details={"partner_nodes": partner_nodes, "host": host, "port": port},
    )
    extra_data: dict | None = None
    if api_key:
        extra_data = {"api_key_comfy_org": api_key}
    if partner_nodes:
        # Only resolve an injected credential when an explicit --api-key hasn't
        # already satisfied the partner node: the resolver may perform a network
        # OAuth refresh, so skipping it here keeps an explicit-key run network-free.
        # Resolved once — the result feeds both the telemetry prop below and the
        # credential gate that follows.
        cred = _resolve_partner_credential() if not extra_data else None
        # Fired BEFORE the reject-for-missing-credential branch so runs that are
        # turned away are still counted: that funnel is exactly what the metric
        # is for, and `credential_present: False` marks them. class_types are
        # node names, not PII — the same data `workflow_unknown_nodes` reports.
        # It does sit AFTER the BE-4326 spend gate, so a run refused for lack of
        # `--allow-spend` emits no event: the gate deliberately precedes any
        # credential resolution (a refusal must not trigger a network OAuth
        # refresh), and `credential_present` needs that resolution. The
        # spend-declined funnel wants its own event rather than an early
        # resolve here.
        # class_type strings come verbatim from untrusted workflow JSON, so the
        # names are bounded before they ship. The count stays exact, so the cap
        # never distorts the metric.
        bounded_nodes, omitted_nodes = _bounded_node_names(partner_nodes)
        tracking.track_event(
            "partner_nodes_detected",
            {
                "partner_nodes": bounded_nodes,
                "partner_node_count": len(partner_nodes),
                "where": "local",
                "credential_present": bool(api_key) or cred is not None,
            },
        )
        if not extra_data:
            if cred is None:
                # Same bounded list in the prose and in `details` — a graph with
                # hundreds of partner nodes would otherwise render an unreadable
                # wall of text, and `details` is rendered right below the message
                # in pretty mode, so capping only one of them bounds nothing.
                # `partner_node_count` carries the exact total for consumers.
                # The suffix counts only what the CAP omitted — names collapsed
                # by de-duplication are still listed, so counting them would
                # promise unlisted nodes that don't exist.
                listed = ", ".join(bounded_nodes) + (f", and {omitted_nodes} more" if omitted_nodes > 0 else "")
                msg = (
                    "Workflow uses partner-API node(s) that need an `api_key_comfy_org` "
                    "credential the local server doesn't have: " + listed + "."
                )
                renderer.error(
                    code="partner_node_requires_credential",
                    message=msg,
                    hint=(
                        "run: comfy cloud login   (or set COMFY_API_KEY in the environment, "
                        "or persist a key with `comfy cloud set-key --key …`; "
                        "cloud runs auto-inject via --where cloud)"
                    ),
                    details={
                        "partner_nodes": bounded_nodes,
                        "partner_node_count": len(partner_nodes),
                        "host": host,
                        "port": port,
                    },
                )
                raise typer.Exit(code=1)
            extra_data = {cred[0]: cred[1]}

    # Pre-submit validation via pure-Python CQL engine (checks class_types + input shapes).
    _preflight_validate(renderer, workflow, object_info, target_label="server", where="local")

    progress = None
    start = time.time()
    if wait and renderer.is_pretty():
        pprint(f"[dim]▸[/dim] Executing [cyan]{workflow_name}[/cyan]")
        progress = ExecutionProgress()
        progress.start()

    execution = WorkflowExecution(
        workflow,
        host,
        port,
        verbose,
        progress,
        local_paths,
        timeout,
        extra_data=extra_data,
    )
    # Wire SIGINT → close the WebSocket so the loop exits promptly.
    token = cancellation.get_token()
    token.on_cancel(lambda: _safe_close(execution))

    # --wait only: the state written right after a successful submit, kept in
    # scope so the exception handlers below can record what happened to a
    # prompt that was already in flight (e.g. the server dying mid-run). Stays
    # None on the async path — that branch writes and owns its own state — and
    # before ``queue()`` has returned a prompt_id.
    wait_state: jobs_state.JobState | None = None
    # --wait success payload, rendered/emitted AFTER the try below rather than
    # inside it. Writing to a closed stdout (`comfy run --wait … | head`)
    # raises BrokenPipeError — a ConnectionError/OSError subclass the
    # disconnect handler would otherwise catch, rewriting the just-persisted
    # `completed` record to `error` and flipping a successful run's exit to 1.
    completed_payload: dict | None = None

    try:
        if wait:
            execution.connect()
        # Pretty + async: a brief spinner while the submit POST is in flight.
        # Falls through cleanly in machine modes (no rendering at all).
        if not wait and renderer.is_pretty():
            with renderer.console().status("[cyan]Submitting workflow…", spinner="dots"):
                execution.queue()
        else:
            execution.queue()
        _journal_run(workflow_name, execution.prompt_id, "local")
        if wait:
            # Write the state file at SUBMIT time, exactly like the async
            # branch below — if the server dies mid-run the on-disk record is
            # the only place the in-flight prompt_id survives. Status is
            # "running" rather than "queued": this foreground process is
            # actively watching it, not leaving it detached in the queue.
            #
            # `--wait` spawns no background watcher, by design; this process
            # *is* the watcher, and it stamps its own pid + create_time on the
            # submit-time record below to say so. Every ordinary outcome is
            # recorded by the handlers below: a node failure and a cancel via
            # `_mark_watch_exit`/`_mark_cancelled`, and a server that dies
            # mid-run via the `server_died` write in the
            # WebSocketException/OSError handler. This process being killed
            # from OUTSIDE (a caller-imposed timeout SIGKILLing the process
            # group, the terminal going away) runs no handler, but the stamp
            # covers it: the record is left non-terminal with a recorded,
            # now-dead pid, which is exactly what `jobs ls`'s stale-watcher
            # reap finalizes as `watcher_crashed`. Spawning a real watcher
            # here too was considered and rejected: it would put a second,
            # independent writer on the state file this branch already
            # finalizes, add a second server connection per foreground run,
            # and leave a background process behind after a synchronous
            # command returns. See docs/json-output.md, "Known limit:
            # `--wait` has no background watcher".
            wait_state = jobs_state.new(
                prompt_id=execution.prompt_id,
                client_id=execution.client_id,
                workflow=workflow_name,
                where="local",
                host=host,
                port=port,
            )
            wait_state.item_map = (compose_meta or {}).get("items")
            wait_state.status = "running"
            jobs_state.stamp_watcher_identity(wait_state)
            _write_state(wait_state)

            # `watch_execution` reports a terminal server event by rendering
            # the error and raising `typer.Exit` (1 for `execution_error`, 130
            # for `execution_interrupted`) — the ordinary failure path, not an
            # exception the handlers below see. Finalize the submit-time
            # record here: the stale-watcher reap would eventually flip a
            # `running` record with our now-dead pid to a generic
            # `watcher_crashed`, but this process knows the real verdict and
            # must record it while it still can.
            try:
                execution.watch_execution()
            except typer.Exit as exit_exc:
                _mark_watch_exit(wait_state, exit_exc.exit_code, execution)
                raise
            end = time.time()
            if progress is not None:
                progress.stop()
                progress = None

            if token.is_set():
                _mark_cancelled(wait_state)
                renderer.error(
                    code="cancelled",
                    message="Cancelled by user",
                    exit_code=130,
                )
                raise typer.Exit(code=130)

            # Completion updates the record written at submit — same file,
            # same final shape (``jobs_state.write`` stamps the timestamps).
            state = wait_state
            state.status = "completed"
            state.outputs = list(execution.outputs)
            # No fallback to the submit-time path: if the terminal write
            # failed, that file still says `running`, so handing it back as
            # the `completed` record's `state_file` would point the caller at
            # contents contradicting what we just reported.
            state_file = _write_state(state)

            # Grouped views of the same artifacts — local parity with the
            # cloud --wait envelope: by producing node always, and by
            # blueprint foreach item when compose embedded an item map.
            from comfy_cli.comfy_client import _group_outputs

            outputs_by_node, outputs_by_item = _group_outputs(list(execution.output_entries), state.item_map)

            completed_payload = {
                "workflow": workflow_name,
                "status": "completed",
                "prompt_id": execution.prompt_id,
                "client_id": execution.client_id,
                "outputs": list(execution.outputs),
                "outputs_by_node": outputs_by_node,
                "outputs_by_item": outputs_by_item,
                "cached_node_ids": list(execution.cached_node_ids),
                "executed_node_ids": list(execution.executed_node_ids),
                "elapsed_seconds": end - start,
                "host": host,
                "port": port,
                "state_file": str(state_file) if state_file else None,
            }
        else:
            # Async path (the default). Write the initial state file and
            # spawn a detached watcher to keep it updated; the foreground
            # caller returns immediately with the prompt_id.
            state = jobs_state.new(
                prompt_id=execution.prompt_id,
                client_id=execution.client_id,
                workflow=workflow_name,
                where="local",
                host=host,
                port=port,
            )
            state.item_map = (compose_meta or {}).get("items")
            state_file = jobs_state.write(state)
            watcher_spawned = _spawn_watcher(execution.prompt_id, where="local", host=host, port=port, notify=notify)

            if renderer.is_pretty():
                from comfy_cli.output.glyphs import status_glyph

                pprint(
                    f"{status_glyph('queued')} [dim]{execution.prompt_id}[/dim]\n"
                    f"  [dim]workflow [/dim]{workflow_name}\n"
                    f"  [dim]watch    [/dim][cyan]comfy jobs watch {execution.prompt_id}[/cyan]\n"
                    f"  [dim]state    [/dim]{state_file}"
                )
                if not watcher_spawned:
                    pprint(
                        "[yellow]⚠ Background watcher could not start; poll manually with `comfy jobs status`[/yellow]"
                    )
            renderer.emit(
                {
                    "workflow": workflow_name,
                    "status": "queued",
                    "prompt_id": execution.prompt_id,
                    "client_id": execution.client_id,
                    "outputs": [],
                    "elapsed_seconds": None,
                    "host": host,
                    "port": port,
                    "state_file": str(state_file) if state_file else None,
                    "watcher_spawned": watcher_spawned,
                },
                command="run",
                where="local",
            )
            # Pretty mode: brief live tail so the user can see the job
            # move through "allocated → executing → completed" without
            # having to run `comfy jobs watch`. The background watcher
            # keeps writing the state file after we return.
            _tail_state_file(execution.prompt_id)
    except KeyboardInterrupt:
        if progress is not None:
            progress.stop()
            progress = None
        _mark_cancelled(wait_state)
        if renderer.is_pretty():
            pprint("[yellow]Workflow execution was interrupted[/yellow]")
        renderer.error(
            code="cancelled",
            message="Workflow execution was interrupted",
            exit_code=130,
        )
        raise typer.Exit(code=130)
    except WebSocketTimeoutException:
        # The job may genuinely still be running server-side, so the
        # submit-time "running" record is left non-terminal — but the watcher
        # stamp MUST be cleared: this process is about to exit, and a
        # recorded-but-dead pid on a non-terminal record is precisely what the
        # stale-watcher reap finalizes, so the next `jobs ls` after a
        # `--wait --timeout 60` on a long-running job would flip a healthy
        # job's record to `error`/`watcher_crashed` — a regression worse than
        # the external-kill gap the stamp closes. `clear_watcher_identity`
        # re-reads under the record's lock and leaves an already-terminal
        # record alone, so a `comfy jobs cancel` that landed while we were
        # blocked here isn't walked back to `running`. The None guard is
        # required: `execution.connect()` raises this same exception, and it
        # runs before `wait_state` exists.
        #
        # Done BEFORE the pretty-mode render below: that render is blocking
        # (and can itself die on a closed stdout), and every millisecond
        # between "we know we're leaving" and "the stamp is gone" is a window
        # in which an external SIGKILL strands the stale pid.
        #
        # Consequence of the no-watcher limit documented at the submit-time
        # write above: nothing is left watching the prompt after this returns,
        # so if the server dies later no `server_died` is ever written. The
        # record stays `running` until the caller asks
        # `comfy jobs status <prompt_id>`, which infers the death from a
        # server that is down (or came back with no record of the prompt).
        if wait_state is not None:
            jobs_state.clear_watcher_identity(wait_state)
        if renderer.is_pretty():
            pprint(
                f"[bold red]Error: WebSocket timed out after {timeout}s waiting for server response.[/bold red]\n"
                "[yellow]For long-running workflows, increase the timeout: comfy run --workflow <file> --timeout 300[/yellow]"
            )
        details = {"timeout": timeout}
        prompt_id = _submitted_prompt_id(execution)
        if prompt_id is not None:
            details["prompt_id"] = prompt_id
        renderer.error(
            code="ws_timeout",
            message=f"WebSocket timed out after {timeout}s waiting for server response.",
            hint="re-run with a larger --timeout (e.g. --timeout 300)",
            details=details,
        )
        raise typer.Exit(code=1)
    except (WebSocketException, ConnectionError, OSError) as e:
        # If we closed the WebSocket ourselves in response to Ctrl-C, the recv
        # loop exits with a WebSocketException that *looks* like the server
        # vanished. Check the cancellation token first so we emit the right
        # error code (`cancelled`) instead of misleading users with
        # "check the server is still running".
        if token.is_set():
            if progress is not None:
                progress.stop()
            _mark_cancelled(wait_state)
            renderer.error(
                code="cancelled",
                message="Cancelled by user",
                exit_code=130,
            )
            raise typer.Exit(code=130) from e
        if renderer.is_pretty():
            pprint(f"[bold red]Error: Lost connection to ComfyUI server: {sanitize_markup(e)}[/bold red]")
        # The server died with a prompt of ours in flight: record that on the
        # submit-time state file and name the prompt in the emitted error, so
        # `comfy jobs status <id>` has something to consult afterwards.
        prompt_id = _submitted_prompt_id(execution)
        if prompt_id is not None and wait_state is not None:
            wait_state.status = "error"
            wait_state.error = {
                "code": "server_died",
                "message": f"Lost connection to ComfyUI while job {prompt_id} was running: {e}",
                "details": {},
            }
            # Same rule as the success path: report a state_file only when
            # this terminal write actually landed.
            path = _write_state(wait_state)
            renderer.error(
                code="ws_disconnected",
                message=f"Lost connection to ComfyUI server while job {prompt_id} was running: {e}",
                hint="check the server is still running; re-run the command",
                details={"prompt_id": prompt_id, "state_file": str(path) if path else None},
            )
            raise typer.Exit(code=1)
        renderer.error(
            code="ws_disconnected",
            message=f"Lost connection to ComfyUI server: {e}",
            hint="check the server is still running; re-run the command",
        )
        raise typer.Exit(code=1)
    finally:
        if progress is not None:
            progress.stop()
        # Best-effort close of the run WebSocket on every exit path. On the
        # async (no --wait) path connect() never ran so ws is None and this is
        # a no-op; it is idempotent with the SIGINT-token close wired above.
        _safe_close(execution)

    # Deliberately outside the try: the job is done and persisted, so a
    # BrokenPipeError from a closed stdout must surface as itself rather than
    # be mistaken for the server disconnecting (see `completed_payload`).
    if completed_payload is not None:
        if renderer.is_pretty():
            if completed_payload["outputs"]:
                pprint("[bold green]\nOutputs:[/bold green]")
                for f in completed_payload["outputs"]:
                    # Output paths are built from server-chosen filenames.
                    pprint(sanitize_markup(f))
            elapsed = timedelta(seconds=completed_payload["elapsed_seconds"])
            pprint(f"[bold green]\nWorkflow execution completed ({elapsed})[/bold green]")
        renderer.emit(completed_payload, command="run", where="local")


def _write_state(state):
    """Best-effort ``jobs_state.write``. Returns the path, or None if the
    write was skipped, the state dir was unwritable, or the prompt_id was
    unusable as a filename.

    A state-file failure must never fail an otherwise-successful run — same
    tolerance the async path gives a watcher that won't spawn. Swallowing
    ``ValueError`` matters now that ``--wait`` writes at SUBMIT: a server
    returning an id ``state_path()`` rejects must not stop us from watching
    the run it just accepted.
    """
    try:
        return jobs_state.write(state)
    except (OSError, ValueError):
        return None


def _mark_cancelled(state):
    """Record a user cancellation on the job state file (best effort).

    Returns the path written, or None when there is nothing to write — a
    Ctrl-C before ``queue()`` returned leaves ``state`` None, so there is no
    prompt_id to file it under, and a record that is ALREADY terminal is left
    alone. That last guard matters because the Ctrl-C handler is shared by the
    whole run: a Ctrl-C after the completion write has landed must not walk a
    ``completed`` job backwards to ``cancelled``.
    """
    if state is None or state.is_terminal:
        return None
    state.status = "cancelled"
    state.error = {"code": "cancelled", "message": "Cancelled by user", "details": {}}
    return _write_state(state)


def _mark_watch_exit(state, exit_code, execution):
    """Finalize the submit-time record when ``watch_execution`` ends the run by
    raising ``typer.Exit``: 130 is a server-side interrupt (``cancelled``),
    anything else is a node failure (``error``).

    The error envelope has already been rendered by the watcher, so this only
    moves the on-disk record off ``running``. Prefers the classified verdict
    ``on_error`` stashed on the execution; falls back to a generic
    ``execution_error`` when there isn't one (or it isn't a real dict — mocked
    executions hand back attribute stubs).
    """
    if state is None:
        return None
    if exit_code == 130:
        return _mark_cancelled(state)
    verdict = getattr(execution, "last_error", None)
    if not isinstance(verdict, dict):
        verdict = {
            "code": "execution_error",
            "message": "Workflow execution failed on the server",
            "details": {},
        }
    state.status = "error"
    state.error = verdict
    return _write_state(state)


def _submitted_prompt_id(execution) -> str | None:
    """The prompt id if the server really returned one, else None.

    Mirrors ``jobs_state.write``'s defensiveness about mocked executions: a
    non-string id means no usable submit happened, so callers fall back to
    their pre-prompt_id behavior.
    """
    prompt_id = getattr(execution, "prompt_id", None)
    if isinstance(prompt_id, str) and prompt_id.strip():
        return prompt_id
    return None


def _journal_run(workflow: str, prompt_id, where: str) -> None:
    """Append the run-submit event to the governing project's run journal
    (anchored at cwd). Wrapped end-to-end: a journaling failure can never
    fail the run."""
    try:
        from comfy_cli import project as project_module

        p = project_module.find_project()
        if p is not None:
            project_module.journal(p, cmd="run", workflow=str(workflow), prompt_id=prompt_id, where=where)
    except Exception:  # noqa: BLE001 — best-effort by contract
        pass


def _count_output_nodes(workflow: dict, object_info: dict) -> int | None:
    """Count nodes in ``workflow`` whose class is an output node, per
    ``object_info``. Returns None when object_info is empty/unknown so callers
    can skip the diff rather than reporting a bogus 0."""
    if not object_info:
        return None
    count = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        spec = object_info.get(ct) if isinstance(ct, str) else None
        if isinstance(spec, dict) and spec.get("output_node") is True:
            count += 1
    return count


def _returned_output_node_count(record: dict) -> int:
    """How many distinct nodes actually produced outputs in the cloud history
    record. The record's ``outputs`` map is keyed by node id."""
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return 0
    return sum(1 for v in outputs.values() if isinstance(v, dict) and v)


def execute_cloud(
    workflow: str,
    *,
    wait: bool = False,
    verbose: bool = False,
    timeout: int = 600,
    notify: bool = False,
    print_prompt: bool = False,
    preloaded: tuple[dict, str, bool, bool] | None = None,
    allow_spend: bool = False,
):
    """Run a workflow against Comfy Cloud via the stored OAuth session.

    Uses the unified :class:`comfy_cli.comfy_client.Client` — same surface as
    local, just a different :class:`comfy_cli.target.Target`.

    ``preloaded`` short-circuits file loading with an in-memory API-format graph
    (the ``comfy run --prompt`` injected default), mirroring :func:`execute`.
    """
    from comfy_cli.comfy_client import Client, HTTPError, Unauthenticated, _group_outputs
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    if preloaded is not None:
        raw_workflow, workflow_name, is_ui, checkpoint_user_set = preloaded
    else:
        checkpoint_user_set = False
        try:
            raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
        except WorkflowLoadError as e:
            renderer.error(code=e.code, message=str(e), hint=e.hint)
            raise typer.Exit(code=1) from e

    # The cloud object_info snapshot is used twice below (UI→API conversion and
    # checkpoint resolution/preflight). `_load_from_target` is a live, uncached
    # HTTPS fetch, so load it at most once and share it across both. `None`
    # means "not fetched yet" — distinct from a fetched-but-empty snapshot,
    # which must NOT trigger a second round-trip.
    cloud_object_info: dict | None = None

    if is_ui:
        # Frontend-format workflows (the `nodes`+`links` shape from the canvas
        # exporter and `comfy templates fetch`) have to be lowered to the API
        # shape before submit. We do it client-side using the cloud snapshot
        # of object_info — the cloud server has no /workflow/convert endpoint.
        from comfy_cli.cql.engine import _load_from_target

        if renderer.is_pretty():
            pprint("[yellow]Detected UI-format workflow, converting to API format…[/yellow]")
        try:
            object_info = cloud_object_info = _load_from_target(mode="cloud")
        except Exception as e:  # noqa: BLE001
            renderer.error(
                code="cql_no_graph",
                message=f"could not load cloud object_info for conversion: {e}",
                hint="object_info is fetched live — check your cloud sign-in and connection then retry, or run against a local server",
            )
            raise typer.Exit(code=1) from e
        try:
            raw_workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            renderer.error(
                code="conversion_error",
                message=f"Workflow conversion failed: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format and retry",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:  # noqa: BLE001
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
            )
            raise typer.Exit(code=1) from e
        if not raw_workflow:
            renderer.error(
                code="workflow_empty",
                message="Workflow conversion produced no executable nodes",
            )
            raise typer.Exit(code=1)

    kind, parsed_workflow = _classify_api_workflow(raw_workflow)
    if kind != "ok":
        renderer.error(
            code="workflow_not_api_format",
            message="Specified workflow does not appear to be an API workflow json file",
            hint="use 'File > Export (API)' in the ComfyUI frontend",
        )
        raise typer.Exit(code=1)

    # Strip the compose/1 provenance block before preflight + submit, keeping
    # its foreach item map to stash on the job state at submit time.
    compose_meta = pop_compose_meta(parsed_workflow)

    # Cloud path uses cached/bundled object_info (no live server needed). Load
    # it up front so checkpoint resolution can run BEFORE the preview/print
    # below — the audit trail must advertise the graph we actually submit.
    # Already fetched above when the workflow arrived in UI format.
    if cloud_object_info is None:
        try:
            from comfy_cli.cql.engine import _load_from_target

            cloud_object_info = _load_from_target(mode="cloud")
        except Exception:  # noqa: BLE001
            cloud_object_info = {}

    # Runtime checkpoint resolution for the bundled `--prompt` default (mirrors
    # the local path): swap the pinned checkpoint for one Comfy Cloud actually
    # has. Guarded to the bundled default and skipped when the user pinned the
    # checkpoint explicitly. Cloud fails open on an empty enum (per-job models).
    if preloaded is not None and workflow_name == "default_text2img" and not checkpoint_user_set:
        _resolve_default_checkpoint_or_exit(renderer, parsed_workflow, cloud_object_info, where="cloud")

    if print_prompt:
        # Documented dry-run: show the API-format graph that WOULD be sent and
        # exit WITHOUT POSTing. Mirrors local execute()'s print_prompt branch.
        if renderer.is_pretty():
            print(json.dumps(parsed_workflow, indent=2, ensure_ascii=False))
        else:
            renderer.event("prompt_preview", prompt=parsed_workflow)
            renderer.emit(
                {"workflow": workflow_name, "status": "preview", "prompt": parsed_workflow},
                command="run",
                where="cloud",
            )
        raise typer.Exit(code=0)

    # Pre-submit validation via pure-Python CQL engine.
    _preflight_validate(renderer, parsed_workflow, cloud_object_info, target_label="cloud", where="cloud")

    # Spend gate (BE-4326): the cloud also bills partner-API nodes, so apply the
    # same consent interlock as the local path before authenticating/submitting.
    # Fail-open on detection (empty cloud object_info → no gate), and fire before
    # Client() so a refusal never triggers cloud auth.
    partner_nodes = _detect_partner_nodes(parsed_workflow, cloud_object_info)
    _spend_gate(
        renderer,
        partner_nodes,
        allow_spend,
        details={"partner_nodes": partner_nodes, "where": "cloud"},
    )

    target = resolve_target(where="cloud")
    try:
        client = Client(target, timeout=float(timeout))
    except Unauthenticated as e:
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy cloud login")
        raise typer.Exit(code=1) from e

    client_id = str(uuid.uuid4())
    start = time.time()

    if wait:
        if renderer.is_pretty():
            pprint(f"[dim]▸[/dim] Executing [cyan]{workflow_name}[/cyan] on Comfy Cloud")
            pprint(f"[dim]  base_url: {target.base_url}[/dim]")
        else:
            renderer.event("executing", workflow=workflow_name, base_url=target.base_url)
    elif not renderer.is_pretty():
        renderer.event("queued", workflow=workflow_name, base_url=target.base_url)

    try:
        if not wait and renderer.is_pretty():
            with renderer.console().status("[cyan]Submitting to Comfy Cloud…", spinner="dots"):
                submit = client.submit_prompt(parsed_workflow, client_id)
        else:
            submit = client.submit_prompt(parsed_workflow, client_id)
    except Unauthenticated as e:
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy cloud login")
        raise typer.Exit(code=1) from e
    except HTTPError as e:
        renderer.error(
            code="cloud_http_error",
            message=f"Cloud server rejected the workflow (HTTP {e.status}): {e.message}",
            hint="check the workflow is valid and the cloud server has the required nodes",
            details={"status": e.status, "body": e.body[:2000]},
        )
        raise typer.Exit(code=1) from e

    if submit.node_errors:
        # Parse per-node errors into readable hint lines
        hint_lines = []
        for nid, record in submit.node_errors.items():
            if not isinstance(record, dict):
                continue
            ct = record.get("class_type", "unknown")
            for err in record.get("errors") or []:
                detail = err.get("details", "") or err.get("message", "")
                hint_lines.append(f"node {nid} ({ct}): {detail}")
        renderer.error(
            code="prompt_rejected",
            message=f"Cloud server rejected {len(submit.node_errors)} node(s)",
            hint="\n".join(hint_lines) if hint_lines else "inspect node_errors in details",
            details={"node_errors": submit.node_errors},
        )
        raise typer.Exit(code=1)

    if not wait:
        state = jobs_state.new(
            prompt_id=submit.prompt_id,
            client_id=client_id,
            workflow=workflow_name,
            where="cloud",
            base_url=target.base_url,
        )
        state.item_map = (compose_meta or {}).get("items")
        state_file = jobs_state.write(state)
        _journal_run(workflow_name, submit.prompt_id, "cloud")
        watcher_spawned = _spawn_watcher(submit.prompt_id, where="cloud", notify=notify)

        if renderer.is_pretty():
            from comfy_cli.output.glyphs import status_glyph

            pprint(
                f"{status_glyph('queued')} [dim]{submit.prompt_id}[/dim]\n"
                f"  [dim]workflow [/dim]{workflow_name}\n"
                f"  [dim]watch    [/dim][cyan]comfy jobs watch {submit.prompt_id} --where cloud[/cyan]\n"
                f"  [dim]state    [/dim]{state_file}"
            )
            if not watcher_spawned:
                pprint("[yellow]⚠ Background watcher could not start; poll manually with `comfy jobs status`[/yellow]")
        renderer.emit(
            {
                "workflow": workflow_name,
                "status": "queued",
                "prompt_id": submit.prompt_id,
                "client_id": client_id,
                "outputs": [],
                "elapsed_seconds": None,
                "base_url": target.base_url,
                "state_file": str(state_file) if state_file else None,
            },
            command="run",
            where="cloud",
        )
        # Pretty mode only: short live tail of the state file so the human
        # sees status transitions before the foreground exits.
        _tail_state_file(submit.prompt_id)
        return

    # --wait: poll the cloud API directly from the foreground process. No
    # watcher subprocess is spawned — this process is the watcher, so it
    # stamps its own pid + create_time on the submit-time record. Every
    # in-process exit below (timeout, auth failure, HTTP error, Ctrl-C,
    # failure, success) writes a terminal state the reap ignores; only an
    # external kill leaves the record non-terminal, and its now-dead pid is
    # what lets `jobs ls`'s stale-watcher reap finalize it as
    # `watcher_crashed` instead of stranding it `running` forever.
    state = jobs_state.new(
        prompt_id=submit.prompt_id,
        client_id=client_id,
        workflow=workflow_name,
        where="cloud",
        base_url=target.base_url,
    )
    state.item_map = (compose_meta or {}).get("items")
    jobs_state.stamp_watcher_identity(state)
    state_file = jobs_state.write(state)
    _journal_run(workflow_name, submit.prompt_id, "cloud")

    # Guard every exit from here on. `Client._request` only converts
    # `urllib.error.HTTPError`, so a DNS failure, a connection reset, a TLS
    # error or a non-JSON body from `wait_for_completion` escapes as a bare
    # `URLError`/`OSError`/`ValueError` and matches none of the handlers
    # below; `extract_outputs` and the rendering after them are likewise
    # unguarded. Such an escape kills this process with the record still
    # non-terminal AND stamped, which is exactly what the next `jobs ls`
    # reaps to `error`/`watcher_crashed` — asserting a crash verdict about a
    # CLOUD job that is very likely still running server-side and that
    # `comfy jobs status <id> --where cloud` could otherwise reconcile
    # against the API. Dropping the stamp on the way out leaves what an
    # un-stamped `--wait` left before: a non-terminal record nobody is
    # watching. Every handled exit below writes a terminal record first, and
    # `clear_watcher_identity` re-reads under the lock and no-ops on those.
    try:
        try:

            def _probe():
                st = client.get_job_status(submit.prompt_id)
                if not st:
                    return None
                return (st.get("status"), st.get("progress"), st.get("queue_position"))

            record = client.wait_for_completion(submit.prompt_id, timeout=float(timeout), progress_probe=_probe)
        except TimeoutError as e:
            state.status = "error"
            state.error = {"code": "cloud_timeout", "message": str(e)}
            jobs_state.write(state)
            renderer.error(
                code="cloud_timeout",
                message=str(e),
                hint=f"the cloud job went silent for {timeout}s; raise --timeout or watch via `comfy jobs watch {submit.prompt_id} --where cloud`",
                details={"prompt_id": submit.prompt_id},
            )
            raise typer.Exit(code=1) from e
        except Unauthenticated as e:
            state.status = "error"
            state.error = {"code": "cloud_unauthorized", "message": str(e)}
            jobs_state.write(state)
            renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy cloud login")
            raise typer.Exit(code=1) from e
        except HTTPError as e:
            state.status = "error"
            state.error = {"code": "cloud_http_error", "message": str(e)}
            jobs_state.write(state)
            renderer.error(
                code="cloud_http_error",
                message=f"Cloud server error while polling (HTTP {e.status}): {e.message}",
                details={"status": e.status, "prompt_id": submit.prompt_id},
            )
            raise typer.Exit(code=1) from e
        except KeyboardInterrupt:
            state.status = "cancelled"
            jobs_state.write(state)
            renderer.error(code="cancelled", message="Cancelled by user", exit_code=130)
            raise typer.Exit(code=130)

        # Determine the terminal status from the record.
        node_outputs = client.extract_outputs(record)
        output_urls = [o["url"] for o in node_outputs]
        exec_status = record.get("status") or record.get("execution_status") or {}
        if isinstance(exec_status, dict):
            status_str = exec_status.get("status_str", "")
        else:
            status_str = str(exec_status).lower()

        if status_str in ("error", "failed"):
            verdict = execution_errors.classify(record.get("error_message") or status_str)
            state.status = "error"
            state.error = {
                "code": verdict["code"],
                "message": verdict["message"],
                "details": verdict["details"],
            }
            state_file = jobs_state.write(state)
            renderer.error(
                code=verdict["code"],
                message=verdict["message"],
                hint=verdict["hint"],
                details={"prompt_id": submit.prompt_id, "status": status_str, **verdict["details"]},
            )
            raise typer.Exit(code=1)

        # Success path.
        state.status = "completed"
        state.outputs = output_urls
        # Stash the full node-keyed history record for downstream consumers
        # (grouped outputs, item-named downloads).
        state.record = record
        state_file = jobs_state.write(state)

        end = time.time()

        # Silent-partial-execution guard: the cloud prunes branches that fail
        # server-side validation and still reports `completed`. Diff the output
        # nodes we submitted against the ones that actually returned outputs so a
        # vanished branch surfaces instead of passing as a clean success.
        warnings: list[dict] = []
        submitted_outputs = _count_output_nodes(parsed_workflow, cloud_object_info)
        returned_outputs = _returned_output_node_count(record)
        if submitted_outputs is not None and returned_outputs < submitted_outputs:
            warnings.append(
                {
                    "code": "partial_execution",
                    "message": (
                        f"submitted {submitted_outputs} output node(s) but the cloud returned outputs "
                        f"for only {returned_outputs}; {submitted_outputs - returned_outputs} branch(es) "
                        "were pruned server-side (likely failed validation) and produced nothing"
                    ),
                    "submitted_output_nodes": submitted_outputs,
                    "returned_output_nodes": returned_outputs,
                }
            )

        if renderer.is_pretty():
            if output_urls:
                pprint("[bold green]\nOutputs:[/bold green]")
                for u in output_urls:
                    pprint(sanitize_markup(u))
            for w in warnings:
                pprint(f"[yellow]⚠ {sanitize_markup(w['message'])}[/yellow]")
            pprint(f"[bold green]\nCloud workflow completed ({timedelta(seconds=end - start)})[/bold green]")

        # Grouped views of the same artifacts: by producing node always, and by
        # blueprint foreach item when compose stashed an item_map at submit.
        outputs_by_node, outputs_by_item = _group_outputs(node_outputs, state.item_map)

        renderer.emit(
            {
                "workflow": workflow_name,
                "status": state.status,
                "prompt_id": submit.prompt_id,
                "client_id": client_id,
                "outputs": output_urls,
                "outputs_by_node": outputs_by_node,
                "outputs_by_item": outputs_by_item,
                "warnings": warnings,
                "elapsed_seconds": end - start,
                "base_url": target.base_url,
                "state_file": str(state_file) if state_file else None,
            },
            command="run",
            where="cloud",
        )
    except BaseException:
        jobs_state.clear_watcher_identity(state)
        raise
