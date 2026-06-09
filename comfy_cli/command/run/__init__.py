"""``comfy run`` — submit workflows to local or cloud.

This module is the public surface (``execute``, ``execute_cloud``) plus a
re-export hub for the helpers used by tests via
``patch("comfy_cli.command.run.X")``. The implementation lives in the
sibling submodules; nothing else should import from those directly so the
patch surface stays stable.
"""

import json
import time
import uuid
from datetime import timedelta
from urllib import request  # noqa: F401 — patch target for tests (run.request.urlopen)

import typer
from websocket import (  # noqa: F401 — patch target for tests (run.WebSocket)
    WebSocket,
    WebSocketException,
    WebSocketTimeoutException,
)

from comfy_cli import cancellation, jobs_state

# Re-exports — names patched by tests live at this namespace.
from comfy_cli.command.run.credentials import _resolve_partner_credential as _resolve_partner_credential
from comfy_cli.command.run.emitter import JsonEmitter as JsonEmitter
from comfy_cli.command.run.execution import ExecutionProgress as ExecutionProgress
from comfy_cli.command.run.execution import WorkflowExecution as WorkflowExecution
from comfy_cli.command.run.execution import _safe_close as _safe_close
from comfy_cli.command.run.loader import _MAX_BODY_PREVIEW as _MAX_BODY_PREVIEW
from comfy_cli.command.run.loader import SCHEMA_VERSION as SCHEMA_VERSION
from comfy_cli.command.run.loader import WorkflowLoadError as WorkflowLoadError
from comfy_cli.command.run.loader import _classify_api_workflow as _classify_api_workflow
from comfy_cli.command.run.loader import _load_workflow_file as _load_workflow_file
from comfy_cli.command.run.loader import _node_errors_to_list as _node_errors_to_list
from comfy_cli.command.run.loader import is_ui_workflow as is_ui_workflow
from comfy_cli.command.run.preflight import PARTNER_NODE_CATEGORY_PREFIX as PARTNER_NODE_CATEGORY_PREFIX
from comfy_cli.command.run.preflight import _detect_partner_nodes as _detect_partner_nodes
from comfy_cli.command.run.preflight import _fetch_object_info as _fetch_object_info
from comfy_cli.command.run.preflight import _preflight_validate as _preflight_validate
from comfy_cli.command.run.preflight import fetch_object_info as fetch_object_info
from comfy_cli.command.run.watcher import _spawn_watcher as _spawn_watcher
from comfy_cli.command.run.watcher import _tail_state_file as _tail_state_file
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()


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
    json_mode: bool = False,
    print_prompt: bool = False,
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
    emitter = JsonEmitter(json_mode=json_mode)

    try:
        raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
    except WorkflowLoadError as e:
        if json_mode:
            emitter.emit_failed(e.kind, str(e))
            raise typer.Exit(code=1) from e
        renderer.error(code=e.kind, message=str(e), hint=e.hint)
        raise typer.Exit(code=1) from e

    if not print_prompt and not check_comfy_server_running(port, host, timeout=timeout):
        if json_mode:
            emitter.emit_failed("connection_error", f"ComfyUI not running on specified address ({host}:{port})")
            raise typer.Exit(code=1)
        renderer.error(
            code="server_not_running",
            message=f"ComfyUI not running on specified address ({host}:{port})",
            hint="run: comfy launch",
            details={"host": host, "port": port},
        )
        raise typer.Exit(code=1)

    if is_ui:
        if not json_mode and renderer.is_pretty():
            pprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        object_info = fetch_object_info(host, port, timeout, emitter=emitter if json_mode else None)
        try:
            workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            if json_mode:
                emitter.emit_failed("conversion_error", f"Workflow conversion failed: {e}")
                raise typer.Exit(code=1) from e
            renderer.error(
                code="conversion_error",
                message=f"Workflow conversion failed: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:
            if json_mode:
                emitter.emit_failed(
                    "conversion_crash",
                    f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                    exception_type=type(e).__name__,
                )
                raise typer.Exit(code=1) from e
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
            )
            raise typer.Exit(code=1) from e
        if not workflow:
            if json_mode:
                emitter.emit_failed("workflow_empty", "Workflow conversion produced no executable nodes")
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_empty",
                message="Workflow conversion produced no executable nodes",
            )
            raise typer.Exit(code=1)
        if json_mode:
            emitter.emit_converted(len(workflow))
    else:
        kind, validated = _classify_api_workflow(raw_workflow)
        if kind == "empty":
            if json_mode:
                emitter.emit_failed("workflow_empty", "API workflow contains no nodes")
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_empty",
                message="API workflow contains no nodes",
            )
            raise typer.Exit(code=1)
        if kind == "invalid":
            if json_mode:
                emitter.emit_failed(
                    "workflow_format_invalid",
                    "Specified workflow does not appear to be an API workflow json file",
                )
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_not_api_format",
                message="Specified workflow does not appear to be an API workflow json file",
                hint="use 'File > Export (API)' in the ComfyUI frontend",
            )
            raise typer.Exit(code=1)
        workflow = validated

    emitter.set_workflow(workflow)

    # In JSON mode, emit the workflow so agents have a complete audit trail.
    if json_mode:
        emitter.emit_prompt_preview(workflow)

    # --print-prompt: emit/print the workflow and exit without submitting.
    if print_prompt:
        if not json_mode:
            print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return

    # Partner-API node preflight. Reject up-front when the workflow
    # depends on a partner node (Veo/Kling/BFL/Gemini/…) and we have no
    # credential to inject. If we DO have a credential, plumb it into
    # extra_data so the partner node finds it server-side — same shape
    # the cloud submit path uses.
    object_info = _fetch_object_info(host, port, timeout)
    partner_nodes = _detect_partner_nodes(workflow, object_info)
    extra_data: dict | None = None
    if api_key:
        extra_data = {"api_key_comfy_org": api_key}
    if partner_nodes:
        cred = _resolve_partner_credential()
        if cred is None and not extra_data:
            msg = (
                "Workflow uses partner-API node(s) that need an `api_key_comfy_org` "
                "credential the local server doesn't have: " + ", ".join(partner_nodes) + "."
            )
            if json_mode:
                emitter.emit_failed(
                    "partner_node_requires_credential",
                    msg,
                    partner_nodes=partner_nodes,
                )
                raise typer.Exit(code=1)
            renderer.error(
                code="partner_node_requires_credential",
                message=msg,
                hint=(
                    "re-submit with `--where cloud` (the CLI auto-injects the key there), "
                    "or store the key locally with `comfy auth set comfy-cloud-api-key --key …`"
                ),
                details={
                    "partner_nodes": partner_nodes,
                    "host": host,
                    "port": port,
                },
            )
            raise typer.Exit(code=1)
        elif cred is not None and not extra_data:
            extra_data = {cred[0]: cred[1]}

    # Pre-submit validation via pure-Python CQL engine (checks class_types + input shapes).
    _preflight_validate(renderer, workflow, object_info, target_label="server", emitter=emitter)

    progress = None
    start = time.time()
    if wait:
        if not json_mode and renderer.is_pretty():
            pprint(f"[dim]▸[/dim] Executing [cyan]{workflow_name}[/cyan]")
            progress = ExecutionProgress()
            progress.start()
        elif not json_mode:
            renderer.event("executing", workflow=workflow_name, host=host, port=port)
    else:
        if not json_mode and not renderer.is_pretty():
            renderer.event("queued", workflow=workflow_name, host=host, port=port)

    execution = WorkflowExecution(
        workflow,
        host,
        port,
        verbose,
        progress,
        local_paths,
        timeout,
        extra_data=extra_data,
        emitter=emitter,
    )
    emitter.set_client_id(execution.client_id)
    # Wire SIGINT → close the WebSocket so the loop exits promptly.
    token = cancellation.get_token()
    token.on_cancel(lambda: _safe_close(execution))

    try:
        if wait:
            execution.connect()
        # Pretty + async: a brief spinner while the submit POST is in flight.
        # Falls through cleanly on JSON mode (no rendering at all).
        if not wait and not json_mode and renderer.is_pretty():
            with renderer.console().status("[cyan]Submitting workflow…", spinner="dots"):
                execution.queue()
        else:
            execution.queue()
        if wait:
            execution.watch_execution()
            end = time.time()
            if progress is not None:
                progress.stop()
                progress = None

            if token.is_set():
                if json_mode:
                    emitter.emit_failed("execution_interrupted", "Cancelled by user")
                    raise typer.Exit(code=130)
                renderer.error(
                    code="cancelled",
                    message="Cancelled by user",
                    exit_code=130,
                )
                raise typer.Exit(code=130)

            if json_mode:
                emitter.emit_completed()

            # Foreground (--wait) completion path also writes the state
            # file so the on-disk record is consistent regardless of which
            # mode the user ran in.
            state = jobs_state.new(
                prompt_id=execution.prompt_id,
                client_id=execution.client_id,
                workflow=workflow_name,
                where="local",
                host=host,
                port=port,
            )
            state.status = "completed"
            state.outputs = list(execution.outputs)
            state_file = jobs_state.write(state)

            if not json_mode:
                if renderer.is_pretty():
                    if len(execution.outputs) > 0:
                        pprint("[bold green]\nOutputs:[/bold green]")
                        for f in execution.outputs:
                            pprint(f)
                    elapsed = timedelta(seconds=end - start)
                    pprint(f"[bold green]\nWorkflow execution completed ({elapsed})[/bold green]")
                renderer.emit(
                    {
                        "workflow": workflow_name,
                        "status": "completed",
                        "prompt_id": execution.prompt_id,
                        "client_id": execution.client_id,
                        "outputs": list(execution.outputs),
                        "elapsed_seconds": end - start,
                        "host": host,
                        "port": port,
                        "state_file": str(state_file) if state_file else None,
                    },
                    command="run",
                    where="local",
                )
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
            state_file = jobs_state.write(state)
            watcher_spawned = _spawn_watcher(execution.prompt_id, where="local", host=host, port=port, notify=notify)

            if not json_mode:
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
        if json_mode:
            emitter.emit_failed("execution_interrupted", "Workflow execution was interrupted")
            raise typer.Exit(code=1)
        pprint("[yellow]Workflow execution was interrupted[/yellow]")
        raise typer.Exit(code=1)
    except WebSocketTimeoutException:
        if json_mode:
            emitter.emit_failed(
                "timeout",
                f"WebSocket timed out after {timeout}s waiting for server response.",
                timeout_seconds=float(timeout),
            )
            raise typer.Exit(code=1)
        if renderer.is_pretty():
            pprint(
                f"[bold red]Error: WebSocket timed out after {timeout}s waiting for server response.[/bold red]\n"
                "[yellow]For long-running workflows, increase the timeout: comfy run --workflow <file> --timeout 300[/yellow]"
            )
        renderer.error(
            code="ws_timeout",
            message=f"WebSocket timed out after {timeout}s waiting for server response.",
            hint="re-run with a larger --timeout (e.g. --timeout 300)",
            details={"timeout": timeout},
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
            if json_mode:
                emitter.emit_failed("execution_interrupted", "Cancelled by user")
                raise typer.Exit(code=130) from e
            renderer.error(
                code="cancelled",
                message="Cancelled by user",
                exit_code=130,
            )
            raise typer.Exit(code=130) from e
        if json_mode:
            emitter.emit_failed("connection_lost", f"Lost connection to ComfyUI server: {e}")
            raise typer.Exit(code=1)
        if renderer.is_pretty():
            pprint(f"[bold red]Error: Lost connection to ComfyUI server: {e}[/bold red]")
        renderer.error(
            code="ws_disconnected",
            message=f"Lost connection to ComfyUI server: {e}",
            hint="check the server is still running; re-run the command",
        )
        raise typer.Exit(code=1)
    finally:
        if progress is not None:
            progress.stop()


def execute_cloud(
    workflow: str,
    *,
    wait: bool = False,
    verbose: bool = False,
    timeout: int = 600,
    notify: bool = False,
):
    """Run a workflow against Comfy Cloud via the stored OAuth session.

    Uses the unified :class:`comfy_cli.comfy_client.Client` — same surface as
    local, just a different :class:`comfy_cli.target.Target`.
    """
    from comfy_cli.comfy_client import Client, HTTPError, Unauthenticated
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    try:
        raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
    except WorkflowLoadError as e:
        renderer.error(code=e.kind, message=str(e), hint=e.hint)
        raise typer.Exit(code=1) from e

    if is_ui:
        # Frontend-format workflows (the `nodes`+`links` shape from the canvas
        # exporter and `comfy templates fetch`) have to be lowered to the API
        # shape before submit. We do it client-side using the cloud snapshot
        # of object_info — the cloud server has no /workflow/convert endpoint.
        from comfy_cli.cql.engine import _load_from_target

        if renderer.is_pretty():
            pprint("[yellow]Detected UI-format workflow, converting to API format…[/yellow]")
        try:
            object_info = _load_from_target(mode="cloud")
        except Exception as e:  # noqa: BLE001
            renderer.error(
                code="cql_no_graph",
                message=f"could not load cloud object_info for conversion: {e}",
                hint="run `comfy nodes refresh --where cloud` to populate the cache",
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

    # Pre-submit validation via pure-Python CQL engine.
    # Cloud path uses cached/bundled object_info (no live server needed).
    try:
        from comfy_cli.cql.engine import _load_from_target

        cloud_object_info = _load_from_target(mode="cloud")
    except Exception:  # noqa: BLE001
        cloud_object_info = {}

    _preflight_validate(renderer, parsed_workflow, cloud_object_info, target_label="cloud")

    target = resolve_target(where="cloud")
    try:
        client = Client(target, timeout=float(timeout))
    except Unauthenticated as e:
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy auth login")
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
        state_file = jobs_state.write(state)
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

    # --wait: poll the cloud API directly from the foreground process.
    # No watcher subprocess needed — simpler, no liveness/crash concerns,
    # and the state file is written exactly once when the job finishes.
    state = jobs_state.new(
        prompt_id=submit.prompt_id,
        client_id=client_id,
        workflow=workflow_name,
        where="cloud",
        base_url=target.base_url,
    )
    state_file = jobs_state.write(state)

    try:
        def _probe():
            st = client.get_job_status(submit.prompt_id)
            if not st:
                return None
            return (st.get("status"), st.get("progress"), st.get("queue_position"))

        record = client.wait_for_completion(
            submit.prompt_id, timeout=float(timeout), progress_probe=_probe
        )
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
    output_urls = client.extract_output_urls(record)
    exec_status = record.get("status") or record.get("execution_status") or {}
    if isinstance(exec_status, dict):
        status_str = exec_status.get("status_str", "")
    else:
        status_str = str(exec_status).lower()

    if status_str in ("error", "failed"):
        state.status = "error"
        state.error = {
            "code": "execution_error",
            "message": record.get("error_message", status_str),
        }
        state_file = jobs_state.write(state)
        renderer.error(
            code="cloud_http_error",
            message=f"Workflow execution failed: {state.error['message']}",
            details={"prompt_id": submit.prompt_id, "status": status_str},
        )
        raise typer.Exit(code=1)

    # Success path.
    state.status = "completed"
    state.outputs = output_urls
    state_file = jobs_state.write(state)

    end = time.time()

    if renderer.is_pretty():
        if output_urls:
            pprint("[bold green]\nOutputs:[/bold green]")
            for u in output_urls:
                pprint(u)
        pprint(f"[bold green]\nCloud workflow completed ({timedelta(seconds=end - start)})[/bold green]")

    renderer.emit(
        {
            "workflow": workflow_name,
            "status": state.status,
            "prompt_id": submit.prompt_id,
            "client_id": client_id,
            "outputs": output_urls,
            "elapsed_seconds": end - start,
            "base_url": target.base_url,
            "state_file": str(state_file) if state_file else None,
        },
        command="run",
        where="cloud",
    )
