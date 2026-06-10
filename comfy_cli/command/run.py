import json
import os
import sys
import time
import urllib.error
import urllib.parse
import uuid
from datetime import timedelta
from urllib import request

import typer
from rich import print as pprint
from rich.progress import BarColumn, Column, Progress, Table, TimeElapsedColumn
from websocket import WebSocket, WebSocketException, WebSocketTimeoutException

from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()

# JSON output schema version. Bumped only for breaking changes per docs/json-output.md.
SCHEMA_VERSION = 1

# Maximum bytes of a server response body we surface to the user (or
# embed in a `failed.error.body` field). Anything longer is truncated.
_MAX_BODY_PREVIEW = 500


def _node_errors_to_list(node_errors) -> list[dict]:
    """Transform ComfyUI's dict-keyed `node_errors` payload into a list of self-contained records.
    Each record carries `node_id` as a field, so agents can iterate the result
    directly without indirecting through dict keys."""
    if not isinstance(node_errors, dict):
        return []
    result = []
    for node_id, record in node_errors.items():
        if not isinstance(record, dict):
            continue
        entry = {"node_id": str(node_id)}
        entry.update(record)
        result.append(entry)
    return result


def is_ui_workflow(workflow) -> bool:
    return (
        isinstance(workflow, dict)
        and isinstance(workflow.get("nodes"), list)
        and isinstance(workflow.get("links"), list)
    )


def _classify_api_workflow(workflow):
    """Classify a parsed JSON object as API workflow / empty / invalid.

    Returns one of:
        ("ok", workflow_dict)   — well-formed API workflow with ≥1 node
        ("empty", None)         — empty dict (caller routes to workflow_empty)
        ("invalid", None)       — not a dict, or first node lacks class_type
    """
    if not isinstance(workflow, dict):
        return ("invalid", None)
    if not workflow:
        return ("empty", None)
    first_key = next(iter(workflow))
    node = workflow[first_key]
    if not isinstance(node, dict) or "class_type" not in node:
        return ("invalid", None)
    return ("ok", workflow)


class JsonEmitter:
    """NDJSON event emitter for ``comfy run --json``.

    Every ``emit_*`` method is a no-op when ``json_mode=False``, so the
    same call sites work for both modes. See ``docs/json-output.md``.
    """

    def __init__(self, json_mode: bool):
        self.json_mode = json_mode
        self.start_time = time.monotonic()
        self.client_id: str | None = None
        self.prompt_id: str | None = None
        self.workflow: dict | None = None
        self.cached_node_ids: list[str] = []
        self.executed_node_ids: list[str] = []
        self.outputs: list[dict] = []

    def set_workflow(self, workflow):
        self.workflow = workflow

    def set_client_id(self, client_id):
        self.client_id = client_id

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def get_title(self, node_id):
        if not isinstance(self.workflow, dict):
            return str(node_id)
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return str(node_id)
        meta = node.get("_meta")
        if isinstance(meta, dict):
            title = meta.get("title")
            if isinstance(title, str) and title:
                return title
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) and class_type else str(node_id)

    def get_class_type(self, node_id):
        if not isinstance(self.workflow, dict):
            return ""
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return ""
        return node.get("class_type", "")

    def _emit(self, event: dict) -> None:
        if not self.json_mode:
            return
        line = json.dumps(event, ensure_ascii=True)
        print(line, flush=True)

    def emit_converted(self, node_count: int) -> None:
        self._emit(
            {
                "event": "converted",
                "schema_version": SCHEMA_VERSION,
                "node_count": node_count,
            }
        )

    def workflow_manifest(self) -> list[dict]:
        """Build the `nodes` array for the `queued` event — one entry per
        node in the submitted (post-conversion) workflow."""
        if not isinstance(self.workflow, dict):
            return []
        manifest: list[dict] = []
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            class_type = class_type if isinstance(class_type, str) else ""
            manifest.append(
                {
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "title": self.get_title(node_id),
                }
            )
        return manifest

    def emit_prompt_preview(self, prompt: dict) -> None:
        self._emit(
            {
                "event": "prompt_preview",
                "schema_version": SCHEMA_VERSION,
                "prompt": prompt,
            }
        )

    def emit_queued(self, prompt_id: str, validation_warnings: list[dict]) -> None:
        self.prompt_id = prompt_id
        self._emit(
            {
                "event": "queued",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "client_id": self.client_id,
                "validation_warnings": validation_warnings,
                "nodes": self.workflow_manifest(),
            }
        )

    def emit_node_cached(self, node_id) -> None:
        node_id = str(node_id)
        self.cached_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_cached",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_executing(self, node_id) -> None:
        node_id = str(node_id)
        # `executed_node_ids` aggregates everything the executor touched —
        # including intermediate nodes that never fire a server-side `executed` WS event.
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_executing",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_progress(self, node_id, value, max_val) -> None:
        node_id = str(node_id)
        self._emit(
            {
                "event": "node_progress",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "value": value,
                "max": max_val,
            }
        )

    def emit_node_executed(self, node_id, outputs: list[dict]) -> None:
        node_id = str(node_id)
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self.outputs.extend(outputs)
        self._emit(
            {
                "event": "node_executed",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "outputs": outputs,
            }
        )

    def emit_completed(self) -> None:
        self._emit(
            {
                "event": "completed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "outputs": self.outputs,
                "cached_node_ids": self.cached_node_ids,
                "executed_node_ids": self.executed_node_ids,
            }
        )

    def fail(self, kind: str, message: str, *, rich_message: str | None = None, **extras) -> typer.Exit:
        """Emit a `failed` event (in JSON mode) or print a red text message
        (otherwise), then return the `typer.Exit(code=1)` for the caller to
        raise. Returning rather than raising keeps `raise ... from e`
        chaining clean at call sites. `rich_message` overrides `message`
        for the human-readable text only — it is auto-wrapped in
        `[bold red]...[/bold red]`. Sites that need multi-colour Rich
        markup should emit the failure event explicitly."""
        self.emit_failed(kind, message, **extras)
        if not self.json_mode:
            pprint(f"[bold red]{rich_message if rich_message is not None else message}[/bold red]")
        return typer.Exit(code=1)

    def emit_failed(self, kind: str, message: str, **extras) -> None:
        error = {"kind": kind, "message": message}
        error.update(extras)
        self._emit(
            {
                "event": "failed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "error": error,
            }
        )


def fetch_object_info(host, port, timeout, emitter=None):
    """GET ``/object_info`` from the running ComfyUI server.

    The response describes every loaded node class's input schema and is what
    the converter uses to map widget values to input names, fill defaults, etc.

    In JSON mode, failures emit a structured ``failed`` event via ``emitter``.
    Either way, a ``typer.Exit(code=1)`` is raised.
    """
    url = f"http://{host}:{port}/object_info"
    json_mode = bool(emitter and emitter.json_mode)
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace").strip()
        if json_mode:
            emitter.emit_failed(
                "object_info_unavailable",
                f"Failed to fetch /object_info (HTTP {e.code})",
                status_code=e.code,
                body=body_text[:_MAX_BODY_PREVIEW],
            )
        else:
            pprint(
                f"[bold red]Failed to fetch /object_info (HTTP {e.code}): {body_text[:_MAX_BODY_PREVIEW]}[/bold red]"
            )
        raise typer.Exit(code=1) from e
    except urllib.error.URLError as e:
        msg = f"Failed to fetch /object_info from {host}:{port}: {e.reason} (override with --host / --port)"
        if json_mode:
            emitter.emit_failed("connection_error", msg)
        else:
            pprint(f"[bold red]{msg}[/bold red]")
        raise typer.Exit(code=1) from e
    except TimeoutError as e:
        msg = f"Failed to fetch /object_info from {host}:{port}: timed out after {timeout}s (override with --host / --port)"
        if json_mode:
            emitter.emit_failed("connection_error", msg)
        else:
            pprint(f"[bold red]{msg}[/bold red]")
        raise typer.Exit(code=1) from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        if json_mode:
            emitter.emit_failed(
                "object_info_unavailable",
                "Server returned invalid JSON for /object_info",
                status_code=200,
                body=body.decode("utf-8", errors="replace")[:_MAX_BODY_PREVIEW],
            )
        else:
            pprint("[bold red]Failed to fetch /object_info: server returned invalid JSON[/bold red]")
        raise typer.Exit(code=1) from e


def execute(
    workflow: str,
    host,
    port,
    wait=True,
    verbose=False,
    timeout=120,
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

    emitter = JsonEmitter(json_mode=json_mode)
    workflow_name = os.path.abspath(os.path.expanduser(workflow))

    if not os.path.isfile(workflow_name):
        if json_mode:
            emitter.emit_failed("workflow_not_found", f"Workflow file not found: {workflow_name}")
        else:
            pprint(
                f"[bold red]Specified workflow file not found: {workflow_name}[/bold red]",
                file=sys.stderr,
            )
        raise typer.Exit(code=1)

    # Under --print-prompt we skip this pre-flight probe. API-format input
    # makes no server calls downstream so it works fully offline; UI-format
    # input still needs /object_info for the converter, but if it's
    # unreachable, fetch_object_info() surfaces the same connection_error
    # kind a few lines later.
    if not print_prompt and not check_comfy_server_running(port, host, timeout=timeout):
        raise emitter.fail(
            "connection_error",
            f"ComfyUI not running at {host}:{port} (override with --host / --port)",
        )

    try:
        with open(workflow_name, encoding="utf-8") as f:
            raw_workflow = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise emitter.fail("workflow_read_error", f"Unable to read workflow file: {e}") from e
    except json.JSONDecodeError as e:
        raise emitter.fail(
            "workflow_invalid_json",
            f"Specified workflow file is not valid JSON: {e}",
        ) from e

    if is_ui_workflow(raw_workflow):
        if not json_mode:
            pprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        object_info = fetch_object_info(host, port, timeout, emitter=emitter)
        try:
            workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            raise emitter.fail("conversion_error", f"Workflow conversion failed: {e}") from e
        except Exception as e:
            if json_mode:
                emitter.emit_failed(
                    "conversion_crash",
                    f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                    exception_type=type(e).__name__,
                )
            else:
                pprint(
                    f"[bold red]Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}[/bold red]\n"
                    "[yellow]The UI-to-API converter is experimental. Please report this at[/yellow]\n"
                    "[yellow]  https://github.com/Comfy-Org/comfy-cli/issues[/yellow]\n"
                    "[yellow]and attach the workflow file if possible.[/yellow]"
                )
                if verbose:
                    import traceback as _tb

                    _tb.print_exc()
            raise typer.Exit(code=1) from e
        if not workflow:
            raise emitter.fail("workflow_empty", "Workflow conversion produced no executable nodes")
        emitter.set_workflow(workflow)
        if json_mode:
            emitter.emit_converted(len(workflow))
    else:
        kind, validated = _classify_api_workflow(raw_workflow)
        if kind == "empty":
            raise emitter.fail(
                "workflow_empty",
                "API workflow contains no nodes",
                rich_message="Specified API workflow has no nodes",
            )
        if kind == "invalid":
            raise emitter.fail(
                "workflow_format_invalid",
                "Workflow file is neither a ComfyUI API workflow nor an exported UI workflow",
                rich_message=("Specified workflow is neither a ComfyUI API workflow nor an exported UI workflow"),
            )
        workflow = validated
        emitter.set_workflow(workflow)

    # In JSON mode, always emit the converted workflow graph so agents have
    # a complete audit trail of what the CLI is about to submit. The event
    # is non-terminal in normal flow and terminal under --print-prompt.
    if json_mode:
        emitter.emit_prompt_preview(workflow)

    if print_prompt:
        if not json_mode:
            print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return

    progress = None
    start = time.time()
    if wait and not json_mode:
        pprint(f"Executing workflow: {workflow_name}")
        progress = ExecutionProgress()
        progress.start()
    elif not wait and not json_mode:
        print(f"Queuing workflow: {workflow_name}")

    execution = WorkflowExecution(
        workflow,
        host,
        port,
        verbose,
        progress,
        timeout,
        api_key=api_key,
        emitter=emitter,
    )
    emitter.set_client_id(execution.client_id)

    try:
        if wait:
            execution.connect()
        execution.queue()
        if wait:
            execution.watch_execution()
            end = time.time()
            if progress is not None:
                progress.stop()
                progress = None

            if json_mode:
                emitter.emit_completed()
            else:
                if len(execution.outputs) > 0:
                    pprint("[bold green]\nOutputs:[/bold green]")
                    for f in execution.outputs:
                        pprint(f)
                elapsed = timedelta(seconds=end - start)
                pprint(f"[bold green]\nWorkflow execution completed ({elapsed})[/bold green]")
        else:
            # --no-wait: queued was already emitted by execution.queue().
            if not json_mode:
                pprint("[bold green]Workflow queued[/bold green]")
    except WebSocketTimeoutException:
        # Not migrated to emitter.fail(): the text-mode message combines
        # a red error line and a yellow remediation hint, which the
        # single-colour auto-wrap in fail() can't express.
        msg = f"WebSocket timed out after {timeout}s waiting for server response"
        if json_mode:
            emitter.emit_failed("timeout", msg, timeout_seconds=float(timeout))
        else:
            pprint(
                f"[bold red]Error: {msg}.[/bold red]\n"
                "[yellow]For long-running workflows, increase the timeout: comfy run --workflow <file> --timeout 300[/yellow]"
            )
        raise typer.Exit(code=1)
    except (WebSocketException, ConnectionError, OSError) as e:
        raise emitter.fail(
            "connection_lost",
            f"Lost connection to ComfyUI server: {e}",
            rich_message=f"Error: Lost connection to ComfyUI server: {e}",
        )
    except KeyboardInterrupt:
        raise emitter.fail("execution_interrupted", "Interrupted by user") from None
    finally:
        if progress is not None:
            progress.stop()


class ExecutionProgress(Progress):
    def get_renderables(self):
        table_columns = (
            (Column(no_wrap=True) if isinstance(_column, str) else _column.get_table_column().copy())
            for _column in self.columns
        )

        for task in self.tasks:
            percent = "[progress.percentage]{task.percentage:>3.0f}%".format(task=task)  # noqa
            if task.fields.get("progress_type") == "overall":
                overall_table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)
                overall_table.add_row(BarColumn().render(task), percent, TimeElapsedColumn().render(task))
                yield overall_table
            else:
                yield self.make_tasks_table([task])


class WorkflowExecution:
    def __init__(
        self,
        workflow,
        host,
        port,
        verbose,
        progress,
        timeout=120,
        api_key: str | None = None,
        emitter: JsonEmitter | None = None,
    ):
        self.workflow = workflow
        self.host = host
        self.port = port
        self.verbose = verbose
        self.client_id = str(uuid.uuid4())
        self.outputs: list = []
        self.progress = progress
        self.remaining_nodes = set(self.workflow.keys())
        self.total_nodes = len(self.remaining_nodes)
        if progress is not None:
            self.overall_task = self.progress.add_task("", total=self.total_nodes, progress_type="overall")
        self.current_node = None
        self.progress_task = None
        self.progress_node = None
        self.prompt_id = None
        self.ws = None
        self.timeout = timeout
        self.api_key = api_key
        # Default to a no-op emitter so internal call sites don't need to
        # branch on whether json mode is active.
        self.emitter = emitter if emitter is not None else JsonEmitter(json_mode=False)

    def connect(self):
        self.ws = WebSocket()
        # Timeout on the handshake too: a server busy loading a model
        # can otherwise leave the CLI hung with no terminal event.
        self.ws.connect(
            f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}",
            timeout=self.timeout,
        )

    def queue(self):
        data: dict = {"prompt": self.workflow, "client_id": self.client_id}
        data["extra_data"] = {"comfy_usage_source": "comfy-cli"}
        if self.api_key:
            data["extra_data"]["api_key_comfy_org"] = self.api_key
        req = request.Request(
            f"http://{self.host}:{self.port}/prompt",
            json.dumps(data).encode("utf-8"),
        )
        req.add_header("Comfy-Usage-Source", "comfy-cli")
        try:
            resp = request.urlopen(req, timeout=self.timeout)
            raw_body = resp.read()
        except urllib.error.HTTPError as e:
            self._handle_submit_http_error(e)
            raise typer.Exit(code=1) from e
        except urllib.error.URLError as e:
            self._stop_progress()
            raise self.emitter.fail(
                "connection_error",
                f"Cannot reach server at {self.host}:{self.port}: {e.reason}",
            ) from e
        except TimeoutError as e:
            self._stop_progress()
            raise self.emitter.fail(
                "connection_error",
                f"Connection to {self.host}:{self.port} timed out: {e}",
            ) from e
        except OSError as e:
            self._stop_progress()
            raise self.emitter.fail(
                "connection_error",
                f"Network error contacting {self.host}:{self.port}: {e}",
            ) from e

        try:
            body = json.loads(raw_body) if raw_body else None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._stop_progress()
            body_str = raw_body.decode("utf-8", errors="replace")[:_MAX_BODY_PREVIEW]
            raise self.emitter.fail(
                "invalid_response",
                "Server returned HTTP 200 with unparseable body",
                rich_message=f"Server returned HTTP 200 with unparseable body: {body_str}",
                status_code=200,
                body=body_str,
            ) from e

        prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            self._stop_progress()
            body_str = json.dumps(body)[:_MAX_BODY_PREVIEW] if body is not None else ""
            raise self.emitter.fail(
                "invalid_response",
                "Server returned HTTP 200 without a prompt_id",
                rich_message=f"Server returned HTTP 200 without a prompt_id: {body_str}",
                status_code=200,
                body=body_str,
            )

        self.prompt_id = prompt_id

        # 200 may still carry node_errors if some output chains failed
        # validation but others passed — surface as warnings, not a failure.
        node_errors = body.get("node_errors") if isinstance(body, dict) else None
        validation_warnings = _node_errors_to_list(node_errors)

        if self.emitter.json_mode:
            self.emitter.emit_queued(prompt_id, validation_warnings)

    def _handle_submit_http_error(self, e: urllib.error.HTTPError) -> None:
        raw = b""
        try:
            raw = e.read()
        except Exception:
            pass
        try:
            body = json.loads(raw) if raw else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        body_str = (raw or b"").decode("utf-8", errors="replace")
        self._stop_progress()

        code = e.code
        if code == 400 and isinstance(body, dict) and isinstance(body.get("node_errors"), dict) and body["node_errors"]:
            self._emit_validation_error(body["node_errors"])
            return
        if 400 <= code < 500:
            kind = "client_error"
        elif 500 <= code < 600:
            kind = "server_error"
        else:
            kind = "client_error"

        if self.emitter.json_mode:
            self.emitter.emit_failed(
                kind,
                f"Server returned HTTP {code}",
                status_code=code,
                body=body_str[:_MAX_BODY_PREVIEW],
            )
        else:
            if code == 500:
                pprint(f"[bold red]Error running workflow\n{body_str}[/bold red]")
            elif code == 400 and isinstance(body, dict):
                pprint(f"[bold red]Error running workflow\n{json.dumps(body, indent=2)}[/bold red]")
            else:
                pprint(f"[bold red]Error running workflow (HTTP {code})\n{body_str[:_MAX_BODY_PREVIEW]}[/bold red]")

    def _emit_validation_error(self, node_errors: dict) -> None:
        if self.emitter.json_mode:
            message = "Workflow failed validation"
            try:
                first_node = next(iter(node_errors.values()))
                errs = first_node.get("errors") if isinstance(first_node, dict) else None
                if isinstance(errs, list) and errs:
                    first = errs[0]
                    if isinstance(first, dict) and isinstance(first.get("message"), str):
                        message = first["message"]
            except StopIteration:
                pass
            self.emitter.emit_failed(
                "validation_error",
                message,
                node_errors=_node_errors_to_list(node_errors),
            )
        else:
            pprint(f"[bold red]Error running workflow\n{json.dumps(node_errors, indent=2)}[/bold red]")

    def _stop_progress(self) -> None:
        if self.progress is not None:
            try:
                self.progress.stop()
            except Exception:
                pass

    def watch_execution(self):
        self.ws.settimeout(self.timeout)
        while True:
            message = self.ws.recv()
            if not isinstance(message, str):
                continue
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                # Tolerate malformed frames from misbehaving proxies.
                continue
            if not self.on_message(parsed):
                break

    def update_overall_progress(self):
        if self.progress is None:
            return
        self.progress.update(self.overall_task, completed=self.total_nodes - len(self.remaining_nodes))

    def log_node(self, type, node_id):
        if not self.verbose:
            return
        if self.emitter.json_mode:
            # --verbose is a no-op in JSON mode; Rich output would corrupt the stream.
            return

        node = self.workflow.get(node_id)
        if node is None:
            pprint(f"{type} : [bright_black]({node_id})[/]")
            return
        class_type = node["class_type"]
        title = self.emitter.get_title(node_id)

        if title != class_type:
            title += f"[bright_black] - {class_type}[/]"
        title += f"[bright_black] ({node_id})[/]"

        pprint(f"{type} : {title}")

    def format_image_path(self, img):
        """Build a single human-readable path string for the legacy text
        output. Prefers a clickable absolute filesystem path when the
        host is a known loopback, the workspace resolves, the path stays
        inside the workspace's per-type output dir, and the file exists
        on disk. Otherwise falls back to a /view URL."""
        filename = img["filename"]
        subfolder = img.get("subfolder") or ""
        output_type = img.get("type") or "output"

        if self.host in ("127.0.0.1", "localhost", "::1", "[::1]"):
            ws_path = self._text_mode_workspace_path()
            if ws_path:
                parts = [subfolder, filename] if subfolder else [filename]
                type_root = os.path.normpath(os.path.join(ws_path, output_type))
                candidate = os.path.normpath(os.path.join(type_root, *parts))
                if (candidate == type_root or candidate.startswith(type_root + os.sep)) and os.path.isfile(candidate):
                    return candidate

        return self._view_url(filename, subfolder, output_type)

    def _view_url(self, filename: str, subfolder: str, file_type: str) -> str:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        return f"http://{self.host}:{self.port}/view?{urllib.parse.urlencode(params)}"

    def _text_mode_workspace_path(self) -> str | None:
        # workspace_manager.get_workspace_path() can print a warning and
        # write config on the stale-recent path. Memoize so a workflow
        # with N outputs doesn't repeat the side effects N times.
        if not hasattr(self, "_ws_path_cached"):
            try:
                self._ws_path_cached = workspace_manager.get_workspace_path()[0]
            except Exception:
                self._ws_path_cached = None
        return self._ws_path_cached

    def _build_output_object(self, node_id, category, item) -> dict:
        """Construct a structured Output dict for the JSON contract."""
        node_id = str(node_id)
        filename = item["filename"]
        subfolder = item.get("subfolder") or ""
        file_type = item.get("type") or "output"

        return {
            "category": category,
            "node_id": node_id,
            "class_type": self.emitter.get_class_type(node_id),
            "title": self.emitter.get_title(node_id),
            "filename": filename,
            "subfolder": subfolder,
            "type": file_type,
            "url": self._view_url(filename, subfolder, file_type),
        }

    def on_message(self, message):
        # Defensive: a malformed (non-object) JSON frame from the server
        # must not raise out of the recv loop — that would tear down the
        # run without a terminal `failed` event and break the contract.
        if not isinstance(message, dict):
            return True
        data = message.get("data")
        if not isinstance(data, dict):
            return True
        if data.get("prompt_id") != self.prompt_id:
            return True

        msg_type = message.get("type")
        if msg_type == "executing":
            return self.on_executing(data)
        elif msg_type == "execution_cached":
            self.on_cached(data)
        elif msg_type == "progress":
            self.on_progress(data)
        elif msg_type == "executed":
            self.on_executed(data)
        elif msg_type == "execution_error":
            self.on_error(data)
        elif msg_type == "execution_interrupted":
            self.on_interrupted(data)

        return True

    def on_executing(self, data):
        if self.progress_task is not None and self.progress is not None:
            self.progress.remove_task(self.progress_task)
            self.progress_task = None

        # `node: null` is the documented "execution done" signal. A
        # missing key is a protocol violation — skip the frame and keep
        # listening rather than prematurely terminating.
        if "node" not in data:
            return True
        if data["node"] is None:
            return False

        node_id = data["node"]
        if self.current_node:
            self.remaining_nodes.discard(self.current_node)
            self.update_overall_progress()
        self.current_node = node_id
        self.log_node("Executing", node_id)
        if self.emitter.json_mode:
            self.emitter.emit_node_executing(node_id)
        return True

    def on_cached(self, data):
        nodes = data.get("nodes") or []
        for n in nodes:
            if n is None:
                continue
            self.remaining_nodes.discard(n)
            self.log_node("Cached", n)
            if self.emitter.json_mode:
                self.emitter.emit_node_cached(n)
        self.update_overall_progress()

    def on_progress(self, data):
        node = data.get("node")
        if node is None:
            return
        value = data.get("value", 0)
        max_val = data.get("max", 0)
        if self.progress is not None:
            if self.progress_node != node:
                self.progress_node = node
                if self.progress_task is not None:
                    self.progress.remove_task(self.progress_task)
                self.progress_task = self.progress.add_task(
                    self.emitter.get_title(node), total=max_val, progress_type="node"
                )
            self.progress.update(self.progress_task, completed=value)
        if self.emitter.json_mode:
            self.emitter.emit_node_progress(node, value, max_val)

    def on_executed(self, data):
        node_id = data.get("node")
        if node_id is None:
            return
        self.remaining_nodes.discard(node_id)
        self.update_overall_progress()

        # node_executed fires whenever the server emits `executed`, even
        # when there are no file-shaped outputs (outputs=[] in that case).
        structured_outputs: list[dict] = []
        output = data.get("output")
        if isinstance(output, dict):
            for category, items in output.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    obj = self._build_output_object(node_id, category, item)
                    structured_outputs.append(obj)
                    if not self.emitter.json_mode:
                        # Legacy string list, only consumed by the Rich path.
                        self.outputs.append(self.format_image_path(item))

        if self.emitter.json_mode:
            self.emitter.emit_node_executed(node_id, structured_outputs)

    def on_error(self, data):
        raw_node_id = data.get("node_id", "")
        node_id = str(raw_node_id) if raw_node_id is not None else ""
        class_type = data.get("node_type") or data.get("class_type") or ""
        exception_type = data.get("exception_type", "")
        raw_tb = data.get("traceback", "")
        if isinstance(raw_tb, list):
            traceback_str = "".join(str(x) for x in raw_tb)
        elif isinstance(raw_tb, str):
            traceback_str = raw_tb
        else:
            traceback_str = ""
        message = data.get("exception_message") or "Workflow execution failed"

        self._stop_progress()
        if self.emitter.json_mode:
            title = self.emitter.get_title(node_id) if node_id else ""
            if not class_type and node_id:
                class_type = self.emitter.get_class_type(node_id)
            self.emitter.emit_failed(
                "execution_error",
                message,
                node_id=node_id,
                class_type=class_type,
                title=title,
                exception_type=exception_type,
                traceback=traceback_str,
            )
        else:
            pprint(f"[bold red]Error running workflow\n{json.dumps(data, indent=2)}[/bold red]")
        raise typer.Exit(code=1)

    def on_interrupted(self, data):
        self._stop_progress()
        if self.emitter.json_mode:
            self.emitter.emit_failed(
                "execution_interrupted",
                "Workflow execution was interrupted",
            )
        else:
            pprint("[yellow]Workflow execution was interrupted[/yellow]")
        raise typer.Exit(code=1)
