"""WorkflowExecution — the local websocket execution loop.

The class owns one workflow run end-to-end against a local ComfyUI server:
HTTP submit to `/prompt`, then a WebSocket session translating server
events into structured renderer calls. Also defines ``ExecutionProgress``
(a Rich Progress subclass with a custom overall row) and ``_safe_close``
for cancellation cleanup.

Machine-output dialect (single source of truth — the renderer):

    server WS message     renderer event (`schema: event/1`)
    -----------------     -----------------------------------------------
    executing             executing        {node, class_type, title, prompt_id}
    execution_cached      execution_cached {node, class_type, title, prompt_id}
    progress              progress         {node, completed, total, prompt_id}  (throttled)
    executed              executed         {node, class_type, title, outputs, prompt_id}
                          output           {url, prompt_id}  (one per file output)
    execution_error       execution_error event + `execution_error` error envelope
    execution_interrupted `cancelled` error envelope, exit 130

These are the same event names `comfy jobs watch` emits, so the run stream
and the watch stream speak one dialect. Failures go through
``renderer.error(code=…)`` with codes from ``comfy_cli.error_codes``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import uuid
from urllib import request

import typer
from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.table import Column, Table

from comfy_cli import execution_errors
from comfy_cli.command.run.loader import _MAX_BODY_PREVIEW, _node_errors_to_list
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()


def _safe_close(execution: WorkflowExecution) -> None:
    """Best-effort WebSocket close on cancellation."""
    try:
        if execution.ws is not None:
            execution.ws.close()
    except Exception:  # noqa: BLE001
        pass


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
        local_paths,
        timeout=30,
        *,
        extra_data: dict | None = None,
        api_key: str | None = None,
    ):
        self.workflow = workflow
        self.host = host
        self.port = port
        self.verbose = verbose
        self.client_id = str(uuid.uuid4())
        self.outputs: list = []
        # Node-keyed companion to the flat `outputs` URLs — one
        # {"node_id", "url", "filename", "type"} entry per recorded URL, in
        # the same order. Local parity with the cloud history record so
        # `run --wait` can group outputs by node / foreach item.
        self.output_entries: list[dict] = []
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
        # Credentials injected into ``extra_data`` so partner-API nodes
        # (partner/* category) can authenticate at execute time —
        # mirrors what ``comfy_client.submit_prompt`` does for cloud.
        self.extra_data = dict(extra_data) if extra_data else None
        self.api_key = api_key
        self.renderer = get_renderer()
        # Aggregated node bookkeeping surfaced in the final envelope:
        # `cached_node_ids` — server said the node came from cache;
        # `executed_node_ids` — everything the executor *ran* (union of
        # nodes seen in `executing` or `executed`), including intermediate
        # compute nodes that never fire a server-side `executed` event.
        self.cached_node_ids: list[str] = []
        self.executed_node_ids: list[str] = []

    def connect(self):
        # Resolve via the package namespace so tests can patch
        # ``comfy_cli.command.run.WebSocket`` and have it take effect here.
        from comfy_cli.command import run as _run_pkg

        self.ws = _run_pkg.WebSocket()
        # The local executor POSTs to http://{host}:{port}/prompt (see queue()),
        # so the websocket must use the matching plaintext ws:// scheme. Only
        # upgrade to wss:// when the host is explicitly an https/wss URL.
        scheme = "wss" if self.host.lower().startswith(("https://", "wss://")) else "ws"
        bare_host = self.host.split("://", 1)[-1]
        self.ws.connect(
            f"{scheme}://{bare_host}:{self.port}/ws?clientId={self.client_id}",
            timeout=self.timeout,
        )

    def workflow_manifest(self) -> list[dict]:
        """Build the `nodes` array for the `queued` event — one entry per
        node in the submitted (post-conversion) workflow."""
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
                    "title": self.get_node_title(node_id),
                }
            )
        return manifest

    def queue(self):
        data: dict = {"prompt": self.workflow, "client_id": self.client_id}
        # Usage-source attribution rides extra_data so the server can tell
        # CLI-originated executions apart from web-UI ones (upstream #468).
        data["extra_data"] = {"comfy_usage_source": "comfy-cli"}
        if self.extra_data:
            data["extra_data"].update(self.extra_data)
        elif self.api_key:
            data["extra_data"]["api_key_comfy_org"] = self.api_key
        req = request.Request(f"http://{self.host}:{self.port}/prompt", json.dumps(data).encode("utf-8"))
        req.add_header("Comfy-Usage-Source", "comfy-cli")
        try:
            resp = request.urlopen(req, timeout=self.timeout)
            raw_body = resp.read()
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            body_text = body_bytes.decode("utf-8", errors="replace").strip() if body_bytes else ""

            if self.progress is not None:
                self.progress.stop()

            if e.status == 400:
                try:
                    parsed = json.loads(body_bytes) if body_bytes else {}
                except (json.JSONDecodeError, ValueError):
                    parsed = {}
                node_errors_raw = parsed.get("node_errors", {}) if isinstance(parsed, dict) else {}
                if isinstance(node_errors_raw, dict) and node_errors_raw:
                    node_errors = _node_errors_to_list(node_errors_raw)
                    if self.renderer.is_pretty():
                        pprint(f"[bold red]Error running workflow\n{json.dumps(node_errors_raw, indent=2)}[/bold red]")
                    self.renderer.error(
                        code="prompt_rejected",
                        message=f"Workflow has {len(node_errors_raw)} validation error(s)",
                        hint="inspect `details.node_errors` and fix the workflow",
                        details={"status": e.status, "node_errors": node_errors},
                    )
                    raise typer.Exit(code=1)

            if self.renderer.is_pretty():
                pprint(f"[bold red]Error running workflow (HTTP {e.status})\n{body_text}[/bold red]")
            if e.status < 500:
                self.renderer.error(
                    code="client_error",
                    message=f"Error running workflow (HTTP {e.status})",
                    hint="check `details.body` for the server's message",
                    details={"status": e.status, "body": body_text[:_MAX_BODY_PREVIEW]},
                )
            else:
                self.renderer.error(
                    code="server_error",
                    message=f"Error running workflow (HTTP {e.status})",
                    hint="check the ComfyUI server logs",
                    details={"status": e.status, "body": body_text[:_MAX_BODY_PREVIEW]},
                )
            raise typer.Exit(code=1)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if self.progress is not None:
                self.progress.stop()
            reason = str(e.reason) if isinstance(e, urllib.error.URLError) else str(e)
            if self.renderer.is_pretty():
                pprint(f"[bold red]Error: Failed to submit workflow: {reason}[/bold red]")
            self.renderer.error(
                code="connection_error",
                message=f"Failed to submit workflow: {reason}",
                hint="check the server is still running; re-run the command",
            )
            raise typer.Exit(code=1)

        try:
            body = json.loads(raw_body) if raw_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None

        prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            if self.progress is not None:
                self.progress.stop()
            if self.renderer.is_pretty():
                pprint("[bold red]Error: Server returned HTTP 200 without a prompt_id[/bold red]")
            self.renderer.error(
                code="invalid_response",
                message="Server returned HTTP 200 without a prompt_id",
                hint="check that the host:port really is a ComfyUI server",
                details={"status": 200},
            )
            raise typer.Exit(code=1)

        self.prompt_id = prompt_id

        # 200 may still carry node_errors if some output chains failed
        # validation but others passed — surface as warnings, not a failure.
        node_errors = body.get("node_errors") if isinstance(body, dict) else None
        validation_warnings = _node_errors_to_list(node_errors)

        self.renderer.event(
            "queued",
            prompt_id=prompt_id,
            client_id=self.client_id,
            validation_warnings=validation_warnings,
            nodes=self.workflow_manifest(),
        )

    def watch_execution(self):
        if self.ws is None:
            raise RuntimeError("watch_execution called before the websocket was connected")
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

    def get_node_title(self, node_id):
        """Display label: ``_meta.title`` if present, else ``class_type``,
        else the node id. Defensive against unknown ids and non-dict nodes."""
        node = self.workflow.get(node_id)
        if node is None and not isinstance(node_id, str):
            node = self.workflow.get(str(node_id))
        if not isinstance(node, dict):
            return str(node_id)
        meta = node.get("_meta")
        if isinstance(meta, dict):
            title = meta.get("title")
            if isinstance(title, str) and title:
                return title
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) and class_type else str(node_id)

    def _class_type(self, node_id):
        node = self.workflow.get(node_id)
        if node is None and not isinstance(node_id, str):
            node = self.workflow.get(str(node_id))
        if not isinstance(node, dict):
            return ""
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) else ""

    def _track_executed(self, node_id) -> None:
        node_id = str(node_id)
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)

    def log_node(self, type, node_id):
        if not self.verbose:
            return
        if not self.renderer.is_pretty():
            # --verbose is a no-op in machine modes; the event stream already
            # carries the same information.
            return

        node = self.workflow.get(node_id)
        if node is None:
            pprint(f"{type} : [bright_black]({node_id})[/]")
            return
        class_type = node["class_type"]
        title = self.get_node_title(node_id)

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
        """Construct a structured Output dict carried on `executed` events."""
        node_id = str(node_id)
        filename = item["filename"]
        subfolder = item.get("subfolder") or ""
        file_type = item.get("type") or "output"

        return {
            "category": category,
            "node_id": node_id,
            "class_type": self._class_type(node_id),
            "title": self.get_node_title(node_id),
            "filename": filename,
            "subfolder": subfolder,
            "type": file_type,
            "url": self._view_url(filename, subfolder, file_type),
        }

    def on_message(self, message):
        # Defensive: a malformed (non-object) JSON frame from the server
        # must not raise out of the recv loop — that would tear down the
        # run without a terminal envelope and break the contract.
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
        else:
            if self.current_node:
                self.remaining_nodes.discard(self.current_node)
                self.update_overall_progress()
            self.current_node = data["node"]
            self.log_node("Executing", data["node"])
            self._track_executed(data["node"])
            self.renderer.event(
                "executing",
                node=str(data["node"]),
                title=self.get_node_title(data["node"]),
                class_type=self._class_type(data["node"]),
                prompt_id=self.prompt_id,
            )
        return True

    def on_cached(self, data):
        nodes = data.get("nodes") or []
        for n in nodes:
            if n is None:
                continue
            self.remaining_nodes.discard(n)
            self.log_node("Cached", n)
            node_id = str(n)
            if node_id not in self.cached_node_ids:
                self.cached_node_ids.append(node_id)
            self.renderer.event(
                "execution_cached",
                node=node_id,
                title=self.get_node_title(n),
                class_type=self._class_type(n),
                prompt_id=self.prompt_id,
            )
        self.update_overall_progress()

    def on_progress(self, data):
        node = data.get("node")
        if node is None:
            return
        if self.progress_node != node:
            self.progress_node = node
            if self.progress is not None:
                if self.progress_task:
                    self.progress.remove_task(self.progress_task)
                self.progress_task = self.progress.add_task(
                    self.get_node_title(node), total=data["max"], progress_type="node"
                )
        if self.progress is not None and self.progress_task is not None:
            self.progress.update(self.progress_task, completed=data["value"])
        # Throttle the NDJSON torrent (samplers can fire 30+/s).
        self.renderer.throttled_event(
            f"progress:{node}",
            "progress",
            max_hz=10,
            node=str(node),
            completed=data["value"],
            total=data["max"],
            prompt_id=self.prompt_id,
        )

    def on_executed(self, data):
        node_id = data.get("node")
        if node_id is None:
            return
        self.remaining_nodes.discard(node_id)
        self.update_overall_progress()
        self._track_executed(node_id)

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

        self.renderer.event(
            "executed",
            node=str(node_id),
            title=self.get_node_title(node_id),
            class_type=self._class_type(node_id),
            outputs=structured_outputs,
            prompt_id=self.prompt_id,
        )

        for obj in structured_outputs:
            url = self.format_image_path(
                {"filename": obj["filename"], "subfolder": obj["subfolder"], "type": obj["type"]}
            )
            if url not in self.outputs:
                # Always record for the state file; emit the NDJSON
                # output event at the same point so json consumers see it.
                self.outputs.append(url)
                self.output_entries.append(
                    {
                        "node_id": obj["node_id"],
                        "url": url,
                        "filename": obj["filename"],
                        "type": obj["type"],
                    }
                )
                self.renderer.event("output", url=url, prompt_id=self.prompt_id)

    def on_error(self, data):
        self._stop_progress()
        data = data if isinstance(data, dict) else {}
        node_id = str(data.get("node_id", ""))
        if self.renderer.is_pretty():
            pprint(f"[bold red]Error running workflow\n{json.dumps(data, indent=2)}[/bold red]")
        # The event keeps the full server payload (incl. complete traceback);
        # the error envelope carries the classified one-line verdict.
        self.renderer.event("execution_error", prompt_id=self.prompt_id, details=data)
        verdict = execution_errors.classify(data)
        self.renderer.error(
            code=verdict["code"],
            message=verdict["message"],
            hint=verdict["hint"],
            details={
                **verdict["details"],
                "class_type": data.get("node_type") or self._class_type(node_id),
                "title": self.get_node_title(node_id),
            },
        )
        raise typer.Exit(code=1)

    def on_interrupted(self, data):
        self._stop_progress()
        if self.renderer.is_pretty():
            pprint("[yellow]Workflow execution was interrupted[/yellow]")
        self.renderer.error(
            code="cancelled",
            message="Workflow execution was interrupted",
            exit_code=130,
        )
        raise typer.Exit(code=130)

    def _stop_progress(self) -> None:
        if self.progress is not None:
            try:
                self.progress.stop()
            except Exception:
                pass
