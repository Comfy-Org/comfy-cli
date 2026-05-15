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


def is_ui_workflow(workflow) -> bool:
    return (
        isinstance(workflow, dict)
        and isinstance(workflow.get("nodes"), list)
        and isinstance(workflow.get("links"), list)
    )


def _validate_api_workflow(workflow):
    """Return the workflow dict if it has the shape of API format, else None."""
    if not isinstance(workflow, dict) or not workflow:
        return None
    node = workflow[next(iter(workflow))]
    if not isinstance(node, dict) or "class_type" not in node:
        return None
    return workflow


def fetch_object_info(host: str, port: int, timeout: int) -> dict:
    """GET ``/object_info`` from the running ComfyUI server.

    The response describes every loaded node class's input schema and is what
    the converter uses to map widget values to input names, fill defaults, etc.
    """
    url = f"http://{host}:{port}/object_info"
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace").strip()
        pprint(f"[bold red]Failed to fetch /object_info (HTTP {e.code}): {body[:500]}[/bold red]")
        raise typer.Exit(code=1) from e
    except urllib.error.URLError as e:
        pprint(f"[bold red]Failed to fetch /object_info: {e.reason}[/bold red]")
        raise typer.Exit(code=1) from e
    except TimeoutError as e:
        pprint(f"[bold red]Failed to fetch /object_info: timed out after {timeout}s[/bold red]")
        raise typer.Exit(code=1) from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        pprint("[bold red]Failed to fetch /object_info: server returned invalid JSON[/bold red]")
        raise typer.Exit(code=1) from e


def execute(
    workflow: str,
    host,
    port,
    wait=True,
    verbose=False,
    local_paths=False,
    timeout=30,
    api_key: str | None = None,
):
    workflow_name = os.path.abspath(os.path.expanduser(workflow))
    if not os.path.isfile(workflow):
        pprint(
            f"[bold red]Specified workflow file not found: {workflow}[/bold red]",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    if not check_comfy_server_running(port, host):
        pprint(f"[bold red]ComfyUI not running on specified address ({host}:{port})[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(workflow_name, encoding="utf-8") as f:
            raw_workflow = json.load(f)
    except OSError as e:
        pprint(f"[bold red]Unable to read workflow file: {e}[/bold red]")
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        pprint(f"[bold red]Specified workflow file is not valid JSON: {e}[/bold red]")
        raise typer.Exit(code=1) from e

    if is_ui_workflow(raw_workflow):
        pprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        object_info = fetch_object_info(host, port, timeout)
        try:
            workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            pprint(f"[bold red]Workflow conversion failed: {e}[/bold red]")
            raise typer.Exit(code=1) from e
        except Exception as e:
            # The converter is experimental; an unexpected crash here is a bug
            # in our code, not user error. Show a clean message and a pointer.
            pprint(
                f"[bold red]Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}[/bold red]\n"
                "[yellow]The UI-to-API converter is experimental. Please report this at[/yellow]\n"
                "[yellow]  https://github.com/Comfy-Org/comfy-cli/issues[/yellow]\n"
                "[yellow]and attach the workflow file if possible.[/yellow]"
            )
            if verbose:
                import traceback

                traceback.print_exc()
            raise typer.Exit(code=1) from e
        if not workflow:
            pprint("[bold red]Workflow conversion produced no executable nodes[/bold red]")
            raise typer.Exit(code=1)
    else:
        workflow = _validate_api_workflow(raw_workflow)
        if not workflow:
            pprint("[bold red]Specified workflow does not appear to be an API workflow json file[/bold red]")
            raise typer.Exit(code=1)

    progress = None
    start = time.time()
    if wait:
        pprint(f"Executing workflow: {workflow_name}")
        progress = ExecutionProgress()
        progress.start()
    else:
        print(f"Queuing workflow: {workflow_name}")

    execution = WorkflowExecution(workflow, host, port, verbose, progress, local_paths, timeout, api_key=api_key)

    try:
        if wait:
            execution.connect()
        execution.queue()
        if wait:
            execution.watch_execution()
            end = time.time()
            progress.stop()
            progress = None

            if len(execution.outputs) > 0:
                pprint("[bold green]\nOutputs:[/bold green]")

                for f in execution.outputs:
                    pprint(f)

            elapsed = timedelta(seconds=end - start)
            pprint(f"[bold green]\nWorkflow execution completed ({elapsed})[/bold green]")
        else:
            pprint("[bold green]Workflow queued[/bold green]")
    except WebSocketTimeoutException:
        pprint(
            f"[bold red]Error: WebSocket timed out after {timeout}s waiting for server response.[/bold red]\n"
            "[yellow]For long-running workflows, increase the timeout: comfy run --workflow <file> --timeout 300[/yellow]"
        )
        raise typer.Exit(code=1)
    except (WebSocketException, ConnectionError, OSError) as e:
        pprint(f"[bold red]Error: Lost connection to ComfyUI server: {e}[/bold red]")
        raise typer.Exit(code=1)
    finally:
        if progress:
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
    def __init__(self, workflow, host, port, verbose, progress, local_paths, timeout=30, api_key: str | None = None):
        self.workflow = workflow
        self.host = host
        self.port = port
        self.verbose = verbose
        self.local_paths = local_paths
        self.client_id = str(uuid.uuid4())
        self.outputs = []
        self.progress = progress
        self.remaining_nodes = set(self.workflow.keys())
        self.total_nodes = len(self.remaining_nodes)
        if progress:
            self.overall_task = self.progress.add_task("", total=self.total_nodes, progress_type="overall")
        self.current_node = None
        self.progress_task = None
        self.progress_node = None
        self.prompt_id = None
        self.ws = None
        self.timeout = timeout
        self.api_key = api_key

    def connect(self):
        self.ws = WebSocket()
        self.ws.connect(f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}")

    def queue(self):
        data: dict = {"prompt": self.workflow, "client_id": self.client_id}
        if self.api_key:
            data["extra_data"] = {"api_key_comfy_org": self.api_key}
        req = request.Request(
            f"http://{self.host}:{self.port}/prompt",
            json.dumps(data).encode("utf-8"),
        )
        try:
            resp = request.urlopen(req)
            body = json.loads(resp.read())

            self.prompt_id = body["prompt_id"]
        except urllib.error.HTTPError as e:
            message = "An unknown error occurred"
            if e.status == 500:
                # This is normally just the generic internal server error
                message = e.read().decode()
            elif e.status == 400:
                # Bad Request - workflow failed validation on the server
                body = json.loads(e.read())
                if body["node_errors"].keys():
                    message = json.dumps(body["node_errors"], indent=2)

            self.progress.stop()

            pprint(f"[bold red]Error running workflow\n{message}[/bold red]")
            raise typer.Exit(code=1)

    def watch_execution(self):
        self.ws.settimeout(self.timeout)
        while True:
            message = self.ws.recv()
            if isinstance(message, str):
                message = json.loads(message)
                if not self.on_message(message):
                    break

    def update_overall_progress(self):
        self.progress.update(self.overall_task, completed=self.total_nodes - len(self.remaining_nodes))

    def get_node_title(self, node_id):
        node = self.workflow.get(node_id)
        if node is None:
            return str(node_id)
        if "_meta" in node and "title" in node["_meta"]:
            return node["_meta"]["title"]
        return node["class_type"]

    def log_node(self, type, node_id):
        if not self.verbose:
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
        filename = img["filename"]
        subfolder = img["subfolder"] if "subfolder" in img else None
        output_type = img["type"] or "output"

        if self.local_paths:
            if subfolder:
                filename = os.path.join(subfolder, filename)

            return os.path.join(workspace_manager.get_workspace_path()[0], output_type, filename)

        query = urllib.parse.urlencode(img)
        return f"http://{self.host}:{self.port}/view?{query}"

    def on_message(self, message):
        data = message["data"] if "data" in message else {}
        # Skip any messages that aren't about our prompt
        if "prompt_id" not in data or data["prompt_id"] != self.prompt_id:
            return True

        if message["type"] == "executing":
            return self.on_executing(data)
        elif message["type"] == "execution_cached":
            self.on_cached(data)
        elif message["type"] == "progress":
            self.on_progress(data)
        elif message["type"] == "executed":
            self.on_executed(data)
        elif message["type"] == "execution_error":
            self.on_error(data)

        return True

    def on_executing(self, data):
        if self.progress_task:
            self.progress.remove_task(self.progress_task)
            self.progress_task = None

        if data["node"] is None:
            return False
        else:
            if self.current_node:
                self.remaining_nodes.discard(self.current_node)
                self.update_overall_progress()
            self.current_node = data["node"]
            self.log_node("Executing", data["node"])
        return True

    def on_cached(self, data):
        nodes = data["nodes"]
        for n in nodes:
            self.remaining_nodes.discard(n)
            self.log_node("Cached", n)
        self.update_overall_progress()

    def on_progress(self, data):
        node = data["node"]
        if self.progress_node != node:
            self.progress_node = node
            if self.progress_task:
                self.progress.remove_task(self.progress_task)

            self.progress_task = self.progress.add_task(
                self.get_node_title(node), total=data["max"], progress_type="node"
            )
        self.progress.update(self.progress_task, completed=data["value"])

    def on_executed(self, data):
        self.remaining_nodes.discard(data["node"])
        self.update_overall_progress()

        if "output" not in data:
            return

        output = data["output"]

        if output is None or "images" not in output:
            return

        for img in output["images"]:
            self.outputs.append(self.format_image_path(img))

    def on_error(self, data):
        pprint(f"[bold red]Error running workflow\n{json.dumps(data, indent=2)}[/bold red]")
        raise typer.Exit(code=1)
