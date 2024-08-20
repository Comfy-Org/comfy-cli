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
from websocket import WebSocket

from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()


def load_api_workflow(file: str):
    with open(file, encoding="utf-8") as f:
        workflow = json.load(f)
        # Check for litegraph properties to ensure this isnt a UI workflow file
        if "nodes" in workflow and "links" in workflow:
            return None

        # Try validating the first entry to ensure it has a node class property
        node_id = next(iter(workflow))
        node = workflow[node_id]
        if "class_type" not in node:
            return None

        return workflow


def execute(workflow: str, host, port, wait=True, verbose=False, local_paths=False, timeout=30):
    workflow_name = os.path.abspath(os.path.expanduser(workflow))
    if not os.path.isfile(workflow):
        pprint(
            f"[bold red]Specified workflow file not found: {workflow}[/bold red]",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    workflow = load_api_workflow(workflow)

    if not workflow:
        pprint("[bold red]Specified workflow does not appear to be an API workflow json file[/bold red]")
        raise typer.Exit(code=1)

    if not check_comfy_server_running(port, host):
        pprint(f"[bold red]ComfyUI not running on specified address ({host}:{port})[/bold red]")
        raise typer.Exit(code=1)

    progress = None
    start = time.time()
    if wait:
        pprint(f"Executing workflow: {workflow_name}")
        progress = ExecutionProgress()
        progress.start()
    else:
        print(f"Queuing workflow: {workflow_name}")

    execution = WorkflowExecution(workflow, host, port, verbose, progress, local_paths, timeout)

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
            percent = "[progress.percentage]{task.percentage:>3.0f}%".format(task=task)
            if task.fields.get("progress_type") == "overall":
                overall_table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)
                overall_table.add_row(BarColumn().render(task), percent, TimeElapsedColumn().render(task))
                yield overall_table
            else:
                yield self.make_tasks_table([task])


class WorkflowExecution:
    def __init__(self, workflow, host, port, verbose, progress, local_paths, timeout=30):
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

    def connect(self):
        self.ws = WebSocket()
        self.ws.connect(f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}")

    def queue(self):
        data = {"prompt": self.workflow, "client_id": self.client_id}
        req = request.Request(f"http://{self.host}:{self.port}/prompt", json.dumps(data).encode("utf-8"))
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
        node = self.workflow[node_id]
        if "_meta" in node and "title" in node["_meta"]:
            return node["_meta"]["title"]
        return node["class_type"]

    def log_node(self, type, node_id):
        if not self.verbose:
            return

        node = self.workflow[node_id]
        class_type = node["class_type"]
        title = self.get_node_title(node_id)

        if title != class_type:
            title += f"[bright_black] - {class_type}[/]"
        title += f"[bright_black] ({node_id})[/]"

        pprint(f"{type} : {title}")

    def format_image_path(self, img):
        filename = img["filename"]
        subfolder = img["subfolder"]
        output_type = img["type"] or "output"

        if self.local_paths:
            if subfolder:
                filename = os.path.join(subfolder, filename)

            filename = os.path.join(workspace_manager.get_workspace_path()[0], output_type, filename)
            return filename

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

        if "images" not in output:
            return

        for img in output["images"]:
            self.outputs.append(self.format_image_path(img))

    def on_error(self, data):
        pprint(f"[bold red]Error running workflow\n{json.dumps(data, indent=2)}[/bold red]")
        raise typer.Exit(code=1)
