import os
import typer
from typing_extensions import Annotated
from comfy_cli import tracking, ui
from comfy_cli.workspace_manager import WorkspaceManager
from comfy_cli.command import custom_nodes
from rich import print
import uuid
import yaml

workspace_manager = WorkspaceManager()

app = typer.Typer()


@app.command(help="Save current snapshot to .yaml file")
@tracking.track_command()
def save(
    output: Annotated[
        str,
        "--output",
        typer.Option(show_default=False, help="Specify the output file path. (.yaml)"),
    ],
):

    if not output.endswith(".yaml"):
        print("[bold red]The output path must end with '.yaml'.[/bold red]")
        raise typer.Exit(code=1)

    output_path = os.path.abspath(output)

    config_manager = workspace_manager.config_manager
    tmp_path = (
        os.path.join(config_manager.get_config_path(), "tmp", str(uuid.uuid4()))
        + ".yaml"
    )
    tmp_path = os.path.abspath(tmp_path)
    custom_nodes.command.execute_cm_cli(
        ["save-snapshot", "--output", tmp_path], silent=True
    )

    with open(tmp_path, "r", encoding="UTF-8") as yaml_file:
        info = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    os.remove(tmp_path)

    info["basic"] = "N/A"  # TODO:
    info["models"] = []  # TODO:

    with open(output_path, "w", encoding="UTF-8") as yaml_file:
        yaml.dump(info, yaml_file, allow_unicode=True)

    print(f"Snapshot file is saved as `{output_path}`")


@app.command(help="Restore from snapshot file")
@tracking.track_command()
def restore(
    input: Annotated[
        str,
        "--input",
        typer.Option(show_default=False, help="Specify the input file path. (.yaml)"),
    ],
):
    input_path = os.path.abspath(input)
    apply_snapshot(input_path)

    # TODO: restore other properties


def apply_snapshot(filepath):
    if not os.path.exists(filepath):
        print(f"[bold red]File not found: {filepath}[/bold red]")
        raise typer.Exit(code=1)

    if workspace_manager.get_comfyui_manager_path() is None or not os.path.exists(
        workspace_manager.get_comfyui_manager_path()
    ):
        print(
            "[bold red]If ComfyUI-Manager is not installed, the snapshot feature cannot be used.[/bold red]"
        )
        raise typer.Exit(code=1)

    custom_nodes.command.restore_snapshot(filepath)
