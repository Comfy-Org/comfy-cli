import typer
from typing_extensions import Annotated
from comfy_cli.command.models import models
from rich import print
import os
import subprocess


from comfy_cli.command import custom_nodes
from comfy_cli.command import install as install_inner
from comfy_cli.command import run as run_inner
from comfy_cli import constants
from comfy_cli.env_checker import EnvChecker
from comfy_cli.meta_data import MetadataManager
from comfy_cli import env_checker
from rich.console import Console
import time

app = typer.Typer()


def init():
    _env_checker = EnvChecker()
    metadata_manager = MetadataManager()
    start_time = time.time()
    metadata_manager.scan_dir()
    end_time = time.time()
    
    print(f"scan_dir took {end_time - start_time:.2f} seconds to run")


@app.callback(invoke_without_command=True)
def no_command(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        ctx.exit()


@app.command(help="Download and install ComfyUI and ComfyUI-Manager")
def install(
    url: Annotated[
        str,
        typer.Option(show_default=False)
    ] = constants.COMFY_GITHUB_URL,
    manager_url: Annotated[
        str,
        typer.Option(show_default=False)
    ] = constants.COMFY_MANAGER_GITHUB_URL,
    workspace: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="Path to ComfyUI workspace")
    ] = constants.COMFY_WORKSPACE,
    skip_manager: Annotated[
        bool,
        lambda: typer.Option(
            default=False,
            help="Skip installing the manager component")
    ] = False,
):
    checker = EnvChecker()
    if checker.python_version.major < 3:
        print(
            "[bold red]Python version 3.6 or higher is required to run ComfyUI.[/bold red]"
        )
        print(
            f"You are currently using Python version {env_checker.format_python_version(checker.python_version)}."
        )
    if checker.currently_in_comfy_repo:
        console = Console()
        # TODO: warn user that you are teh

    install_inner.execute(url, manager_url, workspace, skip_manager)



def update(self):
    print(f"Updating ComfyUI in {self.workspace}...")
    os.chdir(self.workspace)
    subprocess.run(["git", "pull"])
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


@app.command(help="Run workflow file")
def run(
    workflow_file: Annotated[str, typer.Option(help="Path to the workflow file.")],
    ):
    run_inner.execute(workflow_file)


@app.command(help="Launch ComfyUI")
def launch():
    os.chdir(self.workspace)
    run_inner.execute(workflow_file)


@app.command(help="Print out current environment variables.")
def env():
    env_checker = EnvChecker()
    env_checker.print()


app.add_typer(models.app, name="models", help="Manage models.")
app.add_typer(custom_nodes.app, name="nodes", help="Manage custom nodes.")
