import sys

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


def launch_comfyui(_env_checker, cpu):
    _env_checker.config['DEFAULT']['recent_path'] = os.getcwd()
    _env_checker.write_config()

    env_path = _env_checker.get_isolated_env()
    reboot_path = None

    new_env = os.environ.copy()

    if env_path is not None:
        new_env['__COMFY_CLI_SESSION__'] = os.path.join(env_path, 'comfy-cli')
        reboot_path = os.path.join(env_path, 'comfy-cli', '.reboot')

    while True:
        if cpu:
            subprocess.run(["python", "main.py", "--cpu"], env=new_env)
        else:
            subprocess.run(["python", "main.py"], env=new_env)

        if not os.path.exists(reboot_path):
            return
        os.remove(reboot_path)


@app.command(help="Launch ComfyUI: ?[--workspace <path>] ?[--cpu]")
def launch(workspace: Annotated[
                str,
                typer.Option(
                    show_default=False,
                    help="Path to ComfyUI workspace")
            ] = None,
           cpu: Annotated[
               bool,
               lambda: typer.Option(
                   default=False,
                   help="enable CPU mode")
           ] = False,
           ):
    _env_checker = EnvChecker()
    if workspace is not None:
        comfyui_path = os.path.join(workspace, 'ComfyUI')
        if os.path.exists(comfyui_path):
            os.chdir(comfyui_path)
            print(f"\nLaunch ComfyUI from repo: {_env_checker.comfy_repo.working_dir}\n")
            launch_comfyui(_env_checker, cpu)
        else:
            print(f"\nInvalid ComfyUI not found in specified workspace: {workspace}\n", file=sys.stderr)
            raise typer.Exit(code=1)

    elif _env_checker.comfy_repo is not None:
        print(f"\nLaunch ComfyUI from current repo: {_env_checker.comfy_repo.working_dir}\n")
        launch_comfyui(_env_checker, cpu)
    elif _env_checker.config['DEFAULT'].get('recent_path') is not None:
        comfy_path = _env_checker.config['DEFAULT'].get('recent_path')
        print(f"\nLaunch ComfyUI from recent repo: {comfy_path}\n")
        os.chdir(comfy_path)
        launch_comfyui(_env_checker, cpu)
    else:
        print(f"\nComfyUI is not available.\n", file=sys.stderr)
        raise typer.Exit(code=1)


@app.command(help="Print out current environment variables.")
def env():
    _env_checker = EnvChecker()
    _env_checker.print()


app.add_typer(models.app, name="models", help="Manage models.")
app.add_typer(custom_nodes.app, name="nodes", help="Manage custom nodes.")
app.add_typer(custom_nodes.manager_app, name="manager", help="Manager ComfyUI-Manager.")
