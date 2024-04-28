import sys

import typer
from typing_extensions import List, Annotated
from comfy_cli.command.models import models as models_command
from rich import print
import os
import subprocess


from comfy_cli.command import custom_nodes
from comfy_cli.command import install as install_inner
from comfy_cli.command import run as run_inner
from comfy_cli import constants, tracking
from comfy_cli.env_checker import EnvChecker
from comfy_cli.meta_data import MetadataManager
from comfy_cli import env_checker
from rich.console import Console
import time

app = typer.Typer()


def main():
    init()
    app()


def init():
    # TODO(yoland): after this 
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
@tracking.track_command()
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
    ] = "~/comfy",
    restore: Annotated[
        bool,
        lambda: typer.Option(
            default=False,
            help="Restore dependencies for installed ComfyUI if not installed")
    ] = False,
    skip_manager: Annotated[
        bool,
        typer.Option(
            help="Skip installing the manager component")
    ] = False,
    amd: Annotated[
        bool,
        typer.Option(
            help="Install for AMD gpu")
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

    torch_mode = None
    if amd:
        torch_mode = 'amd'

    install_inner.execute(url, manager_url, workspace, restore, skip_manager, torch_mode)


def update(self):
    _env_checker = EnvChecker()
    print(f"Updating ComfyUI in {self.workspace}...")
    os.chdir(self.workspace)
    subprocess.run(["git", "pull"], check=True)
    subprocess.run([sys.executable, '-m', "pip", "install", "-r", "requirements.txt"], check=True)


@app.command(help="Run workflow file")
@tracking.track_command()
def run(
        workflow_file: Annotated[str, typer.Option(help="Path to the workflow file.")],
        ):
    run_inner.execute(workflow_file)


def validate_comfyui(_env_checker):
    if _env_checker.comfy_repo is None:
        print(f"[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)


def launch_comfyui(_env_checker, extra):
    # TODO(yoland/data): Disabled config writing for now, checking with @data
    # We need to find a viable place to write config file, e.g. standard place
    # for macOS:   ~/Library/Application Support/
    # for linx:    ~/.config/
    # for windows: C:\Users\<username>\AppData\Local\
    #_env_checker.config['DEFAULT']['recent_path'] = os.getcwd()
    #_env_checker.write_config()


    validate_comfyui(_env_checker)

    env_path = _env_checker.get_isolated_env()
    reboot_path = None

    new_env = os.environ.copy()

    if env_path is not None:
        new_env['__COMFY_CLI_SESSION__'] = os.path.join(env_path, 'comfy-cli')
        reboot_path = os.path.join(env_path, 'comfy-cli', '.reboot')

    extra = extra if extra is not None else []

    while True:
        subprocess.run([sys.executable, "main.py"] + extra, env=new_env, check=False)

        if not os.path.exists(reboot_path):
            return

        os.remove(reboot_path)


@app.command(help="Launch ComfyUI: ?[--workspace <path>] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
           extra: List[str] = typer.Argument(None)):

    _env_checker = EnvChecker()

    if workspace is not None:
        comfyui_path = os.path.join(workspace, 'ComfyUI')
        if os.path.exists(comfyui_path):
            os.chdir(comfyui_path)
            _env_checker.check()  # update env

            print(f"\nLaunch ComfyUI from repo: {_env_checker.comfy_repo.working_dir}\n")
            launch_comfyui(_env_checker, extra)
        else:
            print(f"\nInvalid ComfyUI not found in specified workspace: {workspace}\n", file=sys.stderr)
            raise typer.Exit(code=1)

    elif _env_checker.comfy_repo is not None:
        os.chdir(_env_checker.comfy_repo.working_dir)
        print(f"\nLaunch ComfyUI from current repo: {_env_checker.comfy_repo.working_dir}\n")
        launch_comfyui(_env_checker, extra)

    elif _env_checker.config['DEFAULT'].get('recent_path') is not None:
        comfy_path = _env_checker.config['DEFAULT'].get('recent_path')
        print(f"\nLaunch ComfyUI from recent repo: {comfy_path}\n")

        os.chdir(comfy_path)
        _env_checker.check()  # update env

        launch_comfyui(_env_checker, extra)
    else:
        print("\nComfyUI is not available.\nTo install ComfyUI, you can run:\n\n\tcomfy install\n\n", file=sys.stderr)
        raise typer.Exit(code=1)


@app.command(help="Print out current environment variables.")
@tracking.track_command()
def env():
    _env_checker = EnvChecker()
    _env_checker.print()


@app.command(hidden=True)
@tracking.track_command()
def nodes():
    print("\n[bold red] No such command, did you mean 'comfy node' instead?[/bold red]\n")


@app.command(hidden=True)
@tracking.track_command()
def models():
    print("\n[bold red] No such command, did you mean 'comfy model' instead?[/bold red]\n")


app.add_typer(models_command.app, name="model", help="Manage models.")
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(custom_nodes.manager_app, name="manager", help="Manager ComfyUI-Manager.")
