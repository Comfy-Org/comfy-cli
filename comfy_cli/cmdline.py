import os
import subprocess
import sys
import time
import uuid
import webbrowser
from typing import Optional

import questionary
import typer
from rich import print
from rich.console import Console
from typing_extensions import Annotated, List

from comfy_cli import constants, env_checker, logging, tracking, ui, utils
from comfy_cli.command import custom_nodes
from comfy_cli.command import install as install_inner
from comfy_cli.command import run as run_inner
from comfy_cli.command.models import models as models_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import EnvChecker, check_comfy_server_running
from comfy_cli.meta_data import MetadataManager
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()
workspace_manager = WorkspaceManager()


def main():
    app()


@app.callback(invoke_without_command=True)
def entry(
    ctx: typer.Context,
    workspace: Optional[str] = typer.Option(
        default=None, show_default=False, help="Path to ComfyUI workspace"
    ),
    recent: Optional[bool] = typer.Option(
        default=False, show_default=False, is_flag=True, help="Execute from recent path"
    ),
    here: Optional[bool] = typer.Option(
        default=False,
        show_default=False,
        is_flag=True,
        help="Execute from current path",
    ),
):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        ctx.exit()

    ctx.ensure_object(dict)  # Ensure that ctx.obj exists and is a dict
    workspace_manager.update_context(ctx, workspace, recent, here)
    init()


def init():
    # TODO(yoland): after this
    metadata_manager = MetadataManager()
    start_time = time.time()
    metadata_manager.scan_dir()
    end_time = time.time()
    logging.setup_logging()
    tracking.prompt_tracking_consent()

    print(f"scan_dir took {end_time - start_time:.2f} seconds to run")


@app.command(help="Download and install ComfyUI and ComfyUI-Manager")
@tracking.track_command()
def install(
    ctx: typer.Context,
    url: Annotated[str, typer.Option(show_default=False)] = constants.COMFY_GITHUB_URL,
    manager_url: Annotated[
        str, typer.Option(show_default=False)
    ] = constants.COMFY_MANAGER_GITHUB_URL,
    restore: Annotated[
        bool,
        lambda: typer.Option(
            default=False,
            help="Restore dependencies for installed ComfyUI if not installed",
        ),
    ] = False,
    skip_manager: Annotated[
        bool, typer.Option(help="Skip installing the manager component")
    ] = False,
    amd: Annotated[bool, typer.Option(help="Install for AMD gpu")] = False,
    commit: Annotated[str, typer.Option(help="Specify commit hash for ComfyUI")] = None,
):
    checker = EnvChecker()

    # In the case of installation, since it involves installing in a non-existent path, get_workspace_path is not used.
    specified_workspace = ctx.obj.get(constants.CONTEXT_KEY_WORKSPACE)
    use_recent = ctx.obj.get(constants.CONTEXT_KEY_RECENT)
    use_here = ctx.obj.get(constants.CONTEXT_KEY_HERE)

    if specified_workspace:
        workspace_path = specified_workspace
    elif use_recent:
        workspace_path = workspace_manager.config_manager.get(
            constants.CONFIG_KEY_RECENT_WORKSPACE
        )
    elif use_here:
        workspace_path = os.getcwd()
    else:  # For installation, if not explicitly specified, it will only install in the default path.
        workspace_path = os.path.expanduser("~/comfy")

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
        torch_mode = "amd"

    install_inner.execute(
        url,
        manager_url,
        workspace_path,
        restore,
        skip_manager,
        torch_mode,
        commit=commit,
    )
    workspace_manager.set_recent_workspace(workspace_path)


def update(self):
    _env_checker = EnvChecker()
    print(f"Updating ComfyUI in {self.workspace}...")
    os.chdir(self.workspace)
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )


# @app.command(help="Run workflow file")
# @tracking.track_command()
# def run(
#   workflow_file: Annotated[str, typer.Option(help="Path to the workflow file.")],
# ):
#   run_inner.execute(workflow_file)


def validate_comfyui(_env_checker):
    if _env_checker.comfy_repo is None:
        print(
            f"[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]"
        )
        raise typer.Exit(code=1)


def launch_comfyui(_env_checker, _config_manager, extra, background=False):
    validate_comfyui(_env_checker)

    if background:
        if _config_manager.background is not None and utils.is_running(
            _config_manager.background[2]
        ):
            print(
                f"[bold red]ComfyUI is already running in background.\nYou cannot start more than one background service.[/bold red]\n"
            )
            raise typer.Exit(code=1)

        port = 8188
        listen = "127.0.0.1"

        if extra is not None:
            for i in range(len(extra) - 1):
                if extra[i] == "--port":
                    port = extra[i + 1]
                if listen[i] == "--listen":
                    listen = extra[i + 1]

            if check_comfy_server_running(port):
                print(
                    f"[bold red]The {port} port is already in use. A new ComfyUI server cannot be launched.\n[bold red]\n"
                )
                raise typer.Exit(code=1)

            if len(extra) > 0:
                extra = ["--"] + extra
        else:
            extra = []

        cmd = [
            "comfy",
            f'--workspace={os.path.join(os.getcwd(), "..")}',
            "launch",
        ] + extra

        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(
            f"[bold yellow]Run ComfyUI in the background.[/bold yellow] ({listen}:{port})"
        )
        _config_manager.config["DEFAULT"][
            constants.CONFIG_KEY_BACKGROUND
        ] = f"{(listen, port, process.pid)}"
        _config_manager.write_config()
        return

    env_path = _env_checker.get_isolated_env()
    reboot_path = None

    new_env = os.environ.copy()

    if env_path is not None:
        session_path = os.path.join(
            _config_manager.get_config_path(), "tmp", str(uuid.uuid4())
        )
        new_env["__COMFY_CLI_SESSION__"] = session_path

        # To minimize the possibility of leaving residue in the tmp directory, use files instead of directories.
        reboot_path = os.path.join(session_path + ".reboot")

    extra = extra if extra is not None else []

    while True:
        subprocess.run([sys.executable, "main.py"] + extra, env=new_env, check=False)

        if not os.path.exists(reboot_path):
            return

        os.remove(reboot_path)


@app.command(help="Stop background ComfyUI")
def stop():
    _config_manager = ConfigManager()

    if constants.CONFIG_KEY_BACKGROUND not in _config_manager.config["DEFAULT"]:
        print(f"[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)

    bg_info = _config_manager.background
    is_killed = utils.kill_all(bg_info[2])

    print(
        f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})"
    )

    _config_manager.remove_background()


@app.command(help="Launch ComfyUI: ?[--background] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(
    ctx: typer.Context,
    background: Annotated[
        bool, typer.Option(help="Launch ComfyUI in background")
    ] = False,
    extra: List[str] = typer.Argument(None),
):
    _env_checker = EnvChecker()
    _config_manager = ConfigManager()

    resolved_workspace = workspace_manager.get_workspace_path(ctx)
    if not resolved_workspace:
        print(
            "\nComfyUI is not available.\nTo install ComfyUI, you can run:\n\n\tcomfy install\n\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    print(f"\nLaunching ComfyUI from: {resolved_workspace}\n")

    os.chdir(resolved_workspace + "/ComfyUI")
    _env_checker.check()  # update environment checks

    # Update the recent workspace
    workspace_manager.set_recent_workspace(resolved_workspace)

    launch_comfyui(_env_checker, _config_manager, extra, background=background)


@app.command("set-default", help="Set default workspace")
@tracking.track_command()
def set_default(workspace_path: str):
    workspace_path = os.path.expanduser(workspace_path)
    comfy_path = os.path.join(workspace_path, "ComfyUI")
    if not os.path.exists(comfy_path):
        print(
            f"Invalid workspace path: {workspace_path}\nThe workspace path must contain 'ComfyUI'."
        )
        raise typer.Exit(code=1)

    workspace_manager.set_default_workspace(workspace_path)


@app.command(help="Show which ComfyUI is selected.")
@tracking.track_command()
def which(ctx: typer.Context):
    comfy_path = workspace_manager.get_workspace_path(ctx)
    if not os.path.exists(comfy_path) or not os.path.exists(
        os.path.join(comfy_path, "ComfyUI")
    ):
        print(
            f"ComfyUI not found, please run 'comfy install', run 'comfy' in a ComfyUI directory, or specify the workspace path with '--workspace'."
        )
        raise typer.Exit(code=1)

    print(f"Target ComfyUI path: {comfy_path}")


@app.command(help="Print out current environment variables.")
@tracking.track_command()
def env():
    _env_checker = EnvChecker()
    _env_checker.print()


@app.command(hidden=True)
@tracking.track_command()
def nodes():
    print(
        "\n[bold red] No such command, did you mean 'comfy node' instead?[/bold red]\n"
    )


@app.command(hidden=True)
@tracking.track_command()
def models():
    print(
        "\n[bold red] No such command, did you mean 'comfy model' instead?[/bold red]\n"
    )


@app.command(help="Provide feedback on the Comfy CLI tool.")
@tracking.track_command()
def feedback():
    print("Feedback Collection for Comfy CLI Tool\n")

    # General Satisfaction
    general_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5, how satisfied are you with the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
    )
    tracking.track_event(
        "feedback_general_satisfaction", {"score": general_satisfaction_score}
    )

    # Usability and User Experience
    usability_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5,  how satisfied are you with the usability and user experience of the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
    )
    tracking.track_event(
        "feedback_usability_satisfaction", {"score": usability_satisfaction_score}
    )

    # Additional Feature-Specific Feedback
    if questionary.confirm(
        "Do you want to provide additional feature-specific feedback on our GitHub page?"
    ).ask():
        tracking.track_event("feedback_additional")
        webbrowser.open("https://github.com/Comfy-Org/comfy-cli/issues/new/choose")

    print("Thank you for your feedback!")


app.add_typer(models_command.app, name="model", help="Manage models.")
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(custom_nodes.manager_app, name="manager", help="Manager ComfyUI-Manager.")
