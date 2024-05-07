import os
import subprocess
import sys
import time
import uuid
import webbrowser
from typing import Optional
from comfy_cli.constants import GPU_OPTION

import questionary
import typer
from rich import print
from typing_extensions import Annotated, List

from comfy_cli import constants, env_checker, logging, tracking, ui, utils
from comfy_cli.command import custom_nodes
from comfy_cli.command import install as install_inner
from comfy_cli.command.models import models as models_command
from comfy_cli.update import check_for_updates
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import EnvChecker, check_comfy_server_running
from comfy_cli.workspace_manager import (
    WorkspaceManager,
    check_comfy_repo,
    WorkspaceType,
)

logging.setup_logging()
app = typer.Typer()
workspace_manager = WorkspaceManager()


def main():
    app()


def mutually_exclusive_group_options():
    group = []

    def callback(_ctx: typer.Context, param: typer.CallbackParam, value: str):
        # Add cli option to group if it was called with a value
        if value is not None and param.name not in group:
            group.append(param.name)
        if len(group) > 1:
            raise typer.BadParameter(
                f"option `{param.name}` is mutually exclusive with option `{group[0]}`"
            )
        return value

    return callback


exclusivity_callback = mutually_exclusive_group_options()


@app.command(help="Display help for commands")
def help(ctx: typer.Context):
    print(ctx.find_root().get_help())
    ctx.exit(0)


@app.callback(invoke_without_command=True)
def entry(
    ctx: typer.Context,
    workspace: Optional[str] = typer.Option(
        default=None,
        show_default=False,
        help="Path to ComfyUI workspace",
        callback=exclusivity_callback,
    ),
    recent: Optional[bool] = typer.Option(
        default=None,
        show_default=False,
        is_flag=True,
        help="Execute from recent path",
        callback=exclusivity_callback,
    ),
    here: Optional[bool] = typer.Option(
        default=None,
        show_default=False,
        is_flag=True,
        help="Execute from current path",
        callback=exclusivity_callback,
    ),
):
    workspace_manager.setup_workspace_manager(workspace, here, recent)

    tracking.prompt_tracking_consent()

    if ctx.invoked_subcommand is None:
        print(
            "[bold yellow]Welcome to Comfy CLI![/bold yellow]: https://github.com/Comfy-Org/comfy-cli"
        )
        print(ctx.get_help())
        ctx.exit()
    start_time = time.time()
    workspace_manager.scan_dir()
    end_time = time.time()

    logging.info(f"scan_dir took {end_time - start_time:.2f} seconds to run")


gpu_exclusivity_callback = mutually_exclusive_group_options()


@app.command(help="Download and install ComfyUI and ComfyUI-Manager")
@tracking.track_command()
def install(
    url: Annotated[str, typer.Option(show_default=False)] = constants.COMFY_GITHUB_URL,
    manager_url: Annotated[
        str, typer.Option(show_default=False)
    ] = constants.COMFY_MANAGER_GITHUB_URL,
    restore: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Restore dependencies for installed ComfyUI if not installed",
        ),
    ] = False,
    skip_manager: Annotated[
        bool,
        typer.Option(show_default=False, help="Skip installing the manager component"),
    ] = False,
    skip_torch_or_directml: Annotated[
        bool,
        typer.Option(show_default=False, help="Skip installing PyTorch Or DirectML"),
    ] = False,
    skip_requirement: Annotated[
        bool, typer.Option(show_default=False, help="Skip installing requirements.txt")
    ] = False,
    nvidia: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Install for Nvidia gpu",
            callback=gpu_exclusivity_callback,
        ),
    ] = None,
    amd: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Install for AMD gpu",
            callback=gpu_exclusivity_callback,
        ),
    ] = None,
    m_series: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Install for Mac M-Series gpu",
            callback=gpu_exclusivity_callback,
        ),
    ] = None,
    commit: Annotated[
        Optional[str], typer.Option(help="Specify commit hash for ComfyUI")
    ] = None,
):
    check_for_updates()
    checker = EnvChecker()

    comfy_path = workspace_manager.get_specified_workspace()

    if comfy_path is None:
        comfy_path = utils.get_not_user_set_default_workspace()

    is_comfy_path, repo_dir = check_comfy_repo(comfy_path)
    if is_comfy_path and not restore:
        print(
            f"[bold red]ComfyUI is already installed at the specified path:[/bold red] {comfy_path}\n"
        )
        print(
            "[bold yellow]If you want to restore dependencies, add the '--restore' option.[/bold yellow]",
        )
        raise typer.Exit(code=1)

    if repo_dir is not None:
        comfy_path = str(repo_dir.working_dir)

    if checker.python_version.major < 3 or checker.python_version.minor < 9:
        print(
            "[bold red]Python version 3.9 or higher is required to run ComfyUI.[/bold red]"
        )
        print(
            f"You are currently using Python version {env_checker.format_python_version(checker.python_version)}."
        )

    platform = utils.get_os()
    if nvidia and platform == constants.OS.MACOS:
        print(
            "[bold red]Nvidia GPU is never on MacOS. What are you smoking? 🤔[/bold red]"
        )
        raise typer.Exit(code=1)

    if platform != constants.OS.MACOS and m_series:
        print(f"[bold red]You are on {platform} bruh [/bold red]")

    gpu = None

    if nvidia:
        gpu = GPU_OPTION.NVIDIA
    elif amd:
        gpu = GPU_OPTION.AMD
    elif m_series:
        gpu = GPU_OPTION.M_SERIES
    else:
        if platform == constants.OS.MACOS:
            gpu = ui.prompt_select_enum(
                "What type of Mac do you have?",
                [GPU_OPTION.M_SERIES, GPU_OPTION.MAC_INTEL],
            )
        else:
            gpu = ui.prompt_select_enum(
                "What GPU do you have?",
                [GPU_OPTION.NVIDIA, GPU_OPTION.AMD, GPU_OPTION.INTEL_ARC],
            )

    if gpu == GPU_OPTION.INTEL_ARC:
        print("[bold yellow]Installing on Intel ARC is not yet supported[/bold yellow]")
        print(
            "[bold yellow]Feel free to follow this thread to manually install:\nhttps://github.com/comfyanonymous/ComfyUI/discussions/476[/bold yellow]"
        )

    install_inner.execute(
        url,
        manager_url,
        comfy_path,
        restore,
        skip_manager,
        commit=commit,
        gpu=gpu,
        platform=platform,
        skip_torch_or_directml=skip_torch_or_directml,
        skip_requirement=skip_requirement,
    )

    print(f"ComfyUI is installed at: {comfy_path}")


@app.command(help="Update ComfyUI Environment [all|comfy]")
@tracking.track_command()
def update(target: str = typer.Argument("comfy", help="[all|comfy]")):
    if target not in ["all", "comfy"]:
        typer.echo(
            f"Invalid target: {target}. Allowed targets are 'all', 'comfy'.",
            err=True,
        )
        raise typer.Exit(code=1)

    _env_checker = EnvChecker()
    comfy_path = workspace_manager.workspace_path

    if "all" == target:
        custom_nodes.command.execute_cm_cli(["update", "all"])
    else:
        print(f"Updating ComfyUI in {comfy_path}...")
        if comfy_path is None:
            print("ComfyUI path is not found.")
            raise typer.Exit(code=1)
        os.chdir(comfy_path)
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
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
            "[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]"
        )
        raise typer.Exit(code=1)


def launch_comfyui(extra, background=False):
    if background:
        config_background = ConfigManager().background
        if config_background is not None and utils.is_running(config_background[2]):
            print(
                "[bold red]ComfyUI is already running in background.\nYou cannot start more than one background service.[/bold red]\n"
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
            f"--workspace={os.path.abspath(os.getcwd())}",
            "launch",
        ] + extra

        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(
            f"[bold yellow]Run ComfyUI in the background.[/bold yellow] ({listen}:{port})"
        )
        ConfigManager().config["DEFAULT"][
            constants.CONFIG_KEY_BACKGROUND
        ] = f"{(listen, port, process.pid)}"
        ConfigManager().write_config()
        return

    env_path = EnvChecker().get_isolated_env()
    reboot_path = None

    new_env = os.environ.copy()

    if env_path is not None:
        session_path = os.path.join(
            ConfigManager().get_config_path(), "tmp", str(uuid.uuid4())
        )
        new_env["__COMFY_CLI_SESSION__"] = session_path

        # To minimize the possibility of leaving residue in the tmp directory, use files instead of directories.
        reboot_path = os.path.join(session_path + ".reboot")

    extra = extra if extra is not None else []

    while True:
        subprocess.run([sys.executable, "main.py"] + extra, env=new_env, check=False)

        if not reboot_path:
            print("[bold red]ComfyUI is not installed.[/bold red]\n")
            return

        if not os.path.exists(reboot_path):
            return

        os.remove(reboot_path)


@app.command(help="Stop background ComfyUI")
@tracking.track_command()
def stop():
    if constants.CONFIG_KEY_BACKGROUND not in ConfigManager().config["DEFAULT"]:
        print("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)

    bg_info = ConfigManager().background
    if not bg_info:
        print("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)
    is_killed = utils.kill_all(bg_info[2])

    if not is_killed:
        print("[bold red]Failed to stop ComfyUI in the background.[/bold red]\n")
    else:
        print(
            f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})"
        )

    ConfigManager().remove_background()


@app.command(help="Launch ComfyUI: ?[--background] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(
    background: Annotated[
        bool, typer.Option(help="Launch ComfyUI in background")
    ] = False,
    extra: List[str] = typer.Argument(None),
):
    check_for_updates()
    resolved_workspace = workspace_manager.workspace_path
    if not resolved_workspace:
        print(
            "\nComfyUI is not available.\nTo install ComfyUI, you can run:\n\n\tcomfy install\n\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    if (
        extra is None or len(extra) == 0
    ) and workspace_manager.workspace_type == WorkspaceType.DEFAULT:
        launch_extras = workspace_manager.config_manager.config["DEFAULT"].get(
            constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, ""
        )

        if launch_extras != "":
            extra = launch_extras.split(" ")

    print(f"\nLaunching ComfyUI from: {resolved_workspace}\n")

    # Update the recent workspace
    workspace_manager.set_recent_workspace(resolved_workspace)

    os.chdir(resolved_workspace)
    launch_comfyui(extra, background=background)


@app.command("set-default", help="Set default ComfyUI path")
@tracking.track_command()
def set_default(
    workspace_path: str,
    launch_extras: Annotated[
        str, typer.Option(help="Specify extra options for launch")
    ] = "",
):
    comfy_path = os.path.abspath(os.path.expanduser(workspace_path))

    if not os.path.exists(comfy_path):
        print(f"Path not found: {comfy_path}.")
        raise typer.Exit(code=1)

    is_comfy_repo, comfy_repo = check_comfy_repo(comfy_path)
    if not is_comfy_repo:
        print(f"Specified path is not a ComfyUI path: {comfy_path}.")
        raise typer.Exit(code=1)

    comfy_path = comfy_repo.working_dir

    print(f"Specified path is set as default ComfyUI path: {comfy_path} ")
    workspace_manager.set_default_workspace(comfy_path)
    workspace_manager.set_default_launch_extras(launch_extras)


@app.command(help="Show which ComfyUI is selected.")
@tracking.track_command()
def which():
    comfy_path = workspace_manager.workspace_path
    if comfy_path is None:
        print(
            "ComfyUI not found, please run 'comfy install', run 'comfy' in a ComfyUI directory, or specify the workspace path with '--workspace'."
        )
        raise typer.Exit(code=1)

    print(f"Target ComfyUI path: {comfy_path}")


@app.command(help="Print out current environment variables.")
@tracking.track_command()
def env():
    check_for_updates()
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
app.add_typer(custom_nodes.manager_app, name="manager", help="Manage ComfyUI-Manager.")
app.add_typer(tracking.app, name="tracking", help="Manage analytics tracking settings.")
