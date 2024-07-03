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
from rich.panel import Panel
from typing_extensions import Annotated, List
import asyncio
import threading

from comfy_cli import constants, env_checker, logging, tracking, ui, utils
from comfy_cli.command import custom_nodes
from comfy_cli.command import run as run_inner
from comfy_cli.command import install as install_inner
from comfy_cli.command.models import models as models_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import GPU_OPTION, CUDAVersion
from comfy_cli.env_checker import EnvChecker, check_comfy_server_running
from comfy_cli.update import check_for_updates
from comfy_cli.workspace_manager import (
    WorkspaceManager,
    WorkspaceType,
    check_comfy_repo,
)

logging.setup_logging()
app = typer.Typer()
workspace_manager = WorkspaceManager()

console = Console()


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
                f"option `{param.name}` is mutually exclusive with option `{group.pop()}`"
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
    workspace: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Path to ComfyUI workspace",
            callback=exclusivity_callback,
        ),
    ] = None,
    recent: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Execute from recent path",
            callback=exclusivity_callback,
        ),
    ] = None,
    here: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Execute from current path",
            callback=exclusivity_callback,
        ),
    ] = None,
    skip_prompt: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Do not prompt user for input, use default options",
        ),
    ] = None,
    enable_telemetry: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            hidden=True,
            is_flag=True,
            help="Enable tracking",
        ),
    ] = True,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print version and exit",
        is_flag=True,
    ),
):
    if version:
        print(ConfigManager().get_cli_version())
        ctx.exit(0)

    workspace_manager.setup_workspace_manager(workspace, here, recent, skip_prompt)

    tracking.prompt_tracking_consent(skip_prompt, default_value=enable_telemetry)

    if ctx.invoked_subcommand is None:
        print(
            "[bold yellow]Welcome to Comfy CLI![/bold yellow]: https://github.com/Comfy-Org/comfy-cli"
        )
        print(ctx.get_help())
        ctx.exit()

    # TODO: Move this to proper place
    # start_time = time.time()
    # workspace_manager.scan_dir()
    # end_time = time.time()
    #
    # logging.info(f"scan_dir took {end_time - start_time:.2f} seconds to run")


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
    cuda_version: Annotated[
        CUDAVersion, typer.Option(show_default=True)
    ] = CUDAVersion.v12_1,
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
    intel_arc: Annotated[
        bool,
        typer.Option(
            hidden=True,
            show_default=False,
            help="(Beta support) install for Intel Arc gpu, based on https://github.com/comfyanonymous/ComfyUI/pull/3439",
            callback=gpu_exclusivity_callback,
        ),
    ] = None,
    cpu: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Install for CPU",
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
        comfy_path = workspace_manager.workspace_path

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
    elif intel_arc:
        gpu = GPU_OPTION.INTEL_ARC
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
        print(
            "[bold yellow]Installing on Intel ARC is not yet completely supported[/bold yellow]"
        )
        env_check = env_checker.EnvChecker()
        if env_check.conda_env is None:
            print(
                "[bold red]Intel ARC support requires conda environment to be activated.[/bold red]"
            )
            raise typer.Exit(code=1)
        if intel_arc is None:
            confirm_result = ui.prompt_confirm_action(
                "Are you sure you want to try beta install feature on Intel ARC?", True
            )
            if not confirm_result:
                raise typer.Exit(code=0)
        print("[bold yellow]Installing on Intel ARC is in beta stage.[/bold yellow]")

    if gpu is None and not cpu:
        print(
            "[bold red]No GPU option selected or `--cpu` enabled, use --\\[gpu option] flag (e.g. --nvidia) to pick GPU. use `--cpu` to install for CPU. Exiting...[/bold red]"
        )
        raise typer.Exit(code=1)

    install_inner.execute(
        url,
        manager_url,
        comfy_path,
        restore,
        skip_manager,
        commit=commit,
        gpu=gpu,
        cuda_version=cuda_version,
        plat=platform,
        skip_torch_or_directml=skip_torch_or_directml,
        skip_requirement=skip_requirement,
    )

    print(f"ComfyUI is installed at: {comfy_path}")


@app.command(help="Update ComfyUI Environment [all|comfy]")
@tracking.track_command()
def update(
    target: str = typer.Argument(
        "comfy",
        help="[all|comfy]",
        autocompletion=utils.create_choice_completer(["all", "comfy"]),
    )
):
    if target not in ["all", "comfy"]:
        typer.echo(
            f"Invalid target: {target}. Allowed targets are 'all', 'comfy'.",
            err=True,
        )
        raise typer.Exit(code=1)

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

    custom_nodes.command.update_node_id_cache()


@app.command(
    help="Run API workflow file using the ComfyUI launched by `comfy launch --background`"
)
@tracking.track_command()
def run(
    workflow: Annotated[str, typer.Option(help="Path to the workflow API json file.")],
    wait: Annotated[
        Optional[bool],
        typer.Option(help="If the command should wait until execution completes."),
    ] = True,
    verbose: Annotated[
        Optional[bool],
        typer.Option(help="Enables verbose output of the execution process."),
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option(
            help="The IP/hostname where the ComfyUI instance is running, e.g. 127.0.0.1 or localhost."
        ),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="The port where the ComfyUI instance is running, e.g. 8188."),
    ] = None,
):
    config = ConfigManager()

    if host:
        s = host.split(":")
        host = s[0]
        if not port and len(s) == 2:
            port = int(s[1])

    local_paths = False
    if config.background:
        if not host:
            host = config.background[0]
            local_paths = True
        if port:
            local_paths = False
        else:
            port = config.background[1]

    if not host:
        host = "127.0.0.1"
    if not port:
        port = 8188

    run_inner.execute(workflow, host, port, wait, verbose, local_paths)


def validate_comfyui(_env_checker):
    if _env_checker.comfy_repo is None:
        print(
            "[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]"
        )
        raise typer.Exit(code=1)


async def launch_and_monitor(cmd, listen, port):
    """
    Monitor the process during the background launch.

    If a success message is captured, exit;
    otherwise, return the log in case of failure.
    """
    logging_flag = False
    log = []
    logging_lock = threading.Lock()

    # NOTE: To prevent encoding error on Windows platform
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["COMFY_CLI_BACKGROUND"] = "true"

    if sys.platform == "win32":
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            encoding="utf-8",
            shell=True,  # win32 only
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # win32 only
        )
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            encoding="utf-8",
        )

    def msg_hook(stream):
        nonlocal log
        nonlocal logging_flag

        while True:
            line = stream.readline()
            if "Launching ComfyUI from:" in line:
                logging_flag = True
            elif "To see the GUI go to:" in line:
                print(
                    f"[bold yellow]ComfyUI is successfully launched in the background.[/bold yellow]\nTo see the GUI go to: http://{listen}:{port}"
                )
                ConfigManager().config["DEFAULT"][
                    constants.CONFIG_KEY_BACKGROUND
                ] = f"{(listen, port, process.pid)}"
                ConfigManager().write_config()

                # NOTE: os.exit(0) doesn't work.
                os._exit(0)

            with logging_lock:
                if logging_flag:
                    log.append(line)

    stdout_thread = threading.Thread(target=msg_hook, args=(process.stdout,))
    stderr_thread = threading.Thread(target=msg_hook, args=(process.stderr,))

    stdout_thread.start()
    stderr_thread.start()

    res = process.wait()

    return log


def background_launch(extra):
    config_background = ConfigManager().background
    if config_background is not None and utils.is_running(config_background[2]):
        console.print(
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

        if len(extra) > 0:
            extra = ["--"] + extra
    else:
        extra = []

    if check_comfy_server_running(port):
        console.print(
            f"[bold red]The {port} port is already in use. A new ComfyUI server cannot be launched.\n[bold red]\n"
        )
        raise typer.Exit(code=1)

    cmd = [
        "comfy",
        f"--workspace={os.path.abspath(os.getcwd())}",
        "launch",
    ] + extra

    loop = asyncio.get_event_loop()
    log = loop.run_until_complete(launch_and_monitor(cmd, listen, port))

    if log is not None:
        console.print(
            Panel(
                "".join(log),
                title="[bold red]Error log during ComfyUI execution[/bold red]",
                border_style="bright_red",
            )
        )

    console.print(f"\n[bold red]Execution error: failed to launch ComfyUI[/bold red]\n")
    # NOTE: os.exit(0) doesn't work
    os._exit(1)


def launch_comfyui(extra):
    reboot_path = None

    new_env = os.environ.copy()

    session_path = os.path.join(
        ConfigManager().get_config_path(), "tmp", str(uuid.uuid4())
    )
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["PYTHONENCODING"] = "utf-8"

    # To minimize the possibility of leaving residue in the tmp directory, use files instead of directories.
    reboot_path = os.path.join(session_path + ".reboot")

    extra = extra if extra is not None else []

    process = None

    if "COMFY_CLI_BACKGROUND" not in os.environ:
        # If not running in background mode, there's no need to use popen. This can prevent the issue of linefeeds occurring with tqdm.
        while True:
            res = subprocess.run(
                [sys.executable, "main.py"] + extra, env=new_env, check=False
            )

            if reboot_path is None:
                print("[bold red]ComfyUI is not installed.[/bold red]\n")
                exit(res)

            if not os.path.exists(reboot_path):
                exit(res)

            os.remove(reboot_path)
    else:
        # If running in background mode without using a popen, broken pipe errors may occur when flushing stdout/stderr.
        def redirector_stderr():
            while True:
                if process is not None:
                    print(process.stderr.readline(), end="")

        def redirector_stdout():
            while True:
                if process is not None:
                    print(process.stdout.readline(), end="")

        threading.Thread(target=redirector_stderr).start()
        threading.Thread(target=redirector_stdout).start()

        try:
            while True:
                if sys.platform == "win32":
                    process = subprocess.Popen(
                        [sys.executable, "main.py"] + extra,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=new_env,
                        encoding="utf-8",
                        shell=True,  # win32 only
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # win32 only
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, "main.py"] + extra,
                        text=True,
                        env=new_env,
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                process.wait()

                if reboot_path is None:
                    print("[bold red]ComfyUI is not installed.[/bold red]\n")
                    os._exit(process.pid)

                if not os.path.exists(reboot_path):
                    os._exit(process.pid)

                os.remove(reboot_path)
        except KeyboardInterrupt:
            if process is not None:
                os._exit(1)


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
    if background:
        background_launch(extra)
    else:
        launch_comfyui(extra)


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
        print(
            f"\nPath not found: {comfy_path}.\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    is_comfy_repo, comfy_repo = check_comfy_repo(comfy_path)
    if not is_comfy_repo:
        print(
            f"\nSpecified path is not a ComfyUI path: {comfy_path}.\n",
            file=sys.stderr,
        )
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
    table = _env_checker.fill_print_table()
    workspace_manager.fill_print_table(table)
    console.print(table)


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
        force_prompting=True,
    )
    tracking.track_event(
        "feedback_general_satisfaction", {"score": general_satisfaction_score}
    )

    # Usability and User Experience
    usability_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5,  how satisfied are you with the usability and user experience of the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
        force_prompting=True,
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
