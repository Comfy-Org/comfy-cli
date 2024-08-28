import os
import subprocess
import sys
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
from comfy_cli.command.launch import launch as launch_command
from comfy_cli.command.models import models as models_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import GPU_OPTION, CUDAVersion
from comfy_cli.env_checker import EnvChecker
from comfy_cli.standalone import StandalonePython
from comfy_cli.update import check_for_updates
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

logging.setup_logging()
app = typer.Typer()
workspace_manager = WorkspaceManager()

console = Console()


def main():
    app()


class MutuallyExclusiveValidator:
    def __init__(self):
        self.group = []

    def reset_for_testing(self):
        self.group.clear()

    def validate(self, _ctx: typer.Context, param: typer.CallbackParam, value: str):
        # Add cli option to group if it was called with a value
        if value is not None and param.name not in self.group:
            self.group.append(param.name)
        if len(self.group) > 1:
            raise typer.BadParameter(f"option `{param.name}` is mutually exclusive with option `{self.group.pop()}`")
        return value


g_exclusivity = MutuallyExclusiveValidator()
g_gpu_exclusivity = MutuallyExclusiveValidator()


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
            callback=g_exclusivity.validate,
        ),
    ] = None,
    recent: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Execute from recent path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    here: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Execute from current path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    skip_prompt: Annotated[
        bool,
        typer.Option(
            show_default=False,
            is_flag=True,
            help="Do not prompt user for input, use default options",
        ),
    ] = False,
    enable_telemetry: Annotated[
        bool,
        typer.Option(
            show_default=False,
            hidden=True,
            is_flag=True,
            help="Enable tracking",
        ),
    ] = False,
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
        print("[bold yellow]Welcome to Comfy CLI![/bold yellow]: https://github.com/Comfy-Org/comfy-cli")
        print(ctx.get_help())
        ctx.exit()

    # TODO: Move this to proper place
    # start_time = time.time()
    # workspace_manager.scan_dir()
    # end_time = time.time()
    #
    # logging.info(f"scan_dir took {end_time - start_time:.2f} seconds to run")


@app.command(help="Download and install ComfyUI and ComfyUI-Manager")
@tracking.track_command()
def install(
    url: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="url or local path pointing to the ComfyUI core git repo to be installed. A specific branch can optionally be specified using a setuptools-like syntax, eg https://foo.git@bar",
        ),
    ] = constants.COMFY_GITHUB_URL,
    manager_url: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="url or local path pointing to the ComfyUI-Manager git repo to be installed. A specific branch can optionally be specified using a setuptools-like syntax, eg https://foo.git@bar",
        ),
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
        Optional[bool],
        typer.Option(
            show_default=False,
            help="Install for Nvidia gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cuda_version: Annotated[CUDAVersion, typer.Option(show_default=True)] = CUDAVersion.v12_1,
    amd: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            help="Install for AMD gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    m_series: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            help="Install for Mac M-Series gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    intel_arc: Annotated[
        Optional[bool],
        typer.Option(
            hidden=True,
            show_default=False,
            help="(Beta support) install for Intel Arc gpu, based on https://github.com/comfyanonymous/ComfyUI/pull/3439",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cpu: Annotated[
        Optional[bool],
        typer.Option(
            show_default=False,
            help="Install for CPU",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    commit: Annotated[Optional[str], typer.Option(help="Specify commit hash for ComfyUI")] = None,
    fast_deps: Annotated[
        Optional[bool],
        typer.Option(
            "--fast-deps",
            show_default=False,
            help="Use new fast dependency installer",
        ),
    ] = False,
):
    check_for_updates()
    checker = EnvChecker()

    comfy_path, _ = workspace_manager.get_workspace_path()

    is_comfy_installed_at_path, repo_dir = check_comfy_repo(comfy_path)
    if is_comfy_installed_at_path and not restore:
        print(f"[bold red]ComfyUI is already installed at the specified path:[/bold red] {comfy_path}\n")
        print(
            "[bold yellow]If you want to restore dependencies, add the '--restore' option.[/bold yellow]",
        )
        raise typer.Exit(code=1)

    if repo_dir is not None:
        comfy_path = str(repo_dir.working_dir)

    if checker.python_version.major < 3 or checker.python_version.minor < 9:
        print("[bold red]Python version 3.9 or higher is required to run ComfyUI.[/bold red]")
        print(f"You are currently using Python version {env_checker.format_python_version(checker.python_version)}.")
    platform = utils.get_os()
    if cpu:
        print("[bold yellow]Installing for CPU[/bold yellow]")
        install_inner.execute(
            url,
            manager_url,
            comfy_path,
            restore,
            skip_manager,
            commit=commit,
            gpu=None,
            cuda_version=cuda_version,
            plat=platform,
            skip_torch_or_directml=skip_torch_or_directml,
            skip_requirement=skip_requirement,
            fast_deps=fast_deps,
        )
        print(f"ComfyUI is installed at: {comfy_path}")
        return None

    if nvidia and platform == constants.OS.MACOS:
        print("[bold red]Nvidia GPU is never on MacOS. What are you smoking? 🤔[/bold red]")
        raise typer.Exit(code=1)

    if platform != constants.OS.MACOS and m_series:
        print(f"[bold red]You are on {platform} bruh [/bold red]")

    gpu = None

    if nvidia:
        gpu = GPU_OPTION.NVIDIA
    elif amd:
        gpu = GPU_OPTION.AMD
    elif m_series:
        gpu = GPU_OPTION.MAC_M_SERIES
    elif intel_arc:
        gpu = GPU_OPTION.INTEL_ARC
    else:
        if platform == constants.OS.MACOS:
            gpu = ui.prompt_select_enum(
                "What type of Mac do you have?",
                [GPU_OPTION.MAC_M_SERIES, GPU_OPTION.MAC_INTEL],
            )
        else:
            gpu = ui.prompt_select_enum(
                "What GPU do you have?",
                [GPU_OPTION.NVIDIA, GPU_OPTION.AMD, GPU_OPTION.INTEL_ARC],
            )

    if gpu == GPU_OPTION.INTEL_ARC:
        print("[bold yellow]Installing on Intel ARC is not yet completely supported[/bold yellow]")
        env_check = env_checker.EnvChecker()
        if env_check.conda_env is None:
            print("[bold red]Intel ARC support requires conda environment to be activated.[/bold red]")
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
        fast_deps=fast_deps,
    )

    print(f"ComfyUI is installed at: {comfy_path}")


@app.command(help="Update ComfyUI Environment [all|comfy]")
@tracking.track_command()
def update(
    target: str = typer.Argument(
        "comfy",
        help="[all|comfy]",
        autocompletion=utils.create_choice_completer(["all", "comfy"]),
    ),
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


@app.command(help="Run API workflow file using the ComfyUI launched by `comfy launch --background`")
@tracking.track_command()
def run(
    workflow: Annotated[str, typer.Option(help="Path to the workflow API json file.")],
    wait: Annotated[
        bool,
        typer.Option(help="If the command should wait until execution completes."),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(help="Enables verbose output of the execution process."),
    ] = False,
    host: Annotated[
        Optional[str],
        typer.Option(help="The IP/hostname where the ComfyUI instance is running, e.g. 127.0.0.1 or localhost."),
    ] = None,
    port: Annotated[
        Optional[int],
        typer.Option(help="The port where the ComfyUI instance is running, e.g. 8188."),
    ] = None,
    timeout: Annotated[
        Optional[int],
        typer.Option(help="The timeout in seconds for the workflow execution."),
    ] = 30,
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

    run_inner.execute(workflow, host, port, wait, verbose, local_paths, timeout)


def validate_comfyui(_env_checker):
    if _env_checker.comfy_repo is None:
        print("[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)


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
        print(f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})")

    ConfigManager().remove_background()


@app.command(help="Launch ComfyUI: ?[--background] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(
    background: Annotated[bool, typer.Option(help="Launch ComfyUI in background")] = False,
    extra: List[str] = typer.Argument(None),
):
    launch_command(background, extra)


@app.command("set-default", help="Set default ComfyUI path")
@tracking.track_command()
def set_default(
    workspace_path: str,
    launch_extras: Annotated[str, typer.Option(help="Specify extra options for launch")] = "",
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
    print("\n[bold red] No such command, did you mean 'comfy node' instead?[/bold red]\n")


@app.command(hidden=True)
@tracking.track_command()
def models():
    print("\n[bold red] No such command, did you mean 'comfy model' instead?[/bold red]\n")


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
    tracking.track_event("feedback_general_satisfaction", {"score": general_satisfaction_score})

    # Usability and User Experience
    usability_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5,  how satisfied are you with the usability and user experience of the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
        force_prompting=True,
    )
    tracking.track_event("feedback_usability_satisfaction", {"score": usability_satisfaction_score})

    # Additional Feature-Specific Feedback
    if questionary.confirm("Do you want to provide additional feature-specific feedback on our GitHub page?").ask():
        tracking.track_event("feedback_additional")
        webbrowser.open("https://github.com/Comfy-Org/comfy-cli/issues/new/choose")

    print("Thank you for your feedback!")


@app.command(help="Download a standalone Python interpreter and dependencies based on an existing comfyui workspace")
@tracking.track_command()
def standalone(
    cli_spec: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="setuptools-style requirement specificer pointing to an instance of comfy-cli",
        ),
    ] = "comfy-cli",
    platform: Annotated[
        Optional[constants.OS],
        typer.Option(
            show_default=False,
            help="Create standalone Python for specified platform",
        ),
    ] = None,
    proc: Annotated[
        Optional[constants.PROC],
        typer.Option(
            show_default=False,
            help="Create standalone Python for specified processor",
        ),
    ] = None,
    rehydrate: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Create standalone Python for CPU",
        ),
    ] = False,
):
    comfy_path, _ = workspace_manager.get_workspace_path()

    platform = utils.get_os() if platform is None else platform
    proc = utils.get_proc() if proc is None else proc

    if rehydrate:
        sty = StandalonePython.FromTarball(fpath="python.tgz")
        sty.rehydrate_comfy_deps()
    else:
        sty = StandalonePython.FromDistro(platform=platform, proc=proc)
        sty.dehydrate_comfy_deps(comfyDir=comfy_path, extraSpecs=cli_spec)
        sty.to_tarball()


app.add_typer(models_command.app, name="model", help="Manage models.")
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(custom_nodes.manager_app, name="manager", help="Manage ComfyUI-Manager.")
app.add_typer(tracking.app, name="tracking", help="Manage analytics tracking settings.")
