import os
import subprocess
import sys
import webbrowser
from typing import Annotated

import questionary
import typer
from rich import print as rprint
from rich.console import Console

from comfy_cli import constants, env_checker, logging, tracking, ui, utils
from comfy_cli.command import custom_nodes, pr_command
from comfy_cli.command import install as install_inner
from comfy_cli.command import run as run_inner
from comfy_cli.command.install import validate_version
from comfy_cli.command.launch import launch as launch_command
from comfy_cli.command.models import models as models_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import GPU_OPTION, CUDAVersion
from comfy_cli.env_checker import EnvChecker
from comfy_cli.standalone import StandalonePython
from comfy_cli.update import check_for_updates
from comfy_cli.uv import DependencyCompiler
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
    rprint(ctx.find_root().get_help())
    ctx.exit(0)


@app.callback(invoke_without_command=True)
def entry(
    ctx: typer.Context,
    workspace: Annotated[
        str | None,
        typer.Option(
            show_default=False,
            help="Path to ComfyUI workspace",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    recent: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Execute from recent path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    here: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Execute from current path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    skip_prompt: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Do not prompt user for input, use default options",
        ),
    ] = False,
    enable_telemetry: Annotated[
        bool,
        typer.Option(
            show_default=False,
            hidden=True,
            help="Enable tracking",
        ),
    ] = False,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print version and exit",
    ),
):
    if version:
        rprint(ConfigManager().get_cli_version())
        ctx.exit(0)

    workspace_manager.setup_workspace_manager(workspace, here, recent, skip_prompt)

    tracking.prompt_tracking_consent(skip_prompt, default_value=enable_telemetry)

    if ctx.invoked_subcommand is None:
        rprint("[bold yellow]Welcome to Comfy CLI![/bold yellow]: https://github.com/Comfy-Org/comfy-cli")
        rprint(ctx.get_help())
        ctx.exit()

    # TODO: Move this to proper place
    # start_time = time.time()
    # workspace_manager.scan_dir()
    # end_time = time.time()
    #
    # logging.info(f"scan_dir took {end_time - start_time:.2f} seconds to run")


def validate_commit_and_version(commit: str | None, ctx: typer.Context) -> str | None:
    """
    Validate that the commit is not specified unless the version is 'nightly'.
    """
    version = ctx.params.get("version")
    if commit and version != "nightly":
        raise typer.BadParameter("You can only specify the commit if the version is 'nightly'.")
    return commit


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
    version: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="Specify version of ComfyUI to install. Default is nightl, which is the latest commit on master branch. Other options include: latest, which is the latest stable release. Or a specific version number, eg. 0.2.0",
            callback=validate_version,
        ),
    ] = "nightly",
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
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for Nvidia gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cuda_version: Annotated[CUDAVersion, typer.Option(show_default=True)] = CUDAVersion.v12_6,
    amd: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for AMD gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    m_series: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for Mac M-Series gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    intel_arc: Annotated[
        bool | None,
        typer.Option(
            hidden=True,
            show_default=False,
            help="Install for Intel Arc gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cpu: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for CPU",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    commit: Annotated[
        str | None, typer.Option(help="Specify commit hash for ComfyUI", callback=validate_commit_and_version)
    ] = None,
    fast_deps: Annotated[
        bool,
        typer.Option(
            "--fast-deps",
            show_default=False,
            help="Use new fast dependency installer",
        ),
    ] = False,
    manager_commit: Annotated[
        str | None,
        typer.Option(help="Specify commit hash for ComfyUI-Manager"),
    ] = None,
    pr: Annotated[
        str | None,
        typer.Option(
            show_default=False,
            help="Install from a specific PR. Supports formats: username:branch, #123, or PR URL",
        ),
    ] = None,
):
    check_for_updates()
    checker = EnvChecker()

    comfy_path, _ = workspace_manager.get_workspace_path()

    is_comfy_installed_at_path, repo_dir = check_comfy_repo(comfy_path)
    if is_comfy_installed_at_path and not restore:
        rprint(f"[bold red]ComfyUI is already installed at the specified path:[/bold red] {comfy_path}\n")
        rprint(
            "[bold yellow]If you want to restore dependencies, add the '--restore' option.[/bold yellow]",
        )
        raise typer.Exit(code=1)

    if repo_dir is not None:
        comfy_path = str(repo_dir.working_dir)

    if checker.python_version.major < 3 or checker.python_version.minor < 9:
        rprint("[bold red]Python version 3.9 or higher is required to run ComfyUI.[/bold red]")
        rprint(f"You are currently using Python version {env_checker.format_python_version(checker.python_version)}.")
    platform = utils.get_os()
    if cpu:
        rprint("[bold yellow]Installing for CPU[/bold yellow]")
        install_inner.execute(
            url,
            manager_url,
            comfy_path,
            restore,
            skip_manager,
            commit=commit,
            version=version,
            gpu=None,
            cuda_version=cuda_version,
            plat=platform,
            skip_torch_or_directml=skip_torch_or_directml,
            skip_requirement=skip_requirement,
            fast_deps=fast_deps,
            manager_commit=manager_commit,
        )
        rprint(f"ComfyUI is installed at: {comfy_path}")
        return None

    if nvidia and platform == constants.OS.MACOS:
        rprint("[bold red]Nvidia GPU is never on MacOS. What are you smoking? 🤔[/bold red]")
        raise typer.Exit(code=1)

    if platform != constants.OS.MACOS and m_series:
        rprint(f"[bold red]You are on {platform} bruh [/bold red]")

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

    if gpu is None and not cpu:
        rprint(
            "[bold red]No GPU option selected or `--cpu` enabled, use --\\[gpu option] flag (e.g. --nvidia) to pick GPU. use `--cpu` to install for CPU. Exiting...[/bold red]"
        )
        raise typer.Exit(code=1)

    if pr and version not in {None, "nightly"} or commit:
        rprint("--pr cannot be used with --version or --commit")
        raise typer.Exit(code=1)

    install_inner.execute(
        url,
        manager_url,
        comfy_path,
        restore,
        skip_manager,
        commit=commit,
        gpu=gpu,
        version=version,
        cuda_version=cuda_version,
        plat=platform,
        skip_torch_or_directml=skip_torch_or_directml,
        skip_requirement=skip_requirement,
        fast_deps=fast_deps,
        manager_commit=manager_commit,
        pr=pr,
    )

    rprint(f"ComfyUI is installed at: {comfy_path}")


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
        rprint(f"Updating ComfyUI in {comfy_path}...")
        if comfy_path is None:
            rprint("ComfyUI path is not found.")
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
        str | None,
        typer.Option(help="The IP/hostname where the ComfyUI instance is running, e.g. 127.0.0.1 or localhost."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(help="The port where the ComfyUI instance is running, e.g. 8188."),
    ] = None,
    timeout: Annotated[
        int | None,
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
        rprint("[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)


@app.command(help="Stop background ComfyUI")
@tracking.track_command()
def stop():
    if constants.CONFIG_KEY_BACKGROUND not in ConfigManager().config["DEFAULT"]:
        rprint("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)

    bg_info = ConfigManager().background
    if not bg_info:
        rprint("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)
    is_killed = utils.kill_all(bg_info[2])

    if not is_killed:
        rprint("[bold red]Failed to stop ComfyUI in the background.[/bold red]\n")
    else:
        rprint(f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})")

    ConfigManager().remove_background()


@app.command(help="Launch ComfyUI: ?[--background] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(
    extra: list[str] = typer.Argument(None),
    background: Annotated[bool, typer.Option(help="Launch ComfyUI in background")] = False,
    frontend_pr: Annotated[
        str | None,
        typer.Option(
            "--frontend-pr",
            show_default=False,
            help="Use a specific frontend PR. Supports formats: username:branch, #123, or PR URL",
        ),
    ] = None,
):
    launch_command(background, extra, frontend_pr)


@app.command("set-default", help="Set default ComfyUI path")
@tracking.track_command()
def set_default(
    workspace_path: str,
    launch_extras: Annotated[str, typer.Option(help="Specify extra options for launch")] = "",
):
    comfy_path = os.path.abspath(os.path.expanduser(workspace_path))

    if not os.path.exists(comfy_path):
        rprint(
            f"\nPath not found: {comfy_path}.\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    is_comfy_repo, comfy_repo = check_comfy_repo(comfy_path)
    if not is_comfy_repo:
        rprint(
            f"\nSpecified path is not a ComfyUI path: {comfy_path}.\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    comfy_path = comfy_repo.working_dir

    rprint(f"Specified path is set as default ComfyUI path: {comfy_path} ")
    workspace_manager.set_default_workspace(comfy_path)
    workspace_manager.set_default_launch_extras(launch_extras)


@app.command(help="Show which ComfyUI is selected.")
@tracking.track_command()
def which():
    comfy_path = workspace_manager.workspace_path
    if comfy_path is None:
        rprint(
            "ComfyUI not found, please run 'comfy install', run 'comfy' in a ComfyUI directory, or specify the workspace path with '--workspace'."
        )
        raise typer.Exit(code=1)

    rprint(f"Target ComfyUI path: {comfy_path}")


@app.command(help="Print out current environment variables.")
@tracking.track_command()
def env():
    check_for_updates()
    env_data = EnvChecker().fill_print_table()
    workspace_data = workspace_manager.fill_print_table()
    all_data = env_data + workspace_data
    ui.display_table(
        data=all_data,
        column_names=[":laptop_computer: Environment", "Value"],
        title="Environment Information",
    )


@app.command(hidden=True)
@tracking.track_command()
def nodes():
    rprint("\n[bold red] No such command, did you mean 'comfy node' instead?[/bold red]\n")


@app.command(hidden=True)
@tracking.track_command()
def models():
    rprint("\n[bold red] No such command, did you mean 'comfy model' instead?[/bold red]\n")


@app.command(help="Provide feedback on the Comfy CLI tool.")
@tracking.track_command()
def feedback():
    rprint("Feedback Collection for Comfy CLI Tool\n")

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

    rprint("Thank you for your feedback!")


@app.command(hidden=True)
@app.command(
    help="Given an existing installation of comfy core and any custom nodes, installs any needed python dependencies"
)
@tracking.track_command()
def dependency():
    comfy_path, _ = workspace_manager.get_workspace_path()

    depComp = DependencyCompiler(cwd=comfy_path)
    depComp.compile_deps()
    depComp.install_deps()


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
    pack_wheels: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Pack requirement wheels in archive when creating standalone bundle",
        ),
    ] = False,
    platform: Annotated[
        constants.OS | None,
        typer.Option(
            show_default=False,
            help="Create standalone Python for specified platform",
        ),
    ] = None,
    proc: Annotated[
        constants.PROC | None,
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
        sty.rehydrate_comfy_deps(packWheels=pack_wheels)
    else:
        sty = StandalonePython.FromDistro(platform=platform, proc=proc)
        sty.dehydrate_comfy_deps(comfyDir=comfy_path, extraSpecs=[], packWheels=pack_wheels)
        sty.to_tarball()


app.add_typer(models_command.app, name="model", help="Manage models.")
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(custom_nodes.manager_app, name="manager", help="Manage ComfyUI-Manager.")

app.add_typer(pr_command.app, name="pr-cache", help="Manage PR cache.")

app.add_typer(tracking.app, name="tracking", help="Manage analytics tracking settings.")
