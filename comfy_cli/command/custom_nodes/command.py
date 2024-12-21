import os
import pathlib
import platform
import subprocess
import sys
import uuid
from typing import Optional

import typer
from rich import print
from typing_extensions import Annotated, List

from comfy_cli import logging, tracking, ui, utils
from comfy_cli.command.custom_nodes.bisect_custom_nodes import bisect_app
from comfy_cli.command.custom_nodes.cm_cli_util import execute_cm_cli
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import NODE_ZIP_FILENAME
from comfy_cli.file_utils import (
    download_file,
    extract_package_as_zip,
    upload_file_to_signed_url,
    zip_files,
)
from comfy_cli.registry import (
    RegistryAPI,
    extract_node_configuration,
    initialize_project_config,
)
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()
app.add_typer(bisect_app, name="bisect", help="Bisect custom nodes for culprit node.")
manager_app = typer.Typer()
workspace_manager = WorkspaceManager()
registry_api = RegistryAPI()


def validate_comfyui_manager(_env_checker):
    manager_path = _env_checker.get_comfyui_manager_path()

    if manager_path is None:
        print("[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)
    elif not os.path.exists(manager_path):
        print(
            f"[bold red]If ComfyUI-Manager is not installed, this feature cannot be used.[/bold red] \\[{manager_path}]"
        )
        raise typer.Exit(code=1)
    elif not os.path.exists(os.path.join(manager_path, ".git")):
        print(
            f"[bold red]The ComfyUI-Manager installation is invalid. This feature cannot be used.[/bold red] \\[{manager_path}]"
        )
        raise typer.Exit(code=1)


def run_script(cmd, cwd="."):
    if len(cmd) > 0 and cmd[0].startswith("#"):
        print(f"[ComfyUI-Manager] Unexpected behavior: `{cmd}`")
        return 0

    subprocess.check_call(cmd, cwd=cwd)

    return 0


pip_map = None


def get_installed_packages():
    global pip_map

    if pip_map is None:
        try:
            result = subprocess.check_output([sys.executable, "-m", "pip", "list"], universal_newlines=True)

            pip_map = {}
            for line in result.split("\n"):
                x = line.strip()
                if x:
                    y = line.split()
                    if y[0] == "Package" or y[0].startswith("-"):
                        continue

                    pip_map[y[0]] = y[1]
        except subprocess.CalledProcessError:
            print("[ComfyUI-Manager] Failed to retrieve the information of installed pip packages.")
            return set()

    return pip_map


def try_install_script(repo_path, install_cmd, instant_execution=False):
    startup_script_path = os.path.join(workspace_manager.workspace_path, "startup-scripts")
    if not instant_execution and (
        (len(install_cmd) > 0 and install_cmd[0].startswith("#"))
        or (
            platform.system() == "Windows"
            # From Yoland: disable commit compare
            # and comfy_ui_commit_datetime.date()
            # >= comfy_ui_required_commit_datetime.date()
        )
    ):
        if not os.path.exists(startup_script_path):
            os.makedirs(startup_script_path)

        script_path = os.path.join(startup_script_path, "install-scripts.txt")
        with open(script_path, "a", encoding="utf-8") as file:
            obj = [repo_path] + install_cmd
            file.write(f"{obj}\n")

        return True
    else:
        # From Yoland: Disable blacklisting
        # if len(install_cmd) == 5 and install_cmd[2:4] == ['pip', 'install']:
        #     if is_blacklisted(install_cmd[4]):
        #         print(f"[ComfyUI-Manager] skip black listed pip installation: '{install_cmd[4]}'")
        #         return True

        print(f"\n## ComfyUI-Manager: EXECUTE => {install_cmd}")
        code = run_script(install_cmd, cwd=repo_path)

        # From Yoland: Disable warning
        # if platform.system() != "Windows":
        #     try:
        #         if comfy_ui_commit_datetime.date() < comfy_ui_required_commit_datetime.date():
        #             print("\n\n###################################################################")
        #             print(f"[WARN] ComfyUI-Manager: Your ComfyUI version ({comfy_ui_revision})[{comfy_ui_commit_datetime.date()}] is too old. Please update to the latest version.")
        #             print(f"[WARN] The extension installation feature may not work properly in the current installed ComfyUI version on Windows environment.")
        #             print("###################################################################\n\n")
        #     except:
        #         pass

        if code != 0:
            print("install script failed")
            return False


def execute_install_script(repo_path):
    install_script_path = os.path.join(repo_path, "install.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")

    # From Yoland: disable lazy mode
    # if lazy_mode:
    #     install_cmd = ["#LAZY-INSTALL-SCRIPT",  sys.executable]
    #     try_install_script(repo_path, install_cmd)
    # else:

    if os.path.exists(requirements_path):
        # import pdb
        # pdb.set_trace()
        print("Install: pip packages")
        with open(requirements_path, "r", encoding="utf-8") as requirements_file:
            for line in requirements_file:
                # From Yoland: disable pip override
                # package_name = remap_pip_package(line.strip())
                package_name = line.strip()
                if package_name and not package_name.startswith("#"):
                    install_cmd = [sys.executable, "-m", "pip", "install", package_name]
                    if package_name.strip() != "":
                        try_install_script(repo_path, install_cmd)

    if os.path.exists(install_script_path):
        print("Install: install script")
        install_cmd = [sys.executable, "install.py"]
        try_install_script(repo_path, install_cmd)


@app.command("save-snapshot", help="Save a snapshot of the current ComfyUI environment")
@tracking.track_command("node")
def save_snapshot(
    output: Annotated[
        Optional[str],
        typer.Option(show_default=False, help="Specify the output file path. (.json/.yaml)"),
    ] = None,
):
    if output is None:
        execute_cm_cli(["save-snapshot"])
    else:
        output = os.path.abspath(output)  # to compensate chdir
        execute_cm_cli(["save-snapshot", "--output", output])


@app.command("restore-snapshot", help="Restore snapshot from snapshot file")
@tracking.track_command("node")
def restore_snapshot(
    path: str,
    pip_non_url: Optional[bool] = typer.Option(
        default=None,
        show_default=False,
        is_flag=True,
        help="Restore for pip packages registered on PyPI.",
    ),
    pip_non_local_url: Optional[bool] = typer.Option(
        default=None,
        show_default=False,
        is_flag=True,
        help="Restore for pip packages registered at web URLs.",
    ),
    pip_local_url: Optional[bool] = typer.Option(
        default=None,
        show_default=False,
        is_flag=True,
        help="Restore for pip packages specified by local paths.",
    ),
):
    extras = []

    if pip_non_url:
        extras += ["--pip-non-url"]

    if pip_non_local_url:
        extras += ["--pip-non-local-url"]

    if pip_local_url:
        extras += ["--pip-local-url"]

    path = os.path.abspath(path)
    execute_cm_cli(["restore-snapshot", path] + extras)


@app.command("restore-dependencies", help="Restore dependencies from installed custom nodes")
@tracking.track_command("node")
def restore_dependencies():
    execute_cm_cli(["restore-dependencies"])


@manager_app.command("disable-gui", help="Disable GUI mode of ComfyUI-Manager")
@tracking.track_command("node")
def disable_gui():
    execute_cm_cli(["cli-only-mode", "enable"])


@manager_app.command("enable-gui", help="Enable GUI mode of ComfyUI-Manager")
@tracking.track_command("node")
def enable_gui():
    execute_cm_cli(["cli-only-mode", "disable"])


@manager_app.command(help="Clear reserved startup action in ComfyUI-Manager")
@tracking.track_command("node")
def clear():
    execute_cm_cli(["clear"])


# completers
show_completer = utils.create_choice_completer(
    [
        "installed",
        "enabled",
        "not-installed",
        "disabled",
        "all",
        "snapshot",
        "snapshot-list",
    ]
)


mode_completer = utils.create_choice_completer(["remote", "local", "cache"])


channel_completer = utils.create_choice_completer(["default", "recent", "dev", "forked", "tutorial", "legacy"])


def node_completer(incomplete: str) -> list[str]:
    try:
        config_manager = ConfigManager()
        tmp_path = os.path.join(config_manager.get_config_path(), "tmp", "node-cache.list")

        with open(tmp_path, "r", encoding="UTF-8", errors="ignore") as cache_file:
            return [node_id for node_id in cache_file.readlines() if node_id.startswith(incomplete)]

    except Exception:
        return []


def node_or_all_completer(incomplete: str) -> list[str]:
    try:
        config_manager = ConfigManager()
        tmp_path = os.path.join(config_manager.get_config_path(), "tmp", "node-cache.list")

        all_opt = []
        if "all".startswith(incomplete):
            all_opt = ["all"]

        with open(tmp_path, "r", encoding="UTF-8", errors="ignore") as cache_file:
            return [node_id for node_id in cache_file.readlines() if node_id.startswith(incomplete)] + all_opt

    except Exception:
        return []


def validate_mode(mode):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command(help="Show node list")
@tracking.track_command("node")
def show(
    arg: str = typer.Argument(
        help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]",
        autocompletion=show_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    valid_commands = [
        "installed",
        "enabled",
        "not-installed",
        "disabled",
        "all",
        "snapshot",
        "snapshot-list",
    ]
    if arg not in valid_commands:
        typer.echo(f"Invalid command: `show {arg}`", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["show", arg], channel=channel, mode=mode)


@app.command("simple-show", help="Show node list (simple mode)")
@tracking.track_command("node")
def simple_show(
    arg: str = typer.Argument(
        help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]",
        autocompletion=show_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    valid_commands = [
        "installed",
        "enabled",
        "not-installed",
        "disabled",
        "all",
        "snapshot",
        "snapshot-list",
    ]
    if arg not in valid_commands:
        typer.echo(f"Invalid command: `show {arg}`", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["simple-show", arg], channel=channel, mode=mode)


# install, reinstall, uninstall
@app.command(help="Install custom nodes")
@tracking.track_command("node")
def install(
    nodes: List[str] = typer.Argument(..., help="List of custom nodes to install", autocompletion=node_completer),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    fast_deps: Annotated[
        Optional[bool],
        typer.Option(
            "--fast-deps",
            show_default=False,
            help="Use new fast dependency installer",
        ),
    ] = False,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    if "all" in nodes:
        typer.echo(f"Invalid command: {mode}. `install all` is not allowed", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["install"] + nodes, channel=channel, fast_deps=fast_deps, mode=mode)


@app.command(help="Reinstall custom nodes")
@tracking.track_command("node")
def reinstall(
    nodes: List[str] = typer.Argument(..., help="List of custom nodes to reinstall", autocompletion=node_completer),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    fast_deps: Annotated[
        Optional[bool],
        typer.Option(
            "--fast-deps",
            show_default=False,
            help="Use new fast dependency installer",
        ),
    ] = False,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    if "all" in nodes:
        typer.echo(f"Invalid command: {mode}. `reinstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["reinstall"] + nodes, channel=channel, fast_deps=fast_deps, mode=mode)


@app.command(help="Uninstall custom nodes")
@tracking.track_command("node")
def uninstall(
    nodes: List[str] = typer.Argument(..., help="List of custom nodes to uninstall", autocompletion=node_completer),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    if "all" in nodes:
        typer.echo(f"Invalid command: {mode}. `uninstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["uninstall"] + nodes, channel=channel, mode=mode)


def update_node_id_cache():
    config_manager = ConfigManager()
    workspace_path = workspace_manager.workspace_path

    cm_cli_path = os.path.join(workspace_path, "custom_nodes", "ComfyUI-Manager", "cm-cli.py")

    tmp_path = os.path.join(config_manager.get_config_path(), "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    cache_path = os.path.join(tmp_path, "node-cache.list")
    cmd = [sys.executable, cm_cli_path, "export-custom-node-ids", cache_path]

    new_env = os.environ.copy()
    new_env["COMFYUI_PATH"] = workspace_path
    subprocess.run(cmd, env=new_env, check=True)


# `update, disable, enable, fix` allows `all` param
@app.command(help="Update custom nodes or ComfyUI")
@tracking.track_command("node")
def update(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to update]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    execute_cm_cli(["update"] + nodes, channel=channel, mode=mode)

    update_node_id_cache()


@app.command(help="Disable custom nodes")
@tracking.track_command("node")
def disable(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to disable]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    execute_cm_cli(["disable"] + nodes, channel=channel, mode=mode)


@app.command(help="Enable custom nodes")
@tracking.track_command("node")
def enable(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to enable]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    execute_cm_cli(["enable"] + nodes, channel=channel, mode=mode)


@app.command(help="Fix dependencies of custom nodes")
@tracking.track_command("node")
def fix(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to fix]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    execute_cm_cli(["fix"] + nodes, channel=channel, mode=mode)


@app.command(
    "install-deps",
    help="Install dependencies from dependencies file(.json) or workflow(.png/.json)",
)
@tracking.track_command("node")
def install_deps(
    deps: Annotated[
        Optional[str],
        typer.Option(show_default=False, help="Dependency spec file (.json)"),
    ] = None,
    workflow: Annotated[
        Optional[str],
        typer.Option(show_default=False, help="Workflow file (.json/.png)"),
    ] = None,
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    if deps is None and workflow is None:
        print("[bold red]One of --deps or --workflow must be provided as an argument.[/bold red]\n")

    tmp_path = None
    if workflow is not None:
        workflow = os.path.abspath(os.path.expanduser(workflow))
        tmp_path = os.path.join(workspace_manager.config_manager.get_config_path(), "tmp")
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        tmp_path = os.path.join(tmp_path, str(uuid.uuid4())) + ".json"

        execute_cm_cli(
            ["deps-in-workflow", "--workflow", workflow, "--output", tmp_path],
            channel,
            mode,
        )

        deps_file = tmp_path
    else:
        deps_file = os.path.abspath(os.path.expanduser(deps))

    execute_cm_cli(["install-deps", deps_file], channel=channel, mode=mode)

    if tmp_path is not None and os.path.exists(tmp_path):
        os.remove(tmp_path)


@app.command("deps-in-workflow", help="Generate dependencies file from workflow (.json/.png)")
@tracking.track_command("node")
def deps_in_workflow(
    workflow: Annotated[str, typer.Option(show_default=False, help="Workflow file (.json/.png)")],
    output: Annotated[str, typer.Option(show_default=False, help="Output file (.json)")],
    channel: Annotated[
        Optional[str],
        typer.Option(
            show_default=False,
            help="Specify the operation mode",
            autocompletion=channel_completer,
        ),
    ] = None,
    mode: str = typer.Option(
        None,
        help="[remote|local|cache]",
        autocompletion=mode_completer,
    ),
):
    validate_mode(mode)

    workflow = os.path.abspath(os.path.expanduser(workflow))
    output = os.path.abspath(os.path.expanduser(output))

    execute_cm_cli(
        ["deps-in-workflow", "--workflow", workflow, "--output", output],
        channel,
        mode,
    )


def validate_node_for_publishing():
    """
    Validates node configuration and runs security checks.
    Returns the validated config if successful, raises typer.Exit if validation fails.
    """
    # Perform some validation logic here
    typer.echo("Validating node configuration...")
    config = extract_node_configuration()

    # Run security checks first
    typer.echo("Running security checks...")
    try:
        # Run ruff check with security rules and --exit-zero to only warn
        cmd = ["ruff", "check", ".", "-q", "--select", "S102,S307", "--exit-zero"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print("[yellow]Security warnings found:[/yellow]")
            print(result.stdout)
            print("[bold yellow]We will soon disable exec and eval, so this will be an error soon.[/bold yellow]")

    except FileNotFoundError:
        print("[red]Ruff is not installed. Please install it with 'pip install ruff'[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]Error running security check: {e}[/red]")
        raise typer.Exit(code=1)

    return config


@app.command("validate", help="Run validation checks for publishing")
@tracking.track_command("publish")
def validate():
    """
    Run validation checks that would be performed during publishing.
    """
    validate_node_for_publishing()
    print("[green]✓ All validation checks passed successfully[/green]")


@app.command("publish", help="Publish node to registry")
@tracking.track_command("publish")
def publish(
    token: Optional[str] = typer.Option(None, "--token", help="Personal Access Token for publishing", hide_input=True),
):
    """
    Publish a node with optional validation.
    """
    config = validate_node_for_publishing()

    # Prompt for API Key
    if not token:
        token = typer.prompt(
            "Please enter your API Key (can be created on https://registry.comfy.org)",
            hide_input=True,
        )

    # Call API to fetch node version with the token in the body
    typer.echo("Publishing node version...")
    try:
        response = registry_api.publish_node_version(config, token)
        # Zip up all files in the current directory, respecting .gitignore files.
        signed_url = response.signedUrl
        zip_filename = NODE_ZIP_FILENAME
        typer.echo("Creating zip file...")
        zip_files(zip_filename)

        # Upload the zip file to the signed URL
        typer.echo("Uploading zip file...")
        upload_file_to_signed_url(signed_url, zip_filename)
    except Exception as e:
        ui.display_error_message({str(e)})
        raise typer.Exit(code=1)


@app.command("init", help="Init scaffolding for custom node")
@tracking.track_command("node")
def scaffold():
    if os.path.exists("pyproject.toml"):
        typer.echo("Warning: 'pyproject.toml' already exists. Will not overwrite.")
        raise typer.Exit(code=1)

    typer.echo("Initializing metadata...")
    initialize_project_config()
    typer.echo("pyproject.toml created successfully. Defaults were filled in. Please check before publishing.")


@app.command("registry-list", help="List all nodes in the registry", hidden=True)
@tracking.track_command("node")
def display_all_nodes():
    """
    Display all nodes in the registry.
    """

    nodes = None
    try:
        nodes = registry_api.list_all_nodes()
    except Exception as e:
        logging.error(f"Failed to fetch nodes from the registry: {str(e)}")
        ui.display_error_message("Failed to fetch nodes from the registry.")

    # Map Node data class instances to tuples for display
    node_data = [
        (
            node.id,
            node.name,
            node.description,
            node.author or "N/A",
            node.license or "N/A",
            ", ".join(node.tags),
            node.latest_version.version if node.latest_version else "N/A",
        )
        for node in nodes
    ]
    ui.display_table(
        node_data,
        [
            "ID",
            "Name",
            "Description",
            "Author",
            "License",
            "Tags",
            "Latest Version",
        ],
        title="List of All Nodes",
    )


@app.command(
    "registry-install",
    help="Install a node from the registry",
    hidden=True,
)
@tracking.track_command("node")
def registry_install(
    node_id: str,
    version: Optional[str] = None,
    force_download: Annotated[
        bool,
        typer.Option(
            "--force-download",
            help="Force download the node even if it is already installed",
        ),
    ] = False,
):
    """
    Install a node from the registry.
    Args:
      node_id: The ID of the node to install.
      version: The version of the node to install. If not provided, the latest version will be installed.
    """

    # If the node ID is not provided, prompt the user to enter it
    if not node_id:
        node_id = typer.prompt("Enter the ID of the node you want to install")

    node_version = None
    try:
        # Call the API to install the node
        node_version = registry_api.install_node(node_id, version)
        if not node_version.download_url:
            logging.error("Download URL not provided from the registry.")
            ui.display_error_message(f"Failed to download the custom node {node_id}.")
            return

    except Exception as e:
        logging.error(f"Encountered an error while installing the node. error: {str(e)}")
        ui.display_error_message(f"Failed to download the custom node {node_id}.")
        return

    # Download the node archive
    custom_nodes_path = pathlib.Path(workspace_manager.workspace_path) / "custom_nodes"
    node_specific_path = custom_nodes_path / node_id  # Subdirectory for the node
    if node_specific_path.exists():
        print(
            f"[bold red] The node {node_id} already exists in the workspace. This migit delete any model files in the node.[/bold red]"
        )

        confirm = ui.prompt_confirm_action(
            "Do you want to overwrite it?",
            force_download,
        )
        if not confirm:
            return
    node_specific_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    local_filename = node_specific_path / f"{node_id}-{node_version.version}.zip"
    logging.debug(f"Start downloading the node {node_id} version {node_version.version} to {local_filename}")
    download_file(node_version.download_url, local_filename)

    # Extract the downloaded archive to the custom_node directory on the workspace.
    logging.debug(f"Start extracting the node {node_id} version {node_version.version} to {custom_nodes_path}")
    extract_package_as_zip(local_filename, node_specific_path)

    # TODO: temoporary solution to run requirement.txt and install script
    execute_install_script(node_specific_path)

    # Delete the downloaded archive
    logging.debug(f"Deleting the downloaded archive {local_filename}")
    os.remove(local_filename)

    logging.info(f"Node {node_id} version {node_version.version} has been successfully installed.")


@app.command(
    "pack",
    help="Pack the current node into a zip file. Ignorining .gitignore files.",
)
@tracking.track_command("pack")
def pack():
    typer.echo("Validating node configuration...")
    config = extract_node_configuration()
    if not config:
        raise typer.Exit(code=1)

    zip_filename = NODE_ZIP_FILENAME
    zip_files(zip_filename)
    typer.echo(f"Created zip file: {NODE_ZIP_FILENAME}")
    logging.info("Node has been packed successfully.")
