import os
import pathlib
import subprocess
import sys
import uuid
from typing import Optional

import typer
from rich import print
from typing_extensions import List, Annotated

from comfy_cli import ui, logging, tracking, utils
from comfy_cli.config_manager import ConfigManager
from comfy_cli.file_utils import (
    download_file,
    upload_file_to_signed_url,
    zip_files,
    extract_package_as_zip,
)
from comfy_cli.registry import (
    RegistryAPI,
    extract_node_configuration,
    initialize_project_config,
)
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()
manager_app = typer.Typer()
workspace_manager = WorkspaceManager()
registry_api = RegistryAPI()


def execute_cm_cli(args, channel=None, mode=None):
    _config_manager = ConfigManager()

    workspace_path = workspace_manager.workspace_path

    if not os.path.exists(workspace_path):
        print(f"\nComfyUI not found: {workspace_path}\n", file=sys.stderr)
        raise typer.Exit(code=1)

    cm_cli_path = os.path.join(
        workspace_path, "custom_nodes", "ComfyUI-Manager", "cm-cli.py"
    )
    if not os.path.exists(cm_cli_path):
        print(f"\nComfyUI-Manager not found: {cm_cli_path}\n", file=sys.stderr)
        raise typer.Exit(code=1)

    cmd = [sys.executable, cm_cli_path] + args
    if channel is not None:
        cmd += ["--channel", channel]

    if mode is not None:
        cmd += ["--mode", mode]

    new_env = os.environ.copy()
    session_path = os.path.join(
        _config_manager.get_config_path(), "tmp", str(uuid.uuid4())
    )
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["COMFYUI_PATH"] = workspace_path

    print(f"Execute from: {workspace_path}")

    try:
        subprocess.run(cmd, env=new_env, check=True)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print(f"Execution error: {cmd}", file=sys.stderr)
        elif e.returncode == 2:
            pass
        else:
            raise e

    workspace_manager.set_recent_workspace(workspace_path)


def validate_comfyui_manager(_env_checker):
    manager_path = _env_checker.get_comfyui_manager_path()

    if manager_path is None:
        print(
            "[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]"
        )
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


@app.command("save-snapshot", help="Save a snapshot of the current ComfyUI environment")
@tracking.track_command("node")
def save_snapshot(
    output: Annotated[
        str,
        typer.Option(
            show_default=False, help="Specify the output file path. (.json/.yaml)"
        ),
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


@app.command(
    "restore-dependencies", help="Restore dependencies from installed custom nodes"
)
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


channel_completer = utils.create_choice_completer(
    ["default", "recent", "dev", "forked", "tutorial", "legacy"]
)


def node_completer(incomplete: str) -> list[str]:
    try:
        config_manager = ConfigManager()
        tmp_path = os.path.join(
            config_manager.get_config_path(), "tmp", "node-cache.list"
        )

        with open(tmp_path, "r", encoding="UTF-8", errors="ignore") as cache_file:
            return [
                node_id
                for node_id in cache_file.readlines()
                if node_id.startswith(incomplete)
            ]

    except Exception:
        return []


def node_or_all_completer(incomplete: str) -> list[str]:
    try:
        config_manager = ConfigManager()
        tmp_path = os.path.join(
            config_manager.get_config_path(), "tmp", "node-cache.list"
        )

        all_opt = []
        if "all".startswith(incomplete):
            all_opt = ["all"]

        with open(tmp_path, "r", encoding="UTF-8", errors="ignore") as cache_file:
            return [
                node_id
                for node_id in cache_file.readlines()
                if node_id.startswith(incomplete)
            ] + all_opt

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
        str,
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

    execute_cm_cli(["show", arg], channel, mode)


@app.command("simple-show", help="Show node list (simple mode)")
@tracking.track_command("node")
def simple_show(
    arg: str = typer.Argument(
        help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]",
        autocompletion=show_completer,
    ),
    channel: Annotated[
        str,
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

    execute_cm_cli(["simple-show", arg], channel, mode)


# install, reinstall, uninstall
@app.command(help="Install custom nodes")
@tracking.track_command("node")
def install(
    nodes: List[str] = typer.Argument(
        ..., help="List of custom nodes to install", autocompletion=node_completer
    ),
    channel: Annotated[
        str,
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
        typer.echo(f"Invalid command: {mode}. `install all` is not allowed", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["install"] + nodes, channel, mode)


@app.command(help="Reinstall custom nodes")
@tracking.track_command("node")
def reinstall(
    nodes: List[str] = typer.Argument(
        ..., help="List of custom nodes to reinstall", autocompletion=node_completer
    ),
    channel: Annotated[
        str,
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
        typer.echo(f"Invalid command: {mode}. `reinstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    validate_mode(mode)

    execute_cm_cli(["reinstall"] + nodes, channel, mode)


@app.command(help="Uninstall custom nodes")
@tracking.track_command("node")
def uninstall(
    nodes: List[str] = typer.Argument(
        ..., help="List of custom nodes to uninstall", autocompletion=node_completer
    ),
    channel: Annotated[
        str,
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

    execute_cm_cli(["uninstall"] + nodes, channel, mode)


def update_node_id_cache():
    config_manager = ConfigManager()
    workspace_path = workspace_manager.workspace_path

    cm_cli_path = os.path.join(
        workspace_path, "custom_nodes", "ComfyUI-Manager", "cm-cli.py"
    )

    tmp_path = os.path.join(config_manager.get_config_path(), "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    cache_path = os.path.join(tmp_path, "node-cache.list")
    cmd = [sys.executable, cm_cli_path, "export-custom-node-ids", cache_path]

    new_env = os.environ.copy()
    new_env["COMFYUI_PATH"] = workspace_path
    res = subprocess.run(cmd, env=new_env, check=True)
    if res.returncode != 0:
        typer.echo(
            "Failed to update node id cache.",
            err=True,
        )
        raise typer.Exit(code=1)


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
        str,
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

    execute_cm_cli(["update"] + nodes, channel, mode)

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
        str,
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

    execute_cm_cli(["disable"] + nodes, channel, mode)


@app.command(help="Enable custom nodes")
@tracking.track_command("node")
def enable(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to enable]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        str,
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

    execute_cm_cli(["enable"] + nodes, channel, mode)


@app.command(help="Fix dependencies of custom nodes")
@tracking.track_command("node")
def fix(
    nodes: List[str] = typer.Argument(
        ...,
        help="[all|List of custom nodes to fix]",
        autocompletion=node_or_all_completer,
    ),
    channel: Annotated[
        str,
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

    execute_cm_cli(["fix"] + nodes, channel, mode)


@app.command(
    "install-deps",
    help="Install dependencies from dependencies file(.json) or workflow(.png/.json)",
)
@tracking.track_command("node")
def install_deps(
    deps: Annotated[
        str, typer.Option(show_default=False, help="Dependency spec file (.json)")
    ] = None,
    workflow: Annotated[
        str, typer.Option(show_default=False, help="Workflow file (.json/.png)")
    ] = None,
    channel: Annotated[
        str,
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
        print(
            f"[bold red]One of --deps or --workflow must be provided as an argument.[/bold red]\n"
        )

    tmp_path = None
    if workflow is not None:
        workflow = os.path.abspath(os.path.expanduser(workflow))
        tmp_path = os.path.join(
            workspace_manager.config_manager.get_config_path(), "tmp"
        )
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

    execute_cm_cli(["install-deps", deps_file], channel, mode)

    if tmp_path is not None and os.path.exists(tmp_path):
        os.remove(tmp_path)


@app.command(
    "deps-in-workflow", help="Generate dependencies file from workflow (.json/.png)"
)
@tracking.track_command("node")
def deps_in_workflow(
    workflow: Annotated[
        str, typer.Option(show_default=False, help="Workflow file (.json/.png)")
    ],
    output: Annotated[
        str, typer.Option(show_default=False, help="Output file (.json)")
    ],
    channel: Annotated[
        str,
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


@app.command("publish", help="Publish node to registry")
@tracking.track_command("publish")
def publish(
    token: Optional[str] = typer.Option(
        None, "--token", help="Personal Access Token for publishing", hide_input=True
    )
):
    """
    Publish a node with optional validation.
    """

    # Perform some validation logic here
    typer.echo("Validating node configuration...")
    config = extract_node_configuration()

    # Prompt for Personal Access Token
    if not token:
        token = typer.prompt("Please enter your Personal Access Token", hide_input=True)

    # Call API to fetch node version with the token in the body
    typer.echo("Publishing node version...")
    response = registry_api.publish_node_version(config, token)

    # Zip up all files in the current directory, respecting .gitignore files.
    signed_url = response.signedUrl
    zip_filename = "node.tar.gz"
    typer.echo("Creating zip file...")
    zip_files(zip_filename)

    # Upload the zip file to the signed URL
    typer.echo("Uploading zip file...")
    upload_file_to_signed_url(signed_url, zip_filename)


@app.command("init", help="Init scaffolding for custom node")
@tracking.track_command("node")
def scaffold():
    if os.path.exists("pyproject.toml"):
        typer.echo("Warning: 'pyproject.toml' already exists. Will not overwrite.")
        raise typer.Exit(code=1)

    typer.echo("Initializing metadata...")
    initialize_project_config()
    typer.echo(
        "pyproject.toml created successfully. Defaults were filled in. Please check before publishing."
    )


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


@app.command("registry-install", help="Install a node from the registry", hidden=True)
@tracking.track_command("node")
def registry_install(node_id: str, version: Optional[str] = None):
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
        logging.error(
            f"Encountered an error while installing the node. error: {str(e)}"
        )
        ui.display_error_message(f"Failed to download the custom node {node_id}.")
        return

    # Download the node archive
    custom_nodes_path = pathlib.Path(workspace_manager.workspace_path) / "custom_nodes"
    node_specific_path = custom_nodes_path / node_id  # Subdirectory for the node
    node_specific_path.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it doesn't exist

    local_filename = node_specific_path / f"{node_id}-{node_version.version}.zip"
    logging.debug(
        f"Start downloading the node {node_id} version {node_version.version} to {local_filename}"
    )
    download_file(node_version.download_url, local_filename)

    # Extract the downloaded archive to the custom_node directory on the workspace.
    logging.debug(
        f"Start extracting the node {node_id} version {node_version.version} to {custom_nodes_path}"
    )
    extract_package_as_zip(local_filename, node_specific_path)

    # Delete the downloaded archive
    logging.debug(f"Deleting the downloaded archive {local_filename}")
    os.remove(local_filename)

    logging.info(
        f"Node {node_id} version {node_version.version} has been successfully installed."
    )
