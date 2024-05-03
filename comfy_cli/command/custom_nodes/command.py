import typer
from typing_extensions import List, Annotated

from comfy_cli import tracking
from comfy_cli.env_checker import EnvChecker
import os
import subprocess
import sys
from rich import print
import uuid
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager

# from comfy_cli.registry import (
#     publish_node_version,
#     extract_node_configuration,
#     upload_file_to_signed_url,
#     zip_files,
#     initialize_project_config,
# )

app = typer.Typer()
manager_app = typer.Typer()
workspace_manager = WorkspaceManager()


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
        cmd += ["--mode", channel]

    new_env = os.environ.copy()
    session_path = os.path.join(
        _config_manager.get_config_path(), "tmp", str(uuid.uuid4())
    )
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["COMFYUI_PATH"] = workspace_path

    print(f"Execute from: {workspace_path}")

    subprocess.run(cmd, env=new_env)
    workspace_manager.set_recent_workspace(workspace_path)


def validate_comfyui_manager(_env_checker):
    manager_path = _env_checker.get_comfyui_manager_path()

    if manager_path is None:
        print(
            f"[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]"
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
        "--output",
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


@app.command("restore-snapshot")
@tracking.track_command("node")
def restore_snapshot(path: str):
    path = os.path.abspath(path)
    execute_cm_cli(["restore-snapshot", path])


@app.command("restore-dependencies")
@tracking.track_command("node")
def restore_dependencies():
    execute_cm_cli(["restore-dependencies"])


@manager_app.command("disable-gui")
@tracking.track_command("node")
def disable_gui():
    execute_cm_cli(["cli-only-mode", "enable"])


@manager_app.command("enable-gui")
@tracking.track_command("node")
def enable_gui():
    execute_cm_cli(["cli-only-mode", "disable"])


@manager_app.command()
@tracking.track_command("node")
def clear(path: str):
    path = os.path.abspath(path)
    execute_cm_cli(["clear", path])


@app.command()
@tracking.track_command("node")
def show(
    args: List[str] = typer.Argument(
        ...,
        help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]",
    ),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
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
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["show"] + args, channel, mode)


@app.command("simple-show")
@tracking.track_command("node")
def simple_show(
    args: List[str] = typer.Argument(
        ...,
        help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]",
    ),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
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
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["simple-show"] + args, channel, mode)


# install, reinstall, uninstall
@app.command()
@tracking.track_command("node")
def install(
    args: List[str] = typer.Argument(..., help="install custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    if "all" in args:
        typer.echo(f"Invalid command: {mode}. `install all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["install"] + args, channel, mode)


@app.command()
@tracking.track_command("node")
def reinstall(
    args: List[str] = typer.Argument(..., help="reinstall custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    if "all" in args:
        typer.echo(f"Invalid command: {mode}. `reinstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["reinstall"] + args, channel, mode)


@app.command()
@tracking.track_command("node")
def uninstall(
    args: List[str] = typer.Argument(..., help="uninstall custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    if "all" in args:
        typer.echo(f"Invalid command: {mode}. `uninstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["uninstall"] + args, channel, mode)


# `update, disable, enable, fix` allows `all` param


@app.command()
@tracking.track_command("node")
def update(
    args: List[str] = typer.Argument(..., help="update custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["update"] + args, channel, mode)


@app.command()
@tracking.track_command("node")
def disable(
    args: List[str] = typer.Argument(..., help="disable custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["disable"] + args, channel, mode)


@app.command()
@tracking.track_command("node")
def enable(
    args: List[str] = typer.Argument(..., help="enable custom nodes"),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["enable"] + args, channel, mode)


@app.command()
@tracking.track_command("node")
def fix(
    args: List[str] = typer.Argument(
        ..., help="fix dependencies for specified custom nodes"
    ),
    channel: Annotated[
        str,
        "--channel",
        typer.Option(show_default=False, help="Specify the operation mode"),
    ] = None,
    mode: Annotated[
        str, "--mode", typer.Option(show_default=False, help="[remote|local|cache]")
    ] = None,
):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(
            f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.",
            err=True,
        )
        raise typer.Exit(code=1)

    execute_cm_cli(["fix"] + args, channel, mode)


# TODO: re-enable publish after v0 launch
# @app.command("publish", help="Publish node to registry")
# @tracking.track_command("publish")
# def publish():
#     """
#     Publish a node with optional validation.
#     """

#     # Perform some validation logic here
#     typer.echo("Validating node configuration...")
#     config = extract_node_configuration()

#     # Prompt for Personal Access Token
#     token = typer.prompt("Please enter your Personal Access Token", hide_input=True)

#     # Call API to fetch node version with the token in the body
#     typer.echo("Publishing node version...")
#     response = publish_node_version(config, token)

#     # Zip up all files in the current directory, respecting .gitignore files.
#     signed_url = response.signedUrl
#     zip_filename = "node.tar.gz"
#     typer.echo("Creating zip file...")
#     zip_files(zip_filename)

#     # Upload the zip file to the signed URL
#     typer.echo("Uploading zip file...")
#     upload_file_to_signed_url(signed_url, zip_filename)


# @app.command("init", help="Init scaffolding for custom node")
# @tracking.track_command("node")
# def scaffold():
#     if os.path.exists("comfynode.toml"):
#         typer.echo("Warning: 'comfynode.toml' already exists. Will not overwrite.")
#         raise typer.Exit(code=1)

#     typer.echo("Initializing metadata...")
#     initialize_project_config()
#     typer.echo(
#         "comfynode.toml created successfully. Defaults were filled in. Please check before publishing."
#     )
