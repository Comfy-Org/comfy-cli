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

app = typer.Typer()
manager_app = typer.Typer()


def execute_cm_cli(args, channel=None, mode=None, workspace=None, recent=False):
    _env_checker = EnvChecker()
    _config_manager = ConfigManager()
    _config_manager.write_config()

    if workspace is not None:
        workspace = os.path.expanduser(workspace)
        comfyui_path = os.path.join(workspace, 'ComfyUI')
    elif not recent and _env_checker.comfy_repo is not None:
        os.chdir(_env_checker.comfy_repo.working_dir)
        comfyui_path = _env_checker.comfy_repo.working_dir
    elif not recent and _config_manager.config['DEFAULT'].get('default_workspace') is not None:
        comfyui_path = os.path.join(_config_manager.config['DEFAULT'].get('default_workspace'), 'ComfyUI')
    elif _config_manager.config['DEFAULT'].get('recent_path') is not None:
        comfyui_path = _config_manager.config['DEFAULT'].get('recent_path')
    else:
        print(f"\nComfyUI is not available.\n", file=sys.stderr)
        raise typer.Exit(code=1)

    if not os.path.exists(comfyui_path):
        print(f"\nComfyUI not found: {comfyui_path}\n", file=sys.stderr)
        raise typer.Exit(code=1)

    cm_cli_path = os.path.join(comfyui_path, 'custom_nodes', 'ComfyUI-Manager', 'cm-cli.py')
    if not os.path.exists(cm_cli_path):
        print(f"\nComfyUI-Manager not found: {cm_cli_path}\n", file=sys.stderr)
        raise typer.Exit(code=1)

    cmd = [sys.executable, cm_cli_path] + args
    if channel is not None:
        cmd += ['--channel', channel]

    if mode is not None:
        cmd += ['--mode', channel]

    env_path = _env_checker.get_isolated_env()
    new_env = os.environ.copy()
    if env_path is not None:
        session_path = os.path.join(_config_manager.get_config_path(), 'tmp', str(uuid.uuid4()))
        new_env['__COMFY_CLI_SESSION__'] = session_path
        new_env['COMFYUI_PATH'] = comfyui_path

    print(f"Execute from: {comfyui_path}")

    subprocess.run(cmd, env=new_env)


def validate_comfyui_manager(_env_checker):
    manager_path = _env_checker.get_comfyui_manager_path()

    if manager_path is None:
        print(f"[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)
    elif not os.path.exists(manager_path):
        print(f"[bold red]If ComfyUI-Manager is not installed, this feature cannot be used.[/bold red] \\[{manager_path}]")
        raise typer.Exit(code=1)
    elif not os.path.exists(os.path.join(manager_path, '.git')):
        print(f"[bold red]The ComfyUI-Manager installation is invalid. This feature cannot be used.[/bold red] \\[{manager_path}]")
        raise typer.Exit(code=1)


@app.command('save-snapshot', help="Save a snapshot of the current ComfyUI environment")
@tracking.track_command("node")
def save_snapshot(
        output: Annotated[str, '--output', typer.Option(show_default=False, help="Specify the output file path. (.json/.yaml)")] = None,
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    if output is None:
        execute_cm_cli(['save-snapshot'], workspace=workspace, recent=recent)
    else:
        output = os.path.abspath(output)  # to compensate chdir
        execute_cm_cli(['save-snapshot', '--output', output], workspace=workspace, recent=recent)


@app.command('restore-snapshot')
@tracking.track_command("node")
def restore_snapshot(
        path: str,
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):

    path = os.path.abspath(path)
    execute_cm_cli(['restore-snapshot', path], workspace=workspace, recent=recent)


@app.command('restore-dependencies')
@tracking.track_command("node")
def restore_dependencies(
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    execute_cm_cli(['restore-dependencies'], workspace=workspace, recent=recent)


@manager_app.command('disable-gui')
@tracking.track_command("node")
def disable_gui(
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    execute_cm_cli(['cli-only-mode', 'enable'], workspace=workspace, recent=recent)


@manager_app.command('enable-gui')
@tracking.track_command("node")
def enable_gui(
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    execute_cm_cli(['cli-only-mode', 'disable'], workspace=workspace, recent=recent)


@manager_app.command()
@tracking.track_command("node")
def clear(path: str,
          workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
          recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    path = os.path.abspath(path)
    execute_cm_cli(['clear', path], workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def show(args: List[str] = typer.Argument(..., help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"),
         channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
         mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
         workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
         recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):

    valid_commands = ["installed", "enabled", "not-installed", "disabled", "all", "snapshot", "snapshot-list"]
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['show'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command('simple-show')
@tracking.track_command("node")
def simple_show(args: List[str] = typer.Argument(..., help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"),
                channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
                mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
                workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
                recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):

    valid_commands = ["installed", "enabled", "not-installed", "disabled", "all", "snapshot", "snapshot-list"]
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['simple-show'] + args, channel, mode, workspace=workspace, recent=recent)


# install, reinstall, uninstall
@app.command()
@tracking.track_command("node")
def install(args: List[str] = typer.Argument(..., help="install custom nodes"),
            channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
            mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
            workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
            recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `install all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['install'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def reinstall(args: List[str] = typer.Argument(..., help="reinstall custom nodes"),
              channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
              mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
              workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
              recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `reinstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['reinstall'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def uninstall(args: List[str] = typer.Argument(..., help="uninstall custom nodes"),
              channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
              mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
              workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
              recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `uninstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['uninstall'] + args, channel, mode, workspace=workspace, recent=recent)


# `update, disable, enable, fix` allows `all` param

@app.command()
@tracking.track_command("node")
def update(args: List[str] = typer.Argument(..., help="update custom nodes"),
           channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
           mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
           workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
           recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['update'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def disable(args: List[str] = typer.Argument(..., help="disable custom nodes"),
            channel: Annotated[str, '--channel', typer.Option(show_default=False,help="Specify the operation mode")] = None,
            mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
            workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
            recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['disable'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def enable(args: List[str] = typer.Argument(..., help="enable custom nodes"),
           channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
           mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
           workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
           recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['enable'] + args, channel, mode, workspace=workspace, recent=recent)


@app.command()
@tracking.track_command("node")
def fix(args: List[str] = typer.Argument(..., help="fix dependencies for specified custom nodes"),
        channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
        mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None,
        recent: Annotated[bool, typer.Option(help="Use recent path (--workspace is higher)")] = False):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['fix'] + args, channel, mode, workspace=workspace, recent=recent)
