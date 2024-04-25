import typer
from typing_extensions import List, Annotated
from comfy_cli.env_checker import EnvChecker
import os
import subprocess
import sys

app = typer.Typer()
manager_app = typer.Typer()


def execute_cm_cli(args, channel=None, mode=None, workspace=None):
    _env_checker = EnvChecker()
    _env_checker.write_config()

    if workspace is not None:
        comfyui_path = os.path.join(workspace, 'ComfyUI')
    elif _env_checker.comfy_repo is not None:
        comfyui_path = _env_checker.comfy_repo.working_dir
    elif _env_checker.config['DEFAULT'].get('recent_path') is not None:
        comfyui_path = _env_checker.config['DEFAULT'].get('recent_path')
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
        new_env['__COMFY_CLI_SESSION__'] = os.path.join(env_path, 'comfy-cli')
        new_env['COMFYUI_PATH'] = comfyui_path

    subprocess.run(cmd, env=new_env)


@app.command('save-snapshot')
def save_snapshot():
    execute_cm_cli(['save-snapshot'])


@app.command('restore-snapshot')
def restore_snapshot(path: str):
    execute_cm_cli(['restore-snapshot', path])


@app.command('restore-dependencies')
def restore_dependencies(path: str):
    execute_cm_cli(['restore-dependencies', path])


@manager_app.command('disable-gui')
def disable_gui():
    execute_cm_cli(['cli-only-mode', 'enable'])


@manager_app.command('enable-gui')
def enable_gui():
    execute_cm_cli(['cli-only-mode', 'disable'])


@manager_app.command()
def clear(path: str):
    execute_cm_cli(['clear', path])


@app.command()
def show(args: List[str] = typer.Argument(..., help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"),
         channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
         mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
         workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):

    valid_commands = ["installed", "enabled", "not-installed", "disabled", "all", "snapshot", "snapshot-list"]
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['show'] + args, channel, mode, workspace)


@app.command('simple-show')
def simple_show(args: List[str] = typer.Argument(..., help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"),
                channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
                mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
                workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):

    valid_commands = ["installed", "enabled", "not-installed", "disabled", "all", "snapshot", "snapshot-list"]
    if not args or len(args) > 1 or args[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(args)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['simple-show'] + args, channel, mode, workspace)


# install, reinstall, uninstall
@app.command()
def install(args: List[str] = typer.Argument(..., help="install custom nodes"),
            channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
            mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
            workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `install all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['install'] + args, channel, mode, workspace)


@app.command()
def reinstall(args: List[str] = typer.Argument(..., help="reinstall custom nodes"),
              channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
              mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
              workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `reinstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['reinstall'] + args, channel, mode, workspace)


@app.command()
def uninstall(args: List[str] = typer.Argument(..., help="uninstall custom nodes"),
              channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
              mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
              workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    if 'all' in args:
        typer.echo(f"Invalid command: {mode}. `uninstall all` is not allowed", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['uninstall'] + args, channel, mode, workspace)


# `update, disable, enable, fix` allows `all` param

@app.command()
def update(args: List[str] = typer.Argument(..., help="update custom nodes"),
           channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
           mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
           workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['update'] + args, channel, mode, workspace)


@app.command()
def disable(args: List[str] = typer.Argument(..., help="disable custom nodes"),
            channel: Annotated[str, '--channel', typer.Option(show_default=False,help="Specify the operation mode")] = None,
            mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
            workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['disable'] + args, channel, mode, workspace)


@app.command()
def enable(args: List[str] = typer.Argument(..., help="enable custom nodes"),
           channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
           mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
           workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['enable'] + args, channel, mode, workspace)


@app.command()
def fix(args: List[str] = typer.Argument(..., help="fix dependencies for specified custom nodes"),
        channel: Annotated[str, '--channel', typer.Option(show_default=False, help="Specify the operation mode")] = None,
        mode: Annotated[str, '--mode', typer.Option(show_default=False, help="[remote|local|cache]")] = None,
        workspace: Annotated[str, typer.Option(show_default=False, help="Path to ComfyUI workspace")] = None):
    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['fix'] + args, channel, mode, workspace)
