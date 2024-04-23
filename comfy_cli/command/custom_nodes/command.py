import typer
from typing_extensions import List, Annotated
from comfy_cli.env_checker import EnvChecker
import os
import subprocess
import sys

app = typer.Typer()
manager_app = typer.Typer()


@app.command()
def add(name: str):
    """Add a new custom node"""
    print(f"Adding a new custom node: {name}")


@app.command()
def remove(name: str):
    """Remove a custom node"""
    print(f"Removing a custom node: {name}")


@app.command('save-snapshot')
def save_snapshot():
    execute_cm_cli(['save-snapshot'])


@app.command('restore-snapshot')
def restore_snapshot(path: str):
    execute_cm_cli(['restore-snapshot', path])


@manager_app.command('disable-gui')
def disable_gui():
    execute_cm_cli(['cli-only-mode', 'enable'])


@manager_app.command('enable-gui')
def enable_gui():
    execute_cm_cli(['cli-only-mode', 'disable'])


@app.command()
def show(
        arg: List[str] = typer.Argument(..., help="[installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]"),
        channel: Annotated[
            str,
            '--channel',
            typer.Option(
                show_default=False,
                help="Specify the operation mode")
        ] = None,
        mode: Annotated[
            str,
            '--mode',
            typer.Option(
                show_default=False,
                help="[remote|local|cache]")
        ] = None,
        workspace: Annotated[
                str,
                typer.Option(
                    show_default=False,
                    help="Path to ComfyUI workspace")
        ] = None,
        ):

    valid_commands = ["installed", "enabled", "not-installed", "disabled", "all", "snapshot", "snapshot-list"]
    if not arg or len(arg) > 1 or arg[0] not in valid_commands:
        typer.echo(f"Invalid command: `show {' '.join(arg)}`", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["remote", "local", "cache"]
    if mode and mode.lower() not in valid_modes:
        typer.echo(f"Invalid mode: {mode}. Allowed modes are 'remote', 'local', 'cache'.", err=True)
        raise typer.Exit(code=1)

    execute_cm_cli(['show'] + arg, channel, mode, workspace)


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

    cmd = ["python", cm_cli_path] + args
    if channel is not None:
        cmd += ['--channel', channel]

    if mode is not None:
        cmd += ['--mode', channel]

    env_path = _env_checker.get_isolated_env()
    new_env = os.environ.copy()
    if env_path is not None:
        new_env['__COMFY_CLI_SESSION__'] = os.path.join(env_path, 'comfy-cli')
        new_env['COMFYUI_PATH'] = comfyui_path

    output = subprocess.check_output(cmd, env=new_env, text=True)
    print(output)
