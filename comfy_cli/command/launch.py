from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import uuid

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from comfy_cli import constants, utils
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.update import check_for_updates
from comfy_cli.workspace_manager import WorkspaceManager, WorkspaceType

workspace_manager = WorkspaceManager()
console = Console()


def launch_comfyui(extra):
    reboot_path = None

    new_env = os.environ.copy()

    session_path = os.path.join(ConfigManager().get_config_path(), "tmp", str(uuid.uuid4()))
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["PYTHONENCODING"] = "utf-8"

    # To minimize the possibility of leaving residue in the tmp directory, use files instead of directories.
    reboot_path = os.path.join(session_path + ".reboot")

    extra = extra if extra is not None else []

    process = None

    if "COMFY_CLI_BACKGROUND" not in os.environ:
        # If not running in background mode, there's no need to use popen. This can prevent the issue of linefeeds occurring with tqdm.
        while True:
            res = subprocess.run([sys.executable, "main.py"] + extra, env=new_env, check=False)

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


def launch(
    background: bool = False,
    extra: list[str] | None = None,
):
    check_for_updates()
    resolved_workspace = workspace_manager.workspace_path

    if not resolved_workspace:
        print(
            "\nComfyUI is not available.\nTo install ComfyUI, you can run:\n\n\tcomfy install\n\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    if (extra is None or len(extra) == 0) and workspace_manager.workspace_type == WorkspaceType.DEFAULT:
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
            if extra[i] == "--listen":
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

    console.print("\n[bold red]Execution error: failed to launch ComfyUI[/bold red]\n")
    # NOTE: os.exit(0) doesn't work
    os._exit(1)


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
                ConfigManager().config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND] = f"{(listen, port, process.pid)}"
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

    process.wait()

    return log
