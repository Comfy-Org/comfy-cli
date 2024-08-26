"""
Module for utility functions.
"""

import functools
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import requests
import typer
from rich import print, progress

from comfy_cli.constants import DEFAULT_COMFY_WORKSPACE, OS, PROC
from comfy_cli.typing import PathLike


def singleton(cls):
    """
    Decorator that implements the Singleton pattern for the decorated class.

    e.g.
    @singleton
    class MyClass:
        pass

    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def get_os():
    if sys.platform == "darwin":
        return OS.MACOS
    elif "win" in sys.platform:
        return OS.WINDOWS

    return OS.LINUX


def get_proc():
    proc = platform.processor()

    if proc == "x86_64":
        return PROC.X86_64
    elif "arm" in proc:
        return PROC.ARM
    else:
        raise ValueError


def install_conda_package(package_name):
    try:
        subprocess.check_call(["conda", "install", "-y", package_name])
        print(f"[bold green] Successfully installed {package_name} [/bold green]")
    except subprocess.CalledProcessError as e:
        print(f"[bold red] Failed to install {package_name}. Error: {e} [/bold red]")
        raise typer.Exit(code=1)


def get_not_user_set_default_workspace():
    return DEFAULT_COMFY_WORKSPACE[get_os()]


def kill_all(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        return True
    except Exception:
        return False


def is_running(pid):
    try:
        psutil.Process(pid)
        return True
    except psutil.NoSuchProcess:
        return False


def create_choice_completer(opts: list[str]):
    def f(incomplete: str) -> list[str]:
        return [opt for opt in opts if opt.startswith(incomplete)]

    return f


def download_progress(url: str, fname: PathLike, cwd: PathLike = ".", allow_redirects: bool = True) -> PathLike:
    """download url to local file fname and show a progress bar.
    See https://stackoverflow.com/q/37573483"""
    cwd = Path(cwd).expanduser().resolve()
    fpath = cwd / fname

    response = requests.get(url, stream=True, allow_redirects=allow_redirects)
    if response.status_code != 200:
        response.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {response.status_code}")
    fsize = int(response.headers.get("Content-Length", 0))

    desc = "(Unknown total file size)" if fsize == 0 else ""
    response.raw.read = functools.partial(response.raw.read, decode_content=True)  # Decompress if needed
    with progress.wrap_file(response.raw, total=fsize, description=desc) as response_raw:
        with fpath.open("wb") as f:
            shutil.copyfileobj(response_raw, f)

    return fpath
