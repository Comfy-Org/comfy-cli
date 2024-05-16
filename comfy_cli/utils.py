"""
Module for utility functions.
"""

import subprocess
import sys

import psutil
from rich import print
import typer

from comfy_cli import constants


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
        return constants.OS.MACOS
    elif "win" in sys.platform:
        return constants.OS.WINDOWS

    return constants.OS.LINUX


def install_conda_package(package_name):
    try:
        subprocess.check_call(["conda", "install", "-y", package_name])
        print(f"[bold green] Successfully installed {package_name} [/bold green]")
    except subprocess.CalledProcessError as e:
        print(f"[bold red] Failed to install {package_name}. Error: {e} [/bold red]")
        raise typer.Exit(code=1)


def get_not_user_set_default_workspace():
    return constants.DEFAULT_COMFY_WORKSPACE[get_os()]


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


def create_choice_completer(opts):
    def f(incomplete: str) -> list[str]:
        return [opt for opt in opts if opt.startswith(incomplete)]

    return f
