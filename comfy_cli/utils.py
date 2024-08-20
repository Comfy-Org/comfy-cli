"""
Module for utility functions.
"""

import functools
import os
from pathlib import Path
import platform
import psutil
import shutil
import subprocess
import sys

import requests
from rich import print
from tqdm.auto import tqdm
import typer

from comfy_cli.constants import DEFAULT_COMFY_WORKSPACE, OS, PROC


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


def create_choice_completer(opts):
    def f(incomplete: str) -> list[str]:
        return [opt for opt in opts if opt.startswith(incomplete)]

    return f

def download_progress(url, fname, cwd=".", allow_redirects=True):
    """download url to local file fname and show a progress bar.
    See https://stackoverflow.com/q/37573483"""
    cwd = Path(cwd).expanduser().resolve()
    fpath = cwd / fname

    response = requests.get(url, stream=True, allow_redirects=allow_redirects)
    if response.status_code != 200:
        response.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {response.status_code}")
    fsize = int(response.headers.get('Content-Length', 0))

    desc = "(Unknown total file size)" if fsize == 0 else ""
    response.raw.read = functools.partial(response.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(response.raw, "read", total=fsize, desc=desc) as response_raw:
        with fpath.open("wb") as f:
            shutil.copyfileobj(response_raw, f)

    return fpath

_platform_targets = {
    (OS.MACOS, PROC.ARM): "aarch64-apple-darwin",
    (OS.MACOS, PROC.X86_64): "x86_64-apple-darwin",
    (OS.LINUX, PROC.X86_64): "x86_64_v3-unknown-linux-gnu",  # x86_64_v3 assumes AVX256 support, no AVX512 support
    (OS.WINDOWS, PROC.X86_64): "x86_64-pc-windows-msvc-shared",
}

_latest_release_json_url = "https://raw.githubusercontent.com/indygreg/python-build-standalone/latest-release/latest-release.json"
_asset_url_prefix = "https://github.com/indygreg/python-build-standalone/releases/download/{tag}"

def download_standalone_python(platform=None, proc=None, version="3.12.5", tag="latest", flavor="install_only"):
    """grab a pre-built distro from the python-build-standalone project. See
    https://gregoryszorc.com/docs/python-build-standalone/main/"""
    platform = get_os() if platform is None else platform
    proc = get_proc() if proc is None else proc
    target = _platform_targets[(platform, proc)]

    if tag=="latest":
        # try to fetch json with info about latest release
        response = requests.get(_latest_release_json_url)
        if response.status_code != 200:
            response.raise_for_status()
            raise RuntimeError(f"Request to {_latest_release_json_url} returned status code {response.status_code}")

        latest_release = response.json()
        tag = latest_release["tag"]
        asset_url_prefix = latest_release["asset_url_prefix"]
    else:
        asset_url_prefix = _asset_url_prefix.format(tag=tag)

    name = f"cpython-{version}+{tag}-{target}-{flavor}"
    fname = f"{name}.tar.gz"
    url = os.path.join(asset_url_prefix, fname)

    return download_progress(url, fname)
