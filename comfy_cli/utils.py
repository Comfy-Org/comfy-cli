"""
Module for utility functions.
"""

import functools
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import psutil
import requests
import typer
from rich import print, progress
from rich.live import Live
from rich.table import Table

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
    platform_system = platform.system().lower()

    if platform_system == "darwin":
        return OS.MACOS
    elif platform_system == "windows":
        return OS.WINDOWS
    elif platform_system == "linux":
        return OS.LINUX
    else:
        raise ValueError(f"Running on unsupported os {platform.system()}")


def get_proc():
    proc = platform.machine()

    if proc == "x86_64" or proc == "AMD64":
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


def download_url(
    url: str,
    fname: PathLike,
    cwd: PathLike = ".",
    allow_redirects: bool = True,
    show_progress: bool = True,
) -> PathLike:
    """download url to local file fname and show a progress bar.
    See https://stackoverflow.com/q/37573483"""
    cwd = Path(cwd).expanduser().resolve()
    fpath = cwd / fname

    response = requests.get(url, stream=True, allow_redirects=allow_redirects)
    if response.status_code != 200:
        response.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {response.status_code}")

    response.raw.read = functools.partial(response.raw.read, decode_content=True)  # Decompress if needed
    with fpath.open("wb") as f:
        if show_progress:
            fsize = int(response.headers.get("Content-Length", 0))
            desc = f"downloading {fname}..." + ("(Unknown total file size)" if fsize == 0 else "")

            with progress.wrap_file(response.raw, total=fsize, description=desc) as response_raw:
                shutil.copyfileobj(response_raw, f)
        else:
            shutil.copyfileobj(response.raw, f)

    return fpath


def extract_tarball(
    inPath: PathLike,
    outPath: Optional[PathLike] = None,
    show_progress: bool = True,
):
    inPath = Path(inPath).expanduser().resolve()
    outPath = inPath.with_suffix("") if outPath is None else Path(outPath).expanduser().resolve()

    with tarfile.open(inPath) as tar:
        info = tar.next()
        old_name = info.name.split("/")[0]
    # path to top-level of extraction result
    extractPath = inPath.with_name(old_name)

    # clean both the extraction path and the final target path
    shutil.rmtree(extractPath, ignore_errors=True)
    shutil.rmtree(outPath, ignore_errors=True)

    if show_progress:
        fileSize = inPath.stat().st_size

        barProg = progress.Progress()
        barTask = barProg.add_task("[cyan]extracting tarball...", total=fileSize)
        pathProg = progress.Progress(progress.TextColumn("{task.description}"))
        pathTask = pathProg.add_task("")

        progress_table = Table.grid()
        progress_table.add_row(barProg)
        progress_table.add_row(pathProg)

        _size = 0

        def _filter(tinfo: tarfile.TarInfo, _path: PathLike):
            nonlocal _size
            pathProg.update(pathTask, description=tinfo.path)
            barProg.advance(barTask, _size)
            _size = tinfo.size

            # TODO: ideally we'd use data_filter here, but it's busted: https://github.com/python/cpython/issues/107845
            # return tarfile.data_filter(tinfo, _path)
            return tinfo
    else:
        _filter = None

    with Live(progress_table, refresh_per_second=10):
        with tarfile.open(inPath) as tar:
            tar.extractall(filter=_filter)

        if show_progress:
            barProg.advance(barTask, _size)
            pathProg.update(pathTask, description="")

    shutil.move(extractPath, outPath)


def create_tarball(
    inPath: PathLike,
    outPath: Optional[PathLike] = None,
    cwd: Optional[PathLike] = None,
    show_progress: bool = True,
):
    cwd = Path("." if cwd is None else cwd).expanduser().resolve()
    inPath = Path(inPath).expanduser().resolve()
    outPath = inPath.with_suffix(".tgz") if outPath is None else Path(outPath).expanduser().resolve()

    # clean the archive target path
    outPath.unlink(missing_ok=True)

    if show_progress:
        fileSize = sum(f.stat().st_size for f in inPath.glob("**/*"))

        barProg = progress.Progress()
        barTask = barProg.add_task("[cyan]creating tarball...", total=fileSize)
        pathProg = progress.Progress(progress.TextColumn("{task.description}"))
        pathTask = pathProg.add_task("")

        progress_table = Table.grid()
        progress_table.add_row(barProg)
        progress_table.add_row(pathProg)

        _size = 0

        def _filter(tinfo: tarfile.TarInfo):
            nonlocal _size
            pathProg.update(pathTask, description=tinfo.path)
            barProg.advance(barTask, _size)
            _size = Path(tinfo.path).stat().st_size

            return tinfo
    else:
        _filter = None

    with Live(progress_table, refresh_per_second=10):
        with tarfile.open(outPath, "w:gz") as tar:
            # don't include parent paths in archive
            tar.add(inPath.relative_to(cwd), filter=_filter)

        if show_progress:
            barProg.advance(barTask, _size)
            pathProg.update(pathTask, description="")
