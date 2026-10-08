"""
Module for utility functions.
"""

import functools
import platform
import re
import shutil
import tarfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

from rich import progress
from rich.live import Live
from rich.table import Table

from comfy_cli.constants import DEFAULT_COMFY_WORKSPACE, OS, PROC
from comfy_cli.typing import PathLike

#: Instance cache per ``@singleton`` class, keyed by the wrapper the decorator
#: returns. Kept here rather than on that wrapper because several tests reach
#: the decorated class through ``Wrapper.__closure__[0]``.
_SINGLETON_CACHES: dict[Any, dict[type, Any]] = {}


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

    _SINGLETON_CACHES[get_instance] = instances
    return get_instance


def reset_singleton_for_testing(factory: Any) -> None:
    """Drop the instance cached behind a ``@singleton``-decorated class.

    The instance captures process-wide state when it is constructed
    (``ConfigManager`` reads the config dir exactly once). Tests repoint that
    state per test, so without this the FIRST test's instance silently serves
    every later one.
    """
    cache = _SINGLETON_CACHES.get(factory)
    if cache is not None:
        cache.clear()


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


def get_not_user_set_default_workspace():
    return DEFAULT_COMFY_WORKSPACE[get_os()]


_RFC3339_RE: Final = re.compile(
    r"(?P<head>\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?(?P<tz>[Zz]|[+-]\d{2}:?\d{2})?"
)


def parse_rfc3339(value: str) -> datetime:
    """Parse an RFC 3339 timestamp at any sub-second precision, always aware.

    ``datetime.fromisoformat`` cannot do this alone on Python 3.10, the minimum
    this package supports: there it takes only 3 or 6 fractional digits and no
    bare ``Z``. Our Go services marshal ``time.Time``, whose RFC3339Nano
    encoding trims trailing zeros, so microseconds of ``437450`` ship as
    ``...:09.43745Z`` — five digits, rejected. Roughly one row in ten is stamped
    that way, and a ``createdAt`` never changes, so such a row was permanently
    unreadable rather than intermittently so.

    A missing zone reads as UTC: callers order these against each other, and
    comparing a naive value to an aware one raises ``TypeError``.
    """
    match = _RFC3339_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"not an RFC 3339 timestamp: {value!r}")
    # Pad so ".1" is a tenth of a second rather than a microsecond, and truncate
    # so Go's nanosecond precision degrades instead of failing.
    fraction = (match.group("frac") or "")[:6].ljust(6, "0")
    zone = match.group("tz") or ""
    offset = "+00:00" if zone in ("Z", "z", "") else (zone if ":" in zone else f"{zone[:3]}:{zone[3:]}")
    return datetime.fromisoformat(f"{match.group('head')}.{fraction}{offset}")


def kill_all(pid):
    # Imported here, not at module level: only kill_all/is_running need psutil.
    import psutil

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        return True
    except Exception:
        return False


def is_running(pid):
    # Imported here, not at module level: only kill_all/is_running need psutil.
    import psutil

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
    # Imported lazily: requests costs ~30ms to import and utils is on the
    # import path of every CLI invocation; only downloads need it.
    import requests

    from comfy_cli.http import DOWNLOAD_TIMEOUT  # urllib.request behind it costs ~60ms; downloads only

    cwd = Path(cwd).expanduser().resolve()
    fpath = cwd / fname

    with requests.get(url, stream=True, allow_redirects=allow_redirects, timeout=DOWNLOAD_TIMEOUT) as response:
        if response.status_code != 200:
            response.raise_for_status()  # Will only raise for 4xx codes, so...
            raise RuntimeError(f"Request to {url} returned status code {response.status_code}")

        response.raw.read = functools.partial(response.raw.read, decode_content=True)  # Decompress if needed
        with fpath.open("wb") as f:
            if show_progress:
                fsize = int(response.headers.get("Content-Length", 0))
                desc = f"downloading {fname}..." + ("(Unknown total file size)" if fsize == 0 else "")

                with progress.wrap_file(cast(BinaryIO, response.raw), total=fsize, description=desc) as response_raw:
                    shutil.copyfileobj(response_raw, f)
            else:
                shutil.copyfileobj(response.raw, f)

    return fpath


@contextmanager
def _tarball_progress(description: str, total: int):
    """Yield the shared two-row Live progress scaffold used by
    extract_tarball/create_tarball.

    Builds a byte-progress bar plus a current-path line inside a single
    ``Live`` display and yields the wired-up
    ``(barProg, barTask, pathProg, pathTask)`` so each caller can supply its
    own ``filter`` body and label.
    """
    barProg = progress.Progress()
    barTask = barProg.add_task(f"[cyan]{description}", total=total)
    pathProg = progress.Progress(progress.TextColumn("{task.description}"))
    pathTask = pathProg.add_task("")

    progress_table = Table.grid()
    progress_table.add_row(barProg)
    progress_table.add_row(pathProg)

    with Live(progress_table, refresh_per_second=10):
        yield barProg, barTask, pathProg, pathTask


def extract_tarball(
    inPath: PathLike,
    outPath: PathLike | None = None,
    show_progress: bool = True,
):
    inPath = Path(inPath).expanduser().resolve()
    outPath = inPath.with_suffix("") if outPath is None else Path(outPath).expanduser().resolve()

    with tarfile.open(inPath) as tar:
        info = tar.next()
        if info is None:
            raise ValueError(f"tarball is empty: {inPath}")
        old_name = info.name.split("/")[0]
    # path to top-level of extraction result
    extractPath = inPath.with_name(old_name)

    # clean both the extraction path and the final target path
    shutil.rmtree(extractPath, ignore_errors=True)
    shutil.rmtree(outPath, ignore_errors=True)

    # Both extraction paths below use the stdlib "data" filter so a member with an
    # absolute path or a `../` traversal is rejected instead of being written
    # wherever it points (CVE-2007-4559).
    #
    # This used to be skipped because of https://github.com/python/cpython/issues/107845,
    # where data_filter resolved symlink targets against the destination root rather
    # than against the directory holding the link, and so falsely raised
    # LinkOutsideDestinationError on valid archives. That was a false-rejection bug,
    # never an escape, and it was fixed in 3.10.13 / 3.11.5 / 3.12.0rc2 (2023-08-24).
    # The only affected releases in our range are 3.10.12 and 3.11.4 — and since
    # `extractall(filter=...)` does not exist before 3.10.12 at all, that is the whole
    # window. On those two, the worst case is a loud error rather than a silent escape.
    if not show_progress:
        with tarfile.open(inPath) as tar:
            tar.extractall(filter="data")
        shutil.move(extractPath, outPath)
        return

    fileSize = inPath.stat().st_size

    with _tarball_progress("extracting tarball...", fileSize) as (barProg, barTask, pathProg, pathTask):

        def _reporting_members(tar: tarfile.TarFile):
            """Yield every member, driving the progress bars as we go.

            Progress reporting used to ride on the ``filter`` argument, which
            meant the extraction filter had to be a custom callable that
            returned members unmodified — silently disabling the CVE-2007-4559
            checks. The two concerns are separable: ``members`` drives the UI
            and ``filter`` stays the stdlib ``"data"`` filter.
            """
            size = 0
            for tinfo in tar:
                pathProg.update(pathTask, description=tinfo.path)
                barProg.advance(barTask, size)
                size = tinfo.size
                yield tinfo
            barProg.advance(barTask, size)

        with tarfile.open(inPath) as tar:
            tar.extractall(members=_reporting_members(tar), filter="data")
        pathProg.update(pathTask, description="")

    shutil.move(extractPath, outPath)


def create_tarball(
    inPath: PathLike,
    outPath: PathLike | None = None,
    cwd: PathLike | None = None,
    show_progress: bool = True,
):
    cwd = Path("." if cwd is None else cwd).expanduser().resolve()
    inPath = Path(inPath).expanduser().resolve()
    outPath = inPath.with_suffix(".tgz") if outPath is None else Path(outPath).expanduser().resolve()

    # clean the archive target path
    outPath.unlink(missing_ok=True)

    if not show_progress:
        with tarfile.open(outPath, "w:gz") as tar:
            # don't include parent paths in archive
            tar.add(inPath.relative_to(cwd), filter=None)
        return

    fileSize = sum(f.stat().st_size for f in inPath.glob("**/*"))

    _size = 0

    with _tarball_progress("creating tarball...", fileSize) as (barProg, barTask, pathProg, pathTask):

        def _filter(tinfo: tarfile.TarInfo):
            nonlocal _size
            pathProg.update(pathTask, description=tinfo.path)
            barProg.advance(barTask, _size)
            _size = Path(tinfo.path).stat().st_size

            return tinfo

        with tarfile.open(outPath, "w:gz") as tar:
            # don't include parent paths in archive
            tar.add(inPath.relative_to(cwd), filter=_filter)
        barProg.advance(barTask, _size)
        pathProg.update(pathTask, description="")
