import logging
import re
import shutil
import subprocess
from pathlib import Path

import requests

from comfy_cli.constants import DEFAULT_STANDALONE_PYTHON_MINOR_VERSION, OS, PROC
from comfy_cli.typing import PathLike
from comfy_cli.utils import create_tarball, download_url, extract_tarball, get_os, get_proc
from comfy_cli.uv import DependencyCompiler

logger = logging.getLogger(__name__)

_here = Path(__file__).expanduser().resolve().parent

_platform_targets = {
    (OS.MACOS, PROC.ARM): "aarch64-apple-darwin",
    (OS.MACOS, PROC.X86_64): "x86_64-apple-darwin",
    (OS.LINUX, PROC.X86_64): "x86_64_v3-unknown-linux-gnu",  # x86_64_v3 assumes AVX256 support, no AVX512 support
    (OS.WINDOWS, PROC.X86_64): "x86_64-pc-windows-msvc",
}

_latest_release_json_url = (
    "https://raw.githubusercontent.com/astral-sh/python-build-standalone/latest-release/latest-release.json"
)
_asset_url_prefix = "https://github.com/astral-sh/python-build-standalone/releases/download/{tag}"


def _resolve_python_version(asset_url_prefix: str, minor_version: str) -> str:
    """Resolve the exact patch version for a minor version series from the release SHA256SUMS.

    Downloads the SHA256SUMS file (~45 KB) from the release and parses it to find
    the available patch version for the requested minor series (e.g. "3.12" -> "3.12.13").
    """
    sha256sums_url = f"{asset_url_prefix.rstrip('/')}/SHA256SUMS"
    response = requests.get(sha256sums_url)
    response.raise_for_status()

    pattern = re.compile(rf"cpython-({re.escape(minor_version)}\.\d+)\+")
    versions = set()
    for line in response.text.splitlines():
        match = pattern.search(line)
        if match:
            versions.add(match.group(1))

    if not versions:
        raise RuntimeError(
            f"No Python {minor_version}.x found in release. Available versions can be checked at {sha256sums_url}"
        )

    # There should be exactly one patch version per minor series in a release, but pick the highest just in case.
    resolved = max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
    logger.info("Resolved Python %s -> %s", minor_version, resolved)
    return resolved


def download_standalone_python(
    platform: str | None = None,
    proc: str | None = None,
    version: str = DEFAULT_STANDALONE_PYTHON_MINOR_VERSION,
    tag: str = "latest",
    flavor: str = "install_only",
    cwd: PathLike = ".",
    show_progress: bool = True,
) -> PathLike:
    """grab a pre-built distro from the python-build-standalone project. See
    https://gregoryszorc.com/docs/python-build-standalone/main/"""
    platform = get_os() if platform is None else platform
    proc = get_proc() if proc is None else proc
    target = _platform_targets[(platform, proc)]

    if tag == "latest":
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

    # If version is a minor version (e.g. "3.12"), resolve the exact patch version
    # from the release metadata. Full versions (e.g. "3.12.13") are used as-is.
    if version.count(".") == 1:
        version = _resolve_python_version(asset_url_prefix, version)

    name = f"cpython-{version}+{tag}-{target}-{flavor}"
    fname = f"{name}.tar.gz"
    url = f"{asset_url_prefix.rstrip('/')}/{fname.lstrip('/')}"

    return download_url(url, fname, cwd=cwd, show_progress=show_progress)


class StandalonePython:
    @staticmethod
    def FromDistro(
        platform: str | None = None,
        proc: str | None = None,
        version: str = DEFAULT_STANDALONE_PYTHON_MINOR_VERSION,
        tag: str = "latest",
        flavor: str = "install_only",
        cwd: PathLike = ".",
        name: PathLike = "python",
        show_progress: bool = True,
    ) -> "StandalonePython":
        fpath = download_standalone_python(
            platform=platform,
            proc=proc,
            version=version,
            tag=tag,
            flavor=flavor,
            cwd=cwd,
            show_progress=show_progress,
        )
        return StandalonePython.FromTarball(fpath, name)

    @staticmethod
    def FromTarball(fpath: PathLike, name: PathLike = "python", show_progress: bool = True) -> "StandalonePython":
        fpath = Path(fpath)
        rpath = fpath.parent / name

        extract_tarball(inPath=fpath, outPath=rpath, show_progress=show_progress)

        return StandalonePython(rpath=rpath)

    def __init__(self, rpath: PathLike):
        self.rpath = Path(rpath)
        self.name = self.rpath.name
        if get_os() == OS.WINDOWS:
            self.bin = self.rpath
            self.executable = self.bin / "python.exe"
        else:
            self.bin = self.rpath / "bin"
            self.executable = self.bin / "python"

        # paths to store package artifacts
        self.cache = self.rpath / "cache"
        self.wheels = self.rpath / "wheels"

        self.dep_comp = None

        # upgrade pip if needed, install uv
        self.pip_install("-U", "pip", "uv")

    def clean(self):
        for pycache in self.rpath.glob("**/__pycache__"):
            shutil.rmtree(pycache)

    def run_module(self, mod: str, *args: str):
        cmd: list[str] = [
            str(self.executable),
            "-m",
            mod,
            *args,
        ]

        subprocess.run(cmd, check=True)

    def pip_install(self, *args: str):
        self.run_module("pip", "install", *args)

    def uv_install(self, *args: str):
        self.run_module("uv", "pip", "install", *args)

    def install_comfy_cli(self, dev: bool = False):
        if dev:
            self.uv_install(str(_here.parent))
        else:
            self.uv_install("comfy_cli")

    def run_comfy_cli(self, *args: str):
        self.run_module("comfy_cli", *args)

    def install_comfy(self, *args: str, gpu_arg: str = "--nvidia"):
        self.run_comfy_cli("--here", "--skip-prompt", "install", "--fast-deps", gpu_arg, *args)

    def dehydrate_comfy_deps(
        self,
        comfyDir: PathLike,
        extraSpecs: list[str] | None = None,
        packWheels: bool = False,
    ):
        self.dep_comp = DependencyCompiler(
            cwd=comfyDir,
            executable=self.executable,
            outDir=self.rpath,
            extraSpecs=extraSpecs,
        )
        self.dep_comp.compile_deps()

        if packWheels:
            skip_uv = get_os() == OS.WINDOWS
            self.dep_comp.fetch_dep_wheels(skip_uv=skip_uv)

    def rehydrate_comfy_deps(self, packWheels: bool = False):
        self.dep_comp = DependencyCompiler(
            executable=self.executable, outDir=self.rpath, reqFilesCore=[], reqFilesExt=[]
        )

        if packWheels:
            self.dep_comp.install_wheels_directly()
        else:
            self.dep_comp.install_deps()

    def to_tarball(self, outPath: PathLike | None = None, show_progress: bool = True):
        # remove any __pycache__ before creating archive
        self.clean()

        create_tarball(inPath=self.rpath, outPath=outPath, show_progress=show_progress)
