import os
from pathlib import Path
import shutil
import tarfile
from typing import Optional

import requests

from comfy_cli.constants import OS, PROC
from comfy_cli.utils import download_progress, get_os, get_proc
from comfy_cli.typing import PathLike

_platform_targets = {
    (OS.MACOS, PROC.ARM): "aarch64-apple-darwin",
    (OS.MACOS, PROC.X86_64): "x86_64-apple-darwin",
    (OS.LINUX, PROC.X86_64): "x86_64_v3-unknown-linux-gnu",  # x86_64_v3 assumes AVX256 support, no AVX512 support
    (OS.WINDOWS, PROC.X86_64): "x86_64-pc-windows-msvc-shared",
}

_latest_release_json_url = "https://raw.githubusercontent.com/indygreg/python-build-standalone/latest-release/latest-release.json"
_asset_url_prefix = "https://github.com/indygreg/python-build-standalone/releases/download/{tag}"

def download_standalone_python(
        platform: Optional[str] = None,
        proc: Optional[str] = None,
        version: str = "3.12.5",
        tag: str = "latest",
        flavor: str = "install_only",
        cwd: PathLike = ".",
    ) -> PathLike:
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

    return download_progress(url, fname, cwd=cwd)

class StandalonePython:
    @staticmethod
    def FromDistro(
        platform: Optional[str] = None,
        proc: Optional[str] = None,
        version: str = "3.12.5",
        tag: str = "latest",
        flavor: str = "install_only",
        cwd: PathLike = ".",
        name: PathLike = "python"
    ):
        fpath = download_standalone_python(platform=platform, proc=proc, version=version, tag=tag, flavor=flavor, cwd=cwd)
        return StandalonePython.FromTarball(fpath, name)

    @staticmethod
    def FromTarball(fpath: PathLike, name: PathLike = "python"):
        fpath = Path(fpath)
        with tarfile.open(fpath) as tar:
            info = tar.next()
            old_name = info.name.split("/")[0]
            tar.extractall()

        old_rpath = fpath.parent / old_name
        rpath = fpath.parent / name
        shutil.move(old_rpath, rpath)
        return StandalonePython(rpath=rpath)

    def __init__(self, rpath: PathLike):
        self.rpath = Path(rpath)
        self.name = self.rpath.name
        self.bin = self.rpath / "bin"
        self.executable = self.bin / "python"
