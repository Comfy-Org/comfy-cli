import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional, Union, cast

from comfy_cli import ui
from comfy_cli.constants import GPU_OPTION
from comfy_cli.typing import PathLike


def _run(cmd: list[str], cwd: PathLike, check: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def _check_call(cmd: list[str], cwd: Optional[PathLike] = None):
    """uses check_call to run pip, as reccomended by the pip maintainers.
    see https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program"""
    subprocess.check_call(cmd, cwd=cwd)


_req_name_re: re.Pattern[str] = re.compile(r"require\s([\w-]+)")


def _req_re_closure(name: str) -> re.Pattern[str]:
    return re.compile(rf"({name}\S+)")


def parse_uv_compile_error(err: str) -> tuple[str, list[str]]:
    """takes in stderr from a run of `uv pip compile` that failed due to requirement conflict and spits out
    a tuple of (reqiurement_name, [requirement_spec_in_conflict_a, requirement_spec_in_conflict_b]). Will probably
    fail for stderr produced from other kinds of errors
    """
    if reqNameMatch := _req_name_re.search(err):
        reqName = reqNameMatch[1]
    else:
        raise ValueError

    reqRe = _req_re_closure(reqName)

    return reqName, cast(list[str], reqRe.findall(err))


class DependencyCompiler:
    rocmPytorchUrl = "https://download.pytorch.org/whl/rocm6.1"
    nvidiaPytorchUrl = "https://download.pytorch.org/whl/cu121"

    overrideGpu = dedent(
        """
        # ensure usage of {gpu} version of pytorch
        --extra-index-url {gpuUrl}
        torch
        torchsde
        torchvision
    """
    ).strip()

    reqNames = {
        "requirements.txt",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
    }

    @staticmethod
    def Find_Req_Files(*ders: PathLike) -> list[Path]:
        return [
            file  # fmt: skip
            for der in ders
            for file in Path(der).absolute().iterdir()
            if file.name in DependencyCompiler.reqNames
        ]

    @staticmethod
    def Install_Build_Deps(executable: PathLike = sys.executable):
        """Use pip to install bare minimum requirements for uv to do its thing"""
        cmd = [str(executable), "-m", "pip", "install", "--upgrade", "pip", "uv"]
        _check_call(cmd=cmd)

    @staticmethod
    def Compile(
        cwd: PathLike,
        reqFiles: list[PathLike],
        emit_index_annotation: bool = True,
        emit_index_url: bool = True,
        executable: PathLike = sys.executable,
        index_strategy: str = "unsafe-best-match",
        out: Optional[PathLike] = None,
        override: Optional[PathLike] = None,
        resolve_strategy: Optional[str] = None,
    ) -> subprocess.CompletedProcess[Any]:
        cmd = [
            str(executable),
            "-m",
            "uv",
            "pip",
            "compile",
        ]

        for reqFile in reqFiles:
            cmd.append(str(reqFile))

        if emit_index_annotation:
            cmd.append("--emit-index-annotation")

        if emit_index_url:
            cmd.append("--emit-index-url")

        # ensures that eg tqdm is latest version, even though an old tqdm is on the amd url
        # see https://github.com/astral-sh/uv/blob/main/PIP_COMPATIBILITY.md#packages-that-exist-on-multiple-indexes and https://github.com/astral-sh/uv/issues/171
        if index_strategy is not None:
            cmd.extend(["--index-strategy", "unsafe-best-match"])

        if out is not None:
            cmd.extend(["-o", str(out)])

        if override is not None:
            cmd.extend(["--override", str(override)])

        try:
            return _run(cmd, cwd)
        except subprocess.CalledProcessError as e:
            print(e.__class__.__name__)
            print(e)
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")

            if resolve_strategy == "ask":
                name, reqs = parse_uv_compile_error(e.stderr)
                vers = [req.split(name)[1].strip(",") for req in reqs]

                ver = ui.prompt_select(
                    "Please pick one of the conflicting version specs (or pick latest):",
                    choices=vers + ["latest"],
                    default=vers[0],
                )

                if ver == "latest":
                    req = name
                else:
                    req = name + ver

                e.req = req
            elif resolve_strategy is not None:
                # no other resolve_strategy options implemented yet
                raise ValueError

            raise e

    @staticmethod
    def Install(
        cwd: PathLike,
        executable: PathLike = sys.executable,
        dry: bool = False,
        extra_index_url: Optional[str] = None,
        find_links: Optional[list[str]] = None,
        index_strategy: Optional[str] = "unsafe-best-match",
        no_deps: bool = False,
        no_index: bool = False,
        override: Optional[PathLike] = None,
        reqs: Optional[list[str]] = None,
        reqFile: Optional[list[PathLike]] = None,
    ) -> subprocess.CompletedProcess[Any]:
        cmd = [
            str(executable),
            "-m",
            "uv",
            "pip",
            "install",
        ]

        if dry:
            cmd.append("--dry-run")

        if extra_index_url is not None:
            cmd.extend(["--extra-index-url", extra_index_url])

        if find_links is not None:
            for fl in find_links:
                cmd.extend(["--find-links", fl])

        if index_strategy is not None:
            cmd.extend(["--index-strategy", "unsafe-best-match"])

        if no_deps:
            cmd.append("--no-deps")

        if no_index:
            cmd.append("--no-index")

        if override is not None:
            cmd.extend(["--override", str(override)])

        if reqs is not None:
            cmd.extend(reqs)

        if reqFile is not None:
            for rf in reqFile:
                cmd.extend(["--requirement", rf])

        return _check_call(cmd, cwd)

    @staticmethod
    def Sync(
        cwd: PathLike,
        reqFile: list[PathLike],
        dry: bool = False,
        executable: PathLike = sys.executable,
        extraUrl: Optional[str] = None,
        index_strategy: str = "unsafe-best-match",
    ) -> subprocess.CompletedProcess[Any]:
        cmd = [
            str(executable),
            "-m",
            "uv",
            "pip",
            "sync",
            str(reqFile),
        ]

        if index_strategy is not None:
            cmd.extend(["--index-strategy", "unsafe-best-match"])

        if extraUrl is not None:
            cmd.extend(["--extra-index-url", extraUrl])

        if dry:
            cmd.append("--dry-run")

        return _check_call(cmd, cwd)

    @staticmethod
    def Download(
        cwd: PathLike,
        reqFile: list[PathLike],
        executable: PathLike = sys.executable,
        extraUrl: Optional[str] = None,
        noDeps: bool = False,
        out: Optional[PathLike] = None,
    ) -> subprocess.CompletedProcess[Any]:
        """For now, the `download` cmd has no uv support, so use pip"""
        cmd = [
            str(executable),
            "-m",
            "pip",
            "download",
        ]

        if isinstance(reqFile, (str, Path)):
            cmd.extend(["-r", str(reqFile)])
        elif isinstance(reqFile, list):
            for rf in reqFile:
                cmd.extend(["-r", str(rf)])

        if extraUrl is not None:
            cmd.extend(["--extra-index-url", extraUrl])

        if noDeps:
            cmd.append("--no-deps")

        if out is not None:
            cmd.extend(["-d", str(out)])

        return _check_call(cmd, cwd)

    @staticmethod
    def Wheel(
        cwd: PathLike,
        reqFile: list[PathLike],
        executable: PathLike = sys.executable,
        extraUrl: Optional[str] = None,
        noDeps: bool = False,
        out: Optional[PathLike] = None,
    ) -> subprocess.CompletedProcess[Any]:
        """For now, the `wheel` cmd has no uv support, so use pip"""
        cmd = [str(executable), "-m", "pip", "wheel"]

        if isinstance(reqFile, (str, Path)):
            cmd.extend(["-r", str(reqFile)])
        elif isinstance(reqFile, list):
            for rf in reqFile:
                cmd.extend(["-r", str(rf)])

        if extraUrl is not None:
            cmd.extend(["--extra-index-url", extraUrl])

        if noDeps:
            cmd.append("--no-deps")

        if out is not None:
            cmd.extend(["-w", str(out)])

        return _check_call(cmd, cwd)

    @staticmethod
    def Resolve_Gpu(gpu: Union[GPU_OPTION, None]):
        if gpu is None:
            try:
                tver = metadata.version("torch")
                if "+cu" in tver:
                    return GPU_OPTION.NVIDIA
                elif "+rocm" in tver:
                    return GPU_OPTION.AMD
                else:
                    return None
            except metadata.PackageNotFoundError:
                return None
        else:
            return gpu

    def __init__(
        self,
        cwd: PathLike = ".",
        executable: PathLike = sys.executable,
        gpu: Union[GPU_OPTION, None] = None,
        outDir: PathLike = ".",
        outName: str = "requirements.compiled",
        reqFilesCore: Optional[list[PathLike]] = None,
        reqFilesExt: Optional[list[PathLike]] = None,
        extraSpecs: Optional[list[str]] = None,
    ):
        """Compiler/installer of Python dependencies based on uv

        Args:
            cwd (PathLike): should generally be a comfy workspace dir. Dir that is searched for dependency specification files, and where subprocesses are run in
            executable (PathLike): path to Python executable used to run uv and other subprocesses
            gpu (Union[GPU_OPTION, None]): the gpu against which pytorch and any related dependencies should be built against
            outDir (PathLike): the directory in which to create any output from the compiler itself
            outName (str): the name of the output file containing the compiled requirements
            reqFilesCore (Optional[list[PathLike]]): list of core requirement files (requirements.txt, pyproject.toml, etc) to be included in the compilation. Any requirements determined from these files will override all other requirements
            reqFilesExt (Optional[list[PathLike]]): list of requirement files (requirements.txt, pyproject.toml, etc) to be included in the compilation
            extraSpecs (Optional[list[str]]): list of extra Python requirement specifiers to be included in the compilation
        """
        self.cwd = Path(cwd).expanduser().resolve()
        self.outDir: Path = Path(outDir).expanduser().resolve()
        # use .absolute since .resolve breaks the softlink-is-interpreter assumption of venvs
        self.executable = Path(executable).expanduser().absolute()
        self.gpu = DependencyCompiler.Resolve_Gpu(gpu)
        self.reqFiles = [Path(reqFile) for reqFile in reqFilesExt] if reqFilesExt is not None else None
        self.extraSpecs = [] if extraSpecs is None else extraSpecs

        self.gpuUrl = (
            DependencyCompiler.nvidiaPytorchUrl if self.gpu == GPU_OPTION.NVIDIA else
            DependencyCompiler.rocmPytorchUrl if self.gpu == GPU_OPTION.AMD else
            None
        )  # fmt: skip
        self.out: Path = self.outDir / outName
        self.override = self.outDir / "override.txt"

        self.reqFilesCore = reqFilesCore if reqFilesCore is not None else self.find_core_reqs()
        self.reqFilesExt = reqFilesExt if reqFilesExt is not None else self.find_ext_reqs()

    def find_core_reqs(self):
        return DependencyCompiler.Find_Req_Files(self.cwd)

    def find_ext_reqs(self):
        extDirs = [d for d in (self.cwd / "custom_nodes").iterdir() if d.is_dir() and d.name != "__pycache__"]
        return DependencyCompiler.Find_Req_Files(*extDirs)

    def make_override(self):
        # clean up
        self.override.unlink(missing_ok=True)

        with open(self.override, "w") as f:
            if self.gpu is not None and self.gpuUrl is not None:
                f.write(DependencyCompiler.overrideGpu.format(gpu=self.gpu, gpuUrl=self.gpuUrl))
                f.write("\n\n")

        completed = DependencyCompiler.Compile(
            cwd=self.cwd,
            reqFiles=self.reqFilesCore,
            emit_index_annotation=False,
            emit_index_url=False,
            executable=self.executable,
            override=self.override,
        )

        with open(self.override, "a") as f:
            f.write("# ensure that core comfyui deps take precedence over any 3rd party extension deps\n")
            for line in completed.stdout:
                f.write(line)
            f.write("\n")

    def compile_core_plus_ext(self):
        reqExtras = self.outDir / "requirements.extra"
        # clean up
        reqExtras.unlink(missing_ok=True)
        self.out.unlink(missing_ok=True)

        # make the extra specs file
        if self.extraSpecs:
            with reqExtras.open("w") as f:
                for spec in self.extraSpecs:
                    f.write(spec)
                f.write("\n")

        while True:
            try:
                DependencyCompiler.Compile(
                    cwd=self.cwd,
                    reqFiles=self.reqFilesCore + self.reqFilesExt + ([reqExtras] if self.extraSpecs else []),
                    executable=self.executable,
                    override=self.override,
                    out=self.out,
                    resolve_strategy="ask",
                )

                break
            except subprocess.CalledProcessError as e:
                if hasattr(e, "req"):
                    with open(self.override, "a") as f:
                        f.write(e.req + "\n")
                else:
                    raise AttributeError

    def handle_opencv(self):
        """as per the opencv docs, you should only have exactly one opencv package.
        headless is more suitable for comfy than the gui version, so remove gui if
        headless is present. TODO: add support for contrib pkgs. see: https://github.com/opencv/opencv-python"""

        with open(self.out, "r") as f:
            lines = f.readlines()

        guiFound, headlessFound = False, False
        for line in lines:
            if "opencv-python==" in line:
                guiFound = True
            elif "opencv-python-headless==" in line:
                headlessFound = True

        if headlessFound and guiFound:
            with open(self.out, "w") as f:
                for line in lines:
                    if "opencv-python==" not in line:
                        f.write(line)

    def compile_deps(self):
        self.make_override()
        self.compile_core_plus_ext()
        self.handle_opencv()

    def install_deps(self):
        DependencyCompiler.Install(
            cwd=self.cwd,
            reqFile=[self.out],
            executable=self.executable,
            extra_index_url=self.gpuUrl,
            override=self.override,
        )

    def install_dists(self):
        DependencyCompiler.Install(
            cwd=self.cwd,
            reqFile=[self.out],
            executable=self.executable,
            find_links=[self.outDir / "dists"],
            no_deps=True,
            no_index=True,
        )

    def install_wheels(self):
        DependencyCompiler.Install(
            cwd=self.cwd,
            executable=self.executable,
            find_links=[self.outDir / "wheels"],
            no_deps=True,
            no_index=True,
            reqFile=[self.out],
        )

    def install_wheels_directly(self):
        DependencyCompiler.Install(
            cwd=self.cwd,
            executable=self.executable,
            no_deps=True,
            no_index=True,
            reqs=(self.outDir / "wheels").glob("*.whl"),
        )

    def sync_core_plus_ext(self):
        DependencyCompiler.Sync(
            cwd=self.cwd,
            reqFile=[self.out],
            executable=self.executable,
            extraUrl=self.gpuUrl,
        )

    def fetch_dep_dists(self):
        DependencyCompiler.Download(
            cwd=self.cwd,
            reqFile=[self.out],
            executable=self.executable,
            extraUrl=self.gpuUrl,
            noDeps=True,
            out=self.outDir / "dists",
        )

    def fetch_dep_wheels(self):
        DependencyCompiler.Wheel(
            cwd=self.cwd,
            reqFile=[self.out],
            executable=self.executable,
            extraUrl=self.gpuUrl,
            noDeps=True,
            out=self.outDir / "wheels",
        )
