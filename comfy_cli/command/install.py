import os
import platform
import subprocess
import sys
from typing import Optional

import typer
from rich import print

from comfy_cli import constants, ui, utils
from comfy_cli.command.custom_nodes.command import update_node_id_cache
from comfy_cli.constants import GPU_OPTION
from comfy_cli.uv import DependencyCompiler
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

workspace_manager = WorkspaceManager()


def get_os_details():
    os_name = platform.system()  # e.g., Linux, Darwin (macOS), Windows
    os_version = platform.release()
    return os_name, os_version


def pip_install_comfyui_dependencies(
    repo_dir,
    gpu: GPU_OPTION,
    plat: constants.OS,
    cuda_version: constants.CUDAVersion,
    skip_torch_or_directml: bool,
    skip_requirement: bool,
):
    os.chdir(repo_dir)

    result = None
    if not skip_torch_or_directml:
        # install torch for AMD Linux
        if gpu == GPU_OPTION.AMD and plat == constants.OS.LINUX:
            pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/rocm6.0"]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                ]
                + pip_url,
                check=False,
            )

        # install torch for NVIDIA
        if gpu == GPU_OPTION.NVIDIA:
            base_command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
            ]
            if plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v12_1:
                base_command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu121",
                ]
            elif plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v11_8:
                base_command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu118",
                ]
            result = subprocess.run(
                base_command,
                check=False,
            )
        # Beta support for intel arch based on this PR: https://github.com/comfyanonymous/ComfyUI/pull/3439
        if gpu == GPU_OPTION.INTEL_ARC:
            pip_url = [
                "--extra-index-url",
                "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/",
            ]
            utils.install_conda_package("libuv")
            # TODO: wrap pip install in a function
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "mkl", "mkl-dpcpp"],
                check=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.1.0.post2",
                    "torchvision==0.16.0.post2",
                    "torchaudio==2.1.0.post2",
                    "intel-extension-for-pytorch==2.1.30",
                ]
                + pip_url,
                check=False,
            )
        if result and result.returncode != 0:
            print("Failed to install PyTorch dependencies. Please check your environment (`comfy env`) and try again")
            sys.exit(1)

        # install directml for AMD windows
        if gpu == GPU_OPTION.AMD and plat == constants.OS.WINDOWS:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "torch-directml"], check=True)

        # install torch for Mac M Series
        if gpu == GPU_OPTION.MAC_M_SERIES:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--pre",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/nightly/cpu",
                ],
                check=True,
            )

    # install requirements.txt
    if skip_requirement:
        return
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
    if result.returncode != 0:
        print("Failed to install ComfyUI dependencies. Please check your environment (`comfy env`) and try again.")
        sys.exit(1)


# install requirements for manager
def pip_install_manager_dependencies(repo_dir):
    os.chdir(os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)


def execute(
    url: str,
    manager_url: str,
    comfy_path: str,
    restore: bool,
    skip_manager: bool,
    commit: Optional[str] = None,
    gpu: constants.GPU_OPTION = None,
    cuda_version: constants.CUDAVersion = constants.CUDAVersion.v12_1,
    plat: constants.OS = None,
    skip_torch_or_directml: bool = False,
    skip_requirement: bool = False,
    fast_deps: bool = False,
    *args,
    **kwargs,
):
    if not workspace_manager.skip_prompting:
        res = ui.prompt_confirm_action(f"Install from {url} to {comfy_path}?", True)

        if not res:
            print("Aborting...")
            raise typer.Exit(code=1)

    print(f"Installing from [bold yellow]'{url}'[/bold yellow] to '{comfy_path}'")

    repo_dir = comfy_path
    parent_path = os.path.abspath(os.path.join(repo_dir, ".."))

    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    if not os.path.exists(repo_dir):
        if "@" in url:
            # clone specific branch
            url, branch = url.rsplit("@", 1)
            subprocess.run(["git", "clone", "-b", branch, url, repo_dir], check=True)
        else:
            subprocess.run(["git", "clone", url, repo_dir], check=True)

    elif not check_comfy_repo(repo_dir)[0]:
        print(
            f"[bold red]'{repo_dir}' already exists. But it is an invalid ComfyUI repository. Remove it and retry.[/bold red]"
        )
        exit(-1)

    # checkout specified commit
    if commit is not None:
        os.chdir(repo_dir)
        subprocess.run(["git", "checkout", commit], check=True)

    if not fast_deps:
        pip_install_comfyui_dependencies(repo_dir, gpu, plat, cuda_version, skip_torch_or_directml, skip_requirement)

    WorkspaceManager().set_recent_workspace(repo_dir)
    workspace_manager.setup_workspace_manager(specified_workspace=repo_dir)

    print("")

    # install ComfyUI-Manager
    if skip_manager:
        print("Skipping installation of ComfyUI-Manager. (by --skip-manager)")
    else:
        manager_repo_dir = os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager")

        if os.path.exists(manager_repo_dir):
            if restore and not fast_deps:
                pip_install_manager_dependencies(repo_dir)
            else:
                print(
                    f"Directory {manager_repo_dir} already exists. Skipping installation of ComfyUI-Manager.\nIf you want to restore dependencies, add the '--restore' option."
                )
        else:
            print("\nInstalling ComfyUI-Manager..")

            if "@" in manager_url:
                # clone specific branch
                manager_url, manager_branch = manager_url.rsplit("@", 1)
                subprocess.run(["git", "clone", "-b", manager_branch, manager_url, manager_repo_dir], check=True)
            else:
                subprocess.run(["git", "clone", manager_url, manager_repo_dir], check=True)

            if not fast_deps:
                pip_install_manager_dependencies(repo_dir)

    if fast_deps:
        depComp = DependencyCompiler(cwd=repo_dir, gpu=gpu)
        depComp.compile_deps()
        depComp.install_deps()

    if not skip_manager:
        update_node_id_cache()

    os.chdir(repo_dir)

    print("")
