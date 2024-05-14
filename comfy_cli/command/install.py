import os
import platform
import subprocess
import sys

from rich import print
import typer

from comfy_cli import constants, ui, utils
from comfy_cli.constants import GPU_OPTION
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo
from comfy_cli.command.custom_nodes.command import update_node_id_cache

workspace_manager = WorkspaceManager()


def get_os_details():
    os_name = platform.system()  # e.g., Linux, Darwin (macOS), Windows
    os_version = platform.release()
    return os_name, os_version


def install_comfyui_dependencies(
    repo_dir,
    gpu: GPU_OPTION,
    plat: constants.OS,
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
            pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
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
            print(
                "Failed to install PyTorch dependencies. Please check your environment (`comfy env`) and try again"
            )
            sys.exit(1)

        # install directml for AMD windows
        if gpu == GPU_OPTION.AMD and plat == constants.OS.WINDOWS:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch-directml"], check=True
            )

        # install torch for Mac M Series
        if gpu == GPU_OPTION.M_SERIES:
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
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False
    )
    if result.returncode != 0:
        print(
            "Failed to install ComfyUI dependencies. Please check your environment (`comfy env`) and try again."
        )
        sys.exit(1)


# install requirements for manager
def install_manager_dependencies(repo_dir):
    os.chdir(os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )


def execute(
    url: str,
    manager_url: str,
    comfy_path: str,
    restore: bool,
    skip_manager: bool,
    commit=None,
    gpu: constants.GPU_OPTION = None,
    plat: constants.OS = None,
    skip_torch_or_directml: bool = False,
    skip_requirement: bool = False,
    *args,
    **kwargs,
):

    if not workspace_manager.skip_prompting:
        res = ui.prompt_confirm_action(f"Install from {url} to {comfy_path}?")

        if not res:
            print("Aborting...")
            raise typer.Exit(code=1)

    print(f"Installing from [bold yellow]'{url}'[/bold yellow] to '{comfy_path}'")

    repo_dir = comfy_path
    parent_path = os.path.join(repo_dir, "..")

    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    if os.path.exists(repo_dir):
        subprocess.run(["git", "clone", url, repo_dir])
    elif not check_comfy_repo(repo_dir)[0]:
        print(
            f"[bold red]'{repo_dir}' already exists. But it is an invalid ComfyUI repository. Remove it and retry.[/bold red]"
        )
        exit(-1)

    # checkout specified commit
    if commit is not None:
        os.chdir(repo_dir)
        subprocess.run(["git", "checkout", commit])

    install_comfyui_dependencies(
        repo_dir, gpu, plat, skip_torch_or_directml, skip_requirement
    )

    WorkspaceManager().set_recent_workspace(repo_dir)

    print("")

    # install ComfyUI-Manager
    if skip_manager:
        print("Skipping installation of ComfyUI-Manager. (by --skip-manager)")
    else:
        manager_repo_dir = os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager")

        if os.path.exists(manager_repo_dir):
            if restore:
                install_manager_dependencies(repo_dir)
            else:
                print(
                    f"Directory {manager_repo_dir} already exists. Skipping installation of ComfyUI-Manager.\nIf you want to restore dependencies, add the '--restore' option."
                )
        else:
            print("\nInstalling ComfyUI-Manager..")

            subprocess.run(["git", "clone", manager_url, manager_repo_dir])
            install_manager_dependencies(repo_dir)

        update_node_id_cache()

    os.chdir(repo_dir)

    print("")
