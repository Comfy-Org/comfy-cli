import os
import subprocess
from rich import print
import sys
from comfy_cli.workspace_manager import WorkspaceManager


def install_comfyui_dependencies(repo_dir, torch_mode):
    os.chdir(repo_dir)

    # install torch
    if torch_mode == "amd":
        pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/rocm6.0"]
    else:
        pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        + pip_url,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Failed to install PyTorch dependencies. Please check your environment (`comfy env`) and try again"
        )
        sys.exit(1)

    # install other requirements
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
    torch_mode=None,
    commit=None,
    *args,
    **kwargs,
):
    print(f"Installing from '{url}' to '{comfy_path}'")

    repo_dir = comfy_path

    # install ComfyUI
    if os.path.exists(os.path.join(repo_dir, ".git")):
        if restore or commit is not None:
            if commit is not None:
                os.chdir(repo_dir)
                subprocess.run(["git", "checkout", commit])

            install_comfyui_dependencies(repo_dir, torch_mode)
        else:
            print(
                "ComfyUI is installed already. Skipping installation.\nIf you want to restore dependencies, add the '--restore' option."
            )
    else:
        print("\nInstalling ComfyUI..")
        parent_path = os.path.join(repo_dir, "..")

        if not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)

        subprocess.run(["git", "clone", url, repo_dir])

        # checkout specified commit
        if commit is not None:
            os.chdir(repo_dir)
            subprocess.run(["git", "checkout", commit])

        install_comfyui_dependencies(repo_dir, torch_mode)

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

    os.chdir(repo_dir)

    print("")
