import os
import subprocess
from rich import print
import sys


def install_comfyui_dependencies(repo_dir, torch_mode):
    os.chdir(repo_dir)

    # install torch
    if torch_mode == "amd":
        pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/rocm6.0"]
    else:
        pip_url = ["--extra-index-url", "https://download.pytorch.org/whl/cu121"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        + pip_url
    )

    # install other requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


# install requirements for manager
def install_manager_dependencies(repo_dir):
    os.chdir(os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def execute(
    url: str,
    manager_url: str,
    comfy_workspace: str,
    restore: bool,
    skip_manager: bool,
    torch_mode=None,
    commit=None,
    *args,
    **kwargs,
):
    print(f"Installing from {url}")

    # install ComfyUI
    working_dir = os.path.expanduser(comfy_workspace)
    repo_dir = os.path.join(working_dir, os.path.basename(url).replace(".git", ""))
    repo_dir = os.path.abspath(repo_dir)

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
        os.makedirs(working_dir, exist_ok=True)

        repo_dir = os.path.join(working_dir, os.path.basename(url).replace(".git", ""))
        repo_dir = os.path.abspath(repo_dir)
        subprocess.run(["git", "clone", url, repo_dir])

        # checkout specified commit
        if commit is not None:
            os.chdir(repo_dir)
            subprocess.run(["git", "checkout", commit])

        install_comfyui_dependencies(repo_dir, torch_mode)

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
