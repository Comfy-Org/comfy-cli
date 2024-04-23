import os
import subprocess
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print
from comfy_cli import env_checker


def execute(url: str, manager_url: str, comfy_workspace: str, skip_manager: bool, torch_mode=None, *args, **kwargs):
    print(f"Installing from {url}")

    checker = env_checker.EnvChecker()

    # install ComfyUI
    if checker.currently_in_comfy_repo:
        print(f"Already in comfy repo. Skipping installation.")
        repo_dir = os.getcwd()
    else:
        working_dir = os.path.expanduser(comfy_workspace)
        repo_dir = os.path.join(working_dir, os.path.basename(url).replace(".git", ""))

        if os.path.exists(os.path.join(repo_dir, '.git')):
            print("ComfyUI is installed already. Skipping installation.")
        else:
            print("\nInstalling ComfyUI..")
            os.makedirs(working_dir, exist_ok=True)

            repo_dir = os.path.join(working_dir, os.path.basename(url).replace(".git", ""))
            subprocess.run(["git", "clone", url, repo_dir])

            os.chdir(repo_dir)
            # install torch
            if torch_mode == 'amd':
                pip_url = ['--extra-index-url', 'https://download.pytorch.org/whl/rocm5.7']
            else:
                pip_url = ['--extra-index-url', 'https://download.pytorch.org/whl/cu121']
            subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"] + pip_url)

            # install other requirements
            subprocess.run(["pip", "install", "-r", "requirements.txt"])

    # install ComfyUI-Manager
    if skip_manager:
        print("Skipping installation of ComfyUI-Manager. (by --skip-manager)")
    else:
        manager_repo_dir = os.path.join(repo_dir, 'custom_nodes', 'ComfyUI-Manager')

        if os.path.exists(manager_repo_dir):
            print(f"Directory {manager_repo_dir} already exists. Skipping installation of ComfyUI-Manager.")
        else:
            print("\nInstalling ComfyUI-Manager..")

            subprocess.run(["git", "clone", manager_url, manager_repo_dir])
            os.chdir(os.path.join(repo_dir, 'custom_nodes', 'ComfyUI-Manager'))

            subprocess.run(["pip", "install", "-r", "requirements.txt"])
            os.chdir(os.path.join('..', '..'))

    checker.config['DEFAULT']['recent_path'] = repo_dir
    checker.write_config()


