import os
import subprocess
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print
from comfy import env_checker

def execute(url: str, comfy_workspace: str, *args, **kwargs):
    print(f"Installing from {url}")

    checker = env_checker.EnvChecker()

    if checker.currently_in_comfy_repo:
        print(f"Already in comfy repo. Skipping installation.")
        return

    working_dir = os.path.expanduser("~/comfy")

    if os.path.exists(working_dir):
        print(f"Directory {working_dir} already exists. Skipping creation.")
    else:
        os.makedirs(working_dir)
      
    os.makedirs(working_dir, exist_ok=True)
    
    repo_dir = os.path.join(working_dir, os.path.basename(url).replace(".git", ""))
    subprocess.run(["git", "clone", url, repo_dir])
    
    os.chdir(repo_dir)
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

