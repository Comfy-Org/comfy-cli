import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, TypedDict

import requests
import semver
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from comfy_cli import constants, ui, utils
from comfy_cli.command.custom_nodes.command import update_node_id_cache
from comfy_cli.constants import GPU_OPTION
from comfy_cli.git_utils import git_checkout_tag
from comfy_cli.uv import DependencyCompiler
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

workspace_manager = WorkspaceManager()
console = Console()


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
            rprint("Failed to install PyTorch dependencies. Please check your environment (`comfy env`) and try again")
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
        rprint("Failed to install ComfyUI dependencies. Please check your environment (`comfy env`) and try again.")
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
    version: str,
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
    """
    Install ComfyUI from a given URL.
    """
    if not workspace_manager.skip_prompting:
        res = ui.prompt_confirm_action(f"Install from {url} to {comfy_path}?", True)

        if not res:
            rprint("Aborting...")
            raise typer.Exit(code=1)

    rprint(f"Installing from repository [bold yellow]'{url}'[/bold yellow] to '{comfy_path}'")

    repo_dir = comfy_path
    parent_path = os.path.abspath(os.path.join(repo_dir, ".."))

    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    if not os.path.exists(repo_dir):
        clone_comfyui(url=url, repo_dir=repo_dir)

    if version != "nightly":
        checkout_stable_comfyui(version=version, repo_dir=repo_dir)

    elif not check_comfy_repo(repo_dir)[0]:
        rprint(
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

    rprint("")

    # install ComfyUI-Manager
    if skip_manager:
        rprint("Skipping installation of ComfyUI-Manager. (by --skip-manager)")
    else:
        manager_repo_dir = os.path.join(repo_dir, "custom_nodes", "ComfyUI-Manager")

        if os.path.exists(manager_repo_dir):
            if restore and not fast_deps:
                pip_install_manager_dependencies(repo_dir)
            else:
                rprint(
                    f"Directory {manager_repo_dir} already exists. Skipping installation of ComfyUI-Manager.\nIf you want to restore dependencies, add the '--restore' option."
                )
        else:
            rprint("\nInstalling ComfyUI-Manager..")

            if "@" in manager_url:
                # clone specific branch
                manager_url, manager_branch = manager_url.rsplit("@", 1)
                subprocess.run(
                    ["git", "clone", "-b", manager_branch, manager_url, manager_repo_dir],
                    check=True,
                )
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

    rprint("")


def validate_version(version: str) -> Optional[str]:
    """
    Validates the version string as 'latest', 'nightly', or a semantically version number.

    Args:
    version (str): The version string to validate.

    Returns:
    Optional[str]: The validated version string, or None if invalid.

    Raises:
    ValueError: If the version string is invalid.
    """
    if version.lower() in ["nightly", "latest"]:
        return version.lower()

    # Remove 'v' prefix if present
    if version.startswith("v"):
        version = version[1:]

    try:
        semver.VersionInfo.parse(version)
        return version
    except ValueError as exc:
        raise ValueError(
            f"Invalid version format: {version}. "
            "Please use 'nightly', 'latest', or a valid semantic version (e.g., '1.2.3')."
        ) from exc


def fetch_github_releases(repo_owner: str, repo_name: str) -> List[Dict[str, str]]:
    """
    Fetch the list of releases from the GitHub API.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
    response = requests.get(url)
    return response.json()


class GithubRelease(TypedDict):
    """
    A dictionary representing a GitHub release.

    Fields:
    - version: The version number of the release. (Removed the v prefix)
    - tag: The tag name of the release.
    - download_url: The URL to download the release.
    """

    version: Optional[semver.VersionInfo]
    tag: str
    download_url: str


def parse_releases(releases: List[Dict[str, str]]) -> List[GithubRelease]:
    """
    Parse the list of releases fetched from the GitHub API into a list of GithubRelease objects.
    """
    parsed_releases: List[GithubRelease] = []
    for release in releases:
        tag = release["tag_name"]
        if tag.lower() in ["latest", "nightly"]:
            parsed_releases.append({"version": None, "download_url": release["zipball_url"], "tag": tag})
        else:
            version = semver.VersionInfo.parse(tag.lstrip("v"))
            parsed_releases.append({"version": version, "download_url": release["zipball_url"], "tag": tag})

    return parsed_releases


def select_version(releases: List[GithubRelease], version: str) -> Optional[GithubRelease]:
    """
    Given a list of Github releases, select the release that matches the specified version.
    """
    if version.lower() == "latest":
        return next((r for r in releases if r["tag"].lower() == version.lower()), None)

    version = version.lstrip("v")

    try:
        requested_version = semver.VersionInfo.parse(version)
        return next(
            (r for r in releases if isinstance(r["version"], semver.VersionInfo) and r["version"] == requested_version),
            None,
        )
    except ValueError:
        return None


def clone_comfyui(url: str, repo_dir: str):
    """
    Clone the ComfyUI repository from the specified URL.
    """
    if "@" in url:
        # clone specific branch
        url, branch = url.rsplit("@", 1)
        subprocess.run(["git", "clone", "-b", branch, url, repo_dir], check=True)
    else:
        subprocess.run(["git", "clone", url, repo_dir], check=True)


def checkout_stable_comfyui(version: str, repo_dir: str):
    """
    Supports installing stable releases of Comfy (semantic versioning) or the 'latest' version.
    """
    rprint(f"Looking for ComfyUI version '{version}'...")
    selected_release = None
    if version == "latest":
        selected_release = get_latest_release("comfyanonymous", "ComfyUI")
    else:
        releases = fetch_github_releases("comfyanonymous", "ComfyUI")
        parsed_releases = parse_releases(releases)
        selected_release = select_version(parsed_releases, version)

    if selected_release is None:
        rprint(f"Error: No release found for version '{version}'.")
        sys.exit(1)

    tag = str(selected_release["tag"])
    console.print(
        Panel(
            f"Checking out ComfyUI version: [bold cyan]{selected_release['tag']}[/bold cyan]",
            title="[yellow]ComfyUI Checkout[/yellow]",
            border_style="green",
            expand=False,
        )
    )

    with console.status("[bold green]Checking out tag...", spinner="dots"):
        success = git_checkout_tag(repo_dir, tag)
        if not success:
            console.print("\n[bold red]Failed to checkout tag![/bold red]")
            sys.exit(1)


def get_latest_release(repo_owner: str, repo_name: str) -> Optional[GithubRelease]:
    """
    Fetch the latest release information from GitHub API.

    :param repo_owner: The owner of the repository
    :param repo_name: The name of the repository
    :return: A dictionary containing release information, or None if failed
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()

        return GithubRelease(
            tag=data["tag_name"],
            version=semver.VersionInfo.parse(data["tag_name"].lstrip("v")),
            download_url=data["zipball_url"],
        )

    except requests.RequestException as e:
        rprint(f"Error fetching latest release: {e}")
        return None
