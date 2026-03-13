import os
import platform
import subprocess
import sys
from typing import TypedDict
from urllib.parse import urlparse

import requests
import semver
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from comfy_cli import constants, ui
from comfy_cli.command.custom_nodes.command import update_node_id_cache
from comfy_cli.command.github.pr_info import PRInfo
from comfy_cli.constants import GPU_OPTION
from comfy_cli.git_utils import checkout_pr, git_checkout_tag
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

            if plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v12_9:
                base_command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu129",
                ]
            elif plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v12_6:
                base_command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu126",
                ]
            elif plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v12_4:
                base_command += [
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            elif plat == constants.OS.WINDOWS and cuda_version == constants.CUDAVersion.v12_1:
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
        # Update installation to use upstream torch xpu. ipex is no longer needed for Intel Arc GPUs
        # https://github.com/comfyanonymous/ComfyUI/pull/7767
        if gpu == GPU_OPTION.INTEL_ARC:
            pip_url = [
                "--extra-index-url",
                "https://download.pytorch.org/whl/xpu",
            ]

            # TODO: wrap pip install in a function
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

        # install torch for CPU
        if gpu is None:  # Currently, when install for CPU, gpu is None
            pip_url = [
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
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
    commit: str | None = None,
    manager_commit: str | None = None,
    gpu: constants.GPU_OPTION = None,
    cuda_version: constants.CUDAVersion = constants.CUDAVersion.v12_6,
    plat: constants.OS = None,
    skip_torch_or_directml: bool = False,
    skip_requirement: bool = False,
    fast_deps: bool = False,
    pr: str | None = None,
    *args,
    **kwargs,
):
    # Install ComfyUI from a given PR reference.
    if pr:
        url = handle_pr_checkout(pr, comfy_path)
        version = "nightly"

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
        try:
            checkout_stable_comfyui(version=version, repo_dir=repo_dir)
        except GitHubRateLimitError as e:
            rprint(f"[bold red]Error checking out ComfyUI version: {e}[/bold red]")
            sys.exit(1)

    elif not check_comfy_repo(repo_dir)[0]:
        rprint(
            f"[bold red]'{repo_dir}' already exists. But it is an invalid ComfyUI repository. Remove it and retry.[/bold red]"
        )
        sys.exit(-1)

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
                if manager_commit is not None:
                    subprocess.run(["git", "checkout", manager_commit], check=True, cwd=manager_repo_dir)

            if not fast_deps:
                pip_install_manager_dependencies(repo_dir)

    if fast_deps:
        depComp = DependencyCompiler(cwd=repo_dir, gpu=gpu)
        depComp.compile_deps()
        depComp.install_deps()

    if not skip_manager:
        try:
            update_node_id_cache()
        except subprocess.CalledProcessError as e:
            rprint(f"Failed to update node id cache: {e}")

    os.chdir(repo_dir)

    rprint("")


def handle_pr_checkout(pr_ref: str, comfy_path: str) -> str:
    try:
        repo_owner, repo_name, pr_number = parse_pr_reference(pr_ref)
    except ValueError as e:
        rprint(f"[bold red]Error parsing PR reference: {e}[/bold red]")
        raise typer.Exit(code=1)

    try:
        if pr_number:
            pr_info = fetch_pr_info(repo_owner, repo_name, pr_number)
        else:
            username, branch = pr_ref.split(":", 1)
            pr_info = find_pr_by_branch("comfyanonymous", "ComfyUI", username, branch)

        if not pr_info:
            rprint(f"[bold red]PR not found: {pr_ref}[/bold red]")
            raise typer.Exit(code=1)

    except Exception as e:
        rprint(f"[bold red]Error fetching PR information: {e}[/bold red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]PR #{pr_info.number}[/bold]: {pr_info.title}\n"
            f"[yellow]Author[/yellow]: {pr_info.user}\n"
            f"[yellow]Branch[/yellow]: {pr_info.head_branch}\n"
            f"[yellow]Source[/yellow]: {pr_info.head_repo_url}\n"
            f"[yellow]Mergeable[/yellow]: {'✓' if pr_info.mergeable else '✗'}",
            title="[bold blue]Pull Request Information[/bold blue]",
            border_style="blue",
        )
    )

    if not workspace_manager.skip_prompting:
        if not ui.prompt_confirm_action(f"Install ComfyUI from PR #{pr_info.number}?", True):
            rprint("Aborting...")
            raise typer.Exit(code=1)

    parent_path = os.path.abspath(os.path.join(comfy_path, ".."))

    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)

    if not os.path.exists(comfy_path):
        rprint(f"Cloning base repository to {comfy_path}...")
        clone_comfyui(url=pr_info.base_repo_url, repo_dir=comfy_path)

    rprint(f"Checking out PR #{pr_info.number}: {pr_info.title}")
    success = checkout_pr(comfy_path, pr_info)
    if not success:
        rprint("[bold red]Failed to checkout PR[/bold red]")
        raise typer.Exit(code=1)

    rprint(f"[bold green]✓ Successfully checked out PR #{pr_info.number}[/bold green]")
    rprint(f"[bold yellow]Note:[/bold yellow] You are now on branch pr-{pr_info.number}")

    return pr_info.base_repo_url


def validate_version(version: str) -> str | None:
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


class GitHubRateLimitError(Exception):
    """Raised when GitHub API rate limit is exceeded"""


def handle_github_rate_limit(response):
    # Check rate limit headers
    remaining = int(response.headers.get("x-ratelimit-remaining", 0))
    if remaining == 0:
        reset_time = int(response.headers.get("x-ratelimit-reset", 0))
        message = f"Primary rate limit from Github exceeded! Please retry after: {reset_time})"
        raise GitHubRateLimitError(message)

    if "retry-after" in response.headers:
        wait_seconds = int(response.headers["retry-after"])
        message = f"Rate limit from Github exceeded! Please wait {wait_seconds} seconds before retrying."
        rprint(f"[yellow]{message}[/yellow]")
        raise GitHubRateLimitError(message)


class GithubRelease(TypedDict):
    """
    A dictionary representing a GitHub release.

    Fields:
    - version: The version number of the release. (Removed the v prefix)
    - tag: The tag name of the release.
    - download_url: The URL to download the release.
    """

    version: semver.VersionInfo | None
    tag: str
    download_url: str


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
    if version == "latest":
        selected_release = get_latest_release("comfyanonymous", "ComfyUI")
        if selected_release is None:
            rprint(f"Error: No release found for version '{version}'.")
            sys.exit(1)
        tag = str(selected_release["tag"])
    else:
        # For specific versions, directly construct the tag (add 'v' prefix if needed)
        tag = f"v{version}" if not version.startswith("v") else version

    console.print(
        Panel(
            f"Checking out ComfyUI version: [bold cyan]{tag}[/bold cyan]",
            title="[yellow]ComfyUI Checkout[/yellow]",
            border_style="green",
            expand=False,
        )
    )

    with console.status("[bold green]Checking out tag...", spinner="dots"):
        success = git_checkout_tag(repo_dir, tag)
        if not success:
            console.print(f"\n[bold red]Failed to checkout tag '{tag}'![/bold red]")
            console.print("[yellow]The version may not exist. Please check available versions.[/yellow]")
            sys.exit(1)


def get_latest_release(repo_owner: str, repo_name: str) -> GithubRelease | None:
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


def parse_pr_reference(pr_ref: str) -> tuple[str, str, int | None]:
    """
    support formats：
    - username:branch-name
    - #123
    - https://github.com/comfyanonymous/ComfyUI/pull/123

    Returns:
        (repo_owner, repo_name, pr_number)
    """
    pr_ref = pr_ref.strip()

    if pr_ref.startswith("https://github.com/"):
        parsed = urlparse(pr_ref)
        if "/pull/" in parsed.path:
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 4:
                repo_owner = path_parts[0]
                repo_name = path_parts[1]
                pr_number = int(path_parts[3])
                return repo_owner, repo_name, pr_number

    elif pr_ref.startswith("#"):
        pr_number = int(pr_ref[1:])
        return "comfyanonymous", "ComfyUI", pr_number

    elif ":" in pr_ref:
        username, branch = pr_ref.split(":", 1)
        return username, "ComfyUI", None

    else:
        raise ValueError(f"Invalid PR reference format: {pr_ref}")


def fetch_pr_info(repo_owner: str, repo_name: str, pr_number: int) -> PRInfo:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response is None:
            raise Exception(f"Failed to fetch PR #{pr_number}: No response from GitHub API")

        if response.status_code in (403, 429):
            handle_github_rate_limit(response)

        response.raise_for_status()
        data = response.json()

        return PRInfo(
            number=data["number"],
            head_repo_url=data["head"]["repo"]["clone_url"],
            head_branch=data["head"]["ref"],
            base_repo_url=data["base"]["repo"]["clone_url"],
            base_branch=data["base"]["ref"],
            title=data["title"],
            user=data["head"]["repo"]["owner"]["login"],
            mergeable=data.get("mergeable", True),
        )

    except requests.RequestException as e:
        raise Exception(f"Failed to fetch PR #{pr_number}: {e}")


def find_pr_by_branch(repo_owner: str, repo_name: str, username: str, branch: str) -> PRInfo | None:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    params = {"head": f"{username}:{branch}", "state": "open"}

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            pr_data = data[0]
            return PRInfo(
                number=pr_data["number"],
                head_repo_url=pr_data["head"]["repo"]["clone_url"],
                head_branch=pr_data["head"]["ref"],
                base_repo_url=pr_data["base"]["repo"]["clone_url"],
                base_branch=pr_data["base"]["ref"],
                title=pr_data["title"],
                user=pr_data["head"]["repo"]["owner"]["login"],
                mergeable=pr_data.get("mergeable", True),
            )

        return None

    except requests.RequestException:
        return None


def _print_npm_not_found_help(node_version: str) -> None:
    """Print detailed help when npm is not found, with OS-specific instructions."""
    rprint("[bold red]npm is not installed or not found in PATH.[/bold red]")
    rprint()
    rprint("[yellow]npm is a package manager that usually comes bundled with Node.js.[/yellow]")
    rprint(f"[yellow]Your system has Node.js ({node_version}) but npm was not found.[/yellow]")
    rprint()

    current_os = platform.system()

    if current_os == "Windows":
        rprint("[bold cyan]How to fix this on Windows:[/bold cyan]")
        rprint()
        rprint("  [bold]Step 1:[/bold] Uninstall your current Node.js installation:")
        rprint("    • Open the Start menu and search for 'Add or remove programs'")
        rprint("    • Find 'Node.js' in the list and click 'Uninstall'")
        rprint()
        rprint("  [bold]Step 2:[/bold] Download and reinstall Node.js:")
        rprint("    • Go to: [link=https://nodejs.org/]https://nodejs.org/[/link]")
        rprint("    • Click the green 'Download Node.js (LTS)' button")
        rprint("    • Run the downloaded installer")
        rprint("    • [bold]Important:[/bold] Use all default options - do not uncheck anything")
        rprint()
        rprint("  [bold]Step 3:[/bold] Restart your terminal:")
        rprint("    • Close this Command Prompt or PowerShell window completely")
        rprint("    • Open a new Command Prompt or PowerShell window")
        rprint()
        rprint("  [bold]Step 4:[/bold] Verify the installation worked:")
        rprint("    • Type: [bold]npm --version[/bold]")
        rprint("    • You should see a version number (e.g., '10.8.0')")
        rprint()

    elif current_os == "Darwin":  # macOS
        rprint("[bold cyan]How to fix this on macOS:[/bold cyan]")
        rprint()
        rprint("  [bold]Option A - Reinstall Node.js (recommended):[/bold]")
        rprint()
        rprint("    [bold]Step 1:[/bold] Download Node.js:")
        rprint("      • Go to: [link=https://nodejs.org/]https://nodejs.org/[/link]")
        rprint("      • Click the green 'Download Node.js (LTS)' button")
        rprint("      • Open the downloaded .pkg file and follow the installer")
        rprint()
        rprint("    [bold]Step 2:[/bold] Restart your terminal:")
        rprint("      • Close this Terminal window completely (Cmd+Q)")
        rprint("      • Open a new Terminal window")
        rprint()
        rprint("  [bold]Option B - If you use Homebrew:[/bold]")
        rprint("    • Run: [bold]brew install node[/bold]")
        rprint("    • Then restart your terminal")
        rprint()
        rprint("  [bold]Verify the installation:[/bold]")
        rprint("    • Type: [bold]npm --version[/bold]")
        rprint("    • You should see a version number (e.g., '10.8.0')")
        rprint()

    else:  # Linux
        rprint("[bold cyan]How to fix this on Linux:[/bold cyan]")
        rprint()
        rprint("  [bold]Option A - Install npm separately (Ubuntu/Debian):[/bold]")
        rprint("    • Run: [bold]sudo apt update && sudo apt install npm[/bold]")
        rprint("    • Enter your password when prompted")
        rprint()
        rprint("  [bold]Option B - Reinstall Node.js with npm:[/bold]")
        rprint()
        rprint("    [bold]Step 1:[/bold] Remove current Node.js:")
        rprint("      • Ubuntu/Debian: [bold]sudo apt remove nodejs[/bold]")
        rprint("      • Fedora: [bold]sudo dnf remove nodejs[/bold]")
        rprint()
        rprint("    [bold]Step 2:[/bold] Install Node.js (includes npm):")
        rprint("      • Go to: [link=https://nodejs.org/]https://nodejs.org/[/link]")
        rprint("      • Or use NodeSource repository for latest version:")
        rprint("        [bold]curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -[/bold]")
        rprint("        [bold]sudo apt install -y nodejs[/bold]")
        rprint()
        rprint("    [bold]Step 3:[/bold] Restart your terminal:")
        rprint("      • Close this terminal window and open a new one")
        rprint()
        rprint("  [bold]Verify the installation:[/bold]")
        rprint("    • Type: [bold]npm --version[/bold]")
        rprint("    • You should see a version number (e.g., '10.8.0')")
        rprint()

    rprint("[dim]After fixing npm, run your comfy command again.[/dim]")
    rprint()


def verify_node_tools() -> bool:
    """Verify that Node.js, npm, and pnpm are available for frontend building"""
    try:
        node_result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        rprint("[bold red]Node.js is not installed or not found in PATH.[/bold red]")
        rprint("[yellow]To use --frontend-pr, please install Node.js first:[/yellow]")
        rprint("  • Download from: https://nodejs.org/")
        rprint("  • Or use a package manager:")
        rprint("    - macOS: brew install node")
        rprint("    - Ubuntu/Debian: sudo apt install nodejs npm")
        rprint("    - Windows: winget install OpenJS.NodeJS")
        return False

    if node_result.returncode != 0:
        rprint("[bold red]Node.js is not installed or not working correctly.[/bold red]")
        rprint("[yellow]To use --frontend-pr, please install Node.js first:[/yellow]")
        rprint("  • Download from: https://nodejs.org/")
        rprint("  • Or use a package manager:")
        rprint("    - macOS: brew install node")
        rprint("    - Ubuntu/Debian: sudo apt install nodejs npm")
        rprint("    - Windows: winget install OpenJS.NodeJS")
        return False

    node_version = (node_result.stdout or node_result.stderr or "").strip()
    if node_version:
        rprint(f"[green]Found Node.js {node_version}[/green]")
    else:
        rprint("[green]Found Node.js[/green]")

    try:
        npm_result = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        _print_npm_not_found_help(node_version)
        return False

    if npm_result.returncode != 0:
        _print_npm_not_found_help(node_version)
        return False

    npm_version = npm_result.stdout.strip()
    if npm_version:
        rprint(f"[green]Found npm {npm_version}[/green]")
    else:
        rprint("[green]Found npm[/green]")

    try:
        pnpm_result = subprocess.run(["pnpm", "--version"], capture_output=True, text=True, check=False)
        if pnpm_result.returncode == 0:
            pnpm_version = pnpm_result.stdout.strip()
            if pnpm_version:
                rprint(f"[green]Found pnpm {pnpm_version}[/green]")
            else:
                rprint("[green]Found pnpm[/green]")
            return True
    except FileNotFoundError:
        pass

    rprint("[yellow]pnpm is not installed but is required for the modern frontend.[/yellow]")

    install_pnpm = Confirm.ask(
        "[bold yellow]Install pnpm automatically using npm?[/bold yellow] (This will run: npm install -g pnpm)"
    )
    if not install_pnpm:
        rprint("[bold red]Cannot build frontend without pnpm.[/bold red]")
        rprint("[yellow]To install manually:[/yellow]")
        rprint("  npm install -g pnpm")
        return False

    rprint("[yellow]Installing pnpm...[/yellow]")
    install_result = subprocess.run(["npm", "install", "-g", "pnpm"], capture_output=True, text=True, check=False)

    if install_result.returncode != 0:
        rprint("[bold red]Failed to install pnpm automatically.[/bold red]")
        rprint(f"[red]Error: {install_result.stderr}[/red]")
        rprint("[yellow]Please install manually: npm install -g pnpm[/yellow]")
        return False

    try:
        verify_result = subprocess.run(["pnpm", "--version"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        rprint("[bold red]pnpm installation succeeded but pnpm was not found on PATH.[/bold red]")
        rprint(
            "[yellow]Try restarting your shell or add npm global bin to PATH, then verify with: pnpm --version[/yellow]"
        )
        return False

    if verify_result.returncode != 0:
        rprint("[bold red]pnpm installation failed to verify.[/bold red]")
        if verify_result.stderr:
            rprint(f"[red]{verify_result.stderr.strip()}[/red]")
        return False

    pnpm_version = verify_result.stdout.strip()
    rprint(f"[green]Successfully installed pnpm {pnpm_version}[/green]")
    return True


def handle_temporary_frontend_pr(frontend_pr: str) -> str | None:
    """Handle temporary frontend PR for launch - returns path to built frontend"""
    from comfy_cli.pr_cache import PRCache

    rprint("\n[bold blue]Preparing frontend PR for launch...[/bold blue]")

    # Verify Node.js tools first
    if not verify_node_tools():
        rprint("[bold red]Cannot build frontend without Node.js and npm[/bold red]")
        return None

    # Parse frontend PR reference
    try:
        repo_owner, repo_name, pr_number = parse_frontend_pr_reference(frontend_pr)
    except ValueError as e:
        rprint(f"[bold red]Error parsing frontend PR reference: {e}[/bold red]")
        return None

    # Fetch PR info
    try:
        if pr_number:
            pr_info = fetch_pr_info(repo_owner, repo_name, pr_number)
        else:
            username, branch = frontend_pr.split(":", 1)
            pr_info = find_pr_by_branch(repo_owner, repo_name, username, branch)

        if not pr_info:
            rprint(f"[bold red]Frontend PR not found: {frontend_pr}[/bold red]")
            return None
    except Exception as e:
        rprint(f"[bold red]Error fetching frontend PR information: {e}[/bold red]")
        return None

    # Check cache first
    cache = PRCache()
    cached_path = cache.get_cached_frontend_path(pr_info)
    if cached_path:
        rprint(f"[bold green]Using cached frontend build for PR #{pr_info.number}[/bold green]")
        rprint(f"[bold green]PR #{pr_info.number}: {pr_info.title} by {pr_info.user}[/bold green]")
        return str(cached_path)

    # Need to build - show PR info
    console.print(
        Panel(
            f"[bold]Frontend PR #{pr_info.number}[/bold]: {pr_info.title}\n"
            f"[yellow]Author[/yellow]: {pr_info.user}\n"
            f"[yellow]Branch[/yellow]: {pr_info.head_branch}\n"
            f"[yellow]Source[/yellow]: {pr_info.head_repo_url}",
            title="[bold blue]Building Frontend PR[/bold blue]",
            border_style="blue",
        )
    )

    # Build in cache directory
    cache_path = cache.get_frontend_cache_path(pr_info)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Clone or update repository
    repo_path = cache_path / "repo"
    if not (repo_path / ".git").exists():
        rprint("Cloning frontend repository...")
        clone_comfyui(url=pr_info.base_repo_url, repo_dir=str(repo_path))

    # Checkout PR
    rprint(f"Checking out PR #{pr_info.number}...")
    success = checkout_pr(str(repo_path), pr_info)
    if not success:
        rprint("[bold red]Failed to checkout frontend PR[/bold red]")
        return None

    # Build frontend
    rprint("\n[bold yellow]Building frontend (this may take a moment)...[/bold yellow]")
    original_dir = os.getcwd()
    try:
        os.chdir(repo_path)

        # Run pnpm install
        rprint("Running pnpm install...")
        pnpm_install = subprocess.run(["pnpm", "install"], capture_output=True, text=True, check=False)
        if pnpm_install.returncode != 0:
            rprint(f"[bold red]pnpm install failed:[/bold red]\n{pnpm_install.stderr}")
            return None

        # Build with vite
        rprint("Building with vite...")
        vite_build = subprocess.run(["npx", "vite", "build"], capture_output=True, text=True, check=False)
        if vite_build.returncode != 0:
            rprint(f"[bold red]vite build failed:[/bold red]\n{vite_build.stderr}")
            return None

        # Check if dist exists
        dist_path = repo_path / "dist"
        if dist_path.exists():
            # Save cache info
            cache.save_cache_info(pr_info, cache_path)
            rprint("[bold green]✓ Frontend built and cached successfully[/bold green]")
            rprint(f"[bold green]Using frontend from PR #{pr_info.number}: {pr_info.title}[/bold green]")
            rprint(f"[dim]Cache will expire in {cache.DEFAULT_MAX_CACHE_AGE_DAYS} days[/dim]")
            return str(dist_path)
        else:
            rprint("[bold red]Frontend build completed but dist folder not found[/bold red]")
            return None

    finally:
        os.chdir(original_dir)


def parse_frontend_pr_reference(pr_ref: str) -> tuple[str, str, int | None]:
    """
    Parse frontend PR reference. Similar to parse_pr_reference but defaults to Comfy-Org/ComfyUI_frontend
    """
    pr_ref = pr_ref.strip()

    if pr_ref.startswith("https://github.com/"):
        parsed = urlparse(pr_ref)
        if "/pull/" in parsed.path:
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 4:
                repo_owner = path_parts[0]
                repo_name = path_parts[1]
                pr_number = int(path_parts[3])
                return repo_owner, repo_name, pr_number

    elif pr_ref.startswith("#"):
        pr_number = int(pr_ref[1:])
        return "Comfy-Org", "ComfyUI_frontend", pr_number

    elif ":" in pr_ref:
        username, branch = pr_ref.split(":", 1)
        return "Comfy-Org", "ComfyUI_frontend", None

    else:
        raise ValueError(f"Invalid frontend PR reference format: {pr_ref}")
