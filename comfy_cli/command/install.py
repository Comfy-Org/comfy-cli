import bisect
import os
import platform
import re
import subprocess
import sys
from typing import TypedDict
from urllib.parse import urlparse

import git
import requests
import semver
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from comfy_cli import constants, ui
from comfy_cli.command.custom_nodes.command import update_node_id_cache
from comfy_cli.command.github.pr_info import PRInfo
from comfy_cli.constants import GPU_OPTION
from comfy_cli.cuda_detect import DEFAULT_CUDA_TAG
from comfy_cli.git_utils import checkout_pr, git_checkout_tag
from comfy_cli.output import rprint
from comfy_cli.resolve_python import ensure_workspace_python
from comfy_cli.uv import DependencyCompiler, ensure_pip
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

workspace_manager = WorkspaceManager()
console = Console()


def _pip_install_torch(python: str, index_args: list[str]) -> subprocess.CompletedProcess:
    """Install torch, torchvision, and torchaudio with the given index arguments."""
    return subprocess.run(
        [python, "-m", "pip", "install", "torch", "torchvision", "torchaudio"] + index_args,
        check=False,
    )


def pip_install_comfyui_dependencies(
    repo_dir,
    gpu: GPU_OPTION | None,
    plat: constants.OS,
    cuda_version: constants.CUDAVersion | None,
    skip_torch_or_directml: bool,
    skip_requirement: bool,
    python: str = sys.executable,
    rocm_version: constants.ROCmVersion = constants.ROCmVersion.v7_2,
    cuda_tag: str | None = None,
):
    os.chdir(repo_dir)

    result = None
    if not skip_torch_or_directml:
        # install torch for AMD Linux
        if gpu == GPU_OPTION.AMD and plat == constants.OS.LINUX:
            result = _pip_install_torch(
                python, ["--index-url", f"https://download.pytorch.org/whl/rocm{rocm_version.value}"]
            )

        # install torch for NVIDIA
        if gpu == GPU_OPTION.NVIDIA:
            if cuda_tag is None:
                cuda_tag = f"cu{cuda_version.value.replace('.', '')}" if cuda_version else DEFAULT_CUDA_TAG
            result = _pip_install_torch(python, ["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"])

        # install torch for Intel Arc GPUs (upstream torch xpu)
        # https://github.com/comfyanonymous/ComfyUI/pull/7767
        if gpu == GPU_OPTION.INTEL_ARC:
            result = _pip_install_torch(python, ["--extra-index-url", "https://download.pytorch.org/whl/xpu"])

        # install torch for CPU
        if gpu is None:
            result = _pip_install_torch(python, ["--extra-index-url", "https://download.pytorch.org/whl/cpu"])

        if result and result.returncode != 0:
            rprint("Failed to install PyTorch dependencies. Please check your environment (`comfy env`) and try again")
            raise typer.Exit(code=1)

        # install directml for AMD windows
        if gpu == GPU_OPTION.AMD and plat == constants.OS.WINDOWS:
            subprocess.run([python, "-m", "pip", "install", "torch-directml"], check=True)

        # install torch for Mac M Series
        if gpu == GPU_OPTION.MAC_M_SERIES:
            subprocess.run(
                [
                    python,
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
    result = subprocess.run([python, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
    if result.returncode != 0:
        rprint("Failed to install ComfyUI dependencies. Please check your environment (`comfy env`) and try again.")
        raise typer.Exit(code=1)


def pip_install_manager(repo_dir, python=sys.executable):
    """Install ComfyUI-Manager via manager_requirements.txt."""
    from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli, find_legacy_manager_clone

    manager_req_path = os.path.join(repo_dir, constants.MANAGER_REQUIREMENTS_FILE)
    if not os.path.exists(manager_req_path):
        rprint(
            f"[bold yellow]Warning: {constants.MANAGER_REQUIREMENTS_FILE} not found. "
            "Skipping manager installation (older ComfyUI version?).[/bold yellow]"
        )
        return False
    result = subprocess.run(
        [python, "-m", "pip", "install", "-r", constants.MANAGER_REQUIREMENTS_FILE],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        rprint("[bold red]Failed to install ComfyUI-Manager.[/bold red]")
        if result.stderr:
            rprint(f"[dim]{result.stderr.strip()}[/dim]")
        return False

    # Clear caches so manager detection picks up the newly installed module
    find_cm_cli.cache_clear()
    find_legacy_manager_clone.cache_clear()
    return True


def _install_manager_with_fallback(repo_dir, python, *, bootstrap_pip: bool):
    """Install ComfyUI-Manager, degrading gracefully when it fails.

    On failure, disable the manager GUI mode so a later ``comfy launch`` doesn't
    inject manager flags for a manager that isn't actually installed.

    ``bootstrap_pip`` bootstraps pip first (no-op if already present): the
    fast_deps path leaves a uv-managed venv that may ship no pip, whereas the
    pip path has already bootstrapped it earlier in ``execute``.
    """
    if bootstrap_pip:
        ensure_pip(python)
    if not pip_install_manager(repo_dir, python=python):
        # Manager installation failed - disable to prevent launch issues
        from comfy_cli.config_manager import ConfigManager

        ConfigManager().set(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")
        rprint("[yellow]Manager not installed. Launch will run without manager flags.[/yellow]")


def execute(
    url: str,
    comfy_path: str,
    restore: bool,
    skip_manager: bool,
    version: str,
    commit: str | None = None,
    gpu: constants.GPU_OPTION | None = None,
    cuda_version: constants.CUDAVersion | None = None,
    cuda_tag: str | None = None,
    rocm_version: constants.ROCmVersion = constants.ROCmVersion.v7_2,
    plat: constants.OS = None,
    skip_torch_or_directml: bool = False,
    skip_requirement: bool = False,
    fast_deps: bool = False,
    pr: str | None = None,
):
    """Install ComfyUI from a given URL."""
    # Install ComfyUI from a given PR reference.
    if pr:
        url = handle_pr_checkout(pr, comfy_path)
        version = "nightly"

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
            checkout_stable_comfyui(version=version, repo_dir=repo_dir, url=url)
        except GitHubRateLimitError as e:
            rprint(f"[bold red]Error checking out ComfyUI version: {e}[/bold red]")
            raise typer.Exit(code=1) from e

    elif not check_comfy_repo(repo_dir)[0]:
        # Get actual remote URL for better error message
        try:
            repo = git.Repo(repo_dir)
            remote_urls = [r.url for r in repo.remotes]
            rprint(
                f"[bold red]'{repo_dir}' exists but its remote URL is not a recognized ComfyUI repository.[/bold red]"
            )
            if remote_urls:
                rprint(f"[yellow]Found remotes: {', '.join(remote_urls)}[/yellow]")
            rprint("[yellow]Recognized sources: Comfy-Org, comfyanonymous, drip-art, ltdrdata[/yellow]")
        except git.InvalidGitRepositoryError:
            rprint(f"[bold red]'{repo_dir}' exists but is not a valid git repository.[/bold red]")
        except Exception:
            rprint(
                f"[bold red]'{repo_dir}' already exists. But it is an invalid ComfyUI repository. Remove it and retry.[/bold red]"
            )
        raise typer.Exit(code=1)

    # checkout specified commit
    if commit is not None:
        os.chdir(repo_dir)
        subprocess.run(["git", "checkout", commit], check=True)

    python = ensure_workspace_python(repo_dir)
    rprint(f"Using Python: [bold]{python}[/bold]")

    if not fast_deps:
        # The pip path needs pip; a uv-managed workspace venv may not ship it.
        # Bootstrap it first (no-op if present) so the installs below don't crash.
        ensure_pip(python)
        pip_install_comfyui_dependencies(
            repo_dir,
            gpu,
            plat,
            cuda_version,
            skip_torch_or_directml,
            skip_requirement,
            python=python,
            rocm_version=rocm_version,
            cuda_tag=cuda_tag,
        )

    WorkspaceManager().set_recent_workspace(repo_dir)
    workspace_manager.setup_workspace_manager(specified_workspace=repo_dir)

    rprint("")

    # install ComfyUI-Manager
    if skip_manager:
        rprint("Skipping installation of ComfyUI-Manager. (by --skip-manager)")
        # Save to config so launch doesn't inject --enable-manager
        from comfy_cli.config_manager import ConfigManager

        ConfigManager().set(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")
    else:
        rprint("\nInstalling ComfyUI-Manager..")
        if not fast_deps:
            # pip was already bootstrapped above for the pip path.
            _install_manager_with_fallback(repo_dir, python, bootstrap_pip=False)

    if fast_deps:
        if python != sys.executable:
            # Workspace venv needs uv bootstrapped; for the global Python
            # uv is already available as a comfy-cli dependency.
            DependencyCompiler.Install_Build_Deps(executable=python)
        if cuda_tag:
            # DependencyCompiler expects a dotted version like "13.0", not a tag like "cu130"
            digits = cuda_tag[2:]
            resolved_cuda = f"{digits[:2]}.{digits[2:]}"
        elif cuda_version:
            resolved_cuda = cuda_version.value
        else:
            resolved_cuda = None
        depComp = DependencyCompiler(
            cwd=repo_dir,
            executable=python,
            gpu=gpu,
            cuda_version=resolved_cuda,
            rocm_version=rocm_version.value,
            skip_torch=skip_torch_or_directml,
        )
        depComp.compile_deps()
        depComp.install_deps()
        # Install manager separately (not included in DependencyCompiler).
        # fast_deps leaves a uv-managed venv that may have no pip, but the
        # manager install uses pip — the helper bootstraps it first.
        if not skip_manager:
            _install_manager_with_fallback(repo_dir, python, bootstrap_pip=True)

    if not skip_manager:
        try:
            update_node_id_cache()
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
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


def validate_optional_version(version: str | None) -> str | None:
    """Typer callback for an *optional* ``--version`` flag.

    ``validate_version`` is written for a flag that always has a value (``comfy
    install --version`` defaults to ``nightly``). ``comfy update`` treats the
    flag as opt-in, so ``None`` must pass through untouched. Invalid input is
    re-raised as ``typer.BadParameter`` so a headless caller gets the standard
    CLI usage error instead of a traceback.
    """
    if version is None:
        return None
    try:
        return validate_version(version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


class GitHubRateLimitError(Exception):
    """Raised when GitHub API rate limit is exceeded"""


def handle_github_rate_limit(response):
    # Check rate limit headers
    remaining = int(response.headers.get("x-ratelimit-remaining", 0))
    if remaining == 0:
        reset_time = int(response.headers.get("x-ratelimit-reset", 0))
        message = f"Primary rate limit from Github exceeded! Please retry after: {reset_time}"
        raise GitHubRateLimitError(message)

    if "retry-after" in response.headers:
        wait_seconds = int(response.headers["retry-after"])
        message = f"Rate limit from Github exceeded! Please wait {wait_seconds} seconds before retrying."
        rprint(f"[yellow]{message}[/yellow]")
        raise GitHubRateLimitError(message)


def _github_get(url: str, *, params: dict | None = None, timeout: int) -> dict | list:
    """GET a GitHub REST API URL with optional GITHUB_TOKEN auth.

    SECURITY: must only ever be called with https://api.github.com/ URLs —
    it attaches the user's GITHUB_TOKEN as a Bearer header. Keep module-private.
    Raises GitHubRateLimitError on 403/429 rate limits, requests.HTTPError on
    other non-2xx, requests.RequestException on network errors.
    """
    if not url.startswith("https://api.github.com/"):
        raise ValueError(f"_github_get only accepts api.github.com URLs, got: {url}")

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(url, headers=headers, params=params, timeout=timeout)

    if response.status_code in (403, 429):
        handle_github_rate_limit(response)

    response.raise_for_status()
    return response.json()


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


def _resolve_latest_tag_from_local(repo_dir: str) -> tuple[str | None, bool]:
    """Pick the highest stable semver tag from the local clone.

    Returns ``(tag, fetch_ok)``:
    - ``tag``: the tag string (e.g. ``"v0.20.1"``), or ``None`` when no stable
      semver tag is available (or the directory isn't a git repo).
    - ``fetch_ok``: whether ``git fetch --tags`` succeeded. Callers can use this
      to distinguish "no new releases" from "couldn't reach the remote", which
      changes the right messaging when falling back to the API.

    Pre-release tags (e.g. ``v1.2.3-rc1``) are skipped to mirror GitHub's
    ``releases/latest`` behavior. Note that this picks the highest semver tag,
    which may differ from the release a maintainer has manually marked as
    "Latest" on GitHub — acceptable trade-off given the unauthenticated API's
    60 req/hr per-IP cap; users can pin a specific version with ``--version``
    if needed.

    ``git_checkout_tag`` skips its own ``git fetch --tags`` when the resolved
    tag is already present locally, so on the happy path we fetch exactly once
    here. Crucially, that also lets the cached-tag offline path succeed: if
    fetch above fails (``fetch_ok=False``) but a tag is found from disk,
    ``git_checkout_tag`` will not retry the unreachable fetch.
    """
    fetch_ok = False
    try:
        completed = subprocess.run(
            ["git", "-C", repo_dir, "fetch", "--tags", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        fetch_ok = completed.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # Tolerate timeout / OS-level failure; fall through with whatever's on disk.
        pass

    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "tag", "--list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None, fetch_ok

    best: tuple[semver.VersionInfo, str] | None = None
    for line in result.stdout.splitlines():
        tag = line.strip()
        if not tag:
            continue
        try:
            parsed = semver.VersionInfo.parse(tag.lstrip("v"))
        except ValueError:
            continue
        if parsed.prerelease:
            continue
        if best is None or parsed > best[0]:
            best = (parsed, tag)

    return (best[1] if best else None), fetch_ok


_GITHUB_REPO_RE = re.compile(
    # `github.com[:/]<owner>/<repo>` with optional `.git` and optional setuptools-style
    # `@branch` suffix (matching what ``clone_comfyui`` accepts via ``rsplit("@", 1)``).
    # Branch names may contain slashes (`release/1.0`), so the `@<branch>` group is greedy
    # to end-of-string. The repo segment forbids `@` and `/` to avoid eating those parts.
    r"github\.com[/:]([^/\s]+)/([^/@\s]+?)(?:\.git)?(?:@.+)?/?$",
)


def _parse_github_owner_repo(url: str | None) -> tuple[str, str] | None:
    """Parse a GitHub repo URL into ``(owner, repo)``.

    Handles the URL forms ``clone_comfyui`` accepts:
    - ``https://github.com/owner/repo``
    - ``https://github.com/owner/repo.git``
    - ``https://github.com/owner/repo@branch`` (setuptools-style branch suffix)
    - ``git@github.com:owner/repo`` (SSH form)

    Returns ``None`` for empty input, local paths, or non-GitHub URLs (GitLab,
    self-hosted, etc.) — the caller decides what to do with that.
    """
    if not url:
        return None
    match = _GITHUB_REPO_RE.search(url)
    return (match.group(1), match.group(2)) if match else None


def checkout_stable_comfyui(version: str, repo_dir: str, url: str | None = None):
    """
    Supports installing stable releases of Comfy (semantic versioning) or the 'latest' version.

    For ``version="latest"`` we resolve the highest stable semver tag from the
    local clone first to avoid burning the unauthenticated GitHub API budget
    (60 req/hr per IP). The ``releases/latest`` API is only consulted when local
    resolution turns up nothing.

    The optional ``url`` is the install URL forwarded from ``execute``; it lets
    the API fallback query the same repo we cloned from (forks included)
    instead of always asking upstream. Non-GitHub URLs and missing URLs
    fall back to ``comfyanonymous/ComfyUI`` so the prior behavior is preserved
    for users who pass a local path or a non-GitHub remote.
    """
    rprint(f"Looking for ComfyUI version '{version}'...")
    if version == "latest":
        tag, fetch_ok = _resolve_latest_tag_from_local(repo_dir)
        if tag is None:
            if not fetch_ok:
                rprint(
                    "[yellow]Could not refresh tags from the remote (offline or auth failure); "
                    "trying GitHub API as a last resort.[/yellow]"
                )
            else:
                rprint("[yellow]No stable release tags found locally; querying GitHub API.[/yellow]")
            owner, repo = _parse_github_owner_repo(url) or ("comfyanonymous", "ComfyUI")
            selected_release = get_latest_release(owner, repo)
            if selected_release is None:
                rprint(f"Error: No release found for version '{version}'.")
                raise typer.Exit(code=1)
            tag = str(selected_release["tag"])
        elif not fetch_ok:
            # Tag list comes from a cached state — flag it so the user knows
            # they may not be on the actual newest release.
            rprint(
                f"[yellow]Warning: could not refresh tags from remote; "
                f"using cached tag {tag}. Re-run with network access to get the newest release.[/yellow]"
            )
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
            raise typer.Exit(code=1)


class VersionSwitchError(Exception):
    """A failure while moving an existing workspace to another ComfyUI version.

    Carries the stable envelope ``code`` and its ``hint`` so the command layer
    can hand it straight to ``renderer.error`` without re-deriving them.
    ``stash_ref`` is set when a stash was already created — the caller must tell
    the user their work is still recoverable.
    """

    def __init__(self, code: str, message: str, hint: str | None = None, *, stash_ref: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.stash_ref = stash_ref


class VersionSwitchResult(TypedDict):
    """What ``switch_comfyui_version`` did, in envelope-ready form."""

    previous: str
    current: str
    stashed: bool
    stash_ref: str | None


def _git_capture(repo_dir: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command in ``repo_dir`` and capture it, never raising.

    Uses ``git -C`` rather than ``os.chdir`` so a failure part-way through a
    switch can't leave the process in someone else's directory.
    """
    argv = ["git", "-C", repo_dir, *args]
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr=str(exc))


def _git_error_detail(result: subprocess.CompletedProcess) -> str:
    """Best-effort one-line reason from a failed git invocation."""
    for stream in (result.stderr, result.stdout):
        if stream and stream.strip():
            return stream.strip().splitlines()[-1]
    return f"git exited with code {result.returncode}"


def _describe_head(repo_dir: str) -> str:
    """Human-readable name for the current checkout, e.g. ``a1b2c3d (v0.3.0)``."""
    sha_result = _git_capture(repo_dir, "rev-parse", "--short", "HEAD")
    describe_result = _git_capture(repo_dir, "describe", "--tags", "--always")
    sha = sha_result.stdout.strip() if sha_result.returncode == 0 else ""
    described = describe_result.stdout.strip() if describe_result.returncode == 0 else ""
    if sha and described and described != sha:
        return f"{sha} ({described})"
    return sha or described or "unknown"


def _fetch_tags(repo_dir: str) -> bool:
    """Refresh tags from the remote. Returns whether it succeeded.

    Failure is tolerated by callers: a tag that already exists locally is still
    checkout-able offline (mirrors ``git_utils.git_checkout_tag``).
    """
    return _git_capture(repo_dir, "fetch", "--tags", "--quiet", timeout=60).returncode == 0


def _resolve_default_branch(repo_dir: str) -> str:
    """The remote's default branch, falling back to ComfyUI's historical ``master``."""
    result = _git_capture(repo_dir, "symbolic-ref", "refs/remotes/origin/HEAD")
    if result.returncode == 0:
        ref = result.stdout.strip()
        prefix = "refs/remotes/origin/"
        if ref.startswith(prefix):
            branch = ref[len(prefix) :].strip()
            if branch:
                return branch
    return "master"


def _nearby_version_tags(repo_dir: str, version: str, limit: int = 5) -> list[str]:
    """Up to ``limit`` local ``v*`` tags closest (by semver order) to ``version``.

    Used to turn "that version doesn't exist" into something actionable. Tags
    that don't parse as semver are skipped; if the requested version itself
    doesn't parse we just return the newest tags.
    """
    result = _git_capture(repo_dir, "tag", "--list", "v*", timeout=10)
    if result.returncode != 0:
        return []

    parsed: list[tuple[semver.VersionInfo, str]] = []
    for line in result.stdout.splitlines():
        tag = line.strip()
        if not tag:
            continue
        try:
            parsed.append((semver.VersionInfo.parse(tag.lstrip("v")), tag))
        except ValueError:
            continue
    if not parsed:
        return []

    parsed.sort(key=lambda item: item[0])
    try:
        requested = semver.VersionInfo.parse(version.lstrip("v"))
    except ValueError:
        requested = None

    if requested is None:
        window = parsed[-limit:]
    else:
        index = bisect.bisect_left([item[0] for item in parsed], requested)
        start = max(0, min(index - limit // 2, len(parsed) - limit))
        window = parsed[start : start + limit]

    # Newest first — that's the order a user scanning for "what can I pick?" wants.
    return [tag for _, tag in reversed(window)]


def _stash_note(stash_ref: str | None) -> str:
    if not stash_ref:
        return ""
    return (
        f" Your uncommitted changes are still stashed as {stash_ref} "
        "(recover with `git stash list` then `git stash pop`)."
    )


def switch_comfyui_version(
    comfy_path: str,
    version: str,
    *,
    stash: bool = True,
    url: str | None = None,
) -> VersionSwitchResult:
    """Move an existing ComfyUI workspace to ``version``, headlessly.

    ``version`` is the output of ``validate_version``: ``nightly``, ``latest``,
    or a semver string with or without a ``v`` prefix.

    The target is resolved and validated *before* the working tree is touched,
    so an unknown version leaves the workspace exactly as it was. A dirty tree
    is stashed by default (never auto-popped — the stash ref is reported back);
    ``stash=False`` refuses to proceed instead.

    Checking out a tag leaves a detached HEAD, which is expected; ``nightly``
    checks out the remote's default branch and pulls, which is also how a
    previously rolled-back (detached) workspace rolls forward again.

    Installing dependencies for the new version is the caller's job — this
    function only moves the git tree.

    :raises VersionSwitchError: on any resolution, stash, or checkout failure.
    """
    previous = _describe_head(comfy_path)

    # --- 1. Resolve + validate the target before touching anything -----------
    target_tag: str | None = None
    target_branch: str | None = None

    if version == "nightly":
        target_branch = _resolve_default_branch(comfy_path)
        target_label = target_branch
    elif version == "latest":
        # `_resolve_latest_tag_from_local` runs its own `git fetch --tags`.
        target_tag, fetch_ok = _resolve_latest_tag_from_local(comfy_path)
        if target_tag is None:
            owner, repo = _parse_github_owner_repo(url) or ("comfyanonymous", "ComfyUI")
            try:
                release = get_latest_release(owner, repo)
            except GitHubRateLimitError as exc:
                raise VersionSwitchError(
                    code="version_switch_unknown_version",
                    message=f"Could not resolve the latest ComfyUI release: {exc}",
                    hint="retry later, or pin an exact version with `--version <X.Y.Z>`",
                ) from exc
            if release is None:
                detail = (
                    "no local release tags and the remote could not be reached" if not fetch_ok else "no release found"
                )
                raise VersionSwitchError(
                    code="version_switch_unknown_version",
                    message=f"Could not resolve the latest ComfyUI release ({detail}).",
                    hint="check your network, or pin an exact version with `--version <X.Y.Z>`",
                )
            target_tag = str(release["tag"])
        target_label = target_tag
    else:
        fetch_ok = _fetch_tags(comfy_path)
        target_tag = version if version.startswith("v") else f"v{version}"
        if _git_capture(comfy_path, "rev-parse", "--verify", f"refs/tags/{target_tag}").returncode != 0:
            nearby = _nearby_version_tags(comfy_path, version)
            available = f" Nearest available versions: {', '.join(nearby)}." if nearby else ""
            offline = "" if fetch_ok else " (tags could not be refreshed from the remote — you may be offline)"
            raise VersionSwitchError(
                code="version_switch_unknown_version",
                message=f"ComfyUI version '{version}' not found: no tag '{target_tag}' in {comfy_path}{offline}.{available}",
                hint="run `git tag --list 'v*'` in your ComfyUI workspace to see every available version",
            )
        target_label = target_tag

    # --- 2. Stash by default -------------------------------------------------
    status = _git_capture(comfy_path, "status", "--porcelain")
    if status.returncode != 0:
        raise VersionSwitchError(
            code="version_switch_failed",
            message=f"Could not read the git status of {comfy_path}: {_git_error_detail(status)}",
            hint="make sure the workspace is a healthy git repository",
        )

    stash_ref: str | None = None
    if status.stdout.strip():
        if not stash:
            raise VersionSwitchError(
                code="version_switch_dirty_tree",
                message=f"{comfy_path} has uncommitted changes and --no-stash was passed, so nothing was changed.",
                hint="commit or stash your changes, or re-run without --no-stash to stash them automatically",
            )
        stash_result = _git_capture(
            comfy_path, "stash", "push", "-u", "-m", f"comfy-cli: before switch to {target_label}"
        )
        if stash_result.returncode != 0:
            raise VersionSwitchError(
                code="version_switch_failed",
                message=f"Could not stash uncommitted changes in {comfy_path}: {_git_error_detail(stash_result)}",
                hint="commit or discard your changes manually, then re-run",
            )
        ref_result = _git_capture(comfy_path, "rev-parse", "--short", "refs/stash")
        stash_ref = ref_result.stdout.strip() if ref_result.returncode == 0 else None
        stash_ref = f"stash@{{0}} ({stash_ref})" if stash_ref else "stash@{0}"

    # --- 3. Checkout ---------------------------------------------------------
    checkout_target = target_tag if target_tag is not None else target_branch
    checkout = _git_capture(comfy_path, "checkout", checkout_target)
    if checkout.returncode != 0:
        raise VersionSwitchError(
            code="version_switch_failed",
            message=f"Failed to check out '{checkout_target}': {_git_error_detail(checkout)}.{_stash_note(stash_ref)}",
            hint="resolve the git error in your ComfyUI workspace, then re-run",
            stash_ref=stash_ref,
        )

    if target_branch is not None:
        pull = _git_capture(comfy_path, "pull", timeout=300)
        if pull.returncode != 0:
            raise VersionSwitchError(
                code="version_switch_failed",
                message=f"Checked out '{target_branch}' but `git pull` failed: {_git_error_detail(pull)}. "
                f"The workspace is on '{target_branch}' but not up to date.{_stash_note(stash_ref)}",
                hint="fix the git error (network, conflicts) and re-run — the command is safe to repeat",
                stash_ref=stash_ref,
            )

    return VersionSwitchResult(
        previous=previous,
        current=_describe_head(comfy_path),
        stashed=stash_ref is not None,
        stash_ref=stash_ref,
    )


def get_latest_release(repo_owner: str, repo_name: str) -> GithubRelease | None:
    """
    Fetch the latest release information from GitHub API.

    :param repo_owner: The owner of the repository
    :param repo_name: The name of the repository
    :return: A dictionary containing release information, or None if failed
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

    try:
        data = _github_get(url, timeout=5)

        # Forks may use non-semver tags (e.g. "release-2026-04"); the caller
        # only needs the raw tag string for git checkout, so let `version`
        # fall back to None instead of crashing.
        tag_name = data["tag_name"]
        try:
            parsed_version = semver.VersionInfo.parse(tag_name.lstrip("v"))
        except ValueError:
            parsed_version = None

        return GithubRelease(
            tag=tag_name,
            version=parsed_version,
            download_url=data["zipball_url"],
        )

    except requests.RequestException as e:
        rprint(f"Error fetching latest release: {e}")
        return None


def _parse_pr_reference(
    pr_ref: str,
    default_owner: str,
    default_repo: str,
) -> tuple[str, str, int | None]:
    """Parse a GitHub PR reference into (repo_owner, repo_name, pr_number).

    Supported formats:
    - #123                                          → (default_owner, default_repo, 123)
    - username:branch-name                          → (username, default_repo, None)
    - https://github.com/owner/repo/pull/123        → (owner, repo, 123)
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
        return default_owner, default_repo, pr_number

    elif ":" in pr_ref:
        username, branch = pr_ref.split(":", 1)
        return username, default_repo, None

    else:
        raise ValueError(f"Invalid PR reference format: {pr_ref}")


def parse_pr_reference(pr_ref: str) -> tuple[str, str, int | None]:
    return _parse_pr_reference(pr_ref, "comfyanonymous", "ComfyUI")


def _pr_info_from_github(data: dict) -> PRInfo:
    # GitHub returns head.repo/base.repo as null once the PR's source repo has been
    # deleted; indexing into that would raise an opaque TypeError.
    head_repo = data["head"].get("repo")
    base_repo = data["base"].get("repo")
    if head_repo is None or base_repo is None:
        raise ValueError(f"PR #{data['number']} cannot be installed: its source repository has been deleted.")

    # Absent (list endpoint) and null (mergeability still being computed) both mean
    # "unknown"; keep the pre-existing optimistic default rather than showing ✗.
    mergeable = data.get("mergeable")

    return PRInfo(
        number=data["number"],
        head_repo_url=head_repo["clone_url"],
        head_branch=data["head"]["ref"],
        base_repo_url=base_repo["clone_url"],
        base_branch=data["base"]["ref"],
        title=data["title"],
        user=head_repo["owner"]["login"],
        mergeable=True if mergeable is None else mergeable,
    )


def fetch_pr_info(repo_owner: str, repo_name: str, pr_number: int) -> PRInfo:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"

    try:
        data = _github_get(url, timeout=10)
        return _pr_info_from_github(data)

    except requests.RequestException as e:
        raise Exception(f"Failed to fetch PR #{pr_number}: {e}")


def find_pr_by_branch(repo_owner: str, repo_name: str, username: str, branch: str) -> PRInfo | None:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    params = {"head": f"{username}:{branch}", "state": "open"}

    try:
        data = _github_get(url, params=params, timeout=10)

        if data:
            return _pr_info_from_github(data[0])

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
            pr_info = find_pr_by_branch("Comfy-Org", "ComfyUI_frontend", username, branch)

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
    return _parse_pr_reference(pr_ref, "Comfy-Org", "ComfyUI_frontend")
