import os
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from comfy_cli._safe_exec import BinaryNotFoundError, resolve_required_binary
from comfy_cli.command.github.pr_info import PRInfo

console = Console()


def reject_option_like_ref(ref: str, what: str = "ref") -> None:
    """Refuse a revision that git would parse as an option.

    Git scans for options *before* the first positional and permutes them, so a
    value like ``--upload-pack=<cmd>`` passed where a revision is expected is
    read as an option no matter where a ``--`` separator sits — for ``git
    clone`` the separator works, for ``git checkout <rev>`` it cannot.

    Nothing legitimate is lost: ``git tag`` and ``git branch`` both refuse to
    *create* a name beginning with ``-`` ("'-foo' is not a valid tag name"), and
    a commit SHA never starts with one. So a ``-``-leading value is always either
    a typo or an injection attempt.

    :raises ValueError: ``ref`` begins with ``-``.
    """
    if ref.startswith("-"):
        raise ValueError(f"invalid {what} {ref!r}: git {what}s may not begin with '-'")


def _print_checkout_failure(tag: str, detail: str, stderr: str | None = None) -> None:
    """Render the "Git Checkout Failed" panel for a tag checkout that didn't happen."""
    error_message = Text()
    error_message.append("Git Checkout Error", style="bold red on white")
    error_message.append("\n\nFailed to checkout tag: ", style="bold yellow")
    error_message.append(f"[cyan]{tag}[/cyan]")
    error_message.append("\n\nError details:", style="bold red")
    error_message.append(f"\n{detail}", style="italic")

    if stderr:
        error_message.append("\n\nError output:", style="bold red")
        error_message.append(f"\n{stderr}", style="italic yellow")

    console.print(
        Panel(
            error_message,
            title="[bold white on red]Git Checkout Failed[/bold white on red]",
            border_style="red",
            expand=False,
        )
    )


def sanitize_for_local_branch(branch_name: str) -> str:
    if not branch_name:
        return "unknown"

    sanitized = branch_name.replace("/", "-")

    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")

    sanitized = sanitized.strip("-")

    return sanitized or "unknown"


def git_checkout_tag(repo_path: str, tag: str) -> bool:
    """
    Checkout a specific Git tag in the given repository.

    Skips the network ``git fetch --tags`` when the tag already exists locally.
    This avoids a redundant round-trip on the happy path (the caller usually
    just cloned the repo or just ran a fetch via the resolver) and lets offline
    installs proceed when the tag is already cached. Only when the tag is
    absent locally do we attempt to fetch — and a failed fetch in that case is
    a real, unrecoverable error (``check=True`` surfaces it as before).

    :param repo_path: Path to the Git repository
    :param tag: The tag to checkout
    :return: True if the checkout succeeds, False if any git command failed —
        including git being absent or refused. Both callers
        (``checkout_stable_comfyui``, ``checkout_pr``) turn ``False`` into a
        clean CLI error, so a resolution failure must not escape as a traceback
        past them.
    """
    # Resolved BEFORE the ``os.chdir`` below so the lookup can't see a ``git``
    # planted in the caller-supplied ``repo_path``, and so the absolute path we
    # spawn defeats Windows' current-directory search once we are inside it.
    try:
        reject_option_like_ref(tag, "tag")
        git_bin = resolve_required_binary("git")
    except (BinaryNotFoundError, ValueError) as e:
        _print_checkout_failure(tag, str(e))
        return False
    original_dir = os.getcwd()
    try:
        # Change to the repository directory

        os.chdir(repo_path)

        # Skip the network fetch when the tag is already present locally.
        tag_present_locally = (
            subprocess.run(
                [git_bin, "rev-parse", "--verify", f"refs/tags/{tag}"],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        )
        if not tag_present_locally:
            subprocess.run([git_bin, "fetch", "--tags"], check=True, capture_output=True, text=True)

        # Checkout the specified tag
        subprocess.run([git_bin, "checkout", tag], check=True, capture_output=True, text=True)

        console.print(f"[bold green]Successfully checked out tag: [cyan]{tag}[/cyan][/bold green]")

        return True
    except subprocess.CalledProcessError as e:
        _print_checkout_failure(tag, str(e), stderr=e.stderr)
        return False
    finally:
        # Ensure we always return to the original directory
        os.chdir(original_dir)


def checkout_pr(repo_path: str, pr_info: PRInfo) -> bool:
    """Check out ``pr_info``'s branch in ``repo_path``.

    :return: True on success, False on any git failure — including git being
        absent or refused, which callers handle as a clean CLI error rather than
        a traceback (same contract as :func:`git_checkout_tag`).
    """
    # See ``git_checkout_tag``: resolve before the ``os.chdir`` into ``repo_path``.
    try:
        git_bin = resolve_required_binary("git")
    except BinaryNotFoundError as e:
        _print_pr_checkout_failure(pr_info, str(e))
        return False
    original_dir = os.getcwd()

    try:
        os.chdir(repo_path)

        if pr_info.is_fork:
            remote_name = f"pr-{pr_info.number}-{pr_info.user}"

            result = subprocess.run([git_bin, "remote", "get-url", remote_name], capture_output=True, text=True)

            if result.returncode != 0:
                subprocess.run(
                    [git_bin, "remote", "add", remote_name, pr_info.head_repo_url],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            subprocess.run(
                [git_bin, "fetch", remote_name, pr_info.head_branch], check=True, capture_output=True, text=True
            )

            # fix: "feature/add-support" -> "pr-123-feature-add-support"
            sanitized_branch = sanitize_for_local_branch(pr_info.head_branch)
            local_branch = f"pr-{pr_info.number}-{sanitized_branch}"

            subprocess.run(
                [git_bin, "checkout", "-B", local_branch, f"{remote_name}/{pr_info.head_branch}"],
                check=True,
                capture_output=True,
                text=True,
            )

        else:
            subprocess.run(
                [git_bin, "fetch", "origin", pr_info.head_branch], check=True, capture_output=True, text=True
            )

            sanitized_branch = sanitize_for_local_branch(pr_info.head_branch)
            local_branch = f"pr-{pr_info.number}-{sanitized_branch}"

            subprocess.run(
                [git_bin, "checkout", "-B", local_branch, f"origin/{pr_info.head_branch}"],
                check=True,
                capture_output=True,
                text=True,
            )

        console.print(f"[bold green]Successfully checked out PR #{pr_info.number}: {pr_info.title}[/bold green]")
        console.print(f"[bold yellow]Local branch:[/bold yellow] {local_branch}")
        return True

    except subprocess.CalledProcessError as e:
        _print_pr_checkout_failure(pr_info, stderr=e.stderr)
        return False

    finally:
        os.chdir(original_dir)


def _print_pr_checkout_failure(pr_info: PRInfo, detail: str | None = None, stderr: str | None = None) -> None:
    """Render the "PR Checkout Failed" panel for a PR checkout that didn't happen."""
    error_message = Text()
    error_message.append("Git PR Checkout Error", style="bold red on white")
    error_message.append(f"\n\nFailed to checkout PR #{pr_info.number}", style="bold yellow")
    error_message.append(f"\nTitle: {pr_info.title}", style="italic")
    error_message.append(f"\nBranch: {pr_info.head_branch}", style="italic")

    if detail:
        error_message.append("\n\nError details:", style="bold red")
        error_message.append(f"\n{detail}", style="italic")

    if stderr:
        error_message.append("\n\nError output:", style="bold red")
        error_message.append(f"\n{stderr}", style="italic yellow")

    console.print(
        Panel(
            error_message,
            title="[bold white on red]PR Checkout Failed[/bold white on red]",
            border_style="red",
            expand=False,
        )
    )
