"""Regression tests for the ``git`` / ``ffmpeg`` / ``ffprobe`` spawn sites.

Companion to :mod:`tests.comfy_cli.test_safe_exec`, which unit-tests the resolver
itself. Here the contract under test is per *call site*:

* the argv actually handed to :mod:`subprocess` starts with the **resolved
  absolute path**, never the bare name Windows' ``CreateProcess`` would look up
  in the current working directory;
* a binary planted **directly in the CWD** is never executed — the site either
  fails loudly (required binaries) or degrades exactly as it already did when the
  binary was simply absent (tolerant sites and the best-effort previewer).

"Degrades as if absent" has one deliberate exception, covered below:
:func:`comfy_cli.file_utils.list_git_tracked_files`. There ``[]`` means "not a
git repository" and makes :func:`~comfy_cli.file_utils.zip_files` package the
whole directory, so a *refused* git raises rather than silently widening the
archive ``comfy node publish`` uploads.

The sites that ``os.chdir`` into a user-supplied directory before spawning get an
explicit test each, because there the attacker-controlled directory *is* the CWD
by construction rather than by the user happening to be standing in it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

# ``comfy_cli.command`` must be imported before ``comfy_cli.git_utils``: the two
# form a pre-existing import cycle (``git_utils`` → ``command.github.pr_info`` →
# ``command/__init__`` → ``command.install`` → ``git_utils``) that only resolves
# when the ``command`` package is the entry point, which is how the real CLI
# always reaches it. ``# isort: split`` pins that order against the formatter.
from comfy_cli.command import install as install_cmd
from comfy_cli.command import outdated as outdated_cmd
from comfy_cli.command.github.pr_info import PRInfo

# isort: split
from comfy_cli import _safe_exec, file_utils, git_utils

# --- harness ---------------------------------------------------------------


def _plant(directory: Path, name: str) -> Path:
    """Create a stand-in executable called ``name`` inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / name
    binary.write_text("")
    return binary


def _cwd_first_which(system_dir: Path):
    """A ``shutil.which`` that searches the CWD *before* ``$PATH``.

    That is Windows' lookup order (and POSIX's too when ``$PATH`` holds ``.`` or
    an empty entry), and it is what makes CWD planting exploitable. Emulating it
    here lets these tests reproduce the Windows vector on any host: a plant in
    the CWD wins the lookup, and the resolver's job is to refuse it.
    """

    def which(name, *_args, **_kwargs):
        for candidate in (Path(os.getcwd()) / name, system_dir / name):
            if candidate.exists():
                return str(candidate)
        return None

    return which


class _Recorder:
    """Stand-in for ``subprocess.run`` that records argv and never spawns."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.calls: list[list[str]] = []
        self._result = (returncode, stdout, stderr)

    def __call__(self, argv, *_args, **_kwargs):
        self.calls.append(list(argv))
        returncode, stdout, stderr = self._result
        return subprocess.CompletedProcess(args=list(argv), returncode=returncode, stdout=stdout, stderr=stderr)

    @property
    def argv0s(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.fixture
def system_bin(tmp_path):
    """A directory standing in for a normal, trusted absolute ``$PATH`` entry."""
    directory = tmp_path / "usr_bin"
    directory.mkdir()
    return directory


@pytest.fixture
def neutral_cwd(tmp_path, monkeypatch):
    """A CWD with nothing planted in it."""
    directory = tmp_path / "neutral_cwd"
    directory.mkdir()
    monkeypatch.chdir(directory)
    return directory


# --- git_utils: repo-directory spawn sites ----------------------------------


class TestGitUtilsResolvesBeforeChdir:
    """``git_checkout_tag`` / ``checkout_pr`` spawn ``git`` with ``cwd=`` set to a
    caller-supplied repo directory, without ever ``os.chdir``-ing the process
    itself. Both resolve ``git`` against the process's own CWD *before* touching
    ``repo_path`` at all, so a ``git`` planted in the target repo can neither be
    executed nor shadow the legitimate binary into a hard failure."""

    def test_checkout_tag_spawns_resolved_path_not_the_repo_plant(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        repo = tmp_path / "repo"
        planted = _plant(repo, "git")

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(git_utils.subprocess, "run", recorder),
        ):
            assert git_utils.git_checkout_tag(str(repo), "v1.2.3") is True

        assert recorder.calls, "expected git to be invoked"
        assert set(recorder.argv0s) == {str(legit)}
        assert str(planted) not in recorder.argv0s
        assert "git" not in recorder.argv0s  # never the bare name

    def test_checkout_pr_spawns_resolved_path_not_the_repo_plant(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        repo = tmp_path / "repo"
        planted = _plant(repo, "git")
        pr_info = PRInfo(
            number=7,
            head_repo_url="https://github.com/comfy/comfy.git",
            head_branch="feature/x",
            base_repo_url="https://github.com/comfy/comfy.git",
            base_branch="main",
            title="t",
            user="u",
            mergeable=True,
        )

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(git_utils.subprocess, "run", recorder),
        ):
            assert git_utils.checkout_pr(str(repo), pr_info) is True

        assert recorder.calls, "expected git to be invoked"
        assert set(recorder.argv0s) == {str(legit)}
        assert str(planted) not in recorder.argv0s

    def test_checkout_tag_refuses_a_git_planted_in_the_cwd(self, tmp_path, system_bin, monkeypatch):
        """The remaining shape: the *process* CWD is the attacker's directory.

        ``git_checkout_tag`` is documented to return ``False`` on failure and both
        callers rely on that for a clean CLI error, so a refusal has to come back
        as ``False`` rather than escaping as a traceback past them.
        """
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        planted = _plant(attacker_cwd, "git")
        monkeypatch.chdir(attacker_cwd)
        repo = tmp_path / "repo"
        repo.mkdir()

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(git_utils.subprocess, "run", recorder),
        ):
            assert git_utils.git_checkout_tag(str(repo), "v1.2.3") is False

        assert recorder.calls == [], f"planted {planted} must never be executed"

    def test_checkout_tag_restores_cwd_when_git_is_unresolvable(self, tmp_path, system_bin, neutral_cwd):
        """Resolution happens before the chdir, so the failure cannot strand the
        process inside the repo directory."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)):
            assert git_utils.git_checkout_tag(str(repo), "v1.2.3") is False
        assert Path(os.getcwd()).resolve() == neutral_cwd.resolve()

    def test_checkout_pr_returns_false_when_git_is_unresolvable(self, tmp_path, system_bin, neutral_cwd):
        """Same contract for the PR checkout path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        pr_info = PRInfo(
            number=42,
            head_repo_url="https://github.com/comfy/comfy.git",
            head_branch="feature/x",
            base_repo_url="https://github.com/comfy/comfy.git",
            base_branch="main",
            title="t",
            user="u",
            mergeable=True,
        )
        with patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)):
            assert git_utils.checkout_pr(str(repo), pr_info) is False
        assert Path(os.getcwd()).resolve() == neutral_cwd.resolve()

    def test_checkout_tag_refuses_an_option_like_tag(self, tmp_path, system_bin, neutral_cwd):
        """``git checkout <rev>`` has no end-of-options escape, so a tag that git
        would parse as an option (``--upload-pack=<cmd>``) is rejected outright."""
        _plant(system_bin, "git")
        repo = tmp_path / "repo"
        repo.mkdir()

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(git_utils.subprocess, "run", recorder),
        ):
            assert git_utils.git_checkout_tag(str(repo), "--upload-pack=touch /tmp/pwned") is False
        assert recorder.calls == []


# --- tolerant git sites: degrade exactly as a missing git already did -------


class TestTolerantGitSitesDegrade:
    """``BinaryNotFoundError`` subclasses ``FileNotFoundError``, so the sites that
    already swallowed a missing ``git`` keep their existing degradation instead of
    starting to raise."""

    def test_list_git_tracked_files_uses_resolved_path(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        recorded: list[list[str]] = []

        def fake_check_output(argv, **_kwargs):
            recorded.append(list(argv))
            return "a.py\nb.py\n"

        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(file_utils.subprocess, "check_output", fake_check_output),
        ):
            assert file_utils.list_git_tracked_files(str(tmp_path)) == ["a.py", "b.py"]

        assert recorded[0][0] == str(legit)

    def test_list_git_tracked_files_returns_empty_when_git_is_absent(self, tmp_path, neutral_cwd):
        """An *absent* git still degrades to ``[]`` — that has always meant "no
        git answer", and ``zip_files`` reads it as "not a git repository"."""
        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", lambda *_a, **_k: None),
            patch.object(file_utils.subprocess, "check_output", recorder),
        ):
            assert file_utils.list_git_tracked_files(str(tmp_path)) == []
        assert recorder.calls == []

    def test_list_git_tracked_files_raises_for_cwd_planted_git(self, tmp_path, system_bin, monkeypatch):
        """A *refused* git must not be flattened into the same ``[]``.

        ``zip_files`` treats ``[]`` as "not a git repository" and falls back to
        walking the whole directory — which would sweep untracked and gitignored
        files (``.env``, keys, venvs) into the archive ``comfy node publish``
        uploads. Refusing loudly is the only safe answer here.
        """
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        _plant(attacker_cwd, "git")
        monkeypatch.chdir(attacker_cwd)

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(file_utils.subprocess, "check_output", recorder),
            pytest.raises(_safe_exec.BinaryNotFoundError),
        ):
            file_utils.list_git_tracked_files(str(tmp_path))
        assert recorder.calls == []

    def test_zip_files_does_not_walk_everything_when_git_is_refused(self, tmp_path, system_bin, monkeypatch):
        """End-to-end shape of the above: the publish archive is never widened."""
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        _plant(attacker_cwd, "git")
        (attacker_cwd / ".env").write_text("SECRET=hunter2")
        monkeypatch.chdir(attacker_cwd)

        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            pytest.raises(_safe_exec.BinaryNotFoundError),
        ):
            file_utils.zip_files(str(tmp_path / "node.zip"))

    def test_outdated_git_output_uses_resolved_path(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        recorder = _Recorder(stdout="true\n")
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(outdated_cmd.subprocess, "run", recorder),
        ):
            assert outdated_cmd._git_output(["rev-parse", "--is-inside-work-tree"], str(tmp_path)) == "true"
        assert recorder.argv0s == [str(legit)]

    def test_outdated_git_output_returns_none_for_cwd_planted_git(self, tmp_path, system_bin, monkeypatch):
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        _plant(attacker_cwd, "git")
        monkeypatch.chdir(attacker_cwd)

        recorder = _Recorder(stdout="true\n")
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(outdated_cmd.subprocess, "run", recorder),
        ):
            assert outdated_cmd._git_output(["rev-parse"], str(tmp_path)) is None
        assert recorder.calls == []

    def test_install_git_capture_uses_resolved_path(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        recorder = _Recorder(stdout="abc1234\n")
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(install_cmd.subprocess, "run", recorder),
        ):
            result = install_cmd._git_capture(str(tmp_path), "rev-parse", "--short", "HEAD")
        assert result.returncode == 0
        assert recorder.argv0s == [str(legit)]

    def test_install_git_capture_fails_soft_for_cwd_planted_git(self, tmp_path, system_bin, monkeypatch):
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        _plant(attacker_cwd, "git")
        monkeypatch.chdir(attacker_cwd)

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(install_cmd.subprocess, "run", recorder),
        ):
            result = install_cmd._git_capture(str(tmp_path), "rev-parse")
        assert result.returncode == 1
        assert recorder.calls == []

    def test_resolve_latest_tag_from_local_uses_resolved_path(self, tmp_path, system_bin, neutral_cwd):
        legit = _plant(system_bin, "git")
        recorder = _Recorder(stdout="v0.1.0\nv0.2.0\n")
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(install_cmd.subprocess, "run", recorder),
        ):
            tag, fetch_ok = install_cmd._resolve_latest_tag_from_local(str(tmp_path))
        assert (tag, fetch_ok) == ("v0.2.0", True)
        assert set(recorder.argv0s) == {str(legit)}


class TestCloneComfyuiUsesResolvedGit:
    def test_clone_uses_resolved_path(self, system_bin, neutral_cwd, tmp_path):
        legit = _plant(system_bin, "git")
        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(install_cmd.subprocess, "run", recorder),
        ):
            install_cmd.clone_comfyui("https://github.com/comfy/comfy.git", str(tmp_path / "dest"))
        assert recorder.argv0s == [str(legit)]

    def test_clone_refuses_a_git_planted_in_the_cwd(self, tmp_path, system_bin, monkeypatch):
        _plant(system_bin, "git")
        attacker_cwd = tmp_path / "attacker"
        _plant(attacker_cwd, "git")
        monkeypatch.chdir(attacker_cwd)

        recorder = _Recorder()
        with (
            patch.object(_safe_exec.shutil, "which", _cwd_first_which(system_bin)),
            patch.object(install_cmd.subprocess, "run", recorder),
            pytest.raises(_safe_exec.BinaryNotFoundError),
        ):
            install_cmd.clone_comfyui("https://github.com/comfy/comfy.git", str(tmp_path / "dest"))
        assert recorder.calls == []
