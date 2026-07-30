"""Tests for ``comfy update comfy --version X`` — headless version switch/rollback.

Every git call is mocked (``FakeGit`` dispatches on the argv), so nothing here
touches a real repository or the network. The unit-level cases drive
``install.switch_comfyui_version`` directly; the CLI-level cases drive
``cmdline.update`` so the flag wiring, the dependency reinstall, and the exit
codes are covered too.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli import cmdline
from comfy_cli.command import install as install_inner


class FakeGit:
    """A stand-in for ``subprocess.run`` that answers the git calls we make.

    ``comfy_cli.command.install`` and ``comfy_cli.cmdline`` both do a plain
    ``import subprocess``, so patching ``<module>.subprocess.run`` patches the
    one global function for both. This dispatcher therefore has to serve the
    whole command: git calls are simulated below, anything else (the pip
    install) is recorded in ``self.other_calls`` and answered with
    ``self.pip_returncode``.

    Every argv is recorded in ``self.calls`` so a test can assert on what did
    (and, more importantly, did NOT) run.
    """

    def __init__(
        self,
        *,
        tags: tuple[str, ...] = (),
        status: str = "",
        head_sha: str = "abc1234",
        describe: str = "v0.2.9",
        default_branch: str = "master",
        fail: tuple[str, ...] = (),
        pip_returncode: int = 0,
    ):
        self.tags = list(tags)
        self.status = status
        self.head_sha = head_sha
        self.describe = describe
        self.default_branch = default_branch
        self.fail = set(fail)
        self.pip_returncode = pip_returncode
        self.calls: list[list[str]] = []
        self.other_calls: list[tuple[list[str], dict]] = []

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _sub(call: list[str]) -> list[str]:
        """The git subcommand argv, with any ``-C <repo>`` prefix stripped."""
        return call[3:] if call[1:2] == ["-C"] else call[1:]

    def ran(self, *prefix: str) -> bool:
        return self.call_matching(*prefix) is not None

    @property
    def git_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "git"]

    def call_matching(self, *prefix: str) -> list[str] | None:
        for call in self.git_calls:
            if self._sub(call)[: len(prefix)] == list(prefix):
                return call
        return None

    # -- the dispatcher --------------------------------------------------
    def __call__(self, argv, **kwargs):
        argv = list(argv)
        if not argv or argv[0] != "git":
            self.other_calls.append((argv, kwargs))
            return self._done(argv, self.pip_returncode)

        self.calls.append(argv)
        if "fail_all" in self.fail:
            return self._done(argv, 1, stderr="boom")
        sub = self._sub(argv)

        if sub[:1] == ["fetch"]:
            return self._done(argv, 1 if "fetch" in self.fail else 0)
        if sub[:1] == ["symbolic-ref"]:
            if "symbolic-ref" in self.fail:
                return self._done(argv, 1, stderr="ref HEAD is not a symbolic ref")
            return self._done(argv, 0, stdout=f"refs/remotes/origin/{self.default_branch}\n")
        if sub[:2] == ["rev-parse", "--verify"]:
            tag = sub[2].removeprefix("refs/tags/")
            return self._done(argv, 0 if tag in self.tags else 1)
        if sub[:3] == ["rev-parse", "--short", "HEAD"]:
            return self._done(argv, 0, stdout=f"{self.head_sha}\n")
        if sub[:3] == ["rev-parse", "--short", "refs/stash"]:
            return self._done(argv, 0, stdout="stash01\n")
        if sub[:1] == ["describe"]:
            return self._done(argv, 0, stdout=f"{self.describe}\n")
        if sub[:2] == ["tag", "--list"]:
            return self._done(argv, 0, stdout="\n".join(self.tags) + "\n")
        if sub[:2] == ["status", "--porcelain"]:
            return self._done(argv, 0, stdout=self.status)
        if sub[:2] == ["stash", "push"]:
            if "stash" in self.fail:
                return self._done(argv, 1, stderr="cannot stash")
            self.status = ""
            return self._done(argv, 0)
        if sub[:1] == ["checkout"]:
            if "checkout" in self.fail:
                return self._done(argv, 1, stderr=f"pathspec '{sub[1]}' did not match")
            # A successful checkout moves HEAD — mirror that so the "current"
            # ref reported back is not just an echo of "previous".
            self.describe = sub[1]
            self.head_sha = "def5678"
            return self._done(argv, 0)
        if sub[:1] == ["pull"]:
            return self._done(argv, 1 if "pull" in self.fail else 0, stderr="network down")
        raise AssertionError(f"unexpected git call: {argv}")

    @staticmethod
    def _done(argv, returncode, stdout="", stderr=""):
        return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def fake_git():
    git = FakeGit(tags=("v0.2.7", "v0.2.9", "v0.3.0", "v0.3.1"))
    with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
        yield git


def _run_update(tmp_path, **kwargs):
    """Drive ``cmdline.update``; subprocess is already faked by ``fake_git``."""
    with (
        patch.object(cmdline.workspace_manager, "workspace_path", str(tmp_path)),
        patch("comfy_cli.cmdline.resolve_workspace_python", return_value="/resolved/python"),
        patch("comfy_cli.cmdline.ensure_pip"),
        patch("comfy_cli.cmdline.os.chdir"),
        patch("comfy_cli.cmdline.custom_nodes.command.update_node_id_cache"),
    ):
        cmdline.update(target=kwargs.pop("target", "comfy"), **kwargs)


# ---------------------------------------------------------------------------
# (a) happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_semver_checks_out_tag_and_installs_requirements(self, fake_git, tmp_path):
        _run_update(tmp_path, version="0.3.0")

        assert fake_git.call_matching("checkout", "v0.3.0") is not None
        pip_cmd, pip_kwargs = fake_git.other_calls[0]
        assert pip_cmd == ["/resolved/python", "-m", "pip", "install", "-r", "requirements.txt"]
        assert pip_kwargs["cwd"] == str(tmp_path)

    def test_v_prefix_is_accepted_and_not_doubled(self, fake_git, tmp_path):
        result = install_inner.switch_comfyui_version(str(tmp_path), "v0.3.0")

        assert fake_git.call_matching("checkout", "v0.3.0") is not None
        assert result["stashed"] is False
        assert result["stash_ref"] is None
        assert result["previous"] != result["current"]

    def test_torch_is_never_touched(self, fake_git, tmp_path):
        _run_update(tmp_path, version="0.3.0")

        for argv, _ in fake_git.other_calls:
            assert "torch" not in argv

    def test_reports_previous_and_current_refs(self, fake_git, tmp_path):
        result = install_inner.switch_comfyui_version(str(tmp_path), "0.3.0")

        assert result["previous"] == "abc1234 (v0.2.9)"
        assert result["current"] == "def5678 (v0.3.0)"


# ---------------------------------------------------------------------------
# (b) unknown version
# ---------------------------------------------------------------------------


class TestUnknownVersion:
    def test_unknown_version_raises_and_lists_tags(self, fake_git, tmp_path):
        with pytest.raises(install_inner.VersionSwitchError) as exc:
            install_inner.switch_comfyui_version(str(tmp_path), "9.9.9")

        assert exc.value.code == "version_switch_unknown_version"
        assert "v0.3.1" in exc.value.message

    def test_unknown_version_leaves_the_tree_untouched(self, fake_git, tmp_path):
        with pytest.raises(install_inner.VersionSwitchError):
            install_inner.switch_comfyui_version(str(tmp_path), "9.9.9")

        assert not fake_git.ran("checkout")
        assert not fake_git.ran("stash", "push")
        assert not fake_git.ran("pull")

    def test_unknown_version_exits_nonzero_without_installing_deps(self, fake_git, tmp_path):
        with pytest.raises(typer.Exit) as exc:
            _run_update(tmp_path, version="9.9.9")

        assert exc.value.exit_code == 1
        assert not fake_git.ran("checkout")

    def test_dirty_tree_is_not_stashed_when_validation_fails(self, fake_git, tmp_path):
        fake_git.status = " M comfy_extras/foo.py\n"

        with pytest.raises(install_inner.VersionSwitchError):
            install_inner.switch_comfyui_version(str(tmp_path), "9.9.9")

        assert not fake_git.ran("stash", "push")


# ---------------------------------------------------------------------------
# (c) dirty tree / stash behavior
# ---------------------------------------------------------------------------


class TestDirtyTree:
    def test_dirty_tree_stashes_before_checkout(self, fake_git, tmp_path):
        fake_git.status = " M comfy_extras/foo.py\n"

        result = install_inner.switch_comfyui_version(str(tmp_path), "0.3.0")

        stash_call = fake_git.call_matching("stash", "push")
        assert stash_call is not None
        assert "-u" in stash_call
        assert "comfy-cli: before switch to v0.3.0" in stash_call

        stash_index = fake_git.git_calls.index(stash_call)
        checkout_index = fake_git.git_calls.index(fake_git.call_matching("checkout", "v0.3.0"))
        assert stash_index < checkout_index

        assert result["stashed"] is True
        assert "stash@{0}" in result["stash_ref"]

    def test_stash_is_never_popped_or_dropped(self, fake_git, tmp_path):
        fake_git.status = " M comfy_extras/foo.py\n"

        install_inner.switch_comfyui_version(str(tmp_path), "0.3.0")

        assert not fake_git.ran("stash", "pop")
        assert not fake_git.ran("stash", "drop")

    def test_no_stash_refuses_on_a_dirty_tree(self, fake_git, tmp_path):
        fake_git.status = " M comfy_extras/foo.py\n"

        with pytest.raises(install_inner.VersionSwitchError) as exc:
            install_inner.switch_comfyui_version(str(tmp_path), "0.3.0", stash=False)

        assert exc.value.code == "version_switch_dirty_tree"
        assert not fake_git.ran("stash", "push")
        assert not fake_git.ran("checkout")

    def test_no_stash_is_a_noop_on_a_clean_tree(self, fake_git, tmp_path):
        result = install_inner.switch_comfyui_version(str(tmp_path), "0.3.0", stash=False)

        assert result["stashed"] is False
        assert fake_git.call_matching("checkout", "v0.3.0") is not None

    def test_checkout_failure_after_a_stash_says_the_stash_survives(self, tmp_path):
        git = FakeGit(tags=("v0.3.0",), status=" M foo.py\n", fail=("checkout",))
        with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
            with pytest.raises(install_inner.VersionSwitchError) as exc:
                install_inner.switch_comfyui_version(str(tmp_path), "0.3.0")

        assert exc.value.code == "version_switch_failed"
        assert exc.value.stash_ref is not None
        assert "stashed" in exc.value.message


# ---------------------------------------------------------------------------
# (d) nightly roll-forward from a detached HEAD
# ---------------------------------------------------------------------------


class TestNightly:
    def test_nightly_from_detached_head_checks_out_default_branch_and_pulls(self, tmp_path):
        git = FakeGit(tags=("v0.3.0",), describe="v0.3.0", default_branch="master")
        with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
            result = install_inner.switch_comfyui_version(str(tmp_path), "nightly")

        assert git.call_matching("checkout", "master") is not None
        checkout_index = git.git_calls.index(git.call_matching("checkout", "master"))
        pull_index = git.git_calls.index(git.call_matching("pull"))
        assert checkout_index < pull_index
        assert result["current"] == "def5678 (master)"

    def test_nightly_honors_a_non_master_default_branch(self, tmp_path):
        git = FakeGit(default_branch="main")
        with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
            install_inner.switch_comfyui_version(str(tmp_path), "nightly")

        assert git.call_matching("checkout", "main") is not None

    def test_nightly_falls_back_to_master_when_origin_head_is_unset(self, tmp_path):
        git = FakeGit(fail=("symbolic-ref",))
        with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
            install_inner.switch_comfyui_version(str(tmp_path), "nightly")

        assert git.call_matching("checkout", "master") is not None

    def test_pull_failure_reports_the_branch_is_checked_out(self, tmp_path):
        git = FakeGit(fail=("pull",))
        with patch("comfy_cli.command.install.subprocess.run", side_effect=git):
            with pytest.raises(install_inner.VersionSwitchError) as exc:
                install_inner.switch_comfyui_version(str(tmp_path), "nightly")

        assert exc.value.code == "version_switch_failed"
        assert "not up to date" in exc.value.message


# ---------------------------------------------------------------------------
# (e) --version with the wrong target
# ---------------------------------------------------------------------------


class TestTargetValidation:
    @pytest.mark.parametrize("target", ["all", "cli"])
    def test_version_is_rejected_for_non_comfy_targets(self, fake_git, target, tmp_path):
        with pytest.raises(typer.Exit) as exc:
            _run_update(tmp_path, target=target, version="0.3.0")

        assert exc.value.exit_code == 1
        assert fake_git.calls == []
        assert fake_git.other_calls == []

    def test_plain_update_still_pulls(self, fake_git, tmp_path):
        """The bare ``comfy update comfy`` path must be untouched by --version."""
        _run_update(tmp_path)

        assert fake_git.git_calls[0] == ["git", "pull"]
        assert fake_git.other_calls[0][0] == [
            "/resolved/python",
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]


# ---------------------------------------------------------------------------
# (f) validate_version rejection surfaces through typer
# ---------------------------------------------------------------------------


class TestVersionValidation:
    def test_none_passes_through(self):
        assert install_inner.validate_optional_version(None) is None

    def test_valid_values_are_normalized(self):
        assert install_inner.validate_optional_version("v0.3.0") == "0.3.0"
        assert install_inner.validate_optional_version("NIGHTLY") == "nightly"

    def test_garbage_becomes_a_bad_parameter(self):
        with pytest.raises(typer.BadParameter):
            install_inner.validate_optional_version("not-a-version")

    def test_garbage_exits_nonzero_through_the_cli(self):
        result = CliRunner().invoke(cmdline.app, ["update", "comfy", "--version", "not-a-version"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# (g) pip failure after a successful checkout
# ---------------------------------------------------------------------------


class TestDependencyFailure:
    def test_pip_failure_exits_nonzero_after_the_checkout_landed(self, fake_git, tmp_path, capsys):
        fake_git.pip_returncode = 1

        with pytest.raises(typer.Exit) as exc:
            _run_update(tmp_path, version="0.3.0")

        assert exc.value.exit_code == 1
        assert fake_git.call_matching("checkout", "v0.3.0") is not None

        # The rendered error panel soft-wraps, so match on words, not phrases.
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "dependencies may be incomplete" in combined
        assert "idempotent" in combined
