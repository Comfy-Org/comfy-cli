"""``comfy install`` into an existing folder that isn't a ComfyUI checkout must fail loudly.

Seen on Windows: ``comfy --skip-prompt install --nvidia`` with a leftover,
non-git workspace directory printed "'<path>' exists but is not a valid git
repository." and nothing was installed — yet an agent driving it in ``--json``
mode got no envelope at all (and older releases exited 0). It now emits an
``ok: false`` envelope with a registered code and a hint, and exits 1.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import git
import pytest
import typer
from typer.testing import CliRunner

from comfy_cli import constants
from comfy_cli.cmdline import app, g_exclusivity, g_gpu_exclusivity

runner = CliRunner()


@pytest.fixture
def install_env(tmp_path):
    """Point ``install`` at ``tmp_path/ComfyUI`` and stub every side effect past the target check."""
    g_exclusivity.reset_for_testing()
    g_gpu_exclusivity.reset_for_testing()
    target = tmp_path / "ComfyUI"
    with (
        patch("comfy_cli.cmdline.EnvChecker") as checker,
        patch("comfy_cli.cmdline.workspace_manager.get_workspace_path", return_value=(str(target), None)),
        patch("comfy_cli.cmdline.utils.get_os", return_value=constants.OS.WINDOWS),
        patch("comfy_cli.cmdline._resolve_cuda", return_value=(None, "cu130")),
        patch("comfy_cli.command.install.clone_comfyui") as clone,
        patch("comfy_cli.command.install.checkout_stable_comfyui") as checkout,
        patch("comfy_cli.command.install.ensure_workspace_python") as ensure_python,
    ):
        checker.return_value.python_version = MagicMock(major=3, minor=12)
        yield {"target": target, "clone": clone, "checkout": checkout, "ensure_python": ensure_python}


def _envelope(output: str) -> dict:
    lines = [line for line in output.splitlines() if line.startswith("{")]
    assert lines, f"no JSON envelope on stdout: {output!r}"
    return json.loads(lines[-1])


def _assert_nothing_installed(env):
    env["clone"].assert_not_called()
    env["checkout"].assert_not_called()
    env["ensure_python"].assert_not_called()


def test_non_git_folder_fails_with_error_envelope(install_env):
    target = install_env["target"]
    target.mkdir()
    (target / "leftover.txt").write_text("not comfy")

    result = runner.invoke(app, ["--json", "--skip-prompt", "install", "--nvidia"])

    assert result.exit_code == 1
    env = _envelope(result.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "install_target_not_git_repo"
    assert "not a valid git repository" in env["error"]["message"]
    assert "--workspace" in env["error"]["hint"]
    assert "remove/rename" in env["error"]["hint"]
    assert env["error"]["details"]["path"] == str(target)
    _assert_nothing_installed(install_env)


def test_non_git_folder_fails_in_pretty_mode(install_env):
    install_env["target"].mkdir()

    result = runner.invoke(app, ["--skip-prompt", "install", "--nvidia"])

    assert result.exit_code == 1
    assert "not a valid git repository" in result.output
    assert "ComfyUI is installed at" not in result.output
    _assert_nothing_installed(install_env)


def test_foreign_git_repo_fails_with_not_comfyui(install_env):
    target = install_env["target"]
    repo = git.Repo.init(target)
    repo.create_remote("origin", "https://github.com/someone/not-comfy.git")

    result = runner.invoke(app, ["--json", "--skip-prompt", "install", "--nvidia"])

    assert result.exit_code == 1
    env = _envelope(result.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "install_target_not_comfyui"
    assert env["error"]["details"]["remotes"] == ["https://github.com/someone/not-comfy.git"]
    assert "--workspace" in env["error"]["hint"]
    _assert_nothing_installed(install_env)


def test_remote_credentials_are_redacted_from_error(install_env):
    target = install_env["target"]
    repo = git.Repo.init(target)
    repo.create_remote("origin", "https://someuser:s3cr3t-tok@github.com/someone/not-comfy.git")
    repo.create_remote("mirror", "git@github.com:someone/not-comfy.git")

    result = runner.invoke(app, ["--json", "--skip-prompt", "install", "--nvidia"])

    assert result.exit_code == 1
    env = _envelope(result.stdout)
    assert env["error"]["code"] == "install_target_not_comfyui"
    assert sorted(env["error"]["details"]["remotes"]) == [
        "git@github.com:someone/not-comfy.git",
        "https://***@github.com/someone/not-comfy.git",
    ]
    assert "https://***@github.com/someone/not-comfy.git" in env["error"]["message"]
    assert "s3cr3t-tok" not in result.output
    assert "someuser" not in result.output
    _assert_nothing_installed(install_env)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:tok@host/x", "https://***@host/x"),
        ("https://tok@host:8443/x", "https://***@host:8443/x"),
        ("ssh://git@host/x.git", "ssh://***@host/x.git"),
        ("https://github.com/Comfy-Org/ComfyUI.git", "https://github.com/Comfy-Org/ComfyUI.git"),
        ("git@github.com:owner/repo.git", "git@github.com:owner/repo.git"),
        ("/some/local/path", "/some/local/path"),
    ],
)
def test_redact_remote_url(url, expected):
    from comfy_cli.command.install import _redact_remote_url

    assert _redact_remote_url(url) == expected


@pytest.mark.parametrize("version", ["nightly", "latest"])
def test_non_git_subfolder_of_comfyui_checkout_is_rejected(tmp_path, install_env, capsys, version):
    # check_comfy_repo walks up to a parent ComfyUI checkout, so a leftover
    # non-git folder *inside* one used to pass install.execute's target check
    # (the ``comfy install`` front door already bails out as "already
    # installed", but execute must not rely on its caller for that).
    from comfy_cli.command import install

    parent = git.Repo.init(tmp_path)
    parent.create_remote("origin", "https://github.com/comfyanonymous/ComfyUI")
    target = install_env["target"]
    target.mkdir()

    with patch.object(install.workspace_manager, "skip_prompting", True), pytest.raises(typer.Exit) as exc:
        install.execute(
            url="https://github.com/comfyanonymous/ComfyUI",
            comfy_path=str(target),
            restore=False,
            skip_manager=True,
            version=version,
        )

    assert exc.value.exit_code == 1
    assert "install_target_not_git_repo" in capsys.readouterr().out
    _assert_nothing_installed(install_env)


def test_existing_comfyui_checkout_is_accepted_as_target(install_env):
    target = install_env["target"]
    repo = git.Repo.init(target)
    repo.create_remote("origin", "https://github.com/comfyanonymous/ComfyUI")
    install_env["ensure_python"].side_effect = RuntimeError("past the target check")

    result = runner.invoke(app, ["--json", "--skip-prompt", "install", "--nvidia", "--restore"])

    install_env["ensure_python"].assert_called_once()
    assert "install_target_not" not in result.output


def test_versioned_install_rejects_non_git_folder_before_checkout(install_env):
    # The target check used to live only on the nightly branch; a --version
    # install went on to try a git checkout inside the unrelated folder.
    install_env["target"].mkdir()

    result = runner.invoke(app, ["--json", "--skip-prompt", "install", "--nvidia", "--version", "latest"])

    assert result.exit_code == 1
    assert _envelope(result.stdout)["error"]["code"] == "install_target_not_git_repo"
    _assert_nothing_installed(install_env)
