"""Tests for ``comfy update`` when the ComfyUI workspace does not exist.

The default workspace path is resolved from config even when nothing was ever
installed there, so ``workspace_path`` can be a string naming a missing
directory. ``update`` only guarded against ``None`` and then ``os.chdir``-ed
into the path, crashing with a raw ``FileNotFoundError`` traceback — no
envelope under ``--json``, no hint to run ``comfy install``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli import cmdline, constants
from comfy_cli.workspace_manager import WorkspaceType

runner = CliRunner()

SUBPROCESS_RUN = "comfy_cli.cmdline.subprocess.run"
SWITCH_VERSION = "comfy_cli.cmdline._switch_comfy_version"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def _resolve_to(path):
    # The root callback re-runs ``setup_workspace_manager``, which drops any
    # value patched onto ``workspace_path``; pin the resolver underneath instead.
    return patch.object(cmdline.workspace_manager, "get_workspace_path", return_value=(path, WorkspaceType.DEFAULT))


@pytest.fixture
def missing_workspace(tmp_path):
    path = str(tmp_path / "ComfyUI")
    with _resolve_to(path):
        yield path


class TestMissingWorkspace:
    def test_missing_path_is_a_structured_error_in_json_mode(self, missing_workspace):
        with patch(SUBPROCESS_RUN) as mock_run:
            result = runner.invoke(cmdline.app, ["--json", "update"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "not_in_workspace"
        assert missing_workspace in envelope["error"]["message"]
        assert "comfy install" in envelope["error"]["hint"]
        mock_run.assert_not_called()

    def test_missing_path_is_explained_in_pretty_mode(self, missing_workspace):
        with patch(SUBPROCESS_RUN) as mock_run:
            result = runner.invoke(cmdline.app, ["--no-json", "update", "comfy"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "not_in_workspace" in _strip_ansi(result.stdout)
        mock_run.assert_not_called()

    def test_missing_path_blocks_a_version_switch(self, missing_workspace):
        with patch(SWITCH_VERSION) as mock_switch:
            result = runner.invoke(cmdline.app, ["--json", "update", "comfy", "--version", "nightly"])

        assert result.exit_code == 1
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["error"]["code"] == "not_in_workspace"
        mock_switch.assert_not_called()

    def test_unresolved_path_is_a_structured_error(self):
        with _resolve_to(None):
            result = runner.invoke(cmdline.app, ["--json", "update"])

        assert result.exit_code == 1
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["error"]["code"] == "not_in_workspace"
        assert "comfy install" in envelope["error"]["hint"]


class TestRecentWorkspaceResolution:
    @pytest.mark.parametrize("output_flag", ["--json", "--no-json"])
    @pytest.mark.parametrize("target_args", [[], ["comfy"], ["comfy", "--version", "nightly"]])
    def test_missing_recent_has_structured_comfy_failure(self, monkeypatch, output_flag, target_args):
        manager_config = cmdline.workspace_manager.config_manager
        original_get = manager_config.get
        monkeypatch.setattr(
            manager_config,
            "get",
            lambda key: None if key == constants.CONFIG_KEY_RECENT_WORKSPACE else original_get(key),
        )
        with patch(SUBPROCESS_RUN) as installer, patch(SWITCH_VERSION) as switch:
            result = runner.invoke(cmdline.app, [output_flag, "--recent", "update", *target_args])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        if output_flag == "--json":
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            assert len(lines) == 1
            envelope = json.loads(lines[0])
            assert envelope["ok"] is False
            assert envelope["error"]["code"] == "not_in_workspace"
            assert "comfy install" in envelope["error"]["hint"]
        else:
            assert "not_in_workspace" in _strip_ansi(result.stdout)
        installer.assert_not_called()
        switch.assert_not_called()

    @pytest.mark.parametrize("exit_code", [1, 7])
    def test_comfy_resolution_keeps_resolver_exit_status(self, exit_code):
        with (
            patch.object(cmdline.workspace_manager, "get_workspace_path", side_effect=typer.Exit(code=exit_code)),
            patch(SUBPROCESS_RUN) as installer,
        ):
            result = runner.invoke(cmdline.app, ["--json", "update", "comfy"])
        assert result.exit_code == exit_code
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["error"]["code"] == "not_in_workspace"
        installer.assert_not_called()

    @pytest.mark.parametrize("exit_code", [1, 7])
    def test_all_resolution_keeps_existing_exit_without_manager_execution(self, exit_code):
        with (
            patch.object(cmdline.workspace_manager, "get_workspace_path", side_effect=typer.Exit(code=exit_code)),
            patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as manager,
        ):
            result = runner.invoke(cmdline.app, ["--json", "update", "all"])
        assert result.exit_code == exit_code
        assert "not_in_workspace" not in result.stdout
        manager.assert_not_called()


class TestUpdateCliFailures:
    """`comfy update cli` used to let installer failures escape as tracebacks."""

    def test_unavailable_installer_is_a_structured_error(self):
        from comfy_cli import update

        err = update.UpgradeUnavailable("no pip here", hint="run `uv tool upgrade comfy-cli`")
        with patch("comfy_cli.update.upgrade_cli", side_effect=err):
            result = runner.invoke(cmdline.app, ["--json", "update", "cli"])

        assert result.exit_code == 1
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["error"]["code"] == "update_cli_failed"
        assert envelope["error"]["hint"] == "run `uv tool upgrade comfy-cli`"

    def test_failed_installer_is_a_structured_error(self):
        err = subprocess.CalledProcessError(1, ["uv", "tool", "upgrade", "comfy-cli"])
        with patch("comfy_cli.update.upgrade_cli", side_effect=err):
            result = runner.invoke(cmdline.app, ["--json", "update", "cli"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["error"]["code"] == "update_cli_failed"
        assert envelope["error"]["details"]["returncode"] == 1

    @pytest.mark.parametrize("output_flag", ["--json", "--no-json"])
    @pytest.mark.parametrize(
        "installer_error",
        [
            FileNotFoundError("installer disappeared"),
            PermissionError("installer is not executable"),
            subprocess.TimeoutExpired(["uv", "tool", "upgrade", "comfy-cli"], 600),
        ],
        ids=["missing-installer", "installer-permissions", "timeout"],
    )
    def test_installer_start_and_timeout_failures_are_structured(self, installer_error, output_flag):
        with patch("comfy_cli.update.upgrade_cli", side_effect=installer_error):
            result = runner.invoke(cmdline.app, [output_flag, "update", "cli"])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        if output_flag == "--json":
            envelope = json.loads(result.stdout.strip().splitlines()[-1])
            assert envelope["error"]["code"] == "update_cli_failed"
            assert envelope["error"]["hint"]
            if isinstance(installer_error, subprocess.TimeoutExpired):
                assert envelope["error"]["details"]["timeout"] == 600
        else:
            assert "update_cli_failed" in _strip_ansi(result.stdout)


class TestUpdateCliInstallerOutput:
    @pytest.mark.parametrize("installer", ["pip", "uv"])
    @pytest.mark.parametrize("output_flag", ["--json", "--json-stream", "--no-json"])
    @pytest.mark.parametrize("fails", [False, True])
    def test_installer_diagnostics_follow_output_mode(self, tmp_path, monkeypatch, installer, output_flag, fails):
        from comfy_cli import update

        prefix = tmp_path / "tools" / "comfy-cli"
        prefix.mkdir(parents=True)
        if installer == "uv":
            (prefix / "uv-receipt.toml").write_text('[tool]\nrequirements = [{ name = "comfy-cli" }]\n')
        monkeypatch.setattr(update.sys, "prefix", str(prefix))
        monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
        monkeypatch.setattr(update.importlib.util, "find_spec", lambda name: ModuleSpec("pip", loader=None))

        def simulated_installer(argv, **kwargs):
            print("installer diagnostic sentinel", file=kwargs.get("stdout") or sys.stdout)
            if fails:
                raise subprocess.CalledProcessError(3, argv)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(update.subprocess, "run", simulated_installer)
        result = runner.invoke(cmdline.app, [output_flag, "update", "cli"])
        assert result.exit_code == (1 if fails else 0)
        if output_flag == "--no-json":
            assert "installer diagnostic sentinel" in result.stdout
            assert "installer diagnostic sentinel" not in result.stderr
        else:
            assert "installer diagnostic sentinel" not in result.stdout
            assert "installer diagnostic sentinel" in result.stderr
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            assert len(lines) == 1
            envelope = json.loads(lines[0])
            assert envelope["type"] == "envelope"
            assert envelope["ok"] is not fails
            if fails:
                assert envelope["error"]["code"] == "update_cli_failed"
                assert envelope["error"]["details"]["returncode"] == 3
            else:
                assert envelope["data"] == {"target": "cli"}


@pytest.mark.parametrize("output_flag", ["--json", "--json-stream", "--no-json"])
def test_pinned_uv_receipt_is_a_structured_cli_failure(tmp_path, monkeypatch, output_flag):
    from comfy_cli import update

    prefix = tmp_path / "custom tools" / "comfy-cli"
    prefix.mkdir(parents=True)
    path = prefix / "uv-receipt.toml"
    receipt = '[tool]\nrequirements = [{ name = "comfy-cli", specifier = "==1.2.3" }]\n'
    path.write_text(receipt)
    monkeypatch.setattr(update.sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
    with patch("comfy_cli.update.subprocess.run") as installer:
        result = runner.invoke(cmdline.app, [output_flag, "update", "cli"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    if output_flag == "--no-json":
        assert "update_cli_failed" in _strip_ansi(result.stdout)
        assert "reinstall" in _strip_ansi(result.stdout).lower()
    else:
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 1
        error = json.loads(lines[0])["error"]
        assert error["code"] == "update_cli_failed"
        assert "reinstall" in error["hint"].lower()
        assert str(prefix.parent) in error["hint"]
        assert "UV_TOOL_DIR" in error["hint"]
    assert path.read_text() == receipt
    installer.assert_not_called()
