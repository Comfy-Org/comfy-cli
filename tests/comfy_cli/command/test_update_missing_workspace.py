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
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli import cmdline
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
        import subprocess

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
