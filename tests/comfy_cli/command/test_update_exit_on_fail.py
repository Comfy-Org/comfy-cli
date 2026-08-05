"""Tests for ``comfy update all --exit-on-fail``.

``comfy update all`` shells out to ``cm-cli update all`` via
``execute_cm_cli``, which — with ``raise_on_error`` off — swallows a
``CalledProcessError`` with returncode 1 (prints to stderr, returns ``None``).
The command therefore exited 0 even when pack updates failed, so a wrapper
could report a partial failure as a clean success.

``--exit-on-fail`` opts into the same plumbing ``comfy node install`` already
has: ``raise_on_error=True`` into ``execute_cm_cli`` and a re-raise as
``typer.Exit(code=...)``. Without the flag the old exit-0 behavior is kept, so
existing callers see no change.
"""

from __future__ import annotations

import re
import subprocess
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli import cmdline

runner = CliRunner()

EXECUTE_CM_CLI = "comfy_cli.cmdline.custom_nodes.command.execute_cm_cli"
UPDATE_NODE_ID_CACHE = "comfy_cli.cmdline.custom_nodes.command.update_node_id_cache"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def _cm_cli_exiting(returncode: int):
    """A stand-in for ``execute_cm_cli`` against a cm-cli that exits ``returncode``.

    It reproduces the real handler's contract rather than always raising: with
    ``raise_on_error`` off, exit 1 and 2 are swallowed and ``None`` is returned;
    anything else is re-raised regardless. That way the no-flag assertions below
    exercise the behavior that actually ships, not a mock that only ever raises.
    """

    def _fake(args, **kwargs):
        error = subprocess.CalledProcessError(returncode, ["python", "-m", "cm_cli"] + list(args))
        if kwargs.get("raise_on_error") or returncode not in (1, 2):
            raise error
        return None

    return _fake


class TestExitOnFailFlag:
    def test_failed_pack_update_exits_nonzero(self):
        """The acceptance criterion: cm-cli exit 1 + the flag => non-zero exit."""
        with (
            patch(EXECUTE_CM_CLI, side_effect=_cm_cli_exiting(1)) as mock_execute,
            patch(UPDATE_NODE_ID_CACHE) as mock_cache,
        ):
            result = runner.invoke(cmdline.app, ["update", "all", "--exit-on-fail"])

        assert result.exit_code == 1
        args, kwargs = mock_execute.call_args
        assert args[0] == ["update", "all"]
        assert kwargs.get("raise_on_error") is True
        # A failed update must not go on to refresh the node id cache.
        mock_cache.assert_not_called()

    def test_the_cm_cli_exit_code_is_propagated(self):
        with (
            patch(EXECUTE_CM_CLI, side_effect=_cm_cli_exiting(7)),
            patch(UPDATE_NODE_ID_CACHE),
        ):
            result = runner.invoke(cmdline.app, ["update", "all", "--exit-on-fail"])

        assert result.exit_code == 7

    def test_success_still_exits_zero_with_the_flag(self):
        with (
            patch(EXECUTE_CM_CLI, return_value="ok") as mock_execute,
            patch(UPDATE_NODE_ID_CACHE) as mock_cache,
        ):
            result = runner.invoke(cmdline.app, ["update", "all", "--exit-on-fail"])

        assert result.exit_code == 0
        assert mock_execute.call_args.kwargs.get("raise_on_error") is True
        mock_cache.assert_called_once()

    def test_flag_is_advertised_in_help(self):
        result = runner.invoke(cmdline.app, ["update", "--help"])

        assert result.exit_code == 0
        assert "--exit-on-fail" in _strip_ansi(result.stdout)

    @pytest.mark.parametrize("target", ["comfy", "cli"])
    def test_flag_is_accepted_for_the_non_cm_cli_targets(self, target, tmp_path):
        """``comfy``/``cli`` never reach cm-cli and already fail loudly, but the flag
        must still parse so a wrapper can forward it unconditionally."""
        with (
            patch("comfy_cli.update.upgrade_cli"),
            patch("comfy_cli.cmdline.workspace_manager") as mock_ws,
            patch("comfy_cli.cmdline.resolve_workspace_python", return_value="/resolved/python"),
            patch("comfy_cli.cmdline.ensure_pip"),
            patch("comfy_cli.cmdline.subprocess.run") as mock_run,
            patch(UPDATE_NODE_ID_CACHE),
            patch(EXECUTE_CM_CLI) as mock_execute,
        ):
            mock_ws.workspace_path = str(tmp_path)
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            result = runner.invoke(cmdline.app, ["update", target, "--exit-on-fail"])

        assert result.exit_code == 0
        mock_execute.assert_not_called()


class TestWithoutTheFlagNothingChanges:
    def test_failed_pack_update_still_exits_zero(self):
        """Existing callers must not silently change behavior — they opt in."""
        with (
            patch(EXECUTE_CM_CLI, side_effect=_cm_cli_exiting(1)) as mock_execute,
            patch(UPDATE_NODE_ID_CACHE) as mock_cache,
        ):
            result = runner.invoke(cmdline.app, ["update", "all"])

        assert result.exit_code == 0
        assert mock_execute.call_args.kwargs.get("raise_on_error") is False
        mock_cache.assert_called_once()

    def test_an_unexpected_exit_code_still_surfaces(self):
        """``execute_cm_cli`` re-raises exit codes other than 1/2 even with
        ``raise_on_error`` off. The new try/except must not swallow those."""
        with (
            patch(EXECUTE_CM_CLI, side_effect=_cm_cli_exiting(3)),
            patch(UPDATE_NODE_ID_CACHE),
        ):
            result = runner.invoke(cmdline.app, ["update", "all"])

        assert result.exit_code != 0
        assert isinstance(result.exception, subprocess.CalledProcessError)
