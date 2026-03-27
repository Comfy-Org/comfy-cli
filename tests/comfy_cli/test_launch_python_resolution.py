import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.command import launch


class TestLaunchComfyui:
    def test_uses_python_param(self):
        mock_result = subprocess.CompletedProcess(args=[], returncode=0)

        with (
            patch("comfy_cli.command.launch.ConfigManager"),
            patch("comfy_cli.command.launch.subprocess.run", return_value=mock_result) as mock_run,
        ):
            with pytest.raises(SystemExit):
                launch.launch_comfyui(extra=[], python="/resolved/python")

        mock_run.assert_called()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/resolved/python"
        assert cmd[0] != sys.executable

    @pytest.mark.parametrize("returncode", [0, 1, 42])
    def test_foreground_exit_code_matches_subprocess(self, returncode):
        """exit() should receive the subprocess returncode, not the CompletedProcess object."""
        mock_result = subprocess.CompletedProcess(args=[], returncode=returncode)

        with (
            patch("comfy_cli.command.launch.ConfigManager"),
            patch("comfy_cli.command.launch.subprocess.run", return_value=mock_result),
        ):
            with pytest.raises(SystemExit) as exc_info:
                launch.launch_comfyui(extra=[], python="/resolved/python")

        assert exc_info.value.code == returncode


class TestLaunchResolvesWorkspacePython:
    def test_resolves_and_passes_python(self):
        with (
            patch("comfy_cli.command.launch.resolve_workspace_python", return_value="/resolved/python") as mock_resolve,
            patch.object(launch.workspace_manager, "workspace_path", "/fake/workspace"),
            patch.object(launch.workspace_manager, "workspace_type", launch.WorkspaceType.DEFAULT),
            patch.object(launch.workspace_manager, "config_manager", MagicMock()),
            patch.object(launch.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.launch.check_for_updates"),
            patch("comfy_cli.command.launch.os.chdir"),
            patch("comfy_cli.command.launch.launch_comfyui") as mock_launch_comfyui,
        ):
            launch.launch(background=False)

        mock_resolve.assert_called_once_with("/fake/workspace")
        mock_launch_comfyui.assert_called_once()
        assert mock_launch_comfyui.call_args[1]["python"] == "/resolved/python"
