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
            patch("comfy_cli.command.launch.os.chdir"),
            patch("comfy_cli.command.launch.launch_comfyui") as mock_launch_comfyui,
        ):
            launch.launch(background=False)

        mock_resolve.assert_called_once_with("/fake/workspace")
        mock_launch_comfyui.assert_called_once()
        assert mock_launch_comfyui.call_args[1]["python"] == "/resolved/python"


class TestBackgroundLaunchReexec:
    def test_reexecs_the_invoked_binary_not_path_comfy(self, tmp_path, monkeypatch):
        """Regression test for #691: `comfy launch --background` must re-exec the
        binary that is running. A bare "comfy" resolves through PATH and dies with
        FileNotFoundError when comfy-cli was invoked by its own path from a venv
        that isn't on PATH."""
        invoked = tmp_path / "venv" / "bin" / "comfy"
        invoked.parent.mkdir(parents=True)
        invoked.write_text("#!/bin/sh" + chr(10))
        monkeypatch.setattr(sys, "argv", [str(invoked), "launch", "--background"])
        monkeypatch.setattr(launch.ConfigManager, "background", property(lambda self: None), raising=False)
        monkeypatch.setattr(launch.utils, "is_running", lambda *a, **k: False)
        monkeypatch.setattr(launch, "check_comfy_server_running", lambda port: False)
        captured = {}

        async def fake_monitor(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return None

        monkeypatch.setattr(launch, "launch_and_monitor", fake_monitor)
        monkeypatch.setattr(launch, "_hard_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

        with pytest.raises(SystemExit):
            launch.background_launch(extra=[])

        assert captured["cmd"][0] == str(invoked)

    def test_module_invocation_falls_back_to_path(self, tmp_path, monkeypatch):
        """`python -m comfy_cli` has a .py in argv[0]; PATH resolution stays the fallback."""
        monkeypatch.setattr(sys, "argv", [str(tmp_path / "comfy_cli" / "__main__.py")])
        monkeypatch.setattr(launch.ConfigManager, "background", property(lambda self: None), raising=False)
        monkeypatch.setattr(launch.utils, "is_running", lambda *a, **k: False)
        monkeypatch.setattr(launch, "check_comfy_server_running", lambda port: False)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/found/on/path/comfy")
        captured = {}

        async def fake_monitor(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return None

        monkeypatch.setattr(launch, "launch_and_monitor", fake_monitor)
        monkeypatch.setattr(launch, "_hard_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

        with pytest.raises(SystemExit):
            launch.background_launch(extra=[])

        assert captured["cmd"][0] == "/found/on/path/comfy"


class TestResolveComfyExec:
    def test_keeps_the_invoked_console_script(self, tmp_path):
        exe = tmp_path / "venv" / "bin" / "comfy"
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh" + chr(10))
        assert launch._resolve_comfy_exec(str(exe)) == str(exe)

    def test_module_and_missing_paths_fall_back_to_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/found/on/path/comfy")
        assert launch._resolve_comfy_exec(str(tmp_path / "comfy_cli" / "__main__.py")) == "/found/on/path/comfy"
        assert launch._resolve_comfy_exec(str(tmp_path / "nope")) == "/found/on/path/comfy"
