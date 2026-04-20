from unittest.mock import patch

from comfy_cli.command.custom_nodes import command


class TestGetInstalledPackages:
    def test_uses_resolved_python(self):
        command.pip_map = None

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", "/fake/workspace"),
            patch(
                "comfy_cli.command.custom_nodes.command.subprocess.check_output",
                return_value="Package  Version\n------  -------\npip  24.0\n",
            ) as mock_check_output,
        ):
            command.get_installed_packages()

        cmd = mock_check_output.call_args[0][0]
        assert cmd[0] == "/resolved/python"
        assert cmd == ["/resolved/python", "-m", "pip", "list"]

        command.pip_map = None


class TestExecuteInstallScript:
    def test_pip_uses_resolved_python(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("somepackage\n")

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.command.custom_nodes.command.subprocess.check_call") as mock_check_call,
        ):
            command.execute_install_script(str(tmp_path))

        mock_check_call.assert_called()
        cmd = mock_check_call.call_args[0][0]
        assert cmd[0] == "/resolved/python"
        assert "-m" in cmd and "pip" in cmd

    def test_install_py_uses_resolved_python(self, tmp_path):
        (tmp_path / "install.py").write_text("print('install')\n")

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.command.custom_nodes.command.subprocess.check_call") as mock_check_call,
        ):
            command.execute_install_script(str(tmp_path))

        mock_check_call.assert_called()
        cmd = mock_check_call.call_args[0][0]
        assert cmd == ["/resolved/python", "install.py"]

    def test_inline_comment_not_passed_to_pip(self, tmp_path):
        # Issue #431 regression: inline comments in requirements.txt must not
        # survive into the argv handed to pip. Pre-fix, the line was passed
        # verbatim (e.g. "matplotlib>=3.3.0  # note") and pip rejected it.
        (tmp_path / "requirements.txt").write_text("matplotlib>=3.3.0  # For visualization\n")

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.command.custom_nodes.command.subprocess.check_call") as mock_check_call,
        ):
            command.execute_install_script(str(tmp_path))

        for call in mock_check_call.call_args_list:
            argv = call[0][0]
            assert not any("#" in element for element in argv), f"inline comment leaked into pip argv: {argv!r}"

    def test_uses_pip_install_r(self, tmp_path):
        # Option C: delegate requirements-file parsing to pip via `-r <path>`.
        # This lets pip handle inline comments, line continuations, VCS URL
        # fragments, env markers, -e, -r, --index-url, etc.
        requirements_path = tmp_path / "requirements.txt"
        requirements_path.write_text("numpy>=1.0\n")

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.command.custom_nodes.command.subprocess.check_call") as mock_check_call,
        ):
            command.execute_install_script(str(tmp_path))

        mock_check_call.assert_called_once()
        argv = mock_check_call.call_args[0][0]
        assert argv == ["/resolved/python", "-m", "pip", "install", "-r", str(requirements_path)]


class TestUpdateNodeIdCache:
    def test_uses_resolved_python(self, tmp_path):
        cm_cli_path = tmp_path / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py"
        cm_cli_path.parent.mkdir(parents=True)
        cm_cli_path.touch()

        config_path = tmp_path / "config"
        config_path.mkdir()

        with (
            patch(
                "comfy_cli.command.custom_nodes.command.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(command.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.command.custom_nodes.command.ConfigManager") as MockConfig,
            patch("comfy_cli.command.custom_nodes.command.subprocess.run") as mock_run,
        ):
            MockConfig.return_value.get_config_path.return_value = str(config_path)
            command.update_node_id_cache()

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/resolved/python"
