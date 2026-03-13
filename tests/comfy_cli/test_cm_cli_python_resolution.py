from unittest.mock import MagicMock, patch

from comfy_cli.command.custom_nodes import cm_cli_util


class TestExecuteCmCli:
    def test_uses_resolved_python(self, tmp_path):
        cm_cli_path = tmp_path / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py"
        cm_cli_path.parent.mkdir(parents=True)
        cm_cli_path.touch()

        mock_result = MagicMock()
        mock_result.stdout = "output"

        with (
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/resolved/python",
            ) as mock_resolve,
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as MockConfig,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.run", return_value=mock_result) as mock_run,
        ):
            MockConfig.return_value.get_config_path.return_value = str(tmp_path / "config")
            cm_cli_util.execute_cm_cli(["show", "installed"])

        mock_resolve.assert_called_once_with(str(tmp_path))
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/resolved/python"

    def test_fast_deps_passes_python_to_compiler(self, tmp_path):
        cm_cli_path = tmp_path / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py"
        cm_cli_path.parent.mkdir(parents=True)
        cm_cli_path.touch()

        mock_result = MagicMock()
        mock_result.stdout = "output"

        with (
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as MockConfig,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.run", return_value=mock_result),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.DependencyCompiler") as MockCompiler,
        ):
            MockConfig.return_value.get_config_path.return_value = str(tmp_path / "config")
            mock_instance = MagicMock()
            MockCompiler.return_value = mock_instance

            cm_cli_util.execute_cm_cli(["install", "some-node"], fast_deps=True)

        MockCompiler.assert_called_once()
        assert MockCompiler.call_args[1]["executable"] == "/resolved/python"
