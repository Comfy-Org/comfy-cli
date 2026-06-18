from unittest.mock import MagicMock, patch

from comfy_cli import cmdline


class TestUpdateComfy:
    def test_uses_resolved_python(self, tmp_path):
        with (
            patch("comfy_cli.cmdline.resolve_workspace_python", return_value="/resolved/python") as mock_resolve,
            patch.object(cmdline.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.cmdline.os.chdir"),
            patch("comfy_cli.cmdline.subprocess.run") as mock_run,
            patch("comfy_cli.cmdline.run_pip_install") as mock_install,
            patch("comfy_cli.cmdline.custom_nodes.command.update_node_id_cache"),
        ):
            cmdline.update(target="comfy")

        mock_resolve.assert_called_once_with(str(tmp_path))
        mock_run.assert_called_once_with(["git", "pull"], check=True)
        mock_install.assert_called_once_with(executable="/resolved/python", args=["-r", "requirements.txt"], check=True)

    def test_update_comfy_succeeds_when_cm_cli_missing(self, tmp_path):
        """Regression test for #403: comfy update must not crash when cm-cli is absent."""
        with (
            patch("comfy_cli.cmdline.resolve_workspace_python", return_value="/resolved/python"),
            patch.object(cmdline.workspace_manager, "workspace_path", str(tmp_path)),
            patch("comfy_cli.cmdline.os.chdir"),
            patch("comfy_cli.cmdline.subprocess.run"),
            patch("comfy_cli.cmdline.run_pip_install"),
            patch(
                "comfy_cli.cmdline.custom_nodes.command.update_node_id_cache",
                side_effect=FileNotFoundError("cm-cli not found"),
            ) as mock_cache,
        ):
            cmdline.update(target="comfy")
        mock_cache.assert_called_once()


class TestDependency:
    def test_passes_python_to_compiler(self, tmp_path):
        with (
            patch("comfy_cli.cmdline.resolve_workspace_python", return_value="/resolved/python") as mock_resolve,
            patch.object(cmdline.workspace_manager, "get_workspace_path", return_value=(str(tmp_path), None)),
            patch("comfy_cli.cmdline.DependencyCompiler") as MockCompiler,
        ):
            mock_instance = MagicMock()
            MockCompiler.return_value = mock_instance

            cmdline.dependency()

        mock_resolve.assert_called_once_with(str(tmp_path))
        MockCompiler.assert_called_once()
        assert MockCompiler.call_args[1]["executable"] == "/resolved/python"
