import re
import subprocess
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.command import app
from comfy_cli.file_utils import DownloadException

runner = CliRunner()


def strip_ansi(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_install_no_deps_option_exists():
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    clean_output = strip_ansi(result.stdout)
    assert "--no-deps" in clean_output
    assert "Skip dependency installation" in clean_output


def test_install_fast_deps_and_no_deps_mutually_exclusive():
    result = runner.invoke(app, ["install", "test-node", "--fast-deps", "--no-deps"])
    assert result.exit_code != 0
    assert "Cannot use --fast-deps and --no-deps together" in result.output


def test_install_no_deps_alone_works():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--no-deps"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("no_deps") is True
        assert kwargs.get("fast_deps") is False


def test_install_fast_deps_alone_works():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--fast-deps"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("fast_deps") is True
        assert kwargs.get("no_deps") is False


def test_install_neither_deps_option():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("fast_deps") is False
        assert kwargs.get("no_deps") is False


def test_multiple_commands_work_independently():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli"):
        result1 = runner.invoke(app, ["install", "test-node", "--no-deps"])
        assert result1.exit_code == 0

    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli"):
        result2 = runner.invoke(app, ["install", "test-node2", "--fast-deps"])
        assert result2.exit_code == 0


def test_install_uv_compile_passes_to_execute():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True
        assert kwargs.get("fast_deps") is False
        assert kwargs.get("no_deps") is False


def test_install_no_uv_compile_passes_false():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--no-uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is False


def test_install_uv_compile_and_fast_deps_mutually_exclusive():
    result = runner.invoke(app, ["install", "test-node", "--uv-compile", "--fast-deps"])
    assert result.exit_code != 0
    assert "Cannot use" in result.output


def test_install_uv_compile_and_no_deps_mutually_exclusive():
    result = runner.invoke(app, ["install", "test-node", "--uv-compile", "--no-deps"])
    assert result.exit_code != 0
    assert "Cannot use" in result.output


def test_uv_sync_calls_execute_cm_cli():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["uv-sync"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0] == ["uv-sync"]


def test_reinstall_uv_compile_passes_to_execute():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["reinstall", "test-node", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_reinstall_uv_compile_and_fast_deps_mutually_exclusive():
    result = runner.invoke(app, ["reinstall", "test-node", "--uv-compile", "--fast-deps"])
    assert result.exit_code != 0
    assert "Cannot use" in result.output


def test_reinstall_no_uv_compile_passes_false():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["reinstall", "test-node", "--no-uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is False


def test_install_exit_on_fail_reraises_and_propagates_code():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(7, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node", "--exit-on-fail"])
        assert result.exit_code == 7
        assert mock_execute.called
        args, kwargs = mock_execute.call_args
        assert kwargs.get("raise_on_error") is True
        assert args[0][0] == "install" and "--exit-on-fail" in args[0] and "bad-node" in args[0]


def test_save_snapshot_no_output():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["save-snapshot"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0] == ["save-snapshot"]


def test_save_snapshot_with_output():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["save-snapshot", "--output", "/tmp/snap.json"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0][0] == "save-snapshot"
        assert "--output" in args[0]


def test_restore_snapshot_with_uv_compile():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["restore-snapshot", "/tmp/snap.json", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_restore_snapshot_with_pip_flags():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["restore-snapshot", "/tmp/snap.json", "--pip-non-url", "--pip-local-url"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert "--pip-non-url" in args[0]
        assert "--pip-local-url" in args[0]


def test_restore_dependencies_with_uv_compile():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["restore-dependencies", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_update_with_uv_compile():
    with (
        patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute,
        patch("comfy_cli.command.custom_nodes.command.update_node_id_cache"),
    ):
        result = runner.invoke(app, ["update", "test-node", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_fix_with_uv_compile():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["fix", "test-node", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_uninstall_rejects_all():
    result = runner.invoke(app, ["uninstall", "all"])
    assert result.exit_code != 0
    assert "`uninstall all` is not allowed" in result.output
    assert "Invalid command" not in result.output


def test_reinstall_rejects_all():
    result = runner.invoke(app, ["reinstall", "all"])
    assert result.exit_code != 0
    assert "`reinstall all` is not allowed" in result.output
    assert "Invalid command" not in result.output


def test_validate_mode_rejects_invalid():
    result = runner.invoke(app, ["install", "test-node", "--mode", "invalid-mode"])
    assert result.exit_code != 0
    assert "Invalid mode" in result.output


def test_install_deps_with_deps_file():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install-deps", "--deps", "/tmp/deps.json"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert "install-deps" in args[0]


def test_install_deps_with_uv_compile():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install-deps", "--deps", "/tmp/deps.json", "--uv-compile"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("uv_compile") is True


def test_install_deps_no_args_shows_error():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli"):
        result = runner.invoke(app, ["install-deps"])
        assert "One of --deps or --workflow" in result.output


def test_restore_snapshot_with_pip_non_local_url():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["restore-snapshot", "/tmp/snap.json", "--pip-non-local-url"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert "--pip-non-local-url" in args[0]


def test_update_calls_update_node_id_cache():
    with (
        patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute,
        patch("comfy_cli.command.custom_nodes.command.update_node_id_cache") as mock_cache,
    ):
        result = runner.invoke(app, ["update", "test-node"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        mock_cache.assert_called_once()


def test_uninstall_calls_execute():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["uninstall", "test-node"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0] == ["uninstall", "test-node"]


def test_show_installed():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["show", "installed"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0] == ["show", "installed"]


def test_install_deps_with_workflow(tmp_path):
    workflow_file = tmp_path / "workflow.json"
    workflow_file.write_text("{}")
    with (
        patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute,
        patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws,
    ):
        mock_ws.config_manager.get_config_path.return_value = str(tmp_path)
        result = runner.invoke(app, ["install-deps", "--workflow", str(workflow_file)])
        assert result.exit_code == 0
        assert mock_execute.call_count == 2
        first_call_args = mock_execute.call_args_list[0][0][0]
        second_call_args = mock_execute.call_args_list[1][0][0]
        assert first_call_args[0] == "deps-in-workflow"
        assert second_call_args[0] == "install-deps"


def test_install_rejects_all():
    result = runner.invoke(app, ["install", "all"])
    assert result.exit_code != 0
    assert "`install all` is not allowed" in result.output
    assert "Invalid command" not in result.output


def test_simple_show_installed():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["simple-show", "installed"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        args, _ = mock_execute.call_args
        assert args[0] == ["simple-show", "installed"]


def test_show_with_channel():
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["show", "installed", "--channel", "dev"])
        assert result.exit_code == 0
        mock_execute.assert_called_once()
        _, kwargs = mock_execute.call_args
        assert kwargs.get("channel") == "dev"


class TestRegistryInstallDownloadError:
    """registry-install must catch DownloadException, surface a friendly one-line
    error via ui.display_error_message, and exit cleanly — never raise a traceback."""

    def _invoke(self, tmp_path, download_side_effect):
        fake_version = MagicMock(download_url="http://example.com/node.zip", version="1.0.0")

        with (
            patch("comfy_cli.command.custom_nodes.command.registry_api") as mock_api,
            patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws,
            patch("comfy_cli.command.custom_nodes.command.download_file", side_effect=download_side_effect) as mock_dl,
            patch("comfy_cli.command.custom_nodes.command.ui") as mock_ui,
            patch("comfy_cli.command.custom_nodes.command.extract_package_as_zip") as mock_extract,
            patch("comfy_cli.command.custom_nodes.command.execute_install_script") as mock_script,
        ):
            mock_api.install_node.return_value = fake_version
            mock_ws.workspace_path = str(tmp_path)
            result = runner.invoke(app, ["registry-install", "test-node"])
            return result, mock_ui, mock_dl, mock_extract, mock_script

    def test_download_exception_caught_and_reported(self, tmp_path):
        result, mock_ui, mock_dl, mock_extract, mock_script = self._invoke(
            tmp_path, DownloadException("server unreachable")
        )

        assert result.exit_code == 0
        mock_dl.assert_called_once()
        mock_ui.display_error_message.assert_called_once()
        (msg,), _ = mock_ui.display_error_message.call_args
        assert "test-node" in msg
        assert "server unreachable" in msg

    def test_no_extract_or_install_script_after_failure(self, tmp_path):
        """After a download failure we must not try to unzip or run the install script."""
        result, _mock_ui, _mock_dl, mock_extract, mock_script = self._invoke(tmp_path, DownloadException("boom"))

        assert result.exit_code == 0
        mock_extract.assert_not_called()
        mock_script.assert_not_called()

    def test_no_traceback_in_output(self, tmp_path):
        result, _mock_ui, _mock_dl, _mock_extract, _mock_script = self._invoke(tmp_path, DownloadException("boom"))

        assert "Traceback" not in result.output
        assert "DownloadException" not in result.output
