import json
import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from typer.testing import CliRunner

from comfy_cli import cmdline
from comfy_cli.command.custom_nodes.command import app
from comfy_cli.file_utils import DownloadException
from comfy_cli.registry import RegistryAPIError

runner = CliRunner()


def _split_stream_runner() -> CliRunner:
    """A runner that keeps stdout and stderr separate, so tests can assert the
    JSON-mode contract that stdout carries ONLY the envelope."""
    return CliRunner()


@pytest.fixture
def _reset_renderer():
    """Invoking the root app installs a process-wide JSON renderer; drop it so
    later tests that invoke the sub-app get the default pretty renderer back."""
    yield
    from comfy_cli.output.renderer import reset_renderer_for_testing

    reset_renderer_for_testing()


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


def test_install_exit_on_fail_signal_death_becomes_shell_convention_code():
    """`Popen.wait()` returns -9 when the OOM killer reaps cm-cli mid dependency
    build. Exiting with that raw value would truncate to a fabricated 247, so map
    it to the shell's 128+N convention."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(-9, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node", "--exit-on-fail"])
        assert result.exit_code == 137


def test_install_exit_on_fail_code_that_would_truncate_to_zero_stays_nonzero():
    """Windows can return codes above 255; a multiple of 256 would otherwise
    truncate to 0 and report a failed install as a success."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(256, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node", "--exit-on-fail"])
        assert result.exit_code == 1


def test_install_exit_on_fail_code_2_does_not_masquerade_as_usage_error():
    """Click reserves exit code 2 for its own usage errors, so a cm-cli exit 2 is
    remapped rather than letting a wrapper confuse "you invoked comfy wrong" with
    "cm-cli exited 2"."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(2, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node", "--exit-on-fail"])
        assert result.exit_code == 1


def test_install_exit_on_fail_wide_status_with_low_byte_2_is_remapped():
    """A wide status like 258 truncates to 2 in `sys.exit` — the exact Click
    usage-error collision the remap exists to prevent — so the normalization
    keys on the low byte, not the literal value 2."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(258, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node", "--exit-on-fail"])
        assert result.exit_code == 1


def test_install_without_exit_on_fail_surfaces_unexpected_exit_codes():
    """execute_cm_cli itself swallows cm-cli exits 1/2 when raise_on_error is
    off; anything else (e.g. a signal death) it re-raises, and install must not
    turn that into a silent exit 0 (mirrors the `comfy update all` handler)."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(-9, "cm-cli")
        result = runner.invoke(app, ["install", "bad-node"])
        assert result.exit_code != 0
        assert isinstance(result.exception, subprocess.CalledProcessError)


def test_install_exit_on_fail_json_stdout_carries_only_the_envelope(_reset_renderer):
    """execute_cm_cli streams cm-cli's raw output to sys.stdout; in JSON mode
    install routes that stream to stderr so stdout stays a single parseable
    envelope, labeled with the actual subcommand."""

    def _stream_then_fail(args, **kwargs):
        sys.stdout.write("raw cm-cli progress line\n")
        # click's CliRunner only flushes stdout at invoke exit, so the redirected
        # stderr wrapper would otherwise hold this line in its buffer forever.
        sys.stdout.flush()
        raise subprocess.CalledProcessError(7, ["python", "-m", "cm_cli", "install"])

    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli", side_effect=_stream_then_fail):
        result = _split_stream_runner().invoke(cmdline.app, ["--json", "node", "install", "bad-node", "--exit-on-fail"])

    assert result.exit_code == 7
    envelope = json.loads(result.stdout.strip())
    assert "raw cm-cli progress line\n" in result.stderr
    assert envelope["ok"] is False
    assert envelope["command"] == "node install"
    assert envelope["error"]["code"] == "node_install_failed"
    assert envelope["error"]["details"] == {"cm_cli_returncode": 7, "failed_stage": "cm-cli"}


def test_install_exit_on_fail_signal_death_is_reported_as_a_signal(_reset_renderer):
    """No process exits with -9 — cm-cli was killed by signal 9. The message
    says so instead of leaking Popen's negative-signal encoding."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(-9, ["python", "-m", "cm_cli", "install"])
        result = runner.invoke(cmdline.app, ["--json", "node", "install", "bad-node", "--exit-on-fail"])

    assert result.exit_code == 137
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert "killed by signal 9" in envelope["error"]["message"]
    assert envelope["error"]["details"]["cm_cli_returncode"] == -9


def test_install_fast_deps_dep_failure_is_not_blamed_on_cm_cli(_reset_renderer):
    """With --fast-deps, execute_cm_cli runs the dependency compiler after
    cm-cli succeeds; a pip/uv failure there is a dependency failure — the packs
    themselves installed — and must not be attributed to cm-cli."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        mock_execute.side_effect = subprocess.CalledProcessError(1, ["uv", "pip", "install", "-r", "requirements.txt"])
        result = runner.invoke(cmdline.app, ["--json", "node", "install", "bad-node", "--fast-deps", "--exit-on-fail"])

    assert result.exit_code == 1
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert "dependency installation" in envelope["error"]["message"]
    assert envelope["error"]["details"] == {"returncode": 1, "failed_stage": "dependency-install"}


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

        # Must exit non-zero so automation / CI can detect the failure.
        assert result.exit_code == 1
        mock_dl.assert_called_once()
        mock_ui.display_error_message.assert_called_once()
        (msg,), _ = mock_ui.display_error_message.call_args
        assert "test-node" in msg
        assert "server unreachable" in msg

    def test_no_extract_or_install_script_after_failure(self, tmp_path):
        """After a download failure we must not try to unzip or run the install script."""
        result, _mock_ui, _mock_dl, mock_extract, mock_script = self._invoke(tmp_path, DownloadException("boom"))

        assert result.exit_code == 1
        mock_extract.assert_not_called()
        mock_script.assert_not_called()

    def test_no_traceback_in_output(self, tmp_path):
        result, _mock_ui, _mock_dl, _mock_extract, _mock_script = self._invoke(tmp_path, DownloadException("boom"))

        assert "Traceback" not in result.output
        assert "DownloadException" not in result.output


class TestRegistryInstallApiError:
    """A RegistryAPIError from install_node must surface a machine-readable
    renderer.error(code="registry_install_failed", details={status, body}) and
    exit non-zero — not a bare traceback and not a silent exit 0."""

    def test_api_error_surfaced_with_code_and_exit_1(self, tmp_path):
        with (
            patch("comfy_cli.command.custom_nodes.command.registry_api") as mock_api,
            patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws,
            patch("comfy_cli.command.custom_nodes.command.get_renderer") as mock_get_renderer,
        ):
            mock_ws.workspace_path = str(tmp_path)
            mock_api.install_node.side_effect = RegistryAPIError(
                "Failed to install node: 404 - Not Found", status=404, body="Not Found"
            )

            result = runner.invoke(app, ["registry-install", "test-node"])

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        mock_get_renderer.return_value.error.assert_called_once()
        _, kwargs = mock_get_renderer.return_value.error.call_args
        assert kwargs["code"] == "registry_install_failed"
        assert kwargs["details"] == {"node_id": "test-node", "status": 404, "body": "Not Found"}


class TestRegistryInstallNonApiFailure:
    """Failures that aren't RegistryAPIError — a connection error, a DNS failure,
    a timeout, a JSON decode error, or a registry response carrying no download
    URL — must also emit the registry_install_failed envelope and exit non-zero.
    Exiting 0 here reports a network outage to automation / CI as success."""

    def _invoke(self, tmp_path, *, install_node_side_effect=None, install_node_return=None):
        with (
            patch("comfy_cli.command.custom_nodes.command.registry_api") as mock_api,
            patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws,
            patch("comfy_cli.command.custom_nodes.command.get_renderer") as mock_get_renderer,
            patch("comfy_cli.command.custom_nodes.command.download_file") as mock_dl,
        ):
            mock_ws.workspace_path = str(tmp_path)
            if install_node_side_effect is not None:
                mock_api.install_node.side_effect = install_node_side_effect
            else:
                mock_api.install_node.return_value = install_node_return

            result = runner.invoke(app, ["registry-install", "test-node"])
            return result, mock_get_renderer, mock_dl

    def test_connection_error_exits_1_with_envelope(self, tmp_path):
        result, mock_get_renderer, mock_dl = self._invoke(
            tmp_path, install_node_side_effect=requests.ConnectionError("Name or service not known")
        )

        # Must exit non-zero so automation / CI can detect the failure.
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        mock_dl.assert_not_called()
        mock_get_renderer.return_value.error.assert_called_once()
        _, kwargs = mock_get_renderer.return_value.error.call_args
        assert kwargs["code"] == "registry_install_failed"
        assert kwargs["details"] == {"node_id": "test-node"}

    def test_missing_download_url_exits_1_with_envelope(self, tmp_path):
        result, mock_get_renderer, mock_dl = self._invoke(
            tmp_path, install_node_return=MagicMock(download_url="", version="1.0.0")
        )

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        mock_dl.assert_not_called()
        mock_get_renderer.return_value.error.assert_called_once()
        _, kwargs = mock_get_renderer.return_value.error.call_args
        assert kwargs["code"] == "registry_install_failed"
        assert kwargs["details"] == {"node_id": "test-node"}
