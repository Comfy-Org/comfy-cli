import subprocess
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli.command.custom_nodes import cm_cli_util


def _make_mock_proc(returncode, stdout_lines=None):
    """Create a mock Popen process with given returncode and stdout lines."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines or [])
    mock_proc.wait.return_value = returncode
    return mock_proc


@pytest.fixture(autouse=True)
def _clear_find_cm_cli_cache():
    cm_cli_util.find_cm_cli.cache_clear()
    yield
    cm_cli_util.find_cm_cli.cache_clear()


@pytest.fixture()
def _cm_cli_env(tmp_path):
    mock_proc = MagicMock()
    mock_proc.stdout = iter(["ok\n"])
    mock_proc.wait.return_value = 0
    with (
        patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
        patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
        patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
        patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
        patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/resolved/python"),
        patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
        patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch("comfy_cli.command.custom_nodes.cm_cli_util.DependencyCompiler") as mock_compiler,
    ):
        mock_cfg.return_value.get_config_path.return_value = str(tmp_path / "config")
        mock_compiler.return_value = MagicMock()
        yield {"mock_popen": mock_popen, "mock_proc": mock_proc, "mock_compiler": mock_compiler}


class TestFindCmCli:
    def test_returns_true_when_module_exists(self):
        with patch("comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=MagicMock()):
            assert cm_cli_util.find_cm_cli() is True

    def test_returns_false_when_module_missing(self):
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=None),
            patch.object(cm_cli_util.workspace_manager, "workspace_path", None),
        ):
            assert cm_cli_util.find_cm_cli() is False

    def test_returns_true_when_found_in_workspace_venv(self):
        mock_result = MagicMock(returncode=0)
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=None),
            patch.object(cm_cli_util.workspace_manager, "workspace_path", "/fake/workspace"),
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/fake/venv/python",
            ),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.run", return_value=mock_result),
        ):
            assert cm_cli_util.find_cm_cli() is True

    def test_returns_false_when_missing_from_workspace_venv(self):
        mock_result = MagicMock(returncode=1)
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=None),
            patch.object(cm_cli_util.workspace_manager, "workspace_path", "/fake/workspace"),
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/fake/venv/python",
            ),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.run", return_value=mock_result),
        ):
            assert cm_cli_util.find_cm_cli() is False

    def test_result_is_cached(self):
        with patch(
            "comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=MagicMock()
        ) as mock_spec:
            cm_cli_util.find_cm_cli()
            cm_cli_util.find_cm_cli()
            mock_spec.assert_called_once()

    def test_cache_clear_allows_recheck(self):
        with (
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.importlib.util.find_spec", return_value=None
            ) as mock_spec,
            patch.object(cm_cli_util.workspace_manager, "workspace_path", None),
        ):
            assert cm_cli_util.find_cm_cli() is False
            cm_cli_util.find_cm_cli.cache_clear()
            mock_spec.return_value = MagicMock()
            assert cm_cli_util.find_cm_cli() is True


class TestResolveManagerGuiMode:
    def test_returns_config_mode_when_set(self):
        with patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda k: "disable" if k == "manager_gui_mode" else None
            assert cm_cli_util.resolve_manager_gui_mode() == "disable"

    def test_legacy_false_returns_disable(self):
        with patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda k: "False" if k == "manager_gui_enabled" else None
            assert cm_cli_util.resolve_manager_gui_mode() == "disable"

    def test_legacy_true_returns_enable_gui(self):
        with patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda k: "True" if k == "manager_gui_enabled" else None
            assert cm_cli_util.resolve_manager_gui_mode() == "enable-gui"

    def test_legacy_boolean_0_returns_disable(self):
        with patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda k: "0" if k == "manager_gui_enabled" else None
            assert cm_cli_util.resolve_manager_gui_mode() == "disable"

    def test_no_config_manager_available_returns_enable_gui(self):
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
        ):
            mock_cfg.return_value.get.return_value = None
            assert cm_cli_util.resolve_manager_gui_mode() == "enable-gui"

    def test_no_config_no_manager_returns_not_installed_value(self):
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=False),
        ):
            mock_cfg.return_value.get.return_value = None
            assert cm_cli_util.resolve_manager_gui_mode("not-installed") == "not-installed"

    def test_no_config_no_manager_returns_none_by_default(self):
        with (
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=False),
        ):
            mock_cfg.return_value.get.return_value = None
            assert cm_cli_util.resolve_manager_gui_mode() is None


class TestExecuteCmCli:
    def test_no_workspace_raises_exit(self):
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", None),
            pytest.raises(typer.Exit),
        ):
            cm_cli_util.execute_cm_cli(["show"])

    def test_no_cm_cli_raises_exit(self):
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", "/workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=False),
            pytest.raises(typer.Exit),
        ):
            cm_cli_util.execute_cm_cli(["show"])

    def test_happy_path_returns_stdout(self, _cm_cli_env):
        result = cm_cli_util.execute_cm_cli(["show", "installed"])
        assert result == "ok\n"

    def test_cmd_uses_python_m_cm_cli(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["show"])
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert cmd[:3] == ["/resolved/python", "-m", "cm_cli"]

    def test_channel_appended(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["show"], channel="stable")
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--channel" in cmd
        assert cmd[cmd.index("--channel") + 1] == "stable"

    def test_uv_compile_flag(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["install", "node"], uv_compile=True)
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--uv-compile" in cmd

    def test_fast_deps_adds_no_deps(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["install", "node"], fast_deps=True)
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--no-deps" in cmd

    def test_no_deps_adds_no_deps(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["install", "node"], no_deps=True)
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--no-deps" in cmd

    def test_uv_compile_takes_precedence_over_fast_deps(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["install", "node"], uv_compile=True, fast_deps=True)
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--uv-compile" in cmd
        assert "--no-deps" not in cmd

    def test_mode_appended(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["install", "node"], mode="remote")
        cmd = _cm_cli_env["mock_popen"].call_args[0][0]
        assert "--mode" in cmd
        assert cmd[cmd.index("--mode") + 1] == "remote"

    def test_error_returncode_1_returns_none(self, tmp_path):
        mock_proc = _make_mock_proc(1)
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            result = cm_cli_util.execute_cm_cli(["install", "node"])
            assert result is None

    def test_error_returncode_2_returns_none(self, tmp_path):
        mock_proc = _make_mock_proc(2)
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            result = cm_cli_util.execute_cm_cli(["install", "node"])
            assert result is None

    def test_error_other_returncode_raises(self, tmp_path):
        mock_proc = _make_mock_proc(42)
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
            pytest.raises(subprocess.CalledProcessError, match="42"),
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            cm_cli_util.execute_cm_cli(["install", "node"])

    def test_raise_on_error_reraises(self, tmp_path):
        mock_proc = _make_mock_proc(1)
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
            pytest.raises(subprocess.CalledProcessError),
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            cm_cli_util.execute_cm_cli(["install", "node"], raise_on_error=True)

    def test_fast_deps_triggers_dependency_compiler(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["ok\n"])
        mock_proc.wait.return_value = 0
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.DependencyCompiler") as mock_compiler,
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            mock_instance = MagicMock()
            mock_compiler.return_value = mock_instance
            cm_cli_util.execute_cm_cli(["install", "node"], fast_deps=True)
            mock_compiler.assert_called_once()
            mock_instance.compile_deps.assert_called_once()
            mock_instance.install_deps.assert_called_once()

    def test_fast_deps_non_dependency_cmd_skips_compiler(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["ok\n"])
        mock_proc.wait.return_value = 0
        with (
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as mock_cfg,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python", return_value="/python"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.DependencyCompiler") as mock_compiler,
        ):
            mock_cfg.return_value.get_config_path.return_value = str(tmp_path)
            cm_cli_util.execute_cm_cli(["show", "all"], fast_deps=True)
            mock_compiler.assert_not_called()

    def test_sets_comfyui_path_env(self, _cm_cli_env):
        cm_cli_util.execute_cm_cli(["show"])
        env = _cm_cli_env["mock_popen"].call_args[1]["env"]
        assert "COMFYUI_PATH" in env
