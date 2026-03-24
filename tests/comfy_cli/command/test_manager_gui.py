from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import constants
from comfy_cli.command.launch import _get_manager_flags


@pytest.fixture()
def mock_config_manager():
    with patch("comfy_cli.command.custom_nodes.command.ConfigManager") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture()
def mock_launch_config_manager():
    with patch("comfy_cli.command.launch.ConfigManager") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


class TestManagerCommands:
    def test_disable_manager_sets_config(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import disable_manager

        disable_manager()

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")

    def test_enable_gui_sets_config(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import enable_gui

        enable_gui()

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "enable-gui")

    def test_disable_gui_sets_config(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import disable_gui

        disable_gui()

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable-gui")

    def test_enable_legacy_gui_sets_config(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import enable_legacy_gui

        enable_legacy_gui()

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "enable-legacy-gui")


class TestGetManagerFlags:
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="disable")
    def test_disable_mode_returns_empty(self, mock_resolve):
        result = _get_manager_flags()
        assert result == []

    @patch("comfy_cli.command.launch.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="enable-gui")
    def test_enable_gui_mode_returns_enable_manager(self, mock_resolve, mock_find):
        result = _get_manager_flags()
        assert result == ["--enable-manager"]

    @patch("comfy_cli.command.launch.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="disable-gui")
    def test_disable_gui_mode_returns_both_flags(self, mock_resolve, mock_find):
        result = _get_manager_flags()
        assert result == ["--enable-manager", "--disable-manager-ui"]

    @patch("comfy_cli.command.launch.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="enable-legacy-gui")
    def test_enable_legacy_gui_mode_returns_legacy_flags(self, mock_resolve, mock_find):
        result = _get_manager_flags()
        assert result == ["--enable-manager", "--enable-manager-legacy-ui"]

    @patch("comfy_cli.command.launch.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="unknown-mode")
    def test_unknown_mode_returns_default_with_warning(self, mock_resolve, mock_find, capsys):
        result = _get_manager_flags()
        assert result == ["--enable-manager"]
        captured = capsys.readouterr()
        assert "unknown-mode" in captured.out.lower() or "Unknown manager mode" in captured.out

    @patch("comfy_cli.command.launch.find_cm_cli", return_value=False)
    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value="enable-gui")
    def test_enable_mode_without_cmcli_returns_empty(self, mock_resolve, mock_find):
        """When config is enable-* but cm-cli is not available, return empty list."""
        result = _get_manager_flags()
        assert result == []
        mock_find.assert_called_once()

    @patch("comfy_cli.command.launch.resolve_manager_gui_mode", return_value=None)
    def test_not_installed_returns_empty(self, mock_resolve):
        """When resolve returns None (not installed), return empty list."""
        result = _get_manager_flags()
        assert result == []


class TestResolveManagerGuiMode:
    """Tests for resolve_manager_gui_mode shared helper."""

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_returns_configured_mode(self, mock_cm_cls):
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.return_value = "disable-gui"

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "disable-gui"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_old_config_false_migrates_to_disable(self, mock_cm_cls, mock_find):
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.side_effect = lambda key: {
            constants.CONFIG_KEY_MANAGER_GUI_MODE: None,
            constants.CONFIG_KEY_MANAGER_GUI_ENABLED: "False",
        }.get(key)

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "disable"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_old_config_true_migrates_to_enable_gui(self, mock_cm_cls, mock_find):
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.side_effect = lambda key: {
            constants.CONFIG_KEY_MANAGER_GUI_MODE: None,
            constants.CONFIG_KEY_MANAGER_GUI_ENABLED: "True",
        }.get(key)

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "enable-gui"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_no_config_with_cmcli_defaults_to_enable_gui(self, mock_cm_cls, mock_find):
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.return_value = None

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "enable-gui"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=False)
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_no_config_no_cmcli_returns_not_installed_value(self, mock_cm_cls, mock_find):
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.return_value = None

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode(not_installed_value=None) is None
        assert resolve_manager_gui_mode(not_installed_value="not-installed") == "not-installed"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_old_config_boolean_false_migrates_to_disable(self, mock_cm_cls):
        """Test backward compatibility with actual boolean False value."""
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.side_effect = lambda key: {
            constants.CONFIG_KEY_MANAGER_GUI_MODE: None,
            constants.CONFIG_KEY_MANAGER_GUI_ENABLED: False,
        }.get(key)

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "disable"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli", return_value=True)
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager")
    def test_old_config_boolean_true_migrates_to_enable_gui(self, mock_cm_cls, mock_find):
        """Test backward compatibility with actual boolean True value."""
        instance = MagicMock()
        mock_cm_cls.return_value = instance
        instance.get.side_effect = lambda key: {
            constants.CONFIG_KEY_MANAGER_GUI_MODE: None,
            constants.CONFIG_KEY_MANAGER_GUI_ENABLED: True,
        }.get(key)

        from comfy_cli.command.custom_nodes.cm_cli_util import resolve_manager_gui_mode

        assert resolve_manager_gui_mode() == "enable-gui"


class TestLaunchManagerFlagInjection:
    @patch("comfy_cli.command.launch.launch_comfyui")
    @patch("comfy_cli.command.launch._get_manager_flags", return_value=["--enable-manager"])
    @patch("comfy_cli.command.launch.workspace_manager")
    @patch("comfy_cli.command.launch.check_for_updates")
    @patch("os.chdir")
    def test_launch_injects_enable_manager(
        self, mock_chdir, mock_updates, mock_ws, mock_get_flags, mock_launch_comfyui
    ):
        mock_ws.workspace_path = "/fake/workspace"
        mock_ws.workspace_type = "default"
        mock_ws.config_manager.config = {"DEFAULT": {}}

        from comfy_cli.command.launch import launch

        launch(background=False, extra=["--port", "8188"])

        args, kwargs = mock_launch_comfyui.call_args
        extra_arg = args[0]
        assert "--enable-manager" in extra_arg
        assert "--port" in extra_arg

    @patch("comfy_cli.command.launch.launch_comfyui")
    @patch("comfy_cli.command.launch._get_manager_flags", return_value=[])
    @patch("comfy_cli.command.launch.workspace_manager")
    @patch("comfy_cli.command.launch.check_for_updates")
    @patch("os.chdir")
    def test_launch_no_inject_when_disabled(
        self, mock_chdir, mock_updates, mock_ws, mock_get_flags, mock_launch_comfyui
    ):
        mock_ws.workspace_path = "/fake/workspace"
        mock_ws.workspace_type = "default"
        mock_ws.config_manager.config = {"DEFAULT": {}}

        from comfy_cli.command.launch import launch

        launch(background=False, extra=["--port", "8188"])

        args, kwargs = mock_launch_comfyui.call_args
        extra_arg = args[0]
        assert "--enable-manager" not in extra_arg

    @patch("comfy_cli.command.launch.launch_comfyui")
    @patch("comfy_cli.command.launch._get_manager_flags", return_value=["--enable-manager"])
    @patch("comfy_cli.command.launch.workspace_manager")
    @patch("comfy_cli.command.launch.check_for_updates")
    @patch("os.chdir")
    def test_launch_injects_when_extra_is_none(
        self, mock_chdir, mock_updates, mock_ws, mock_get_flags, mock_launch_comfyui
    ):
        mock_ws.workspace_path = "/fake/workspace"
        mock_ws.workspace_type = "not_default"
        mock_ws.config_manager.config = {"DEFAULT": {}}

        from comfy_cli.command.launch import launch

        launch(background=False, extra=None)

        args, kwargs = mock_launch_comfyui.call_args
        extra_arg = args[0]
        assert extra_arg == ["--enable-manager"]

    @patch("comfy_cli.command.launch.launch_comfyui")
    @patch("comfy_cli.command.launch._get_manager_flags", return_value=["--enable-manager", "--disable-manager-ui"])
    @patch("comfy_cli.command.launch.workspace_manager")
    @patch("comfy_cli.command.launch.check_for_updates")
    @patch("os.chdir")
    def test_launch_injects_disable_gui_flags(
        self, mock_chdir, mock_updates, mock_ws, mock_get_flags, mock_launch_comfyui
    ):
        mock_ws.workspace_path = "/fake/workspace"
        mock_ws.workspace_type = "not_default"
        mock_ws.config_manager.config = {"DEFAULT": {}}

        from comfy_cli.command.launch import launch

        launch(background=False, extra=None)

        args, kwargs = mock_launch_comfyui.call_args
        extra_arg = args[0]
        assert "--enable-manager" in extra_arg
        assert "--disable-manager-ui" in extra_arg

    @patch("comfy_cli.command.launch.launch_comfyui")
    @patch(
        "comfy_cli.command.launch._get_manager_flags", return_value=["--enable-manager", "--enable-manager-legacy-ui"]
    )
    @patch("comfy_cli.command.launch.workspace_manager")
    @patch("comfy_cli.command.launch.check_for_updates")
    @patch("os.chdir")
    def test_launch_injects_legacy_gui_flags(
        self, mock_chdir, mock_updates, mock_ws, mock_get_flags, mock_launch_comfyui
    ):
        mock_ws.workspace_path = "/fake/workspace"
        mock_ws.workspace_type = "not_default"
        mock_ws.config_manager.config = {"DEFAULT": {}}

        from comfy_cli.command.launch import launch

        launch(background=False, extra=None)

        args, kwargs = mock_launch_comfyui.call_args
        extra_arg = args[0]
        assert "--enable-manager" in extra_arg
        assert "--enable-manager-legacy-ui" in extra_arg


class TestMigrateLegacy:
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_no_workspace_exits(self, mock_ws, mock_config_manager):
        """When workspace is not set, migrate-legacy should exit with error."""
        mock_ws.workspace_path = None

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        with pytest.raises(typer.Exit) as exc_info:
            migrate_legacy(yes=True)

        assert exc_info.value.exit_code == 1
        mock_config_manager.set.assert_not_called()

    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_with_cli_only_mode(self, mock_ws, mock_config_manager, tmp_path):
        # Setup: create legacy manager with .enable-cli-only-mode and .git
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        (legacy_manager / ".enable-cli-only-mode").touch()

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")
        # Verify moved to .disabled
        assert not legacy_manager.exists()
        assert (custom_nodes / ".disabled" / "ComfyUI-Manager").exists()

    @patch("comfy_cli.command.custom_nodes.command.subprocess.run")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_without_cli_only_mode(self, mock_ws, mock_subprocess_run, mock_config_manager, tmp_path):
        # Setup: create legacy manager with .git but without .enable-cli-only-mode
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        # Create manager_requirements.txt for successful install
        (tmp_path / constants.MANAGER_REQUIREMENTS_FILE).write_text("comfyui-manager>=4.1b1")

        mock_ws.workspace_path = str(tmp_path)
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "enable-gui")
        # Verify moved to .disabled
        assert not legacy_manager.exists()
        assert (custom_nodes / ".disabled" / "ComfyUI-Manager").exists()

    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_no_legacy_manager(self, mock_ws, mock_config_manager, tmp_path):
        # Setup: no legacy manager
        custom_nodes = tmp_path / "custom_nodes"
        custom_nodes.mkdir(parents=True)

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        # Should not call set when no legacy manager found
        mock_config_manager.set.assert_not_called()

    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_target_exists(self, mock_ws, mock_config_manager, tmp_path):
        # Setup: both source and target exist
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        (custom_nodes / ".disabled" / "ComfyUI-Manager").mkdir(parents=True)

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        with pytest.raises(typer.Exit):
            migrate_legacy(yes=True)

    @patch("comfy_cli.command.custom_nodes.command.subprocess.run")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_lowercase_directory(self, mock_ws, mock_subprocess_run, mock_config_manager, tmp_path):
        # Setup: create legacy manager with lowercase name and .git
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "comfyui-manager"  # lowercase
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        # Create manager_requirements.txt for successful install
        (tmp_path / constants.MANAGER_REQUIREMENTS_FILE).write_text("comfyui-manager>=4.1b1")

        mock_ws.workspace_path = str(tmp_path)
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "enable-gui")
        # Verify moved to .disabled (preserving original case)
        assert not legacy_manager.exists()
        assert (custom_nodes / ".disabled" / "comfyui-manager").exists()

    @patch("comfy_cli.command.custom_nodes.command.resolve_workspace_python", return_value="/workspace/venv/python")
    @patch("comfy_cli.command.custom_nodes.command.subprocess.run")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_installs_manager_requirements(
        self, mock_ws, mock_subprocess_run, mock_resolve_python, mock_config_manager, tmp_path
    ):
        # Setup: create legacy manager with .git and manager_requirements.txt
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        # Create manager_requirements.txt in workspace root
        (tmp_path / constants.MANAGER_REQUIREMENTS_FILE).write_text("comfyui-manager>=4.1b1")

        mock_ws.workspace_path = str(tmp_path)
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        # Verify pip install was called with workspace Python, NOT sys.executable
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]
        assert call_args[0] == "/workspace/venv/python"
        assert "-m" in call_args
        assert "pip" in call_args
        assert "install" in call_args
        assert "-r" in call_args
        # Verify the requirements file path is included
        assert any(constants.MANAGER_REQUIREMENTS_FILE in str(arg) for arg in call_args)

    @patch("comfy_cli.command.custom_nodes.command.subprocess.run")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_no_requirements_file(self, mock_ws, mock_subprocess_run, mock_config_manager, tmp_path):
        # Setup: create legacy manager with .git but NO manager_requirements.txt
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        # Verify pip install was NOT called (no requirements file)
        mock_subprocess_run.assert_not_called()
        # When requirements file is missing, install fails → set to disable
        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")

    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_not_git_repo(self, mock_ws, mock_config_manager, tmp_path):
        # Setup: create directory without .git (not a git repo)
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        # No .git directory

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        # Should not migrate non-git directories
        mock_config_manager.set.assert_not_called()
        # Directory should still exist (not moved)
        assert legacy_manager.exists()

    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_skips_symlink(self, mock_ws, mock_config_manager, tmp_path):
        # Setup: create a symlink instead of real directory
        custom_nodes = tmp_path / "custom_nodes"
        custom_nodes.mkdir(parents=True)
        real_dir = tmp_path / "real-manager"
        real_dir.mkdir()
        (real_dir / ".git").mkdir()
        symlink_path = custom_nodes / "ComfyUI-Manager"
        symlink_path.symlink_to(real_dir)

        mock_ws.workspace_path = str(tmp_path)

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=True)

        # Should not migrate symlinks
        mock_config_manager.set.assert_not_called()
        # Symlink should still exist
        assert symlink_path.is_symlink()

    @patch("comfy_cli.command.custom_nodes.command.shutil.move")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_move_error(self, mock_ws, mock_move, mock_config_manager, tmp_path):
        # Setup: create legacy manager with .git
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()
        (custom_nodes / ".disabled").mkdir()

        mock_ws.workspace_path = str(tmp_path)
        mock_move.side_effect = OSError("Permission denied")

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        with pytest.raises(typer.Exit):
            migrate_legacy(yes=True)

        # Config should not be set on move failure
        mock_config_manager.set.assert_not_called()

    @patch("comfy_cli.command.custom_nodes.command.ui.prompt_confirm_action")
    @patch("comfy_cli.command.custom_nodes.command.workspace_manager")
    def test_migrate_legacy_user_cancels(self, mock_ws, mock_confirm, mock_config_manager, tmp_path):
        # Setup: create legacy manager with .git
        custom_nodes = tmp_path / "custom_nodes"
        legacy_manager = custom_nodes / "ComfyUI-Manager"
        legacy_manager.mkdir(parents=True)
        (legacy_manager / ".git").mkdir()

        mock_ws.workspace_path = str(tmp_path)
        mock_confirm.return_value = False  # User cancels

        from comfy_cli.command.custom_nodes.command import migrate_legacy

        migrate_legacy(yes=False)

        # Should not migrate when user cancels
        mock_config_manager.set.assert_not_called()
        # Directory should still exist
        assert legacy_manager.exists()


class TestInstallSkipManager:
    """Tests for --skip-manager flag setting config to disable."""

    @patch("comfy_cli.command.install.update_node_id_cache")
    @patch("comfy_cli.command.install.pip_install_manager")
    @patch("comfy_cli.command.install.pip_install_comfyui_dependencies")
    @patch("comfy_cli.command.install.workspace_manager")
    @patch("comfy_cli.command.install.WorkspaceManager")
    @patch("comfy_cli.command.install.check_comfy_repo")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.command.install.ui.prompt_confirm_action")
    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.chdir")
    @patch("comfy_cli.command.install.ensure_workspace_python", return_value="/fake/python")
    def test_skip_manager_sets_disable_config(
        self,
        mock_ensure_python,
        mock_chdir,
        mock_makedirs,
        mock_exists,
        mock_config_manager_cls,
        mock_confirm,
        mock_clone,
        mock_check_repo,
        mock_ws_cls,
        mock_ws,
        mock_pip_deps,
        mock_pip_manager,
        mock_update_cache,
    ):
        """When --skip-manager is used, config should be set to disable."""
        # Setup mocks
        mock_exists.side_effect = lambda p: p == "/fake/comfy"  # repo exists
        mock_check_repo.return_value = (True, None)
        mock_ws.skip_prompting = True
        mock_config_manager = MagicMock()
        mock_config_manager_cls.return_value = mock_config_manager

        from comfy_cli.command.install import execute

        execute(
            url="https://github.com/comfyanonymous/ComfyUI",
            comfy_path="/fake/comfy",
            restore=False,
            skip_manager=True,  # Key flag
            version="nightly",
        )

        # Verify config was set to disable
        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")
        # Verify pip_install_manager was NOT called
        mock_pip_manager.assert_not_called()


class TestInstallManagerFailure:
    """Tests for pip_install_manager failure handling."""

    @patch("comfy_cli.command.install.update_node_id_cache")
    @patch("comfy_cli.command.install.pip_install_manager")
    @patch("comfy_cli.command.install.pip_install_comfyui_dependencies")
    @patch("comfy_cli.command.install.workspace_manager")
    @patch("comfy_cli.command.install.WorkspaceManager")
    @patch("comfy_cli.command.install.check_comfy_repo")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.command.install.ui.prompt_confirm_action")
    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.chdir")
    @patch("comfy_cli.command.install.ensure_workspace_python", return_value="/fake/python")
    def test_manager_install_failure_sets_disable_config(
        self,
        mock_ensure_python,
        mock_chdir,
        mock_makedirs,
        mock_exists,
        mock_config_manager_cls,
        mock_confirm,
        mock_clone,
        mock_check_repo,
        mock_ws_cls,
        mock_ws,
        mock_pip_deps,
        mock_pip_manager,
        mock_update_cache,
    ):
        """When pip_install_manager fails, config should be set to disable."""
        # Setup mocks
        mock_exists.side_effect = lambda p: p == "/fake/comfy"  # repo exists
        mock_check_repo.return_value = (True, None)
        mock_ws.skip_prompting = True
        mock_pip_manager.return_value = False  # Manager installation fails
        mock_config_manager = MagicMock()
        mock_config_manager_cls.return_value = mock_config_manager

        from comfy_cli.command.install import execute

        execute(
            url="https://github.com/comfyanonymous/ComfyUI",
            comfy_path="/fake/comfy",
            restore=False,
            skip_manager=False,  # Try to install manager
            version="nightly",
        )

        # Verify pip_install_manager was called
        mock_pip_manager.assert_called_once()
        # Verify config was set to disable due to failure
        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")

    @patch("comfy_cli.command.install.update_node_id_cache")
    @patch("comfy_cli.command.install.pip_install_manager")
    @patch("comfy_cli.command.install.pip_install_comfyui_dependencies")
    @patch("comfy_cli.command.install.workspace_manager")
    @patch("comfy_cli.command.install.WorkspaceManager")
    @patch("comfy_cli.command.install.check_comfy_repo")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.command.install.ui.prompt_confirm_action")
    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.chdir")
    @patch("comfy_cli.command.install.ensure_workspace_python", return_value="/fake/python")
    def test_manager_install_success_does_not_set_disable(
        self,
        mock_ensure_python,
        mock_chdir,
        mock_makedirs,
        mock_exists,
        mock_config_manager_cls,
        mock_confirm,
        mock_clone,
        mock_check_repo,
        mock_ws_cls,
        mock_ws,
        mock_pip_deps,
        mock_pip_manager,
        mock_update_cache,
    ):
        """When pip_install_manager succeeds, config should NOT be set to disable."""
        # Setup mocks
        mock_exists.side_effect = lambda p: p == "/fake/comfy"  # repo exists
        mock_check_repo.return_value = (True, None)
        mock_ws.skip_prompting = True
        mock_pip_manager.return_value = True  # Manager installation succeeds
        mock_config_manager = MagicMock()
        mock_config_manager_cls.return_value = mock_config_manager

        from comfy_cli.command.install import execute

        execute(
            url="https://github.com/comfyanonymous/ComfyUI",
            comfy_path="/fake/comfy",
            restore=False,
            skip_manager=False,
            version="nightly",
        )

        # Verify pip_install_manager was called
        mock_pip_manager.assert_called_once()
        # Verify config was NOT set to disable
        mock_config_manager.set.assert_not_called()

    @patch("comfy_cli.command.install.DependencyCompiler")
    @patch("comfy_cli.command.install.update_node_id_cache")
    @patch("comfy_cli.command.install.pip_install_manager")
    @patch("comfy_cli.command.install.pip_install_comfyui_dependencies")
    @patch("comfy_cli.command.install.workspace_manager")
    @patch("comfy_cli.command.install.WorkspaceManager")
    @patch("comfy_cli.command.install.check_comfy_repo")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.command.install.ui.prompt_confirm_action")
    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("os.path.exists")
    @patch("os.makedirs")
    @patch("os.chdir")
    @patch("comfy_cli.command.install.ensure_workspace_python", return_value="/fake/python")
    def test_fast_deps_manager_failure_sets_disable_config(
        self,
        mock_ensure_python,
        mock_chdir,
        mock_makedirs,
        mock_exists,
        mock_config_manager_cls,
        mock_confirm,
        mock_clone,
        mock_check_repo,
        mock_ws_cls,
        mock_ws,
        mock_pip_deps,
        mock_pip_manager,
        mock_update_cache,
        mock_dep_compiler,
    ):
        """When fast_deps=True and pip_install_manager fails, config should be set to disable."""
        # Setup mocks
        mock_exists.side_effect = lambda p: p == "/fake/comfy"
        mock_check_repo.return_value = (True, None)
        mock_ws.skip_prompting = True
        mock_pip_manager.return_value = False  # Manager installation fails
        mock_config_manager = MagicMock()
        mock_config_manager_cls.return_value = mock_config_manager
        mock_dep_compiler_instance = MagicMock()
        mock_dep_compiler.return_value = mock_dep_compiler_instance

        from comfy_cli.command.install import execute

        execute(
            url="https://github.com/comfyanonymous/ComfyUI",
            comfy_path="/fake/comfy",
            restore=False,
            skip_manager=False,
            version="nightly",
            fast_deps=True,  # Use fast_deps path
        )

        # Verify pip_install_manager was called (fast_deps path)
        mock_pip_manager.assert_called_once()
        # Verify config was set to disable due to failure
        mock_config_manager.set.assert_called_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")


class TestPipInstallManagerCacheClear:
    """Tests for pip_install_manager cache clearing after successful install."""

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli")
    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_pip_install_manager_clears_cache_on_success(self, mock_exists, mock_run, mock_find_cm_cli):
        """When pip install succeeds, find_cm_cli cache should be cleared."""
        from comfy_cli.command.install import pip_install_manager

        # Simulate successful pip install
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Call pip_install_manager
        result = pip_install_manager("/fake/repo")

        # Verify success
        assert result is True
        # Verify cache_clear was called on the mock
        mock_find_cm_cli.cache_clear.assert_called_once()

    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_pip_install_manager_no_cache_clear_on_failure(self, mock_exists, mock_run):
        """When pip install fails, cache should not be affected (function returns early)."""
        from comfy_cli.command.install import pip_install_manager

        # Simulate failed pip install
        mock_run.return_value = MagicMock(returncode=1)

        # Call pip_install_manager
        result = pip_install_manager("/fake/repo")

        # Verify failure
        assert result is False


class TestFillPrintTable:
    """Tests for WorkspaceManager.fill_print_table() method."""

    @pytest.fixture()
    def mock_workspace_config_manager(self):
        with patch("comfy_cli.workspace_manager.ConfigManager") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            yield instance

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="disable")
    def test_fill_print_table_disable_mode(self, mock_resolve, mock_workspace_config_manager):
        """When mode is 'disable', status should show Disabled."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert len(result) == 3
        assert result[0][0] == "Current selected workspace"
        assert result[1][0] == "Manager"
        assert "Disabled" in result[1][1]
        assert result[2][0] == "UV Compile Default"

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-gui")
    def test_fill_print_table_enable_gui_mode(self, mock_resolve, mock_workspace_config_manager):
        """When mode is 'enable-gui', status should show GUI Enabled."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[1][0] == "Manager"
        assert "GUI Enabled" in result[1][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="disable-gui")
    def test_fill_print_table_disable_gui_mode(self, mock_resolve, mock_workspace_config_manager):
        """When mode is 'disable-gui', status should show GUI Disabled."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[1][0] == "Manager"
        assert "GUI Disabled" in result[1][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-legacy-gui")
    def test_fill_print_table_enable_legacy_gui_mode(self, mock_resolve, mock_workspace_config_manager):
        """When mode is 'enable-legacy-gui', status should show Legacy GUI."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[1][0] == "Manager"
        assert "Legacy GUI" in result[1][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="not-installed")
    def test_fill_print_table_not_installed(self, mock_resolve, mock_workspace_config_manager):
        """When resolve returns 'not-installed', status should show Not Installed."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[1][0] == "Manager"
        assert "Not Installed" in result[1][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="unknown-mode")
    def test_fill_print_table_unknown_mode_defaults_to_enable(self, mock_resolve, mock_workspace_config_manager):
        """When mode is unknown, status should default to GUI Enabled."""
        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[1][0] == "Manager"
        assert "GUI Enabled" in result[1][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-gui")
    def test_fill_print_table_uv_compile_enabled(self, mock_resolve, mock_workspace_config_manager):
        """When uv_compile_default is True, status should show Enabled."""
        mock_workspace_config_manager.get.side_effect = lambda key: {
            constants.CONFIG_KEY_UV_COMPILE_DEFAULT: "True",
        }.get(key)

        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[2][0] == "UV Compile Default"
        assert "Enabled" in result[2][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-gui")
    def test_fill_print_table_uv_compile_disabled(self, mock_resolve, mock_workspace_config_manager):
        """When uv_compile_default is not set, status should show Disabled."""
        mock_workspace_config_manager.get.return_value = None

        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[2][0] == "UV Compile Default"
        assert "Disabled" in result[2][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-gui")
    def test_fill_print_table_uv_compile_lowercase_true(self, mock_resolve, mock_workspace_config_manager):
        """When uv_compile_default is 'true' (lowercase), status should show Enabled."""
        mock_workspace_config_manager.get.return_value = "true"

        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[2][0] == "UV Compile Default"
        assert "Enabled" in result[2][1]

    @patch("comfy_cli.command.custom_nodes.cm_cli_util.resolve_manager_gui_mode", return_value="enable-gui")
    def test_fill_print_table_uv_compile_explicit_false(self, mock_resolve, mock_workspace_config_manager):
        """When uv_compile_default is 'False', status should show Disabled."""
        mock_workspace_config_manager.get.return_value = "False"

        from comfy_cli.workspace_manager import WorkspaceManager

        ws = WorkspaceManager()
        ws.workspace_path = "/fake/workspace"

        result = ws.fill_print_table()

        assert result[2][0] == "UV Compile Default"
        assert "Disabled" in result[2][1]


class TestResolveUvCompile:
    """Tests for _resolve_uv_compile() helper function."""

    @pytest.fixture()
    def mock_resolve_config_manager(self):
        with patch("comfy_cli.command.custom_nodes.command.ConfigManager") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            yield instance

    def test_explicit_true_returns_true(self, mock_resolve_config_manager):
        """Explicit --uv-compile overrides everything."""
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(True) is True

    def test_explicit_false_returns_false(self, mock_resolve_config_manager):
        """Explicit --no-uv-compile overrides everything."""
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(False) is False

    def test_explicit_true_ignores_config(self, mock_resolve_config_manager):
        """Explicit flag takes priority over config default."""
        mock_resolve_config_manager.get.return_value = "False"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(True) is True
        mock_resolve_config_manager.get.assert_not_called()

    def test_none_with_config_true(self, mock_resolve_config_manager):
        """None (no flag) + config True → True."""
        mock_resolve_config_manager.get.return_value = "True"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None) is True

    def test_none_with_config_false(self, mock_resolve_config_manager):
        """None (no flag) + config False → False."""
        mock_resolve_config_manager.get.return_value = "False"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None) is False

    def test_none_with_no_config(self, mock_resolve_config_manager):
        """None (no flag) + no config → False."""
        mock_resolve_config_manager.get.return_value = None
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None) is False

    def test_config_true_overridden_by_fast_deps(self, mock_resolve_config_manager):
        """Config True + --fast-deps → False (silent override)."""
        mock_resolve_config_manager.get.return_value = "True"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None, fast_deps=True) is False

    def test_config_true_overridden_by_no_deps(self, mock_resolve_config_manager):
        """Config True + --no-deps → False (silent override)."""
        mock_resolve_config_manager.get.return_value = "True"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None, no_deps=True) is False

    def test_config_false_with_fast_deps_stays_false(self, mock_resolve_config_manager):
        """Config False + --fast-deps → False (no change)."""
        mock_resolve_config_manager.get.return_value = "False"
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(None, fast_deps=True) is False

    def test_explicit_true_not_affected_by_fast_deps(self, mock_resolve_config_manager):
        """Explicit --uv-compile is not silently overridden (mutual exclusivity handled elsewhere)."""
        from comfy_cli.command.custom_nodes.command import _resolve_uv_compile

        assert _resolve_uv_compile(True, fast_deps=True) is True


class TestUvCompileDefaultCommand:
    """Tests for comfy manager uv-compile-default command."""

    def test_uv_compile_default_enable(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import uv_compile_default

        uv_compile_default(enabled=True)

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_UV_COMPILE_DEFAULT, "True")

    def test_uv_compile_default_disable(self, mock_config_manager):
        from comfy_cli.command.custom_nodes.command import uv_compile_default

        uv_compile_default(enabled=False)

        mock_config_manager.set.assert_called_once_with(constants.CONFIG_KEY_UV_COMPILE_DEFAULT, "False")


class TestFindCmCli:
    """Tests for find_cm_cli() function."""

    def test_find_cm_cli_module_found(self):
        """When cm_cli module exists, should return True."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = MagicMock()  # Non-None means module exists
            # Clear cache before test
            from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli

            find_cm_cli.cache_clear()

            result = find_cm_cli()

            assert result is True
            mock_find_spec.assert_called_with("cm_cli")

    def test_find_cm_cli_module_not_found(self):
        """When cm_cli module doesn't exist, should return False."""
        from comfy_cli.command.custom_nodes import cm_cli_util as _cm_cli_util

        with (
            patch("importlib.util.find_spec") as mock_find_spec,
            patch.object(_cm_cli_util.workspace_manager, "workspace_path", None),
        ):
            mock_find_spec.return_value = None  # None means module not found
            from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli

            find_cm_cli.cache_clear()

            result = find_cm_cli()

            assert result is False
            mock_find_spec.assert_called_with("cm_cli")

    def test_find_cm_cli_cache_behavior(self):
        """find_cm_cli should cache results and not call find_spec repeatedly."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = MagicMock()
            from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli

            find_cm_cli.cache_clear()

            # Call multiple times
            result1 = find_cm_cli()
            result2 = find_cm_cli()
            result3 = find_cm_cli()

            # All should return True
            assert result1 is True
            assert result2 is True
            assert result3 is True
            # find_spec should only be called once due to caching
            assert mock_find_spec.call_count == 1


class TestPipInstallManagerEdgeCases:
    """Additional edge case tests for pip_install_manager()."""

    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=False)
    def test_pip_install_manager_requirements_not_found(self, mock_exists, mock_run):
        """When requirements file doesn't exist, should return False without calling pip."""
        from comfy_cli.command.install import pip_install_manager

        result = pip_install_manager("/fake/repo")

        assert result is False
        # subprocess.run should NOT be called
        mock_run.assert_not_called()


class TestValidateComfyuiManager:
    """Tests for validate_comfyui_manager() function."""

    @patch("comfy_cli.command.custom_nodes.command.find_cm_cli", return_value=False)
    def test_validate_comfyui_manager_exits_when_not_found(self, mock_find_cm_cli):
        """When cm-cli is not found, should raise typer.Exit with code 1."""
        from comfy_cli.command.custom_nodes.command import validate_comfyui_manager

        with pytest.raises(typer.Exit) as exc_info:
            validate_comfyui_manager()

        assert exc_info.value.exit_code == 1
        mock_find_cm_cli.assert_called_once()

    @patch("comfy_cli.command.custom_nodes.command.find_cm_cli", return_value=True)
    def test_validate_comfyui_manager_passes_when_found(self, mock_find_cm_cli):
        """When cm-cli is found, should not raise any exception."""
        from comfy_cli.command.custom_nodes.command import validate_comfyui_manager

        # Should not raise
        validate_comfyui_manager()

        mock_find_cm_cli.assert_called_once()
