from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.command.install import _install_manager_with_fallback, pip_install_manager, validate_version


def test_validate_version_nightly():
    assert validate_version("nightly") == "nightly"
    assert validate_version("NIGHTLY") == "nightly"


def test_validate_version_latest():
    assert validate_version("latest") == "latest"
    assert validate_version("LATEST") == "latest"


def test_validate_version_valid_semver():
    assert validate_version("1.2.3") == "1.2.3"
    assert validate_version("v1.2.3") == "1.2.3"
    assert validate_version("1.2.3-alpha") == "1.2.3-alpha"


def test_validate_version_invalid():
    with pytest.raises(ValueError):
        validate_version("invalid_version")


def test_validate_version_empty():
    with pytest.raises(ValueError):
        validate_version("")


class TestPipInstallManager:
    @patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli")
    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_success(self, mock_exists, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=0)
        result = pip_install_manager("/fake/repo")
        assert result is True
        mock_run.assert_called_once()

    @patch("os.path.exists", return_value=False)
    def test_missing_requirements_file(self, mock_exists):
        result = pip_install_manager("/fake/repo")
        assert result is False

    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_pip_failure(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="some error")
        result = pip_install_manager("/fake/repo")
        assert result is False

    @patch("comfy_cli.command.install.subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_pip_failure_no_stderr(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="")
        result = pip_install_manager("/fake/repo")
        assert result is False


class TestInstallManagerWithFallback:
    """The dedupe'd manager-install-and-degrade helper shared by the pip and
    fast_deps paths of ``execute``."""

    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("comfy_cli.command.install.pip_install_manager", return_value=True)
    @patch("comfy_cli.command.install.ensure_pip")
    def test_bootstrap_pip_true_bootstraps_and_installs(self, mock_ensure_pip, mock_install, mock_cfg):
        _install_manager_with_fallback("/fake/repo", "python", bootstrap_pip=True)
        mock_ensure_pip.assert_called_once_with("python")
        mock_install.assert_called_once_with("/fake/repo", python="python")
        # Success: manager GUI mode is left untouched.
        mock_cfg.return_value.set.assert_not_called()

    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("comfy_cli.command.install.pip_install_manager", return_value=True)
    @patch("comfy_cli.command.install.ensure_pip")
    def test_bootstrap_pip_false_skips_bootstrap(self, mock_ensure_pip, mock_install, mock_cfg):
        _install_manager_with_fallback("/fake/repo", "python", bootstrap_pip=False)
        mock_ensure_pip.assert_not_called()
        mock_install.assert_called_once_with("/fake/repo", python="python")

    @patch("comfy_cli.config_manager.ConfigManager")
    @patch("comfy_cli.command.install.pip_install_manager", return_value=False)
    @patch("comfy_cli.command.install.ensure_pip")
    def test_failure_disables_manager_gui_mode(self, mock_ensure_pip, mock_install, mock_cfg):
        _install_manager_with_fallback("/fake/repo", "python", bootstrap_pip=False)
        mock_cfg.return_value.set.assert_called_once_with(constants.CONFIG_KEY_MANAGER_GUI_MODE, "disable")


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
