from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.command.install import pip_install_manager, validate_version


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


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
