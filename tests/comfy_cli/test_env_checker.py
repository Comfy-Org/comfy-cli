import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from comfy_cli.env_checker import EnvChecker, check_comfy_server_running, format_python_version

_EnvCheckerCls = EnvChecker.__closure__[0].cell_contents


class TestFormatPythonVersion:
    def test_modern_python(self):
        v = SimpleNamespace(major=3, minor=12, micro=1)
        assert format_python_version(v) == "3.12.1"

    def test_python_39_is_modern(self):
        v = SimpleNamespace(major=3, minor=9, micro=0)
        assert format_python_version(v) == "3.9.0"

    def test_python_38_is_old(self):
        v = SimpleNamespace(major=3, minor=8, micro=5)
        result = format_python_version(v)
        assert "bold red" in result
        assert "3.8.5" in result

    def test_python_37_is_old(self):
        v = SimpleNamespace(major=3, minor=7, micro=0)
        result = format_python_version(v)
        assert "bold red" in result


class TestCheckComfyServerRunning:
    @patch("comfy_cli.env_checker.requests.get")
    def test_server_running(self, mock_get):
        mock_get.return_value.status_code = 200
        assert check_comfy_server_running() is True

    @patch("comfy_cli.env_checker.requests.get")
    def test_server_not_running(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        assert check_comfy_server_running() is False

    @patch("comfy_cli.env_checker.requests.get")
    def test_non_200_status(self, mock_get):
        mock_get.return_value.status_code = 500
        assert check_comfy_server_running() is False

    @patch("comfy_cli.env_checker.requests.get")
    def test_custom_port_and_host(self, mock_get):
        mock_get.return_value.status_code = 200
        check_comfy_server_running(port=9999, host="0.0.0.0")
        mock_get.assert_called_with("http://0.0.0.0:9999/history")


class TestEnvChecker:
    @pytest.fixture
    def checker(self):
        inst = _EnvCheckerCls.__new__(_EnvCheckerCls)
        inst.python_version = sys.version_info
        inst.virtualenv_path = None
        inst.conda_env = None
        return inst

    def test_check_detects_virtualenv(self, checker):
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/path/to/venv"}):
            checker.check()
        assert checker.virtualenv_path == "/path/to/venv"

    def test_check_detects_conda(self, checker):
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "myenv"}):
            checker.check()
        assert checker.conda_env == "myenv"

    def test_check_no_isolated_env(self, checker):
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("CONDA_DEFAULT_ENV", None)
        with patch.dict(os.environ, env, clear=True):
            checker.check()
        assert checker.virtualenv_path is None
        assert checker.conda_env is None

    def test_get_isolated_env_prefers_venv(self, checker):
        checker.virtualenv_path = "/venv"
        checker.conda_env = "conda"
        assert checker.get_isolated_env() == "/venv"

    def test_get_isolated_env_falls_back_to_conda(self, checker):
        checker.conda_env = "conda"
        assert checker.get_isolated_env() == "conda"

    @patch("comfy_cli.env_checker.check_comfy_server_running", return_value=True)
    @patch("comfy_cli.env_checker.ConfigManager")
    def test_fill_print_table_server_running(self, mock_cm, mock_server, checker):
        mock_cm.return_value.get_env_data.return_value = []
        data = dict(checker.fill_print_table())
        assert "Yes" in data["Comfy Server Running"]

    @patch("comfy_cli.env_checker.check_comfy_server_running", return_value=False)
    @patch("comfy_cli.env_checker.ConfigManager")
    def test_fill_print_table_server_not_running(self, mock_cm, mock_server, checker):
        mock_cm.return_value.get_env_data.return_value = []
        data = dict(checker.fill_print_table())
        assert "No" in data["Comfy Server Running"]
