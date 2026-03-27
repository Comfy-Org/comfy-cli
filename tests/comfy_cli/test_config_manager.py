import configparser
import os
from unittest.mock import patch

import pytest

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager

# Unwrap the singleton to access the original class for testing.
# The singleton decorator stores the original class as the 'cls'
# free variable in the wrapper closure.
_ConfigManagerCls = ConfigManager.__closure__[0].cell_contents


def _make_config_manager(config_dir, is_running_val=True):
    with (
        patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)),
        patch("comfy_cli.config_manager.is_running", return_value=is_running_val),
    ):
        return _ConfigManagerCls()


@pytest.fixture
def config_mgr(tmp_path):
    config_dir = tmp_path / "comfy-cli"
    config_dir.mkdir()
    yield _make_config_manager(config_dir)


class TestLoad:
    def test_creates_tmp_directory(self, tmp_path):
        config_dir = tmp_path / "comfy-cli"
        config_dir.mkdir()
        _make_config_manager(config_dir)
        assert (config_dir / "tmp").is_dir()

    def test_reads_existing_config(self, tmp_path):
        config_dir = tmp_path / "comfy-cli"
        config_dir.mkdir()
        (config_dir / "config.ini").write_text(
            f"[DEFAULT]\n{constants.CONFIG_KEY_DEFAULT_WORKSPACE} = /path/to/comfy\n"
        )
        mgr = _make_config_manager(config_dir)
        assert mgr.get(constants.CONFIG_KEY_DEFAULT_WORKSPACE) == "/path/to/comfy"

    def test_parses_background_info(self, tmp_path):
        config_dir = tmp_path / "comfy-cli"
        config_dir.mkdir()
        (config_dir / "config.ini").write_text(
            f"[DEFAULT]\n{constants.CONFIG_KEY_BACKGROUND} = ('localhost', 8188, 12345)\n"
        )
        mgr = _make_config_manager(config_dir, is_running_val=True)
        assert mgr.background == ("localhost", 8188, 12345)

    def test_removes_background_when_stale_pid(self, tmp_path):
        config_dir = tmp_path / "comfy-cli"
        config_dir.mkdir()
        (config_dir / "config.ini").write_text(
            f"[DEFAULT]\n{constants.CONFIG_KEY_BACKGROUND} = ('localhost', 8188, 99999)\n"
        )
        mgr = _make_config_manager(config_dir, is_running_val=False)
        assert mgr.background is None
        assert constants.CONFIG_KEY_BACKGROUND not in mgr.config["DEFAULT"]


class TestWriteConfig:
    def test_creates_directory_if_missing(self, tmp_path):
        config_dir = tmp_path / "new-dir"
        with patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)):
            mgr = _ConfigManagerCls.__new__(_ConfigManagerCls)
            mgr.config = configparser.ConfigParser()
            mgr.background = None
            mgr.write_config()
        assert (config_dir / "config.ini").exists()

    def test_set_persists_to_file(self, config_mgr):
        config_mgr.set("my_key", "my_value")
        parser = configparser.ConfigParser()
        parser.read(config_mgr.get_config_file_path())
        assert parser["DEFAULT"]["my_key"] == "my_value"


class TestGetBool:
    def test_missing_key_returns_none(self, config_mgr):
        assert config_mgr.get_bool("nonexistent") is None


class TestGetOrOverride:
    def test_set_value_wins(self, config_mgr):
        config_mgr.config["DEFAULT"]["k"] = "from_config"
        with patch.dict(os.environ, {"EK": "from_env"}):
            assert config_mgr.get_or_override("EK", "k", set_value="from_cli") == "from_cli"

    def test_env_var_wins_over_config(self, config_mgr):
        config_mgr.config["DEFAULT"]["k"] = "from_config"
        with patch.dict(os.environ, {"EK": "from_env"}):
            assert config_mgr.get_or_override("EK", "k") == "from_env"

    def test_config_is_fallback(self, config_mgr):
        config_mgr.config["DEFAULT"]["k"] = "from_config"
        env = os.environ.copy()
        env.pop("EK", None)
        with patch.dict(os.environ, env, clear=True):
            assert config_mgr.get_or_override("EK", "k") == "from_config"

    def test_empty_set_value_returns_none(self, config_mgr):
        assert config_mgr.get_or_override("EK", "k", set_value="") is None

    def test_empty_env_var_returns_none(self, config_mgr):
        with patch.dict(os.environ, {"EK": ""}):
            assert config_mgr.get_or_override("EK", "k") is None

    def test_set_value_is_persisted(self, config_mgr):
        config_mgr.get_or_override("EK", "k", set_value="saved")
        assert config_mgr.get("k") == "saved"

    def test_all_missing_returns_none(self, config_mgr):
        env = os.environ.copy()
        env.pop("EK", None)
        with patch.dict(os.environ, env, clear=True):
            assert config_mgr.get_or_override("EK", "k") is None


class TestGetEnvData:
    def test_full_config(self, config_mgr):
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_DEFAULT_WORKSPACE] = "/my/ws"
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS] = "--cpu"
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_RECENT_WORKSPACE] = "/recent"
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_ENABLE_TRACKING] = "true"
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND] = "('localhost', 8188, 42)"
        config_mgr.background = ("localhost", 8188, 42)

        data = dict(config_mgr.get_env_data())
        assert data["Default ComfyUI workspace"] == "/my/ws"
        assert data["Default ComfyUI launch extra options"] == "--cpu"
        assert data["Recent ComfyUI workspace"] == "/recent"
        assert data["Tracking Analytics"] == "Enabled"
        assert "localhost:8188" in data["Background ComfyUI"]
        assert "42" in data["Background ComfyUI"]

    def test_empty_config(self, config_mgr):
        data = dict(config_mgr.get_env_data())
        assert data["Default ComfyUI workspace"] == "No default ComfyUI workspace"
        assert data["Recent ComfyUI workspace"] == "No recent run"
        assert "Tracking Analytics" not in data
        assert "None" in data["Default ComfyUI launch extra options"]

    def test_launch_extras_only_read_when_workspace_set(self, config_mgr):
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS] = "--gpu"
        data = dict(config_mgr.get_env_data())
        assert "None" in data["Default ComfyUI launch extra options"]


class TestRemoveBackground:
    def test_clears_background(self, config_mgr):
        config_mgr.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND] = "('h', 1, 2)"
        config_mgr.background = ("h", 1, 2)
        config_mgr.remove_background()
        assert config_mgr.background is None
        assert constants.CONFIG_KEY_BACKGROUND not in config_mgr.config["DEFAULT"]
