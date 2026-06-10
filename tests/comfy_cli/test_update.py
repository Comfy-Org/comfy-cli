import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from comfy_cli import update
from comfy_cli.update import (
    UPDATE_CHECK_DISABLE_ENV,
    check_for_newer_pypi_version,
    latest_upgrade_version,
    upgrade_cli,
)


def _mock_pypi_response(latest_version):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"info": {"version": latest_version}}
    return mock_resp


class TestCheckForNewerPypiVersion:
    @patch("comfy_cli.update.requests.get")
    def test_newer_version_available(self, mock_get):
        mock_get.return_value = _mock_pypi_response("99.0.0")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is True
        assert ver == "99.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_no_update_when_current(self, mock_get):
        mock_get.return_value = _mock_pypi_response("1.0.0")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is False
        assert ver == "1.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_network_failure_returns_false(self, mock_get):
        mock_get.side_effect = requests.Timeout("connection timed out")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is False
        assert ver == "1.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_timeout_value_is_passed(self, mock_get):
        mock_get.return_value = _mock_pypi_response("1.0.0")
        check_for_newer_pypi_version("comfy-cli", "1.0.0")
        mock_get.assert_called_once_with("https://pypi.org/pypi/comfy-cli/json", timeout=5)


# TestCheckForUpdates was removed alongside the bright-blue
# "🔔 Update Available!" panel (Task 5 of the CLI UX consistency pass).
# ``check_for_newer_pypi_version`` is still tested directly above; the
# welcome banner consumes ``latest_upgrade_version`` (below) inline.


class TestLatestUpgradeVersion:
    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_returns_newer_version_and_writes_cache(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        result = latest_upgrade_version("1.0.0", tmp_path)
        assert result == "99.0.0"
        cache = tmp_path / "update-check.json"
        assert json.loads(cache.read_text())["latest"] == "99.0.0"

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_returns_none_when_current(self, mock_check, tmp_path):
        mock_check.return_value = (False, "1.0.0")
        assert latest_upgrade_version("1.0.0", tmp_path) is None

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_fresh_cache_skips_network(self, mock_check, tmp_path):
        cache = tmp_path / "update-check.json"
        cache.write_text(json.dumps({"checked_at": time.time(), "latest": "99.0.0"}))
        assert latest_upgrade_version("1.0.0", tmp_path) == "99.0.0"
        mock_check.assert_not_called()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_stale_cache_refreshes(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        cache = tmp_path / "update-check.json"
        cache.write_text(json.dumps({"checked_at": 0, "latest": "2.0.0"}))
        assert latest_upgrade_version("1.0.0", tmp_path) == "99.0.0"
        mock_check.assert_called_once()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_disabled_via_env(self, mock_check, tmp_path, monkeypatch):
        monkeypatch.setenv(UPDATE_CHECK_DISABLE_ENV, "1")
        assert latest_upgrade_version("1.0.0", tmp_path) is None
        mock_check.assert_not_called()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_corrupt_cache_does_not_raise(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        cache = tmp_path / "update-check.json"
        cache.write_text("not json")
        assert latest_upgrade_version("1.0.0", tmp_path) is None


class TestUpgradeCli:
    @patch("comfy_cli.update.subprocess.run")
    def test_runs_pip_install_upgrade(self, mock_run):
        upgrade_cli()
        args = mock_run.call_args[0][0]
        assert args[1:] == ["-m", "pip", "install", "-U", "comfy-cli"]
        assert mock_run.call_args[1]["check"] is True


@pytest.mark.parametrize("payload", ["[]", '"x"', '{"checked_at": "abc"}'])
def test_latest_upgrade_version_survives_malformed_cache(tmp_path, payload):
    (tmp_path / "update-check.json").write_text(payload)
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None


@patch("comfy_cli.update.check_for_newer_pypi_version")
def test_latest_upgrade_version_survives_missing_latest_key(mock_check, tmp_path):
    # {"checked_at": 0} is stale — triggers refresh; the payload itself lacks "latest"
    # but refresh returns a valid current_version, so we just confirm no crash.
    mock_check.return_value = (False, "1.0.0")
    (tmp_path / "update-check.json").write_text('{"checked_at": 0}')
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None


def test_latest_upgrade_version_survives_unparseable_latest(tmp_path):
    (tmp_path / "update-check.json").write_text(json.dumps({"checked_at": time.time(), "latest": "unknown"}))
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None
