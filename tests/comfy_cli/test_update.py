from unittest.mock import MagicMock, patch

import requests

from comfy_cli.update import check_for_newer_pypi_version, check_for_updates


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


class TestCheckForUpdates:
    @patch("comfy_cli.update.notify_update")
    @patch("comfy_cli.update.get_version_from_pyproject", return_value="1.0.0")
    @patch("comfy_cli.update.requests.get")
    def test_notifies_when_update_available(self, mock_get, _mock_ver, mock_notify):
        mock_get.return_value = _mock_pypi_response("2.0.0")
        check_for_updates()
        mock_notify.assert_called_once_with("1.0.0", "2.0.0")

    @patch("comfy_cli.update.notify_update")
    @patch("comfy_cli.update.get_version_from_pyproject", return_value="1.0.0")
    @patch("comfy_cli.update.requests.get")
    def test_no_notification_on_network_error(self, mock_get, _mock_ver, mock_notify):
        mock_get.side_effect = requests.ConnectionError("offline")
        check_for_updates()
        mock_notify.assert_not_called()
