from unittest.mock import MagicMock, patch

import requests

from comfy_cli.update import check_for_newer_pypi_version


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
# welcome banner consumes it inline.
