from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.standalone import _resolve_python_version, download_standalone_python

# Minimal SHA256SUMS content matching real format
SAMPLE_SHA256SUMS = """\
aaa  cpython-3.10.20+20260310-aarch64-apple-darwin-install_only.tar.gz
bbb  cpython-3.10.20+20260310-x86_64-pc-windows-msvc-install_only.tar.gz
ccc  cpython-3.12.13+20260310-aarch64-apple-darwin-install_only.tar.gz
ddd  cpython-3.12.13+20260310-x86_64-pc-windows-msvc-install_only.tar.gz
eee  cpython-3.12.13+20260310-x86_64_v3-unknown-linux-gnu-install_only.tar.gz
fff  cpython-3.13.12+20260310-x86_64-pc-windows-msvc-install_only.tar.gz
"""


def _mock_response(text, status_code=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code != 200:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


class TestResolvePythonVersion:
    @patch("comfy_cli.standalone.requests.get")
    def test_resolves_312(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        result = _resolve_python_version("https://example.com/release", "3.12")
        assert result == "3.12.13"

    @patch("comfy_cli.standalone.requests.get")
    def test_resolves_310(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        result = _resolve_python_version("https://example.com/release", "3.10")
        assert result == "3.10.20"

    @patch("comfy_cli.standalone.requests.get")
    def test_resolves_313(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        result = _resolve_python_version("https://example.com/release", "3.13")
        assert result == "3.13.12"

    @patch("comfy_cli.standalone.requests.get")
    def test_missing_version_raises(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        with pytest.raises(RuntimeError, match="No Python 3.14.x found"):
            _resolve_python_version("https://example.com/release", "3.14")

    @patch("comfy_cli.standalone.requests.get")
    def test_http_error_propagates(self, mock_get):
        mock_get.return_value = _mock_response("", status_code=404)
        with pytest.raises(Exception, match="HTTP 404"):
            _resolve_python_version("https://example.com/release", "3.12")

    @patch("comfy_cli.standalone.requests.get")
    def test_picks_highest_patch(self, mock_get):
        """If multiple patch versions exist for a minor series, pick the highest."""
        sha256sums = """\
aaa  cpython-3.12.10+20260310-x86_64-install_only.tar.gz
bbb  cpython-3.12.13+20260310-x86_64-install_only.tar.gz
ccc  cpython-3.12.9+20260310-x86_64-install_only.tar.gz
"""
        mock_get.return_value = _mock_response(sha256sums)
        result = _resolve_python_version("https://example.com/release", "3.12")
        assert result == "3.12.13"

    @patch("comfy_cli.standalone.requests.get")
    def test_url_construction(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        _resolve_python_version("https://example.com/release/", "3.12")
        mock_get.assert_called_once_with("https://example.com/release/SHA256SUMS")

    @patch("comfy_cli.standalone.requests.get")
    def test_no_false_match_across_minor(self, mock_get):
        """3.1 should not match 3.12 or 3.10."""
        mock_get.return_value = _mock_response(SAMPLE_SHA256SUMS)
        with pytest.raises(RuntimeError, match="No Python 3.1.x found"):
            _resolve_python_version("https://example.com/release", "3.1")


class TestDownloadStandalonePython:
    @patch("comfy_cli.standalone.download_url")
    @patch("comfy_cli.standalone.requests.get")
    def test_minor_version_triggers_resolution(self, mock_get, mock_download):
        """When version is a minor version (X.Y), it should resolve the patch."""
        mock_get.side_effect = [
            _mock_response('{"tag": "20260310", "asset_url_prefix": "https://example.com/release"}'),
            _mock_response(SAMPLE_SHA256SUMS),
        ]
        mock_download.return_value = "python.tar.gz"

        download_standalone_python(platform="linux", proc="x86_64", version="3.12")

        # Should have fetched latest-release.json and SHA256SUMS
        assert mock_get.call_count == 2
        # Download URL should contain resolved version
        call_args = mock_download.call_args
        assert "3.12.13" in call_args[1].get("url", "") or "3.12.13" in str(call_args)

    @patch("comfy_cli.standalone.download_url")
    @patch("comfy_cli.standalone.requests.get")
    def test_full_version_skips_resolution(self, mock_get, mock_download):
        """When version is a full version (X.Y.Z), no resolution needed."""
        mock_get.return_value = _mock_response('{"tag": "20260310", "asset_url_prefix": "https://example.com/release"}')
        mock_download.return_value = "python.tar.gz"

        download_standalone_python(platform="linux", proc="x86_64", version="3.12.13")

        # Should have fetched only latest-release.json, not SHA256SUMS
        assert mock_get.call_count == 1
