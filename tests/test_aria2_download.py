"""Tests for aria2 RPC download support."""

import sys
from unittest.mock import Mock, patch

import pytest

from comfy_cli import constants
from comfy_cli.file_utils import DownloadException, _download_file_aria2, download_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aria2_env(monkeypatch):
    """Set the aria2 environment variables."""
    monkeypatch.setenv(constants.ARIA2_SERVER_ENV_KEY, "http://localhost:6800")
    monkeypatch.setenv(constants.ARIA2_SECRET_ENV_KEY, "mysecret")


@pytest.fixture()
def mock_aria2_success(aria2_env):
    """Mock aria2p with a download that completes immediately."""
    mock_download = Mock()
    mock_download.total_length = 1024
    mock_download.completed_length = 1024
    mock_download.is_complete = True
    mock_download.has_failed = False
    mock_download.is_removed = False
    mock_download.update = Mock()

    mock_api = Mock()
    mock_api.add_uris.return_value = mock_download

    with (
        patch("aria2p.Client") as mock_client_cls,
        patch("aria2p.API") as mock_api_cls,
    ):
        mock_api_cls.return_value = mock_api
        yield {
            "api": mock_api,
            "client_cls": mock_client_cls,
            "api_cls": mock_api_cls,
            "download": mock_download,
        }


# ---------------------------------------------------------------------------
# TestAria2Download — unit tests for _download_file_aria2
# ---------------------------------------------------------------------------


class TestAria2Download:
    def test_success(self, tmp_path, mock_aria2_success):
        """Happy path: aria2 download completes successfully."""
        target = tmp_path / "model.safetensors"
        _download_file_aria2("http://example.com/model.safetensors", target)

        mock_aria2_success["api"].add_uris.assert_called_once()
        call_args = mock_aria2_success["api"].add_uris.call_args
        assert call_args[0][0] == ["http://example.com/model.safetensors"]
        opts = call_args[1]["options"]
        assert opts["dir"] == str(tmp_path)
        assert opts["out"] == "model.safetensors"

    def test_passes_headers(self, tmp_path, mock_aria2_success):
        """CivitAI auth headers are forwarded as aria2 header option."""
        target = tmp_path / "model.bin"
        headers = {"Authorization": "Bearer tok123", "Content-Type": "application/json"}
        _download_file_aria2("http://example.com/model.bin", target, headers=headers)

        opts = mock_aria2_success["api"].add_uris.call_args[1]["options"]
        assert "header" in opts
        assert "Authorization: Bearer tok123" in opts["header"]
        assert "Content-Type: application/json" in opts["header"]

    def test_no_headers(self, tmp_path, mock_aria2_success):
        """When no headers provided, 'header' key is absent from options."""
        target = tmp_path / "model.bin"
        _download_file_aria2("http://example.com/model.bin", target)

        opts = mock_aria2_success["api"].add_uris.call_args[1]["options"]
        assert "header" not in opts

    def test_missing_server_env_raises(self, tmp_path, monkeypatch):
        """Error when COMFYUI_MANAGER_ARIA2_SERVER is not set."""
        monkeypatch.delenv(constants.ARIA2_SERVER_ENV_KEY, raising=False)
        with pytest.raises(DownloadException, match=constants.ARIA2_SERVER_ENV_KEY):
            _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")

    def test_import_error_raises(self, tmp_path, aria2_env):
        """Error when aria2p package is not installed."""
        with patch.dict(sys.modules, {"aria2p": None}):
            with pytest.raises(DownloadException, match="aria2p is required"):
                _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")

    def test_download_failure_raises(self, tmp_path, aria2_env):
        """Error when aria2 reports download failed."""
        mock_download = Mock()
        mock_download.total_length = 0
        mock_download.completed_length = 0
        mock_download.is_complete = False
        mock_download.has_failed = True
        mock_download.is_removed = False
        mock_download.error_message = "403 Forbidden"
        mock_download.error_code = "3"
        mock_download.update = Mock()

        mock_api = Mock()
        mock_api.add_uris.return_value = mock_download

        with (
            patch("aria2p.Client"),
            patch("aria2p.API", return_value=mock_api),
        ):
            with pytest.raises(DownloadException, match="403 Forbidden"):
                _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")

    def test_download_removed_raises(self, tmp_path, aria2_env):
        """Error when aria2 download is removed during progress."""
        mock_download = Mock()
        mock_download.total_length = 0
        mock_download.completed_length = 0
        mock_download.is_complete = False
        mock_download.has_failed = False
        mock_download.is_removed = True
        mock_download.update = Mock()

        mock_api = Mock()
        mock_api.add_uris.return_value = mock_download

        with (
            patch("aria2p.Client"),
            patch("aria2p.API", return_value=mock_api),
        ):
            with pytest.raises(DownloadException, match="removed"):
                _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")

    def test_server_url_parsing(self, tmp_path, monkeypatch):
        """Server URL is correctly parsed into host and port."""
        monkeypatch.setenv(constants.ARIA2_SERVER_ENV_KEY, "http://myserver:6800")
        monkeypatch.setenv(constants.ARIA2_SECRET_ENV_KEY, "")

        mock_download = Mock(
            total_length=100,
            completed_length=100,
            is_complete=True,
            has_failed=False,
            is_removed=False,
            update=Mock(),
        )
        mock_api = Mock()
        mock_api.add_uris.return_value = mock_download

        with (
            patch("aria2p.Client") as mock_client_cls,
            patch("aria2p.API", return_value=mock_api),
        ):
            _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")
            mock_client_cls.assert_called_once_with(host="http://myserver", port=6800, secret="")

    def test_server_url_default_port(self, tmp_path, monkeypatch):
        """Default port 6800 is used when not specified in URL."""
        monkeypatch.setenv(constants.ARIA2_SERVER_ENV_KEY, "http://myserver")
        monkeypatch.setenv(constants.ARIA2_SECRET_ENV_KEY, "")

        mock_download = Mock(
            total_length=100,
            completed_length=100,
            is_complete=True,
            has_failed=False,
            is_removed=False,
            update=Mock(),
        )
        mock_api = Mock()
        mock_api.add_uris.return_value = mock_download

        with (
            patch("aria2p.Client") as mock_client_cls,
            patch("aria2p.API", return_value=mock_api),
        ):
            _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")
            mock_client_cls.assert_called_once_with(host="http://myserver", port=6800, secret="")

    def test_secret_passed_to_client(self, tmp_path, monkeypatch):
        """Secret from env var is passed to aria2p.Client."""
        monkeypatch.setenv(constants.ARIA2_SERVER_ENV_KEY, "http://localhost:6800")
        monkeypatch.setenv(constants.ARIA2_SECRET_ENV_KEY, "supersecret")

        mock_download = Mock(
            total_length=100,
            completed_length=100,
            is_complete=True,
            has_failed=False,
            is_removed=False,
            update=Mock(),
        )
        mock_api = Mock()
        mock_api.add_uris.return_value = mock_download

        with (
            patch("aria2p.Client") as mock_client_cls,
            patch("aria2p.API", return_value=mock_api),
        ):
            _download_file_aria2("http://example.com/f.bin", tmp_path / "f.bin")
            mock_client_cls.assert_called_once_with(host="http://localhost", port=6800, secret="supersecret")


# ---------------------------------------------------------------------------
# TestDownloadFileDispatch — dispatch tests for download_file
# ---------------------------------------------------------------------------


class TestDownloadFileDispatch:
    def test_default_downloader_uses_httpx(self, tmp_path):
        """When downloader is not specified, httpx is used."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "4"}
        mock_response.iter_bytes.return_value = [b"data"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)

        with patch("httpx.stream", return_value=mock_response) as mock_stream:
            download_file("http://example.com/f.bin", tmp_path / "f.bin")
            mock_stream.assert_called_once()

    def test_downloader_httpx_explicit(self, tmp_path):
        """When downloader='httpx', httpx is used."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "4"}
        mock_response.iter_bytes.return_value = [b"data"]
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)

        with patch("httpx.stream", return_value=mock_response) as mock_stream:
            download_file("http://example.com/f.bin", tmp_path / "f.bin", downloader="httpx")
            mock_stream.assert_called_once()

    def test_downloader_aria2_dispatches(self, tmp_path):
        """When downloader='aria2', aria2 backend is used."""
        with patch("comfy_cli.file_utils._download_file_aria2") as mock_aria2:
            download_file("http://example.com/f.bin", tmp_path / "f.bin", downloader="aria2")
            mock_aria2.assert_called_once_with("http://example.com/f.bin", tmp_path / "f.bin", None)
