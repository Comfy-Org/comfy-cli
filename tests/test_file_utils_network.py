import json
import pathlib
from unittest.mock import Mock, patch

import httpx
import pytest
import requests

from comfy_cli.file_utils import (
    DownloadException,
    _cleanup_partial,
    _friendly_network_error,
    check_unauthorized,
    download_file,
    extract_package_as_zip,
    guess_status_code_reason,
    upload_file_to_signed_url,
)


def test_guess_status_code_reason_401_with_json():
    message = json.dumps({"message": "API token required"}).encode()
    result = guess_status_code_reason(401, message)
    assert "API token required" in result
    assert "Unauthorized download (401)" in result


def test_guess_status_code_reason_401_without_json():
    result = guess_status_code_reason(401, "not json")
    assert "Unauthorized download (401)" in result
    assert "manually log into a browser" in result


def test_guess_status_code_reason_403():
    result = guess_status_code_reason(403, "")
    assert "Forbidden url (403)" in result


def test_guess_status_code_reason_404():
    result = guess_status_code_reason(404, "")
    assert "not found on server (404)" in result


def test_guess_status_code_reason_unknown():
    result = guess_status_code_reason(500, "")
    assert "Unknown error occurred (status code: 500)" in result


@patch("requests.get")
def test_check_unauthorized_true(mock_get):
    mock_response = Mock()
    mock_response.status_code = 401
    mock_get.return_value = mock_response

    assert check_unauthorized("http://example.com") is True


@patch("requests.get")
def test_check_unauthorized_false(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    assert check_unauthorized("http://example.com") is False


@patch("requests.get")
def test_check_unauthorized_exception(mock_get):
    mock_get.side_effect = requests.RequestException()

    assert check_unauthorized("http://example.com") is False


@patch("httpx.stream")
def test_download_file_success(mock_stream, tmp_path):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Length": "1024"}
    mock_response.iter_bytes.return_value = [b"test data"]
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)
    mock_stream.return_value = mock_response

    test_file = tmp_path / "test.txt"
    download_file("http://example.com", test_file)

    assert test_file.exists()
    assert test_file.read_bytes() == b"test data"


@patch("httpx.stream")
def test_download_file_success_without_content_length(mock_stream, tmp_path):
    """Download should succeed when Content-Length header is missing (e.g. chunked/gzip responses)."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)
    mock_stream.return_value = mock_response

    test_file = tmp_path / "test.txt"
    download_file("http://example.com", test_file)

    assert test_file.exists()
    assert test_file.read_bytes() == b"chunk1chunk2"


@patch("httpx.stream")
def test_download_file_failure(mock_stream):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.read.return_value = ""
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)
    mock_stream.return_value = mock_response

    with pytest.raises(DownloadException) as exc_info:
        download_file("http://example.com", pathlib.Path("test.txt"))

    assert "Failed to download file" in str(exc_info.value)


@patch("requests.put")
def test_upload_file_success(mock_put, tmp_path):
    test_file = tmp_path / "test.zip"
    test_file.write_bytes(b"test data")

    mock_response = Mock()
    mock_response.status_code = 200
    mock_put.return_value = mock_response

    upload_file_to_signed_url("http://example.com", str(test_file))

    mock_put.assert_called_once()


@patch("requests.put")
def test_upload_file_failure(mock_put, tmp_path):
    test_file = tmp_path / "test.zip"
    test_file.write_bytes(b"test data")

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Server error"
    mock_put.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        upload_file_to_signed_url("http://example.com", str(test_file))

    assert "Upload failed" in str(exc_info.value)


def test_extract_package_as_zip(tmp_path):
    # Create a test zip file
    import zipfile

    zip_path = tmp_path / "test.zip"
    extract_path = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as test_zip:
        test_zip.writestr("test.txt", "test content")

    extract_package_as_zip(zip_path, extract_path)

    assert (extract_path / "test.txt").exists()
    assert (extract_path / "test.txt").read_text() == "test content"


# ---------------------------------------------------------------------------
# Helpers for download retry/timeout tests
# ---------------------------------------------------------------------------


def _make_ok_response(content=b"data", content_length=None):
    """Create a mock httpx response that succeeds."""
    mock = Mock()
    mock.status_code = 200
    mock.headers = {}
    if content_length is not None:
        mock.headers["Content-Length"] = str(content_length)
    mock.iter_bytes.return_value = [content]
    mock.__enter__ = Mock(return_value=mock)
    mock.__exit__ = Mock(return_value=None)
    return mock


def _make_failing_iter(data=b"partial", exc=None):
    """Return a callable that creates a generator yielding *data* then raising *exc*."""
    if exc is None:
        exc = httpx.ReadTimeout("read timed out")

    def factory():
        yield data
        raise exc

    return factory


# ---------------------------------------------------------------------------
# TestCleanupPartial
# ---------------------------------------------------------------------------


class TestCleanupPartial:
    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "partial.bin"
        f.write_bytes(b"partial")
        _cleanup_partial(f)
        assert not f.exists()

    def test_noop_when_file_missing(self, tmp_path):
        f = tmp_path / "nonexistent.bin"
        _cleanup_partial(f)  # should not raise
        assert not f.exists()


# ---------------------------------------------------------------------------
# TestFriendlyNetworkError
# ---------------------------------------------------------------------------


class TestFriendlyNetworkError:
    def test_read_timeout(self):
        msg = _friendly_network_error(httpx.ReadTimeout("timed out"))
        assert "read timeout" in msg

    def test_connect_timeout(self):
        msg = _friendly_network_error(httpx.ConnectTimeout("timed out"))
        assert "connect timeout" in msg

    def test_generic_timeout(self):
        msg = _friendly_network_error(httpx.PoolTimeout("pool full"))
        assert "PoolTimeout" in msg

    def test_network_error(self):
        msg = _friendly_network_error(httpx.ReadError("connection reset"))
        assert "ReadError" in msg

    def test_other_exception(self):
        msg = _friendly_network_error(RuntimeError("boom"))
        assert msg == "boom"


# ---------------------------------------------------------------------------
# TestDownloadTimeout
# ---------------------------------------------------------------------------


class TestDownloadTimeout:
    @patch("httpx.stream")
    def test_uses_generous_timeout(self, mock_stream, tmp_path):
        """httpx.stream is called with a 300s read timeout."""
        mock_stream.return_value = _make_ok_response()
        download_file("http://example.com/f.bin", tmp_path / "f.bin")

        _, kwargs = mock_stream.call_args
        timeout = kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 300.0
        assert timeout.connect == 10.0


# ---------------------------------------------------------------------------
# TestDownloadRetry
# ---------------------------------------------------------------------------


class TestDownloadRetry:
    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_succeeds_after_transient_timeout(self, mock_stream, mock_sleep, tmp_path):
        """Download retries on ReadTimeout and eventually succeeds."""
        mock_stream.side_effect = [
            httpx.ReadTimeout("timeout"),
            _make_ok_response(content=b"full data"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"full data"
        assert mock_stream.call_count == 2
        mock_sleep.assert_called_once_with(2)  # backoff: 2 * (0+1)

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_succeeds_after_network_error(self, mock_stream, mock_sleep, tmp_path):
        """Download retries on NetworkError (e.g. connection reset)."""
        mock_stream.side_effect = [
            httpx.ReadError("connection reset"),
            httpx.ConnectError("refused"),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"ok"
        assert mock_stream.call_count == 3

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_all_retries_exhausted_read_timeout(self, mock_stream, mock_sleep, tmp_path):
        """DownloadException after all retries fail with ReadTimeout."""
        mock_stream.side_effect = httpx.ReadTimeout("timeout")

        dest = tmp_path / "model.bin"
        with pytest.raises(DownloadException, match="Download failed after 3 attempts") as exc_info:
            download_file("http://example.com/model.bin", dest)

        assert "read timeout" in str(exc_info.value)
        assert "try again" in str(exc_info.value).lower()
        assert mock_stream.call_count == 3
        assert not dest.exists()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_all_retries_exhausted_connect_error(self, mock_stream, mock_sleep, tmp_path):
        """DownloadException after all retries fail with ConnectError."""
        mock_stream.side_effect = httpx.ConnectError("refused")

        dest = tmp_path / "model.bin"
        with pytest.raises(DownloadException, match="Download failed after 3 attempts") as exc_info:
            download_file("http://example.com/model.bin", dest)

        assert "network error" in str(exc_info.value).lower()
        assert mock_stream.call_count == 3

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_http_error_not_retried(self, mock_stream, mock_sleep, tmp_path):
        """Non-200 HTTP status raises DownloadException immediately, no retry."""
        resp = Mock()
        resp.status_code = 404
        resp.read.return_value = ""
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        with pytest.raises(DownloadException, match="Failed to download file"):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_backoff_increases_with_attempts(self, mock_stream, mock_sleep, tmp_path):
        """Retry backoff is 2s, 4s for attempts 1, 2."""
        mock_stream.side_effect = httpx.ReadTimeout("timeout")

        with pytest.raises(DownloadException):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        # Two sleeps: after attempt 0 and attempt 1 (not after the last attempt)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)  # 2 * (0+1)
        mock_sleep.assert_any_call(4)  # 2 * (1+1)

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_original_exception_chained(self, mock_stream, mock_sleep, tmp_path):
        """The original httpx exception is chained as __cause__."""
        mock_stream.side_effect = httpx.ReadTimeout("the real cause")

        with pytest.raises(DownloadException) as exc_info:
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


# ---------------------------------------------------------------------------
# TestDownloadPartialCleanup
# ---------------------------------------------------------------------------


class TestDownloadPartialCleanup:
    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_partial_file_removed_after_midstream_timeout(self, mock_stream, mock_sleep, tmp_path):
        """A file partially written before a timeout is cleaned up."""
        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"partial data"))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        with pytest.raises(DownloadException):
            download_file("http://example.com/model.bin", dest)

        assert not dest.exists()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_partial_file_removed_between_retries(self, mock_stream, mock_sleep, tmp_path):
        """Partial file from a failed attempt doesn't persist into the next attempt."""
        # First attempt: write partial data then timeout
        fail_resp = Mock()
        fail_resp.status_code = 200
        fail_resp.headers = {}
        fail_resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"stale"))
        fail_resp.__enter__ = Mock(return_value=fail_resp)
        fail_resp.__exit__ = Mock(return_value=None)

        # Second attempt: success
        ok_resp = _make_ok_response(content=b"fresh data")

        mock_stream.side_effect = [fail_resp, ok_resp]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        # File should contain only data from the successful attempt
        assert dest.read_bytes() == b"fresh data"

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_cleanup_on_http_error(self, mock_stream, mock_sleep, tmp_path):
        """Cleanup is attempted even on non-retryable DownloadException."""
        resp = Mock()
        resp.status_code = 403
        resp.read.return_value = ""
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        # Pre-create a file to simulate a previous partial download
        dest.write_bytes(b"leftover")

        with pytest.raises(DownloadException):
            download_file("http://example.com/model.bin", dest)

        assert not dest.exists()

    @patch("httpx.stream")
    def test_cleanup_on_keyboard_interrupt(self, mock_stream, tmp_path):
        """Partial file is cleaned up on KeyboardInterrupt (with skip_prompting)."""
        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"partial", KeyboardInterrupt()))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        with pytest.raises(KeyboardInterrupt):
            download_file("http://example.com/model.bin", dest)

        # File should be cleaned up (default prompt answer is True with skip_prompting)
        assert not dest.exists()
