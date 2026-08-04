import json
import os
import pathlib
import stat
import sys
from unittest.mock import Mock, patch

import httpx
import pytest
import requests

from comfy_cli import file_utils
from comfy_cli.file_utils import (
    DownloadException,
    _cleanup_partial,
    _friendly_network_error,
    _TransientHTTPStatusError,
    check_unauthorized,
    cleanup_partials,
    download_file,
    extract_package_as_zip,
    guess_status_code_reason,
    partial_paths_for,
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
def test_download_file_success_with_garbage_content_length(mock_stream, tmp_path):
    """A non-numeric Content-Length (broken server/proxy) must degrade to an
    indeterminate progress bar, not blow the transfer up with a ValueError out of
    ``int()`` — which escaped `model download`'s handlers as a bare traceback."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Length": "not-a-number"}
    mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)
    mock_stream.return_value = mock_response

    test_file = tmp_path / "test.txt"
    download_file("http://example.com", test_file)

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


def _make_status_response(status_code, body=b""):
    """Create a mock httpx response for a non-200 status."""
    mock = Mock()
    mock.status_code = status_code
    mock.read.return_value = body
    mock.__enter__ = Mock(return_value=mock)
    mock.__exit__ = Mock(return_value=None)
    return mock


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

    def test_protocol_error(self):
        msg = _friendly_network_error(httpx.RemoteProtocolError("peer closed"))
        assert "protocol error" in msg
        assert "RemoteProtocolError" in msg

    def test_proxy_error(self):
        msg = _friendly_network_error(httpx.ProxyError("bad proxy"))
        assert "proxy error" in msg
        assert "ProxyError" in msg

    def test_other_exception(self):
        msg = _friendly_network_error(RuntimeError("boom"))
        assert msg == "boom"

    def test_transient_http_status_known_code_includes_phrase(self):
        # HTTP 503 -> "Service Unavailable" (from stdlib http.HTTPStatus).
        msg = _friendly_network_error(_TransientHTTPStatusError(503, "some reason from body"))
        assert "HTTP 503" in msg
        assert "Service Unavailable" in msg

    def test_transient_http_status_500_includes_phrase(self):
        msg = _friendly_network_error(_TransientHTTPStatusError(500, ""))
        assert "HTTP 500" in msg
        assert "Internal Server Error" in msg

    def test_transient_http_status_unknown_code_falls_back(self):
        # 599 is not a standard HTTPStatus; fall back to just the numeric code.
        msg = _friendly_network_error(_TransientHTTPStatusError(599, "weird"))
        assert "HTTP 599" in msg
        # No crash, no stdlib phrase embedded (since there isn't one).

    def test_invalid_url(self):
        msg = _friendly_network_error(httpx.InvalidURL("Request URL is missing a scheme"))
        assert "invalid URL" in msg
        assert "missing a scheme" in msg


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
    def test_succeeds_after_protocol_error(self, mock_stream, mock_sleep, tmp_path):
        """Download retries on RemoteProtocolError (e.g. peer closed connection mid-stream)."""
        mock_stream.side_effect = [
            httpx.RemoteProtocolError("peer closed connection"),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"ok"
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_succeeds_after_proxy_error(self, mock_stream, mock_sleep, tmp_path):
        """Download retries on ProxyError."""
        mock_stream.side_effect = [
            httpx.ProxyError("bad gateway"),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"ok"
        assert mock_stream.call_count == 2

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
    def test_preexisting_file_preserved_on_http_error(self, mock_stream, mock_sleep, tmp_path):
        """A pre-existing file at the destination is NOT touched when the server returns an HTTP error.

        HTTP errors are raised before _download_file_httpx opens the output file, so there is no
        partial download to clean up. The helper must not destroy unrelated pre-existing data.
        """
        resp = Mock()
        resp.status_code = 403
        resp.read.return_value = ""
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(DownloadException):
            download_file("http://example.com/model.bin", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_preexisting_file_preserved_on_connect_error(self, mock_stream, mock_sleep, tmp_path):
        """A pre-existing file is NOT deleted when all retries fail with a pre-open transient error.

        ConnectError/ConnectTimeout are raised at httpx.stream() entry, before the output file
        is opened. Cleanup must not run in that case, or it would wipe out an unrelated
        pre-existing file at the destination path.
        """
        mock_stream.side_effect = httpx.ConnectError("refused")

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(DownloadException, match="Download failed after 3 attempts"):
            download_file("http://example.com/model.bin", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"

    @patch("comfy_cli.file_utils.ui.prompt_confirm_action", return_value=True)
    @patch("httpx.stream")
    def test_preexisting_file_preserved_on_interrupt_before_open(self, mock_stream, mock_prompt, tmp_path):
        """KeyboardInterrupt during connection setup (before output file is opened) must not
        prompt the user or delete an unrelated pre-existing file.
        """
        mock_stream.side_effect = KeyboardInterrupt()

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(KeyboardInterrupt):
            download_file("http://example.com/model.bin", dest)

        # Prompt should NOT have been shown — the file was never opened this attempt.
        mock_prompt.assert_not_called()
        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"

    @patch("comfy_cli.file_utils.ui.prompt_confirm_action", return_value=True)
    @patch("httpx.stream")
    def test_keyboard_interrupt_cleans_up_when_user_confirms(self, mock_stream, mock_prompt, tmp_path):
        """On KeyboardInterrupt the user is prompted; confirming removes the partial file and re-raises."""
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

        mock_prompt.assert_called_once()
        assert not dest.exists()
        assert partial_paths_for(dest) == []

    @patch("comfy_cli.file_utils.ui.prompt_confirm_action", return_value=False)
    @patch("httpx.stream")
    def test_keyboard_interrupt_keeps_partial_when_user_declines(self, mock_stream, mock_prompt, tmp_path):
        """On KeyboardInterrupt the user is prompted; declining keeps the partial bytes.

        They are kept as the `.part` sibling, never at the destination — the whole
        point of the atomic write is that an interrupted transfer can't leave
        something that looks like a finished model where ComfyUI will load it.
        """
        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"partial data", KeyboardInterrupt()))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        with pytest.raises(KeyboardInterrupt):
            download_file("http://example.com/model.bin", dest)

        mock_prompt.assert_called_once()
        assert not dest.exists()
        parts = partial_paths_for(dest)
        assert len(parts) == 1
        assert parts[0].read_bytes() == b"partial data"


class _HardKill(BaseException):
    """Stands in for a signal: derives from BaseException, so no `except Exception`
    anywhere in the download path can quietly turn it into an ordinary failure."""


class TestAtomicDestination:
    """The destination only ever goes absent→complete (or old-complete→new-complete).

    A killed transfer used to leave a truncated file sitting exactly where a
    finished model belongs, with nothing to mark it — `search_models` and ComfyUI
    both see a plausible file, and loading it fails far from the download that
    caused it.
    """

    @patch("httpx.stream")
    def test_completed_download_lands_via_rename(self, mock_stream, tmp_path):
        """The bytes reach the destination through os.replace from a sibling temp,
        and nothing is left behind."""
        mock_stream.return_value = _make_ok_response(content=b"full model", content_length=10)
        dest = tmp_path / "model.safetensors"

        real_replace = os.replace
        renames = []

        def spy(src, dst, *args, **kwargs):
            renames.append((str(src), str(dst)))
            return real_replace(src, dst, *args, **kwargs)

        with patch("comfy_cli.file_utils.os.replace", side_effect=spy):
            download_file("http://example.com/model.safetensors", dest)

        assert dest.read_bytes() == b"full model"
        assert len(renames) == 1
        src, dst = renames[0]
        assert dst == str(dest)
        assert src.startswith(str(dest) + ".") and src.endswith(".part")
        # The temp was consumed by the rename, not left alongside the model.
        assert partial_paths_for(dest) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.safetensors"]

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_hard_kill_mid_transfer_leaves_nothing_at_the_destination(self, mock_stream, mock_sleep, tmp_path):
        """Mid-stream, the destination does not exist yet — the bytes are in a
        `.part` sibling. That is precisely the on-disk state a SIGKILL freezes,
        and the assertion is made *from inside the stream* so no cleanup handler
        can have run first. The stream then raises a BaseException, which no
        `except Exception` in the download path converts into a tidy failure.
        """
        dest = tmp_path / "checkpoint.safetensors"
        observed = {}

        chunk = b"x" * 65536

        def killed_iter():
            # Keep streaming until bytes have actually reached the disk (a SIGKILL
            # loses the writer's buffer too, and the question here is where the
            # bytes that *did* land ended up), then freeze that instant.
            for _ in range(32):
                yield chunk
                parts = partial_paths_for(dest)
                if parts and parts[0].stat().st_size:
                    # What an operator would find on disk at the moment of the kill.
                    observed["dest_exists"] = dest.exists()
                    observed["parts"] = [(p.name, p.read_bytes()) for p in parts]
                    raise _HardKill("SIGKILL")
            raise AssertionError("no bytes ever reached a .part file")

        resp = Mock()
        resp.status_code = 200
        resp.headers = {"Content-Length": "13000000000"}
        resp.iter_bytes = Mock(side_effect=killed_iter)
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        with pytest.raises(_HardKill):
            download_file("http://example.com/checkpoint.safetensors", dest)

        assert observed["dest_exists"] is False, "a truncated file must never appear at the final path"
        assert len(observed["parts"]) == 1
        part_name, part_bytes = observed["parts"][0]
        assert part_name.endswith(".part")
        # The landed bytes are a prefix of the stream (the tail may still be in the
        # writer's buffer) — and they are in the `.part`, not at the destination.
        assert part_bytes and part_bytes.strip(b"x") == b""
        # And after the unwind the destination is still absent.
        assert not dest.exists()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_retries_never_expose_a_truncated_file_at_the_destination(self, mock_stream, mock_sleep, tmp_path):
        """Between attempts there is nothing at the destination for a reader to
        mistake for a finished model."""
        dest = tmp_path / "model.bin"
        seen_between_attempts = []

        fail_resp = Mock()
        fail_resp.status_code = 200
        fail_resp.headers = {}
        fail_resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"stale bytes"))
        fail_resp.__enter__ = Mock(return_value=fail_resp)
        fail_resp.__exit__ = Mock(return_value=None)

        def stream(*args, **kwargs):
            seen_between_attempts.append((dest.exists(), len(partial_paths_for(dest))))
            return fail_resp if len(seen_between_attempts) == 1 else _make_ok_response(content=b"fresh data")

        mock_stream.side_effect = stream
        download_file("http://example.com/model.bin", dest)

        assert seen_between_attempts == [(False, 0), (False, 0)]
        assert dest.read_bytes() == b"fresh data"
        assert partial_paths_for(dest) == []

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_preexisting_complete_file_survives_a_failed_redownload(self, mock_stream, mock_sleep, tmp_path):
        """A failure *after* bytes started flowing no longer destroys what was
        already at the destination — previously the transfer had truncated it on
        open before it ever knew whether it would succeed."""
        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"half a model"))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"COMPLETE existing model")

        with pytest.raises(DownloadException, match="Download failed after 3 attempts"):
            download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"COMPLETE existing model"
        assert partial_paths_for(dest) == []

    @patch("httpx.stream")
    def test_a_successful_download_replaces_an_existing_file(self, mock_stream, tmp_path):
        """old-complete→new-complete is the other legal transition."""
        mock_stream.return_value = _make_ok_response(content=b"v2")
        dest = tmp_path / "model.bin"
        dest.write_bytes(b"v1")

        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"v2"
        assert partial_paths_for(dest) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    @patch("httpx.stream")
    def test_destination_permissions_match_a_plain_write(self, mock_stream, tmp_path):
        """mkstemp creates 0600 and os.replace carries the mode across, so without
        an explicit chmod every downloaded model would silently become owner-only."""
        mock_stream.return_value = _make_ok_response(content=b"data")
        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        reference = tmp_path / "reference.bin"
        with open(reference, "wb") as f:
            f.write(b"data")

        assert stat.S_IMODE(dest.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    @patch("httpx.stream")
    def test_replacing_a_file_keeps_its_permissions(self, mock_stream, tmp_path):
        mock_stream.return_value = _make_ok_response(content=b"v2")
        dest = tmp_path / "model.bin"
        dest.write_bytes(b"v1")
        dest.chmod(0o640)

        download_file("http://example.com/model.bin", dest)

        assert stat.S_IMODE(dest.stat().st_mode) == 0o640


class TestTempFileNaming:
    """The temp name is derived from the destination name, so the destination's
    own length and the destination's mode both have to survive the round trip."""

    @patch("httpx.stream")
    def test_a_maximum_length_destination_name_still_downloads(self, mock_stream, tmp_path):
        """mkstemp appends 13 bytes to the prefix, so an un-truncated prefix makes
        any name over NAME_MAX-14 fail with ENAMETOOLONG — where the old
        `open(dest, "wb")` succeeded. Names come from remote metadata (CivitAI
        `file["name"]`, the HF path) and `--filename`, none of which cap length.
        """
        mock_stream.return_value = _make_ok_response(content=b"data")
        # 250 bytes: a name the filesystem itself accepts (NAME_MAX is 255), but
        # 9 bytes too long to also carry mkstemp's token and suffix.
        dest = tmp_path / ("m" * 238 + ".safetensors")
        assert 241 < len(dest.name.encode()) <= 255

        download_file("http://example.com/model.safetensors", dest)

        assert dest.read_bytes() == b"data"
        assert partial_paths_for(dest) == []

    @patch("httpx.stream")
    def test_a_long_name_partial_is_still_reclaimable(self, mock_stream, tmp_path):
        """Truncating the prefix is only safe if the matcher truncates identically
        — otherwise `download-cancel` silently stops finding these."""
        dest = tmp_path / ("m" * 238 + ".safetensors")

        def killed_iter():
            yield b"partial"
            raise KeyboardInterrupt()

        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=killed_iter)
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        with (
            patch("comfy_cli.file_utils.ui.prompt_confirm_action", return_value=False),
            pytest.raises(KeyboardInterrupt),
        ):
            download_file("http://example.com/model.safetensors", dest)

        assert [p.read_bytes() for p in partial_paths_for(dest)] == [b"partial"]
        assert cleanup_partials(dest) == 1

    def test_a_non_ascii_name_is_truncated_on_a_character_boundary(self, tmp_path):
        """NAME_MAX counts bytes, so the cut is a byte cut — but it must not
        produce an invalid-UTF-8 name that no filesystem call can round-trip."""
        dest = tmp_path / ("é" * 200 + ".safetensors")
        prefix = file_utils._part_prefix(dest.name)

        assert len(prefix.encode()) <= file_utils._PART_STEM_MAX + 1
        assert prefix.encode().decode() == prefix  # valid UTF-8, no split codepoint
        assert prefix.endswith("."), "the matcher slices on this separator"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    @patch("httpx.stream")
    def test_a_setuid_destination_does_not_pass_its_set_id_bits_on(self, mock_stream, tmp_path):
        """`os.replace` carries the temp's mode onto the destination, so copying
        the old file's full 12-bit mode would stamp setuid/setgid onto freshly
        downloaded, network-controlled bytes — which an in-place write would have
        cleared."""
        mock_stream.return_value = _make_ok_response(content=b"v2")
        dest = tmp_path / "model.bin"
        dest.write_bytes(b"v1")
        # Owner-only under the set-ID bits: the group/other bits are what a
        # real 4755 binary carries, but they play no part in what this asserts
        # (that the 0o777 mask drops set-ID), and granting them here is an
        # overly-permissive chmod in its own right. Group/other preservation is
        # covered by test_replacing_a_file_keeps_its_permissions.
        os.chmod(dest, stat.S_ISUID | stat.S_ISGID | stat.S_IRWXU)
        assert dest.stat().st_mode & stat.S_ISUID, "the fixture must really carry a set-ID bit"

        download_file("http://example.com/model.bin", dest)

        mode = dest.stat().st_mode
        assert stat.S_IMODE(mode) == 0o700
        assert not mode & (stat.S_ISUID | stat.S_ISGID)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    @patch("httpx.stream")
    def test_the_mode_is_applied_through_the_descriptor_not_the_name(self, mock_stream, tmp_path):
        """chmod-by-path follows symlinks, so a name swap between mkstemp and the
        chmod would retarget it at an arbitrary file (CWE-59) — reopening the race
        mkstemp's O_EXCL closed. fchmod on the open fd cannot be redirected."""
        mock_stream.return_value = _make_ok_response(content=b"data")
        dest = tmp_path / "model.bin"

        def refuse(*args, **kwargs):
            raise AssertionError("the temp file must never be chmod'ed by path")

        with patch("comfy_cli.file_utils.os.chmod", side_effect=refuse):
            download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"data"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX umask")
    @patch("httpx.stream")
    def test_a_download_never_touches_the_process_umask(self, mock_stream, tmp_path):
        """Probing the umask means setting it to 0 and restoring it — a window in
        which any *other* thread's new file lands world-writable, and which two
        overlapping probes can leave stranded at 0. It belongs at import, once."""
        mock_stream.return_value = _make_ok_response(content=b"data")

        def refuse(*args, **kwargs):
            raise AssertionError("the umask must not be probed per download")

        with patch("comfy_cli.file_utils.os.umask", side_effect=refuse):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert (tmp_path / "model.bin").read_bytes() == b"data"


class TestPartialPaths:
    """`download-cancel` finds a dead worker's bytes through these, so the match
    has to be tight: too loose and it deletes a user's unrelated file."""

    def _make_part(self, dest, token=b"a1b2c3d4", data=b"bytes"):
        p = dest.parent / f"{dest.name}.{token.decode()}.part"
        p.write_bytes(data)
        return p

    def test_finds_the_temp_a_download_would_create(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        part = self._make_part(dest)
        assert partial_paths_for(dest) == [part]

    def test_ignores_unrelated_part_files(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        keep = [
            tmp_path / "model.safetensors.part",  # no mkstemp token
            tmp_path / "model.safetensors.backup.part",  # wrong token shape
            tmp_path / "model.safetensors.a1b2c3d4.tmp",  # wrong suffix
            tmp_path / "other.safetensors.a1b2c3d4.part",  # another download
            tmp_path / "model.safetensors.A1B2C3D4.part",  # mkstemp never uppercases
        ]
        for f in keep:
            f.write_bytes(b"do not touch")

        assert partial_paths_for(dest) == []
        assert cleanup_partials(dest) == 0
        assert all(f.exists() for f in keep)

    def test_cleanup_removes_every_match_and_reports_the_count(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        self._make_part(dest, token=b"aaaaaaaa")
        self._make_part(dest, token=b"bbbbbbbb")
        unrelated = tmp_path / "keep.bin"
        unrelated.write_bytes(b"keep")

        assert cleanup_partials(dest) == 2
        assert partial_paths_for(dest) == []
        assert unrelated.exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        dest = tmp_path / "nope" / "model.safetensors"
        assert partial_paths_for(dest) == []
        assert cleanup_partials(dest) == 0

    def test_a_real_download_produces_a_matching_name(self, tmp_path):
        """Guards the coupling between mkstemp's naming and the matcher above:
        if either drifts, a cancel silently stops reclaiming anything."""
        dest = tmp_path / "model.safetensors"

        def killed_iter():
            yield b"partial"
            raise KeyboardInterrupt()

        resp = Mock()
        resp.status_code = 200
        resp.headers = {}
        resp.iter_bytes = Mock(side_effect=killed_iter)
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)

        with (
            patch("httpx.stream", return_value=resp),
            patch("comfy_cli.file_utils.ui.prompt_confirm_action", return_value=False),
            pytest.raises(KeyboardInterrupt),
        ):
            download_file("http://example.com/model.safetensors", dest)

        assert [p.read_bytes() for p in partial_paths_for(dest)] == [b"partial"]
        assert cleanup_partials(dest) == 1


class TestDownloadHTTPStatusRetry:
    """Retry behavior for transient HTTP status codes (5xx, 429, 408)."""

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_500_retried_and_succeeds(self, mock_stream, mock_sleep, tmp_path):
        """Download retries on HTTP 500 and succeeds on the next attempt."""
        mock_stream.side_effect = [
            _make_status_response(500),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"ok"
        assert mock_stream.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_502_retried(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = [
            _make_status_response(502),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_503_retried(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = [
            _make_status_response(503),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_504_retried(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = [
            _make_status_response(504),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_429_retried(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = [
            _make_status_response(429),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_408_retried(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = [
            _make_status_response(408),
            _make_ok_response(content=b"ok"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_all_retries_exhausted_on_500(self, mock_stream, mock_sleep, tmp_path):
        """After 3 failed attempts on 500, a DownloadException is raised with a friendly message."""
        mock_stream.side_effect = [
            _make_status_response(500),
            _make_status_response(500),
            _make_status_response(500),
        ]

        dest = tmp_path / "model.bin"
        with pytest.raises(DownloadException, match="Download failed after 3 attempts") as exc_info:
            download_file("http://example.com/model.bin", dest)

        assert "HTTP 500" in str(exc_info.value)
        # The stdlib HTTPStatus phrase is surfaced so the user knows what 500 means.
        assert "Internal Server Error" in str(exc_info.value)
        assert mock_stream.call_count == 3
        # The last transient HTTP error must be chained as __cause__ for debuggability.
        assert isinstance(exc_info.value.__cause__, _TransientHTTPStatusError)
        assert exc_info.value.__cause__.status_code == 500

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_retry_body_read_timeout_still_retries(self, mock_stream, mock_sleep, tmp_path):
        """If reading the 500 response body itself times out, we still retry the request."""
        fail_resp = Mock()
        fail_resp.status_code = 500
        fail_resp.read.side_effect = httpx.ReadTimeout("body read timed out")
        fail_resp.__enter__ = Mock(return_value=fail_resp)
        fail_resp.__exit__ = Mock(return_value=None)

        mock_stream.side_effect = [fail_resp, _make_ok_response(content=b"ok")]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"ok"
        assert mock_stream.call_count == 2

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_mixed_transient_errors_eventually_succeed(self, mock_stream, mock_sleep, tmp_path):
        """Retries work across a mix of network-level and HTTP-status errors."""
        mock_stream.side_effect = [
            _make_status_response(503),
            httpx.ReadTimeout("timeout"),
            _make_ok_response(content=b"finally"),
        ]

        dest = tmp_path / "model.bin"
        download_file("http://example.com/model.bin", dest)

        assert dest.read_bytes() == b"finally"
        assert mock_stream.call_count == 3

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_404_not_retried(self, mock_stream, mock_sleep, tmp_path):
        """404 fails fast without retry."""
        mock_stream.return_value = _make_status_response(404)

        with pytest.raises(DownloadException, match="Failed to download file"):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_401_not_retried(self, mock_stream, mock_sleep, tmp_path):
        """401 fails fast without retry."""
        mock_stream.return_value = _make_status_response(401)

        with pytest.raises(DownloadException, match="Failed to download file"):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_403_not_retried(self, mock_stream, mock_sleep, tmp_path):
        """403 fails fast without retry."""
        mock_stream.return_value = _make_status_response(403)

        with pytest.raises(DownloadException, match="Failed to download file"):
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_preexisting_file_preserved_on_http_status_retry_exhaust(self, mock_stream, mock_sleep, tmp_path):
        """A pre-existing file at the destination is NOT deleted when all retries fail on HTTP 500.

        The retriable HTTP status is raised before _download_file_httpx opens the output file.
        """
        mock_stream.side_effect = [
            _make_status_response(500),
            _make_status_response(500),
            _make_status_response(500),
        ]

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(DownloadException, match="Download failed after 3 attempts"):
            download_file("http://example.com/model.bin", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"


class TestDownloadNonRetriableHTTPError:
    """Non-retriable httpx errors (UnsupportedProtocol, TooManyRedirects, etc.) are wrapped
    as DownloadException so callers only need to handle one error type and users don't
    see a raw Python traceback."""

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_unsupported_protocol_wrapped(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'")

        with pytest.raises(DownloadException, match="Download failed") as exc_info:
            download_file("ftp://example.com/model.bin", tmp_path / "model.bin")

        assert isinstance(exc_info.value.__cause__, httpx.UnsupportedProtocol)
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_too_many_redirects_wrapped(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = httpx.TooManyRedirects("Exceeded maximum allowed redirects")

        with pytest.raises(DownloadException, match="Download failed") as exc_info:
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert isinstance(exc_info.value.__cause__, httpx.TooManyRedirects)
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_decoding_error_wrapped(self, mock_stream, mock_sleep, tmp_path):
        mock_stream.side_effect = httpx.DecodingError("Invalid compressed data")

        with pytest.raises(DownloadException, match="Download failed") as exc_info:
            download_file("http://example.com/model.bin", tmp_path / "model.bin")

        assert isinstance(exc_info.value.__cause__, httpx.DecodingError)
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("comfy_cli.file_utils.time.sleep")
    @patch("httpx.stream")
    def test_invalid_url_wrapped(self, mock_stream, mock_sleep, tmp_path):
        """httpx.InvalidURL does NOT subclass httpx.HTTPError — it must still be wrapped
        as DownloadException so a malformed URL doesn't leak as a Typer traceback."""
        mock_stream.side_effect = httpx.InvalidURL("Request URL is missing a scheme")

        with pytest.raises(DownloadException, match="Download failed") as exc_info:
            download_file("no-scheme-url", tmp_path / "model.bin")

        assert isinstance(exc_info.value.__cause__, httpx.InvalidURL)
        assert "invalid URL" in str(exc_info.value)
        assert mock_stream.call_count == 1
        mock_sleep.assert_not_called()

    @patch("httpx.stream")
    def test_invalid_url_preserves_preexisting_file(self, mock_stream, tmp_path):
        """InvalidURL is raised before the output file is opened — any pre-existing
        file at the destination path must be left intact."""
        mock_stream.side_effect = httpx.InvalidURL("bad")

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(DownloadException):
            download_file("not-a-url", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"

    @patch("httpx.stream")
    def test_preexisting_file_preserved_on_non_retriable_error(self, mock_stream, tmp_path):
        """A non-retriable httpx error before the output file is opened must not delete
        an unrelated pre-existing file at the destination path."""
        mock_stream.side_effect = httpx.UnsupportedProtocol("nope")

        dest = tmp_path / "model.bin"
        dest.write_bytes(b"IMPORTANT pre-existing data")

        with pytest.raises(DownloadException):
            download_file("ftp://example.com/model.bin", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"IMPORTANT pre-existing data"

    @patch("httpx.stream")
    def test_partial_file_cleaned_up_on_mid_stream_non_retriable(self, mock_stream, tmp_path):
        """If a non-retriable error is raised AFTER the output file is opened (mid-stream),
        the partial file is cleaned up."""
        resp = Mock()
        resp.status_code = 200
        resp.headers = {"Content-Length": "100"}
        resp.iter_bytes = Mock(side_effect=_make_failing_iter(b"partial", httpx.DecodingError("bad")))
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=None)
        mock_stream.return_value = resp

        dest = tmp_path / "model.bin"
        with pytest.raises(DownloadException):
            download_file("http://example.com/model.bin", dest)

        assert not dest.exists()
