import json
import pathlib
from unittest.mock import Mock, patch

import pytest
import requests

from comfy_cli.file_utils import (
    DownloadException,
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
    assert "manually log into browser" in result


def test_guess_status_code_reason_403():
    result = guess_status_code_reason(403, "")
    assert "Forbidden url (403)" in result


def test_guess_status_code_reason_404():
    result = guess_status_code_reason(404, "")
    assert "another castle (404)" in result


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
