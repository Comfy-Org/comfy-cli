import io
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.http import DOWNLOAD_TIMEOUT
from comfy_cli.utils import create_tarball, download_url, extract_tarball


class _FakeRaw(io.BytesIO):
    """BytesIO that accepts decode_content kwarg like urllib3 responses.

    The production code does ``response.raw.read = functools.partial(
    response.raw.read, decode_content=True)`` which monkey-patches the
    read method.  A plain BytesIO would blow up because its read() does
    not accept that kwarg.
    """

    def read(self, amt=-1, decode_content=False):
        return super().read(amt)


def _mock_streaming_response(mock_get, content):
    """Wire a mock response that ``download_url`` can use as a context manager."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Length": str(len(content))}
    mock_response.raw = _FakeRaw(content)
    mock_get.return_value.__enter__.return_value = mock_response
    return mock_response


class TestDownloadUrl:
    @patch("comfy_cli.utils.requests.get")
    def test_writes_file(self, mock_get, tmp_path):
        content = b"file contents here"
        _mock_streaming_response(mock_get, content)

        result = download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)
        assert result == tmp_path / "f.bin"
        assert (tmp_path / "f.bin").read_bytes() == content

    @patch("comfy_cli.utils.requests.get")
    def test_passes_download_timeout(self, mock_get, tmp_path):
        """A streaming download must set a (connect, read) timeout so a stalled peer can't hang."""
        _mock_streaming_response(mock_get, b"x")

        download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)
        assert mock_get.call_args.kwargs["timeout"] == DOWNLOAD_TIMEOUT

    @patch("comfy_cli.utils.requests.get")
    def test_releases_connection_on_error_status(self, mock_get, tmp_path):
        """A non-200 must still release the streamed connection rather than leak it until GC."""
        mock_response = _mock_streaming_response(mock_get, b"")
        mock_response.status_code = 500
        mock_response.raise_for_status.return_value = None

        with pytest.raises(RuntimeError):
            download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)

        mock_get.return_value.__exit__.assert_called_once()


class TestTarballRoundTrip:
    def test_create_and_extract(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        src = tmp_path / "mydir"
        src.mkdir()
        (src / "hello.txt").write_text("hello world")
        (src / "sub").mkdir()
        (src / "sub" / "nested.txt").write_text("nested content")

        tarball = tmp_path / "mydir.tgz"
        with patch("comfy_cli.utils.Live"):
            create_tarball(src, tarball, cwd=tmp_path)
        assert tarball.exists()

        dest = tmp_path / "extracted"
        with patch("comfy_cli.utils.Live"):
            extract_tarball(tarball, dest)

        assert (dest / "hello.txt").read_text() == "hello world"
        assert (dest / "sub" / "nested.txt").read_text() == "nested content"
