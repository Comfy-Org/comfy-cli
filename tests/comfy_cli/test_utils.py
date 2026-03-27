import io
from unittest.mock import MagicMock, patch

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


class TestDownloadUrl:
    @patch("comfy_cli.utils.requests.get")
    def test_writes_file(self, mock_get, tmp_path):
        content = b"file contents here"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": str(len(content))}
        mock_response.raw = _FakeRaw(content)
        mock_get.return_value = mock_response

        result = download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)
        assert result == tmp_path / "f.bin"
        assert (tmp_path / "f.bin").read_bytes() == content


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
