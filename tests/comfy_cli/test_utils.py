import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.http import DOWNLOAD_TIMEOUT
from comfy_cli.utils import create_tarball, download_url, extract_tarball, parse_rfc3339


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
    @patch("requests.get")
    def test_writes_file(self, mock_get, tmp_path):
        content = b"file contents here"
        _mock_streaming_response(mock_get, content)

        result = download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)
        assert result == tmp_path / "f.bin"
        assert (tmp_path / "f.bin").read_bytes() == content

    @patch("requests.get")
    def test_passes_download_timeout(self, mock_get, tmp_path):
        """A streaming download must set a (connect, read) timeout so a stalled peer can't hang."""
        _mock_streaming_response(mock_get, b"x")

        download_url("http://example.com/f.bin", "f.bin", cwd=tmp_path, show_progress=False)
        assert mock_get.call_args.kwargs["timeout"] == DOWNLOAD_TIMEOUT

    @patch("requests.get")
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


def _write_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    tinfo = tarfile.TarInfo(name)
    tinfo.size = len(data)
    tar.addfile(tinfo, io.BytesIO(data))


def _write_symlink(tar: tarfile.TarFile, name: str, target: str) -> None:
    tinfo = tarfile.TarInfo(name)
    tinfo.type = tarfile.SYMTYPE
    tinfo.linkname = target
    tar.addfile(tinfo)


class TestExtractTarballFiltering:
    """Regression tests for CVE-2007-4559 (see issue #725).

    ``extract_tarball`` extracts into the current working directory, so a member
    named ``../evil.txt`` lands one level above it unless the stdlib ``data``
    filter rejects the archive.
    """

    @staticmethod
    def _traversal_tarball(workdir):
        tarball = workdir / "payload.tgz"
        with tarfile.open(tarball, "w:gz") as tar:
            _write_member(tar, "payload/keep.txt", b"benign")
            _write_member(tar, "../evil.txt", b"pwned")
        return tarball

    @pytest.mark.parametrize("show_progress", [False, True])
    def test_rejects_path_traversal_member(self, tmp_path, monkeypatch, show_progress):
        """Both extraction paths must refuse to write outside the destination."""
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        tarball = self._traversal_tarball(workdir)
        escaped = tmp_path / "evil.txt"

        rejection = None
        with patch("comfy_cli.utils.Live"):
            try:
                extract_tarball(tarball, workdir / "out", show_progress=show_progress)
            except tarfile.FilterError as exc:
                rejection = exc

        assert not escaped.exists(), f"traversal member escaped the extraction directory: {escaped}"
        assert rejection is not None, "the traversal member was extracted instead of being rejected"

    @pytest.mark.parametrize("show_progress", [False, True])
    def test_allows_internal_symlinks(self, tmp_path, monkeypatch, show_progress):
        """Control: the filter must not reject the layout real payloads use.

        python-build-standalone tarballs (the only thing ``StandalonePython``
        extracts) are full of relative symlinks such as ``bin/python3 ->
        python3.12``. Those stay inside the destination and must survive.
        """
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        tarball = workdir / "python.tgz"
        with tarfile.open(tarball, "w:gz") as tar:
            _write_member(tar, "python/bin/python3.12", b"#!/bin/sh\n")
            _write_symlink(tar, "python/bin/python3", "python3.12")
            _write_symlink(tar, "python/bin/python", "python3")

        dest = workdir / "out"
        with patch("comfy_cli.utils.Live"):
            extract_tarball(tarball, dest, show_progress=show_progress)

        assert (dest / "bin" / "python3.12").read_bytes() == b"#!/bin/sh\n"
        assert (dest / "bin" / "python3").is_symlink()
        assert (dest / "bin" / "python").is_symlink()


class TestParseRfc3339:
    """``datetime.fromisoformat`` alone takes only 3 or 6 fractional digits and
    no bare ``Z`` on Python 3.10, the minimum this package supports, while the
    Go services trim trailing zeros and so emit every width between 0 and 9."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("2026-08-28T03:26:09.43745Z", "2026-08-28T03:26:09.437450+00:00", id="five-digits-trimmed"),
            pytest.param("2026-08-28T03:26:09.1Z", "2026-08-28T03:26:09.100000+00:00", id="one-digit-pads-not-shifts"),
            pytest.param(
                "2026-08-28T03:26:09.806473123Z", "2026-08-28T03:26:09.806473+00:00", id="nanoseconds-truncate"
            ),
            pytest.param("2026-08-28T03:26:09.806473Z", "2026-08-28T03:26:09.806473+00:00", id="microseconds"),
            pytest.param("2026-08-28T03:26:09Z", "2026-08-28T03:26:09+00:00", id="whole-seconds"),
            pytest.param("2026-08-28T03:26:09.806-07:00", "2026-08-28T03:26:09.806000-07:00", id="offset-zone"),
            pytest.param("2026-08-28T03:26:09.806-0700", "2026-08-28T03:26:09.806000-07:00", id="offset-no-colon"),
            pytest.param("2026-08-28T03:26:09", "2026-08-28T03:26:09+00:00", id="missing-zone-reads-as-utc"),
        ],
    )
    def test_every_precision_a_go_service_can_emit_parses(self, value, expected):
        assert parse_rfc3339(value).isoformat() == expected

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("", id="empty"),
            pytest.param("not-a-timestamp", id="not-a-timestamp"),
            pytest.param("2026-08-28", id="date-only"),
            pytest.param("2026-08-28T03:26:09.806Z trailing", id="trailing-junk"),
        ],
    )
    def test_a_value_shaped_nothing_like_rfc3339_is_rejected(self, value):
        with pytest.raises(ValueError):
            parse_rfc3339(value)

    def test_a_trimmed_value_orders_by_instant_not_by_digit_count(self):
        """``.5`` is later than ``.43745`` but sorts below it as text."""
        assert parse_rfc3339("2026-08-28T03:26:09.43745Z") < parse_rfc3339("2026-08-28T03:26:09.5Z")
