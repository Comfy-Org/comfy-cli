"""Tests for the per-file upload helper ``transfer._upload_file``.

The helper is the single place the CLI speaks the server's ``/upload/image``
multipart API — both ``comfy upload`` and ``comfy assets push`` go through
it. The CLI must NEVER touch a ComfyUI install's folders directly; this HTTP
endpoint is the only ingestion path.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest
import typer

from comfy_cli.command import transfer
from comfy_cli.target import Target


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """Captures the request and returns a canned JSON response."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None):
        self.payload = payload or {"name": "ab12.png", "subfolder": "", "type": "input"}
        self.error = error
        self.requests: list = []

    def open(self, req):
        self.requests.append(req)
        if self.error is not None:
            raise self.error
        return _FakeResponse(json.dumps(self.payload).encode())


def _local_target() -> Target:
    return Target(kind="local", base_url="http://127.0.0.1:8188")


def _cloud_target(**kw) -> Target:
    return Target(kind="cloud", base_url="https://cloud.example.com", path_prefix="/api", **kw)


@pytest.fixture
def asset(tmp_path: Path) -> Path:
    p = tmp_path / "frame.png"
    p.write_bytes(b"fake-png-bytes")
    return p


def test_upload_file_posts_multipart_and_returns_response_dict(asset, monkeypatch):
    opener = _FakeOpener(payload={"name": "deadbeef.png", "subfolder": "sub", "type": "input"})
    monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

    result = transfer._upload_file(asset, _local_target(), overwrite=False)

    assert result == {"name": "deadbeef.png", "subfolder": "sub", "type": "input"}
    assert len(opener.requests) == 1
    req = opener.requests[0]
    assert req.full_url == "http://127.0.0.1:8188/upload/image"
    assert req.get_method() == "POST"
    body = req.data
    assert b'name="image"; filename="frame.png"' in body
    assert b"fake-png-bytes" in body
    assert b'name="overwrite"\r\n\r\nfalse\r\n' in body
    assert "multipart/form-data; boundary=" in req.get_header("Content-type")


def test_upload_file_overwrite_true_in_body(asset, monkeypatch):
    opener = _FakeOpener()
    monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

    transfer._upload_file(asset, _local_target(), overwrite=True)

    assert b'name="overwrite"\r\n\r\ntrue\r\n' in opener.requests[0].data


def test_upload_file_cloud_target_attaches_auth_and_prefix(asset, monkeypatch):
    opener = _FakeOpener()
    monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

    transfer._upload_file(asset, _cloud_target(auth_token="tok123"), overwrite=False)

    req = opener.requests[0]
    assert req.full_url == "https://cloud.example.com/api/upload/image"
    assert req.get_header("Authorization") == "Bearer tok123"


def test_upload_file_http_error_propagates(asset, monkeypatch):
    err = urllib.error.HTTPError("http://x/upload/image", 500, "boom", {}, io.BytesIO())
    monkeypatch.setattr(transfer, "_TRANSFER_OPENER", _FakeOpener(error=err))

    with pytest.raises(urllib.error.HTTPError):
        transfer._upload_file(asset, _local_target(), overwrite=False)


def test_upload_file_sanitizes_hostile_filename(tmp_path, monkeypatch):
    hostile = tmp_path / 'a"b.png'
    hostile.write_bytes(b"x")
    opener = _FakeOpener()
    monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

    transfer._upload_file(hostile, _local_target(), overwrite=False)

    assert b'filename="a_b.png"' in opener.requests[0].data


class TestUploadMachineModeStdoutPurity:
    """Same contract as download: in machine modes stdout carries only JSON
    (envelope last) and the human "✓ uploaded" line is pretty-mode-only."""

    @pytest.fixture(autouse=True)
    def reset_renderer(self):
        from comfy_cli.output.renderer import reset_renderer_for_testing

        reset_renderer_for_testing()
        yield
        reset_renderer_for_testing()

    def test_json_mode_stdout_is_pure_json_no_human_line(self, asset, monkeypatch, capsys):
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)
        monkeypatch.setattr(transfer, "resolve_target", lambda where=None: _local_target())
        set_renderer(Renderer(mode=OutputMode.JSON, command="upload"))

        transfer.execute_upload([str(asset)], where="local")

        captured = capsys.readouterr()
        out_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert out_lines, "the envelope must land on stdout"
        parsed = [json.loads(ln) for ln in out_lines]
        assert parsed[-1]["type"] == "envelope"
        assert parsed[-1]["data"]["uploads"][0]["cloud_name"] == "ab12.png"
        assert "uploaded" not in captured.out
        assert "uploaded" not in captured.err


class TestUploadConnectionError:
    """A connection-level failure (``URLError``/``TimeoutError``) on upload must
    surface as a structured ``upload_failed`` envelope, not an unhandled
    traceback that breaks ``--json``/NDJSON consumers (BE-2454)."""

    @pytest.fixture(autouse=True)
    def reset_renderer(self):
        from comfy_cli.output.renderer import reset_renderer_for_testing

        reset_renderer_for_testing()
        yield
        reset_renderer_for_testing()

    def test_urlerror_emits_upload_failed_envelope(self, asset, monkeypatch, capsys):
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        err = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", _FakeOpener(error=err))
        monkeypatch.setattr(transfer, "resolve_target", lambda where=None: _local_target())
        set_renderer(Renderer(mode=OutputMode.JSON, command="upload"))

        with pytest.raises(typer.Exit) as excinfo:
            transfer.execute_upload([str(asset)], where="local")

        assert excinfo.value.exit_code == 1
        env = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
        assert env["ok"] is False
        assert env["error"]["code"] == "upload_failed"
        assert "Connection refused" in env["error"]["message"]
        assert "Connection refused" in env["error"]["details"]["reason"]

    def test_incomplete_read_emits_upload_failed_envelope(self, asset, monkeypatch, capsys):
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        # A truncated response body raises http.client.IncompleteRead — an
        # HTTPException, not a URLError.
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", _FakeOpener(error=http.client.IncompleteRead(b"x", 100)))
        monkeypatch.setattr(transfer, "resolve_target", lambda where=None: _local_target())
        set_renderer(Renderer(mode=OutputMode.JSON, command="upload"))

        with pytest.raises(typer.Exit) as excinfo:
            transfer.execute_upload([str(asset)], where="local")

        assert excinfo.value.exit_code == 1
        env = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
        assert env["error"]["code"] == "upload_failed"
