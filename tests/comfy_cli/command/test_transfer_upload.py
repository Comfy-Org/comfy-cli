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

from comfy_cli.cmdline import app
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
        monkeypatch.setattr(transfer, "resolve_target", lambda **kw: _local_target())
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
        monkeypatch.setattr(transfer, "resolve_target", lambda **kw: _local_target())
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
        monkeypatch.setattr(transfer, "resolve_target", lambda **kw: _local_target())
        set_renderer(Renderer(mode=OutputMode.JSON, command="upload"))

        with pytest.raises(typer.Exit) as excinfo:
            transfer.execute_upload([str(asset)], where="local")

        assert excinfo.value.exit_code == 1
        env = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
        assert env["error"]["code"] == "upload_failed"


class TestUploadHostPortRouting:
    """``comfy upload --host/--port`` aims a LOCAL upload at a specific ComfyUI
    (BE-5662).

    Before this the only lever was the process-wide ``COMFY_LOCAL_URL`` env
    var, so a caller wrapping the CLI (comfy-mcp's ``_with_target``) could not
    route a single invocation. ``execute_upload`` now forwards the pair to
    ``resolve_target``, which applies the documented local precedence:
    explicit value > ``COMFY_LOCAL_URL`` > ``127.0.0.1:8188``.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_local_url(self, monkeypatch):
        # The resolver honors COMFY_LOCAL_URL, so drop any ambient value and
        # let each test state its own precedence inputs.
        monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)

    def test_host_and_port_reach_resolve_target(self, asset, monkeypatch):
        seen: dict = {}

        def fake_resolve_target(**kwargs):
            seen.update(kwargs)
            return _local_target()

        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", _FakeOpener())
        monkeypatch.setattr(transfer, "resolve_target", fake_resolve_target)

        transfer.execute_upload([str(asset)], where="local", host="10.0.0.5", port=9999)

        assert seen == {"where": "local", "host": "10.0.0.5", "port": 9999}

    def test_post_url_targets_the_given_host_and_port(self, asset, monkeypatch):
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local", host="10.0.0.5", port=9999)

        assert opener.requests[0].full_url == "http://10.0.0.5:9999/upload/image"

    def test_no_flags_keeps_the_loopback_default(self, asset, monkeypatch):
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local")

        assert opener.requests[0].full_url == "http://127.0.0.1:8188/upload/image"

    def test_no_flags_still_honors_comfy_local_url(self, asset, monkeypatch):
        monkeypatch.setenv("COMFY_LOCAL_URL", "http://192.168.1.9:8189")
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local")

        assert opener.requests[0].full_url == "http://192.168.1.9:8189/upload/image"

    def test_flags_beat_comfy_local_url(self, asset, monkeypatch):
        monkeypatch.setenv("COMFY_LOCAL_URL", "http://192.168.1.9:8189")
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local", host="10.0.0.5", port=9999)

        assert opener.requests[0].full_url == "http://10.0.0.5:9999/upload/image"

    def test_host_only_flag_keeps_the_env_port(self, asset, monkeypatch):
        # host and port resolve independently, so --host alone must not drop
        # the env var's port back to the 8188 default.
        monkeypatch.setenv("COMFY_LOCAL_URL", "http://192.168.1.9:8189")
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local", host="10.0.0.5")

        assert opener.requests[0].full_url == "http://10.0.0.5:8189/upload/image"

    def test_ipv6_host_is_bracketed_in_the_url(self, asset, monkeypatch):
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)

        transfer.execute_upload([str(asset)], where="local", host="::1", port=8189)

        assert opener.requests[0].full_url == "http://[::1]:8189/upload/image"

    def test_cloud_target_ignores_host_and_port(self, asset, monkeypatch):
        # Belt-and-braces behind the CLI guard below: even if the pair reached a
        # cloud resolve, the cloud address comes from the account, not the flags.
        opener = _FakeOpener()
        monkeypatch.setattr(transfer, "_TRANSFER_OPENER", opener)
        monkeypatch.setattr(transfer, "resolve_target", lambda **kw: _cloud_target())

        transfer.execute_upload([str(asset)], where="cloud", host="10.0.0.5", port=9999)

        assert opener.requests[0].full_url == "https://cloud.example.com/api/upload/image"


class TestUploadCliHostPortFlags:
    """CLI surface of the same feature: `comfy upload --host/--port` mirrors the
    separate options `comfy jobs` takes (not `comfy run`'s combined
    ``host[:port]`` positional), validates the host, and refuses to pair with a
    cloud target."""

    @pytest.fixture
    def runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    @staticmethod
    def _envelope(result) -> dict:
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_flags_are_forwarded_to_execute_upload(self, asset, runner, monkeypatch):
        from comfy_cli import cmdline

        seen: dict = {}

        def fake_execute_upload(files, **kwargs):
            seen["files"] = files
            seen.update(kwargs)
            return ["ab12.png"]

        monkeypatch.setattr(cmdline.transfer_inner, "execute_upload", fake_execute_upload)

        result = runner.invoke(
            app,
            ["upload", str(asset), "--host", "10.0.0.5", "--port", "9999"],
            env={"COMFY_WHERE": "local"},
        )

        assert result.exit_code == 0, result.output
        assert seen["files"] == [str(asset)]
        assert seen["host"] == "10.0.0.5"
        assert seen["port"] == 9999
        assert seen["where"] == "local"

    def test_no_flags_forwards_none(self, asset, runner, monkeypatch):
        from comfy_cli import cmdline

        seen: dict = {}
        monkeypatch.setattr(
            cmdline.transfer_inner,
            "execute_upload",
            lambda files, **kwargs: seen.update(kwargs) or ["ab12.png"],
        )

        result = runner.invoke(app, ["upload", str(asset)], env={"COMFY_WHERE": "local"})

        assert result.exit_code == 0, result.output
        assert seen["host"] is None
        assert seen["port"] is None

    @pytest.mark.parametrize("flags", [["--host", "10.0.0.5"], ["--port", "9999"], ["--host", "h", "--port", "1"]])
    def test_host_or_port_with_cloud_where_flag_is_rejected(self, asset, runner, flags):
        result = runner.invoke(app, ["--json", "upload", str(asset), "--where", "cloud", *flags])

        assert result.exit_code == 1, result.output
        env = self._envelope(result)
        assert env["ok"] is False
        assert env["error"]["code"] == "host_flag_cloud"

    def test_host_with_cloud_where_env_is_rejected(self, asset, runner):
        # COMFY_WHERE is how the top-level `comfy --where cloud` arrives, so the
        # guard has to key off the RESOLVED target, not the flag.
        result = runner.invoke(
            app,
            ["--json", "upload", str(asset), "--host", "10.0.0.5"],
            env={"COMFY_WHERE": "cloud"},
        )

        assert result.exit_code == 1, result.output
        assert self._envelope(result)["error"]["code"] == "host_flag_cloud"

    def test_cloud_without_the_flags_is_untouched(self, asset, runner, monkeypatch):
        from comfy_cli import cmdline

        seen: dict = {}
        monkeypatch.setattr(cmdline.where_module, "cloud_preflight_or_exit", lambda: None)
        monkeypatch.setattr(
            cmdline.transfer_inner,
            "execute_upload",
            lambda files, **kwargs: seen.update(kwargs) or ["ab12.png"],
        )

        result = runner.invoke(app, ["upload", str(asset), "--where", "cloud"])

        assert result.exit_code == 0, result.output
        assert seen["where"] == "cloud"
        assert seen["host"] is None and seen["port"] is None

    @pytest.mark.parametrize("bad", ["evil/host", "user@evil", "host?x", "host#x", "host\rname", "host\nname"])
    def test_invalid_host_is_a_usage_error(self, asset, runner, bad):
        result = runner.invoke(app, ["upload", str(asset), "--host", bad], env={"COMFY_WHERE": "local"})

        # typer.BadParameter -> click UsageError -> exit code 2.
        assert result.exit_code == 2, result.output

    @pytest.mark.parametrize("bad", ["0", "65536", "-1"])
    def test_out_of_range_port_is_a_usage_error(self, asset, runner, bad):
        result = runner.invoke(app, ["upload", str(asset), "--port", bad], env={"COMFY_WHERE": "local"})

        assert result.exit_code == 2, result.output
