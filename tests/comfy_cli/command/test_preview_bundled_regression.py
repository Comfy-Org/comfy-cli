"""Successful bundled rendering must emit one successful result, without an error."""

import inspect
import subprocess
from unittest.mock import Mock

from comfy_cli._safe_exec import BinaryNotFoundError, BinaryRefusal
from comfy_cli.command import preview


def test_bundled_ffmpeg_without_ffprobe_does_not_emit_false_untrusted_error(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "contact-sheet.png"
    renderer = Mock()
    renderer.is_pretty.return_value = False

    def unavailable(name):
        raise BinaryNotFoundError(f"{name} absent", binary=name, reason=BinaryRefusal.ABSENT)

    def render(argv, **kwargs):
        output.write_bytes(b"PNG")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(preview, "get_renderer", lambda: renderer)
    monkeypatch.setattr(preview, "resolve_required_binary", unavailable)
    monkeypatch.setattr(preview, "_bundled_ffmpeg", lambda: "/trusted/bundled/ffmpeg")
    monkeypatch.setattr(preview.subprocess, "run", render)
    inspect.unwrap(preview.preview_cmd)(source, out=output)
    assert output.is_file()
    assert renderer.emit.call_args.args[0]["preview"] == str(output)
    renderer.error.assert_not_called()
