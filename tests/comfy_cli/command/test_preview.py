"""Tests for `comfy preview` — turn a media file into a previewable image."""

from __future__ import annotations

import shutil

import pytest
from typer.testing import CliRunner

from comfy_cli.command.preview import build_preview_cmd, classify_streams

# --- pure: classify ffprobe output -----------------------------------------


def test_classify_video():
    probe = {
        "streams": [
            {"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "30/1", "nb_frames": "150"},
            {"codec_type": "audio"},
        ],
        "format": {"duration": "5.0", "format_name": "mov,mp4,m4a"},
    }
    info = classify_streams(probe)
    assert info["kind"] == "video"
    assert info["width"] == 1280 and info["height"] == 720
    assert abs(info["fps"] - 30) < 0.01
    assert info["duration"] == 5.0
    assert info["has_audio"] is True


def test_classify_image():
    probe = {
        "streams": [{"codec_type": "video", "width": 800, "height": 600, "avg_frame_rate": "0/0", "nb_frames": "1"}],
        "format": {"format_name": "png_pipe"},
    }
    info = classify_streams(probe)
    assert info["kind"] == "image"
    assert info["has_audio"] is False


def test_classify_audio():
    probe = {"streams": [{"codec_type": "audio"}], "format": {"duration": "30.0", "format_name": "flac"}}
    info = classify_streams(probe)
    assert info["kind"] == "audio"
    assert info["has_audio"] is True
    assert info["duration"] == 30.0


# --- pure: build the ffmpeg command ----------------------------------------


def test_build_cmd_video_is_contact_sheet():
    cmd = build_preview_cmd("video", "in.mp4", "out.png", grid=(4, 3), width=300, duration=6.0)
    s = " ".join(cmd)
    assert cmd[0] == "ffmpeg"
    assert "tile=4x3" in s and "in.mp4" in s and s.endswith("out.png")


def test_build_cmd_audio_is_waveform():
    cmd = build_preview_cmd("audio", "a.flac", "out.png", grid=(4, 3), width=600, duration=30.0)
    assert "showwavespic" in " ".join(cmd)


def test_build_cmd_image_is_scaled():
    cmd = build_preview_cmd("image", "i.png", "out.png", grid=(4, 3), width=512, duration=None)
    assert "scale" in " ".join(cmd)


# --- integration: actually run it (needs ffmpeg) ---------------------------


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed")
def test_preview_image_end_to_end(tmp_path, monkeypatch):
    """End-to-end: a real image in → a preview PNG written next to it, exit 0."""
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)
    import subprocess

    from comfy_cli.cmdline import app as cli_app

    src = tmp_path / "src.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:duration=1",
            "-frames:v",
            "1",
            str(src),
        ],
        check=True,
    )
    result = CliRunner().invoke(cli_app, ["preview", str(src)], standalone_mode=False)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "src.preview.png").is_file()


def test_preview_missing_file_errors(tmp_path, monkeypatch):
    """A missing input takes the error path (raises typer.Exit) — no ffmpeg needed."""
    import typer

    from comfy_cli.command.preview import preview_cmd

    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)
    with pytest.raises(typer.Exit):
        preview_cmd(tmp_path / "nope.png")
