"""Tests for `comfy preview` — turn a media file into a previewable image."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

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


_FFMPEG = os.path.join(os.sep, "usr", "bin", "ffmpeg")


def test_build_cmd_video_is_contact_sheet():
    cmd = build_preview_cmd("video", "in.mp4", "out.png", grid=(4, 3), width=300, duration=6.0, ffmpeg_bin=_FFMPEG)
    s = " ".join(cmd)
    assert cmd[0] == _FFMPEG
    assert "tile=4x3" in s and "in.mp4" in s and s.endswith("out.png")


def test_build_cmd_audio_is_waveform():
    cmd = build_preview_cmd("audio", "a.flac", "out.png", grid=(4, 3), width=600, duration=30.0, ffmpeg_bin=_FFMPEG)
    assert "showwavespic" in " ".join(cmd)


def test_build_cmd_image_is_scaled():
    cmd = build_preview_cmd("image", "i.png", "out.png", grid=(4, 3), width=512, duration=None, ffmpeg_bin=_FFMPEG)
    assert "scale" in " ".join(cmd)


def test_build_cmd_requires_an_explicit_ffmpeg_path():
    """``ffmpeg_bin`` is keyword-only and has no default, so a caller cannot
    silently fall back to the bare name this whole change removes."""
    with pytest.raises(TypeError):
        build_preview_cmd("image", "i.png", "out.png", grid=(4, 3), width=512, duration=None)


# --- integration: actually run it (needs ffmpeg) ---------------------------


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed")
def test_preview_image_end_to_end(tmp_path, monkeypatch):
    """End-to-end: a real image in → a preview PNG written next to it, exit 0."""
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)

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


# --- CWD binary-planting guard ---------------------------------------------


def _which_cwd_first(system_dir: Path):
    """A ``shutil.which`` that searches the CWD before ``$PATH`` — Windows' order."""

    def which(name, *_args, **_kwargs):
        for candidate in (Path(os.getcwd()) / name, system_dir / name):
            if candidate.exists():
                return str(candidate)
        return None

    return which


_PROBE_JSON = (
    '{"streams": [{"codec_type": "video", "width": 8, "height": 8, "nb_frames": "1"}],'
    ' "format": {"format_name": "png_pipe"}}'
)


def test_preview_spawns_resolved_absolute_paths(tmp_path, monkeypatch):
    """Both ffprobe and ffmpeg go out as trusted absolute paths, resolved once."""
    from comfy_cli import _safe_exec
    from comfy_cli.command import preview as preview_mod

    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)
    system_bin = tmp_path / "usr_bin"
    system_bin.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (system_bin / name).write_text("")
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG")
    out = tmp_path / "out.png"
    monkeypatch.chdir(tmp_path)

    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        out.write_bytes(b"\x89PNG")
        return subprocess.CompletedProcess(args=list(argv), returncode=0, stdout=_PROBE_JSON, stderr="")

    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(preview_mod.subprocess, "run", fake_run),
    ):
        preview_mod.preview_cmd(src, out=out)

    assert [c[0] for c in calls] == [str(system_bin / "ffprobe"), str(system_bin / "ffmpeg")]


def test_preview_reports_unavailable_for_a_cwd_planted_ffmpeg(tmp_path, monkeypatch):
    """A planted ``ffmpeg`` in the CWD is refused, not executed: the command exits
    with the same ``ffmpeg_unavailable`` error it already used when ffmpeg was
    simply not installed."""
    import typer

    from comfy_cli import _safe_exec
    from comfy_cli.command import preview as preview_mod

    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)
    system_bin = tmp_path / "usr_bin"
    system_bin.mkdir()
    (system_bin / "ffprobe").write_text("")  # ffprobe is legitimately installed
    attacker_cwd = tmp_path / "attacker"
    attacker_cwd.mkdir()
    (attacker_cwd / "ffmpeg").write_text("")  # ...but ffmpeg is only the plant
    monkeypatch.chdir(attacker_cwd)
    src = attacker_cwd / "src.png"
    src.write_bytes(b"\x89PNG")

    calls: list[list[str]] = []
    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(preview_mod.subprocess, "run", lambda argv, **kw: calls.append(list(argv))),
    ):
        with pytest.raises(typer.Exit):
            preview_mod.preview_cmd(src)

    assert calls == [], "the planted ffmpeg must never be spawned"
