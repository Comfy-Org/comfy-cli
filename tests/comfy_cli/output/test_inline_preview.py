"""Tests for the best-effort inline previewer's ffmpeg/ffprobe spawns.

Unlike ``comfy preview`` (which fails loudly when ffmpeg is missing), this module
is documented to skip silently. It therefore uses ``resolve_binary`` — the probe
entry point — so an unresolvable *or* CWD-planted binary degrades to "no
preview", never to a spawn and never to an exception.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from comfy_cli import _safe_exec
from comfy_cli.output import preview as inline_preview


def _which_cwd_first(system_dir: Path):
    """A ``shutil.which`` that searches the CWD before ``$PATH`` — Windows' order."""

    def which(name, *_args, **_kwargs):
        for candidate in (Path(os.getcwd()) / name, system_dir / name):
            if candidate.exists():
                return str(candidate)
        return None

    return which


def _system_bin(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "usr_bin"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_text("")
    return directory


def test_thumbnail_spawns_resolved_ffmpeg(tmp_path, monkeypatch):
    system_bin = _system_bin(tmp_path, "ffmpeg")
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(args=list(argv), returncode=1, stdout=b"", stderr=b"")

    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(inline_preview.subprocess, "run", fake_run),
    ):
        inline_preview._show_video_thumbnail(tmp_path / "clip.mp4")

    assert [c[0] for c in calls] == [str(system_bin / "ffmpeg")]


def test_thumbnail_skips_silently_for_a_cwd_planted_ffmpeg(tmp_path, monkeypatch):
    system_bin = _system_bin(tmp_path)  # nothing legitimate on PATH
    attacker_cwd = tmp_path / "attacker"
    attacker_cwd.mkdir()
    (attacker_cwd / "ffmpeg").write_text("")
    monkeypatch.chdir(attacker_cwd)

    calls: list[list[str]] = []
    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(inline_preview.subprocess, "run", lambda argv, **kw: calls.append(list(argv))),
    ):
        inline_preview._show_video_thumbnail(attacker_cwd / "clip.mp4")

    assert calls == []


def test_video_info_spawns_resolved_ffprobe(tmp_path, monkeypatch):
    system_bin = _system_bin(tmp_path, "ffprobe")
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(args=list(argv), returncode=1, stdout="", stderr="")

    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(inline_preview.subprocess, "run", fake_run),
    ):
        inline_preview._show_video_info(tmp_path / "clip.mp4")

    assert [c[0] for c in calls] == [str(system_bin / "ffprobe")]


def test_video_info_skips_silently_for_a_cwd_planted_ffprobe(tmp_path, monkeypatch):
    system_bin = _system_bin(tmp_path)
    attacker_cwd = tmp_path / "attacker"
    attacker_cwd.mkdir()
    (attacker_cwd / "ffprobe").write_text("")
    monkeypatch.chdir(attacker_cwd)

    calls: list[list[str]] = []
    with (
        patch.object(_safe_exec.shutil, "which", _which_cwd_first(system_bin)),
        patch.object(inline_preview.subprocess, "run", lambda argv, **kw: calls.append(list(argv))),
    ):
        inline_preview._show_video_info(attacker_cwd / "clip.mp4")

    assert calls == []
