"""Inline image/video preview in the terminal.

Uses ``term-image`` (optional dependency) to auto-detect the best
terminal graphics protocol (Kitty, iTerm2, or Unicode half-blocks)
and render images inline.

For videos, extracts a thumbnail frame via ``ffmpeg`` and displays that
alongside a metadata panel.

Skips silently when:
  - term-image is not installed
  - the renderer is in JSON/agentic mode (agents get file paths from the envelope)
  - the file doesn't exist or isn't a supported format
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path

# Suppress term-image's "not running in a terminal" warning — we handle
# the TTY check ourselves before calling into term-image.
warnings.filterwarnings("ignore", message=".*not running within a terminal.*", category=UserWarning)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".apng", ".bmp", ".tiff"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv", ".mpg", ".mpeg"}


def preview(path: str | Path) -> None:
    """Show an inline preview of an image or video file.

    Call this from pretty-mode code paths only. Returns immediately if
    term-image is not installed or the file is unsupported.
    """
    p = Path(path)
    if not p.is_file():
        return

    suffix = p.suffix.lower()
    if suffix in _IMAGE_EXTS:
        _show_image(p)
    elif suffix in _VIDEO_EXTS:
        _show_video(p)


def _show_image(path: Path) -> None:
    try:
        from term_image.image import from_file

        img = from_file(str(path))
        img.draw()
    except (ImportError, Exception):  # noqa: BLE001
        pass


def _show_video(path: Path) -> None:
    """Extract a thumbnail + metadata panel for a video file."""
    _show_video_info(path)
    _show_video_thumbnail(path)


def _show_video_thumbnail(path: Path) -> None:
    """Extract the first frame via ffmpeg and display it."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(  # noqa: S603
            [
                "ffmpeg",
                "-i",
                str(path),
                "-vf",
                "select=eq(n\\,0)",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                tmp_path,
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            _show_image(Path(tmp_path))
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):  # noqa: BLE001
        # ffmpeg not installed or failed — skip silently
        pass
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def _show_video_info(path: Path) -> None:
    """Show a Rich panel with video metadata via ffprobe."""
    try:
        result = subprocess.run(  # noqa: S603
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return

        info = json.loads(result.stdout)
        fmt = info.get("format", {})
        vstream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        if not vstream:
            return

        duration = float(fmt.get("duration", 0))
        w, h = vstream.get("width", "?"), vstream.get("height", "?")

        # Parse fps from "24/1" or "30000/1001" format
        fps_str = vstream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            fps = 0

        codec = vstream.get("codec_name", "?")
        size_bytes = int(fmt.get("size", 0))
        size_mb = size_bytes / 1048576

        from rich import print as rprint
        from rich.panel import Panel

        rprint(
            Panel(
                f"🎬 [bold]{path.name}[/bold]\n   {w}×{h} · {fps:.0f}fps · {codec} · {duration:.1f}s · {size_mb:.1f}MB",
                border_style="blue",
                expand=False,
            )
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):  # noqa: BLE001
        pass
