"""``comfy preview`` — turn a media file into a previewable image.

Creative work is visual: an agent steers the user by showing the result, not by
describing a path (see the ``comfy-relay`` skill). But a clip can't render in a
chat and a long video buries its arc. This command produces a single PNG an
agent (or human) can open/``Read`` immediately:

* **image** → a width-bounded thumbnail,
* **video** → a contact sheet (a grid of frames across the whole timeline),
* **audio** → a waveform image (so you can *see* the dynamics you can't hear).

Pure helpers (:func:`classify_streams`, :func:`build_preview_cmd`) are I/O-free
and unit-tested; the Typer shell runs ffprobe/ffmpeg and renders the envelope.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint

_IMAGE_FORMAT_HINTS = ("_pipe", "image2", "png", "jpeg", "mjpeg", "gif", "webp", "bmp", "apng")


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_fps(rate: Any) -> float | None:
    """``"30/1"`` → 30.0; ``"0/0"`` / junk → None."""
    if not isinstance(rate, str) or "/" not in rate:
        return _to_float(rate)
    num, _, den = rate.partition("/")
    n, d = _to_float(num), _to_float(den)
    if n is None or not d:
        return None
    return n / d


def _is_image_format(format_name: str) -> bool:
    fn = (format_name or "").lower()
    return any(h in fn for h in _IMAGE_FORMAT_HINTS)


def classify_streams(probe: dict) -> dict:
    """Classify an ffprobe ``-show_streams -show_format`` JSON dict.

    Returns ``{kind, width, height, fps, duration, has_audio}`` where ``kind`` is
    ``"image" | "video" | "audio" | "unknown"``.
    """
    streams = probe.get("streams") or []
    fmt = probe.get("format") or {}
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    duration = _to_float(fmt.get("duration"))
    has_audio = bool(astreams)

    if vstreams:
        v = vstreams[0]
        fps = _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate"))
        nb = _to_int(v.get("nb_frames"))
        # A single frame, no real timeline, or an image container = a still.
        is_image = nb == 1 or duration is None or fps is None or _is_image_format(fmt.get("format_name", ""))
        return {
            "kind": "image" if is_image else "video",
            "width": _to_int(v.get("width")),
            "height": _to_int(v.get("height")),
            "fps": fps,
            "duration": duration,
            "has_audio": has_audio,
        }
    if astreams:
        return {"kind": "audio", "width": None, "height": None, "fps": None, "duration": duration, "has_audio": True}
    return {"kind": "unknown", "width": None, "height": None, "fps": None, "duration": duration, "has_audio": has_audio}


def build_preview_cmd(
    kind: str, input_path: str, out_path: str, *, grid: tuple[int, int], width: int, duration: float | None
) -> list[str]:
    """Build the ffmpeg argv for a preview of ``kind``. I/O-free."""
    base = ["ffmpeg", "-v", "error", "-y", "-i", input_path]
    if kind == "video":
        cols, rows = grid
        n = max(1, cols * rows)
        d = duration if duration and duration > 0 else 1.0
        vf = f"fps={n}/{d:.4f},scale={width}:-1,tile={cols}x{rows}"
        return base + ["-frames:v", "1", "-vf", vf, out_path]
    if kind == "audio":
        h = max(160, width // 3)
        return base + ["-filter_complex", f"showwavespic=s={width}x{h}:colors=cyan", "-frames:v", "1", out_path]
    # image (and unknown-but-has-a-frame): a width-bounded thumbnail
    return base + ["-frames:v", "1", "-vf", f"scale='min({width},iw)':-1", out_path]


def _ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout or "{}")


@tracking.track_command("preview")
def preview_cmd(
    file: Annotated[Path, typer.Argument(help="Image, video, or audio file to preview.")],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", show_default=False, help="Output PNG. Defaults to <file>.preview.png"),
    ] = None,
    grid: Annotated[str, typer.Option("--grid", help="Contact-sheet grid for video, COLSxROWS.")] = "4x3",
    width: Annotated[int, typer.Option("--width", help="Preview width in pixels.")] = 480,
):
    """Render a previewable PNG from a media file (image → thumb, video → contact
    sheet, audio → waveform) so the result can be shown, not just described."""
    renderer = get_renderer()
    if not file.is_file():
        renderer.error(code="preview_input_not_found", message=f"File not found: {file}", hint="check the path")
        raise typer.Exit(code=1)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        renderer.error(
            code="ffmpeg_unavailable",
            message="ffmpeg/ffprobe not found on PATH — `comfy preview` needs them.",
            hint="install ffmpeg (e.g. `brew install ffmpeg` / `apt install ffmpeg`)",
        )
        raise typer.Exit(code=1)

    try:
        info = classify_streams(_ffprobe(file))
    except (RuntimeError, json.JSONDecodeError) as e:
        renderer.error(code="preview_failed", message=f"Could not probe {file}: {e}")
        raise typer.Exit(code=1) from e
    if info["kind"] == "unknown":
        renderer.error(
            code="preview_unsupported_media",
            message=f"{file} has no image/video/audio stream to preview.",
            hint="pass an image, video, or audio file",
        )
        raise typer.Exit(code=1)

    try:
        cols, rows = (int(x) for x in grid.lower().split("x", 1))
    except ValueError:
        renderer.error(code="preview_failed", message=f"--grid must be COLSxROWS (got {grid!r})", hint="e.g. 4x3")
        raise typer.Exit(code=1)

    out_path = out or (file.parent / f"{file.stem}.preview.png")
    cmd = build_preview_cmd(
        info["kind"], str(file), str(out_path), grid=(cols, rows), width=width, duration=info["duration"]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.is_file():
        renderer.error(
            code="preview_failed",
            message=f"ffmpeg could not render a preview: {proc.stderr.strip()[:300]}",
            hint="check the file isn't corrupt; try a different --grid/--width",
        )
        raise typer.Exit(code=1)

    payload = {
        "input": str(file),
        "kind": info["kind"],
        "preview": str(out_path),
        "width": info["width"],
        "height": info["height"],
        "duration": info["duration"],
        "fps": info["fps"],
        "has_audio": info["has_audio"],
        "hint": f"open/Read {out_path} to see it"
        + ("" if info["kind"] == "image" else " (a contact sheet across the timeline)"),
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] {info['kind']} preview → [bold]{out_path}[/bold]")
        if info["kind"] != "image":
            dur = f"{info['duration']:.1f}s" if info["duration"] else "?"
            rprint(f"  duration: {dur}   audio: {'yes' if info['has_audio'] else 'no'}")
    renderer.emit(payload, command="preview")
