"""Generate before/after SVG screenshots for the PR."""

from __future__ import annotations

import io

from rich.console import Console
from rich.text import Text

# Shared mock data (resembles real comfy cs output).
RESULTS = [
    {
        "repository": "Comfy-Org/ComfyUI",
        "file": "nodes.py",
        "file_url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/nodes.py",
        "matches": [
            {
                "line": 1698,
                "preview": "class LoadImage:",
                "url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/nodes.py#L1698",
            },
            {
                "line": 1776,
                "preview": "class LoadImageMask:",
                "url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/nodes.py#L1776",
            },
            {
                "line": 1828,
                "preview": "class LoadImageOutput(LoadImage):",
                "url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/nodes.py#L1828",
            },
        ],
    },
    {
        "repository": "Comfy-Org/ComfyUI",
        "file": "comfy_api/input_impl.py",
        "file_url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/comfy_api/input_impl.py",
        "matches": [
            {
                "line": 42,
                "preview": "from .util import ImageInput, LoadImage",
                "url": "https://github.com/Comfy-Org/ComfyUI/blob/abc123/comfy_api/input_impl.py#L42",
            },
        ],
    },
]

STATS_LINE = "5 approximate results, 4 matches returned"


def render_before(console: Console) -> None:
    """Old output: per-line URLs shown as plain text."""
    for r in RESULTS:
        header = Text()
        header.append(r["repository"], style="bold cyan")
        header.append(" / ", style="dim")
        header.append(r["file"], style="bold")
        console.print(header)
        for m in r["matches"]:
            line_text = Text()
            line_text.append(f"  L{m['line']:>5}", style="green")
            line_text.append(f"  {m['preview']}")
            console.print(line_text)
            console.print(f"        [dim]{m['url']}[/dim]")
        console.print()
    console.print(f"[dim]{STATS_LINE}[/dim]")


def render_after_tty(console: Console) -> None:
    """New TTY output: OSC 8 hyperlinks, URLs hidden."""
    for r in RESULTS:
        header = Text()
        header.append(f"{r['repository']} / {r['file']}", style=f"bold cyan link {r['file_url']}")
        console.print(header)
        for m in r["matches"]:
            line_text = Text("  ")
            line_text.append(f"L{m['line']:>5}", style=f"green link {m['url']}")
            line_text.append(f"  {m['preview']}")
            console.print(line_text)
        console.print()
    console.print(f"[dim]{STATS_LINE}[/dim]")


def render_after_pipe(console: Console) -> None:
    """New non-TTY output: one URL per file, compact."""
    for r in RESULTS:
        header = Text()
        header.append(f"{r['repository']} / {r['file']}\n")
        header.append(f"  {r['file_url']}", style="dim")
        console.print(header)
        for m in r["matches"]:
            line_text = Text("  ")
            line_text.append(f"L{m['line']:>5}", style="green")
            line_text.append(f"  {m['preview']}")
            console.print(line_text)
        console.print()
    console.print(f"[dim]{STATS_LINE}[/dim]")


def make_svg(render_fn, title: str) -> str:
    buf = io.StringIO()
    console = Console(
        file=buf,
        record=True,
        force_terminal=True,
        width=100,
        color_system="truecolor",
    )
    render_fn(console)
    return console.export_svg(title=title)


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path("assets")
    out.mkdir(exist_ok=True)

    (out / "before.svg").write_text(make_svg(render_before, "comfy cs LoadImage  (before)"))
    (out / "after_tty.svg").write_text(make_svg(render_after_tty, "comfy cs LoadImage  (after — TTY, links clickable)"))
    (out / "after_pipe.svg").write_text(make_svg(render_after_pipe, "comfy cs LoadImage | cat  (after — piped / AI)"))

    for f in out.iterdir():
        print(f"{f}: {f.stat().st_size:,} bytes")
