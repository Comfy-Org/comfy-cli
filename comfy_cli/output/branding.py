"""Branded output — wordmark, gradient text, welcome card.

Used after ``comfy cloud login`` succeeds and on ``comfy cloud whoami`` so the
human moment of "I just signed into Comfy Cloud" looks like a product, not
a JSON envelope.

Design references:
    - comfy.org brand: orange → pink gradient, modern, minimal
    - Rich truecolor support: every modern terminal does ``#rrggbb`` styles;
      Rich downgrades to the closest ANSI color if it doesn't.
"""

from __future__ import annotations

import time
from typing import Any

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Brand palette — matches the comfy.org orange→pink wordmark.
# ---------------------------------------------------------------------------

BRAND_START = "#FF6E00"  # orange
BRAND_END = "#FF1B6B"  # pink/magenta
BRAND_ACCENT = "#FFAA33"  # warm amber, for accents


# The single tagline used on the welcome banner. Centralized here so a future
# rename happens in one place and the per-command panels (which intentionally
# don't repeat it) stay consistent.
TAGLINE = "the agent-aware ComfyUI CLI"


def branded_panel(
    body,
    *,
    title: str,
    version: str,
    where: str | None = None,
    host: str | None = None,
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    """The canonical Panel wrapper for every pretty-mode screen.

    title    → left-aligned command name (e.g. ``env``, ``jobs``, ``comfy cloud``).
    subtitle → right-aligned, always starts with ``comfy CLI v<version>`` and
               optionally carries routing context. See "Cross-Task Contracts"
               in ``docs/superpowers/plans/2026-05-19-cli-ux-consistency.md``.

    Subtitle composition:
      - Always ``comfy CLI v<version>``.
      - If ``where`` is set, append ``  ·  {where}`` (double-space + dot — same
        weight as the welcome banner uses for its tagline divider).
      - If ``host`` is set, append `` · {host}`` (single-space, since it's a
        sub-qualifier of ``where``). When ``where`` is absent ``host`` rides
        alone behind a single dot.
    """
    parts = [f"comfy CLI v{version}"]
    if where:
        parts.append(f"  ·  {where}")
    if host:
        parts.append(f" · {host}")
    subtitle = "".join(parts)
    return Panel(
        body,
        title=Text(title, style=f"bold {BRAND_START}"),
        subtitle=Text(subtitle, style="dim"),
        title_align="left",
        subtitle_align="right",
        border_style=BRAND_START,
        padding=padding,
    )


# ---------------------------------------------------------------------------
# Wordmark — handcrafted 5-row block art, kept narrow enough for an 80-col tty
# ---------------------------------------------------------------------------

_WORDMARK_ROWS = [
    " ████   ████  ██   ██ █████  ██  ██",
    "██     ██  ██ ███ ███ ██      ████ ",
    "██     ██  ██ ██ █ ██ ████     ██  ",
    "██     ██  ██ ██   ██ ██       ██  ",
    " ████   ████  ██   ██ ██       ██  ",
]


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def gradient_text(
    text: str,
    *,
    start: str = BRAND_START,
    end: str = BRAND_END,
    bold: bool = True,
) -> Text:
    """Render ``text`` so each character is colored along a linear gradient.

    Whitespace is rendered uncolored so the gradient lands on glyphs only.
    """
    sa = _hex_to_rgb(start)
    sb = _hex_to_rgb(end)
    visible_indices = [i for i, ch in enumerate(text) if not ch.isspace()]
    out = Text()
    n = max(len(visible_indices) - 1, 1)
    pos = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            out.append(ch)
            continue
        t = pos / n
        rgb = _lerp_color(sa, sb, t)
        style = f"bold {_rgb_to_hex(*rgb)}" if bold else _rgb_to_hex(*rgb)
        out.append(ch, style=style)
        pos += 1
    return out


def gradient_block(
    rows: list[str],
    *,
    start: str = BRAND_START,
    end: str = BRAND_END,
) -> Text:
    """Apply a top-to-bottom gradient to a multi-line block of text.

    Each *row* gets a single color, interpolated from start to end. Looks
    cleaner than per-character gradient for chunky block letters.
    """
    sa = _hex_to_rgb(start)
    sb = _hex_to_rgb(end)
    n = max(len(rows) - 1, 1)
    out = Text()
    last = len(rows) - 1
    for i, row in enumerate(rows):
        t = i / n
        rgb = _lerp_color(sa, sb, t)
        style = f"bold {_rgb_to_hex(*rgb)}"
        if i < last:
            out.append(row + "\n", style=style)
        else:
            out.append(row, style=style)
    return out


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def _humanize_relative(expires_at: int | None) -> str:
    if expires_at is None:
        return "unknown"
    delta = int(expires_at) - int(time.time())
    if delta < 0:
        return f"expired {_humanize_seconds(-delta)} ago"
    return f"in {_humanize_seconds(delta)}"


def _humanize_seconds(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h, rem = divmod(s, 3600)
    return f"{h}h {rem // 60}m"


# ---------------------------------------------------------------------------
# Welcome banner — the big one, shown after auth login success
# ---------------------------------------------------------------------------


def welcome_banner(
    *,
    base_url: str,
    scope: str,
    client_id: str,
    expires_at: int | None,
    cta: str | None = None,
) -> Panel:
    """The post-login banner. Gradient wordmark + 'CLOUD' subtitle + info.

    Layout::

        ╭──────────────────────────────────────────────────────╮
        │                                                      │
        │   <gradient COMFY block>                             │
        │                  C L O U D                           │
        │                                                      │
        │   ✓ Signed in                                        │
        │     Scope      …                                     │
        │     Expires    in 59m 58s · 2026-05-15T05:00         │
        │     Client     comfy-cli                              │
        │                                                      │
        │   → comfy run …                                      │
        │                                                      │
        ╰─────────── testcloud.comfy.org ──────────────────────╯
    """
    wordmark = gradient_block(_WORDMARK_ROWS)
    subtitle = Text("C  L  O  U  D", style=f"dim {BRAND_END}", justify="center")

    info_tbl = Table.grid(padding=(0, 2), expand=False)
    info_tbl.add_column(justify="right", style="dim", no_wrap=True)
    info_tbl.add_column(overflow="fold")
    info_tbl.add_row("Scope", Text(scope, style="white"))
    info_tbl.add_row(
        "Access token",
        Text.assemble(
            (_humanize_relative(expires_at), f"bold {BRAND_ACCENT}"),
            ("   ", ""),
            (_iso(expires_at), "dim"),
        ),
    )
    info_tbl.add_row("", Text("renews automatically while you keep using the CLI", style="dim"))
    info_tbl.add_row("Client", Text(client_id, style="white"))

    signed_in_line = Text.assemble(
        ("✓ ", "bold green"),
        ("Signed in to Comfy Cloud", "bold white"),
    )

    body_parts: list[Any] = [
        Align.center(wordmark),
        Align.center(subtitle),
        Text(""),
        signed_in_line,
        Text(""),
        info_tbl,
    ]
    if cta:
        body_parts.append(Text(""))
        body_parts.append(
            Text.assemble(
                ("→ ", f"bold {BRAND_ACCENT}"),
                (cta, "yellow"),
            )
        )

    return Panel(
        Group(*body_parts),
        title=Text("", style=""),  # no top title — wordmark is the brand
        subtitle=Text(_pretty_host(base_url), style="dim"),
        subtitle_align="right",
        border_style=BRAND_START,
        padding=(1, 3),
    )


def whoami_banner(
    *,
    base_url: str,
    scope: str,
    client_id: str,
    expires_at: int | None,
    expired: bool,
    version: str = "",
) -> Panel:
    """A compact branded card for ``comfy cloud whoami`` when active.

    Smaller than the post-login banner (no big wordmark) — but still uses
    the brand gradient on the header and the same info table.
    """
    header = gradient_text("comfy cloud", bold=True)
    status = (
        Text.assemble(("⚠ ", "bold yellow"), ("expired", "bold yellow"))
        if expired
        else Text.assemble(("✓ ", "bold green"), ("active", "bold green"))
    )

    head_line = Text.assemble(header, ("   ", ""), status)

    info_tbl = Table.grid(padding=(0, 2), expand=False)
    info_tbl.add_column(justify="right", style="dim", no_wrap=True)
    info_tbl.add_column(overflow="fold")
    info_tbl.add_row("Scope", Text(scope, style="white"))
    info_tbl.add_row(
        "Access token",
        Text.assemble(
            (_humanize_relative(expires_at), f"bold {BRAND_ACCENT}"),
            ("   ", ""),
            (_iso(expires_at), "dim"),
        ),
    )
    if not expired:
        info_tbl.add_row("", Text("renews automatically while you keep using the CLI", style="dim"))
    info_tbl.add_row("Client", Text(client_id, style="white"))

    return branded_panel(
        Group(head_line, Text(""), info_tbl),
        title="comfy cloud",
        version=version,
        host=_pretty_host(base_url),
        padding=(0, 2),
    )


def intro_banner(
    *,
    version: str,
    signed_in: bool,
    base_url: str,
    update_hint: str | None = None,
) -> Panel:
    """The branded landing screen shown when a user types just ``comfy``.

    Gradient wordmark + quick-start list + sign-in status. Pretty mode only;
    JSON callers will hit the help schema or ``discover`` paths.
    """
    wordmark = gradient_block(_WORDMARK_ROWS)
    tagline = Text(
        TAGLINE,
        style=f"dim {BRAND_END}",
        justify="center",
    )

    # Quick start: command + one-line description
    qs = Table.grid(padding=(0, 3), expand=False)
    qs.add_column(style="bold white", no_wrap=True)
    qs.add_column(style="dim", overflow="fold")
    qs.add_row("comfy setup", "get started — pick local/cloud, sign in, install skills")
    qs.add_row("comfy install", "install ComfyUI")
    qs.add_row("comfy launch", "start the local server")
    qs.add_row("comfy cloud login", "sign in to Comfy Cloud")
    qs.add_row("comfy discover", "the agent-facing surface")
    qs.add_row("comfy --help", "everything else")

    if signed_in:
        cloud_line = Text.assemble(
            ("Cloud  ", "dim"),
            ("✓ signed in", "bold green"),
            ("   ", ""),
            (_pretty_host(base_url), "dim"),
        )
    else:
        cloud_line = Text.assemble(
            ("Cloud  ", "dim"),
            ("– not signed in", "dim"),
            ("   ", ""),
            ("→ ", f"bold {BRAND_ACCENT}"),
            ("comfy setup", "yellow"),
        )

    rows = [
        Align.center(wordmark),
        Align.center(tagline),
        Text(""),
        Rule(style=BRAND_END),
        Text(""),
        Text("Quick start", style=f"bold {BRAND_ACCENT}"),
        qs,
        Text(""),
        cloud_line,
    ]
    if update_hint:
        rows.append(
            Text.assemble(
                ("Update ", "dim"),
                (f"v{update_hint} available", f"bold {BRAND_ACCENT}"),
                ("   → ", f"bold {BRAND_ACCENT}"),
                ("comfy update cli", "yellow"),
            )
        )
    body = Group(*rows)
    return Panel(
        body,
        subtitle=Text(f"comfy CLI v{version}", style="dim"),
        subtitle_align="right",
        border_style=BRAND_START,
        padding=(1, 3),
    )


def signed_out_banner(*, base_url: str, version: str = "") -> Panel:
    """Tiny branded card for whoami when not signed in."""
    body = Group(
        Text("not signed in", style="dim"),
        Text(""),
        Text.assemble(
            ("→ ", f"bold {BRAND_ACCENT}"),
            ("comfy cloud login", "yellow"),
        ),
    )
    return branded_panel(
        body,
        title="comfy cloud",
        version=version,
        host=_pretty_host(base_url),
        padding=(0, 2),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iso(expires_at: int | None) -> str:
    if expires_at is None:
        return ""
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return ""


def _pretty_host(url: str) -> str:
    # Strip scheme for the panel subtitle — looks cleaner.
    return url.replace("https://", "").replace("http://", "").rstrip("/")
