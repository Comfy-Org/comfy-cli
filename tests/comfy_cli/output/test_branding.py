"""Tests for the branded welcome / whoami banners."""

from __future__ import annotations

import io
import re
import time

from rich.console import Console
from rich.panel import Panel

from comfy_cli.output.branding import (
    TAGLINE,
    branded_panel,
    gradient_block,
    gradient_text,
    intro_banner,
    signed_out_banner,
    welcome_banner,
    whoami_banner,
)

# ---------------------------------------------------------------------------
# Task 1 — branded_panel + TAGLINE (the screen-wide chrome contract)
# ---------------------------------------------------------------------------


def test_tagline_constant():
    assert TAGLINE == "the agent-aware ComfyUI CLI"


def test_branded_panel_minimal_subtitle():
    panel = branded_panel("body", title="env", version="0.0.0")
    assert isinstance(panel, Panel)
    assert str(panel.title) == "env"
    assert str(panel.subtitle) == "comfy CLI v0.0.0"
    assert panel.title_align == "left"
    assert panel.subtitle_align == "right"


def test_branded_panel_with_where():
    panel = branded_panel("body", title="jobs", version="0.0.0", where="local")
    assert str(panel.subtitle) == "comfy CLI v0.0.0  ·  local"


def test_branded_panel_with_where_and_host():
    panel = branded_panel("body", title="jobs", version="0.0.0", where="local", host="127.0.0.1:8188")
    assert str(panel.subtitle) == "comfy CLI v0.0.0  ·  local · 127.0.0.1:8188"


def test_branded_panel_with_host_no_where():
    """host alone (no where) — used by cloud whoami screen."""
    panel = branded_panel("body", title="comfy cloud", version="0.0.0", host="cloud.comfy.org")
    assert str(panel.subtitle) == "comfy CLI v0.0.0 · cloud.comfy.org"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render(renderable, *, width: int = 100) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, width=width, no_color=True).print(renderable)
    return _ANSI_RE.sub("", buf.getvalue())


# ---------------------------------------------------------------------------
# gradient helpers
# ---------------------------------------------------------------------------


def test_gradient_text_preserves_input_characters():
    text = gradient_text("comfy cloud")
    assert text.plain == "comfy cloud"


def test_gradient_block_preserves_row_content():
    rows = [" ████   ████ ", "██     ██  ██", "██     ██  ██"]
    block = gradient_block(rows)
    # plain (un-styled) text should be the rows joined by newlines.
    assert block.plain == "\n".join(rows)


def test_gradient_block_handles_single_row():
    block = gradient_block(["one row only"])
    assert block.plain == "one row only"


# ---------------------------------------------------------------------------
# welcome_banner
# ---------------------------------------------------------------------------


def test_welcome_banner_includes_wordmark_and_info():
    out = _render(
        welcome_banner(
            base_url="https://testcloud.comfy.org",
            scope="mcp:tools:read mcp:tools:call",
            client_id="comfy-cli",
            expires_at=int(time.time()) + 3600,
            cta="comfy run --workflow X.json",
        )
    )
    # Wordmark glyph rows: at least one row of block characters present.
    assert "████" in out
    # Subtitle below the wordmark.
    assert "C  L  O  U  D" in out
    assert "Signed in to Comfy Cloud" in out
    assert "mcp:tools:read" in out
    assert "comfy-cli" in out
    # CTA arrow.
    assert "→" in out
    assert "comfy run --workflow X.json" in out
    # Host shown in the subtitle.
    assert "testcloud.comfy.org" in out


def test_welcome_banner_renders_in_narrow_terminal():
    # 60 cols is the lower bound we care about — must not raise / wrap badly.
    out = _render(
        welcome_banner(
            base_url="https://testcloud.comfy.org",
            scope="mcp:tools:read",
            client_id="comfy-cli",
            expires_at=int(time.time()) + 600,
            cta=None,
        ),
        width=60,
    )
    assert "Signed in" in out
    assert "comfy-cli" in out


def test_welcome_banner_humanizes_expiry():
    out = _render(
        welcome_banner(
            base_url="https://testcloud.comfy.org",
            scope="x",
            client_id="c",
            expires_at=int(time.time()) + 3597,
            cta=None,
        )
    )
    # "in 59m 57s" (or close — exact seconds may drift in CI)
    assert re.search(r"in \d+m \d+s", out), out


# ---------------------------------------------------------------------------
# whoami_banner / signed_out_banner
# ---------------------------------------------------------------------------


def test_whoami_banner_active_shows_check():
    out = _render(
        whoami_banner(
            base_url="https://testcloud.comfy.org",
            scope="mcp:tools:read",
            client_id="comfy-cli",
            expires_at=int(time.time()) + 600,
            expired=False,
        )
    )
    assert "active" in out
    assert "✓" in out
    assert "comfy-cli" in out


def test_whoami_banner_expired_shows_warning():
    out = _render(
        whoami_banner(
            base_url="https://testcloud.comfy.org",
            scope="mcp:tools:read",
            client_id="comfy-cli",
            expires_at=int(time.time()) - 60,
            expired=True,
        )
    )
    assert "expired" in out
    assert "⚠" in out


def test_intro_banner_signed_out_includes_wordmark_and_login_hint():
    out = _render(
        intro_banner(
            version="0.0.0",
            signed_in=False,
            base_url="https://testcloud.comfy.org",
        )
    )
    assert "████" in out  # wordmark glyph present
    assert "the agent-aware ComfyUI CLI" in out
    assert "Quick start" in out
    assert "comfy install" in out
    assert "comfy launch" in out
    assert "comfy auth login" in out or "comfy cloud login" in out
    assert "comfy discover" in out
    assert "comfy --help" in out
    assert "not signed in" in out
    assert "comfy CLI v0.0.0" in out or "comfy-cli v0.0.0" in out


def test_intro_banner_signed_out_leads_with_setup():
    """A first-time / unconfigured user must be pointed at the one-step wizard."""
    out = _render(
        intro_banner(
            version="0.0.0",
            signed_in=False,
            base_url="https://testcloud.comfy.org",
        )
    )
    assert "comfy setup" in out  # the get-started entry point is surfaced
    # the not-signed-in nudge leads to the wizard, not just raw login
    assert "not signed in" in out


def test_intro_banner_signed_in_shows_check_and_host():
    out = _render(
        intro_banner(
            version="1.2.3",
            signed_in=True,
            base_url="https://testcloud.comfy.org",
        )
    )
    assert "signed in" in out
    assert "✓" in out
    assert "testcloud.comfy.org" in out
    assert "comfy CLI v1.2.3" in out or "comfy-cli v1.2.3" in out


def test_signed_out_banner_points_at_login():
    out = _render(signed_out_banner(base_url="https://testcloud.comfy.org"))
    assert "not signed in" in out
    assert "comfy auth login" in out or "comfy cloud login" in out
    assert "testcloud.comfy.org" in out
