"""Tiny shared vocabulary for status icons + colors across pretty-mode output.

One source of truth so a queued job looks the same in ``comfy run``,
``comfy jobs ls``, and ``comfy jobs watch``. Easy to swap if the brand
changes; easy to teach an agent ("✓ means completed everywhere").
"""

from __future__ import annotations

from comfy_cli.output.sanitize import sanitize_markup

# (glyph, rich-style) per terminal status. Mirrors the JobStatus enum from
# the architecture doc; new statuses must be added here.
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "queued": ("⏳", "cyan"),
    "running": ("◐", "yellow"),
    "completed": ("✓", "bold green"),
    "error": ("✗", "bold red"),
    "cancelled": ("⊘", "yellow"),
    "pending": ("⏳", "cyan"),  # alias used by cloud
}

DEFAULT_STYLE: tuple[str, str] = ("·", "dim")


# Cloud's raw status vocabulary (``executing``, ``success``, ``failed``,
# ``non_retryable_error``, ``retryable_error``) doesn't match the small set
# above. Canonicalize at the rendering boundary so users see one stable set
# regardless of whether the prompt ran on a local server or Comfy Cloud.
_CLOUD_ALIASES = {
    "executing": "running",
    "success": "completed",
    "failed": "error",
    "non_retryable_error": "error",
    "retryable_error": "error",
}


def _canonical(status: str | None) -> str:
    raw = (status or "").strip().lower()
    return _CLOUD_ALIASES.get(raw, raw)


def status_glyph(status: str | None) -> str:
    """Return ``"<glyph> <status>"`` styled with Rich tags.

    Cloud aliases collapse to the local-style vocabulary before lookup —
    no raw ``non_retryable_error`` or ``executing`` strings leak into
    the pretty surface.
    """
    canonical = _canonical(status)
    glyph, style = STATUS_STYLE.get(canonical, DEFAULT_STYLE)
    # An unrecognized status is echoed verbatim, and the return value is markup
    # bound for `Table.add_row` / `Text.from_markup`. Escaping the interpolated
    # half keeps the tags we author live while a server's `[red]` or `[/]` stays
    # inert text — a no-op for every status in `STATUS_STYLE`, all plain words.
    # The `or "unknown"` tests the *sanitized* label, not the raw one: a status
    # made only of control bytes is truthy but sanitizes to empty, and falling
    # back there beats rendering a glyph with a dangling space and no word.
    label = sanitize_markup(canonical) if canonical else ""
    return f"[{style}]{glyph} {label or 'unknown'}[/{style}]"
