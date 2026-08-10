"""Tests for the shared status-glyph vocabulary.

Cloud (``executing``, ``success``, ``failed``, ``non_retryable_error``)
and local (``running``, ``completed``, ``error``) use different status
strings; the pretty surface canonicalizes everything onto the small
local-style set so users see one stable vocabulary across ``comfy
jobs ls`` regardless of routing.
"""

from __future__ import annotations

from comfy_cli.output.glyphs import status_glyph


def test_cloud_executing_renders_as_running():
    assert "◐" in status_glyph("executing")
    assert "running" in status_glyph("executing")  # canonical label


def test_cloud_in_progress_renders_as_running():
    """`/api/jobs` spells an executing job `in_progress`; without the alias it
    renders as the unknown-status fallback dot."""
    out = status_glyph("in_progress")
    assert "◐" in out
    assert "running" in out
    assert "·" not in out


def test_cloud_aliases_agree_with_the_shared_status_map():
    """This module keeps its own alias literal on purpose — it is the leaf of
    the output package and importing ``jobs_state`` here would drag psutil /
    requests / rich in to draw a check mark. That copy is only safe if
    something fails when the two drift, which is this test: every raw status
    ``jobs_state.CLOUD_STATUS_ALIASES`` knows must render exactly as its
    canonical form does.

    Both directions have already drifted once — `lost` and `canceled` were in
    the shared map but not here (rendering as the dim `·` unknown dot instead
    of `✗`/`⊘`), and `retryable_error` was here but not there.
    """
    from comfy_cli import jobs_state

    for raw, canonical in jobs_state.CLOUD_STATUS_ALIASES.items():
        assert status_glyph(raw) == status_glyph(canonical), f"{raw!r} does not render as {canonical!r}"

    # ...and nothing this module aliases may be unknown to the shared map: an
    # entry only here would canonicalize one way on the pretty surface and
    # another in the payload.
    from comfy_cli.output.glyphs import _CLOUD_ALIASES

    assert set(_CLOUD_ALIASES) <= set(jobs_state.CLOUD_STATUS_ALIASES)


def test_cloud_failed_renders_as_error():
    assert "✗" in status_glyph("failed")
    assert "error" in status_glyph("failed")


def test_cloud_non_retryable_error_renders_as_error():
    assert "✗" in status_glyph("non_retryable_error")
    assert "error" in status_glyph("non_retryable_error")


def test_cloud_success_renders_as_completed():
    assert "✓" in status_glyph("success")
    assert "completed" in status_glyph("success")


def test_unknown_status_falls_through_to_default():
    out = status_glyph("definitely_not_a_status")
    assert "·" in out  # default glyph
