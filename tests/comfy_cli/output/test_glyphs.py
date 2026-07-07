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
