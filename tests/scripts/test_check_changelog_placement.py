from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_changelog_placement.py"
_SPEC = importlib.util.spec_from_file_location("check_changelog_placement", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_SCRIPT}")
check_changelog_placement = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_changelog_placement
_SPEC.loader.exec_module(check_changelog_placement)

added_line_numbers = check_changelog_placement.added_line_numbers
enclosing_heading = check_changelog_placement.enclosing_heading
misplaced = check_changelog_placement.misplaced

MERGED = [
    "# Changelog",
    "",
    "## [Unreleased]",
    "",
    "### Fixed",
    "",
    "- a pending entry",
    "",
    "## [1.27.0] - 2026-11-18",
    "",
    "### Fixed",
    "",
    "- a shipped entry",
]


def test_added_line_numbers_counts_every_line_of_a_hunk() -> None:
    assert added_line_numbers("@@ -6,0 +7,3 @@\n+a\n+b\n+c\n") == [7, 8, 9]


def test_added_line_numbers_restarts_at_each_hunk_header() -> None:
    diff = "@@ -6,0 +7,1 @@\n+a\n@@ -20,0 +30,2 @@\n+b\n+c\n"
    assert added_line_numbers(diff) == [7, 30, 31]


def test_added_line_numbers_advances_past_context_without_counting_it() -> None:
    assert added_line_numbers("@@ -6,2 +7,3 @@\n ctx\n+a\n ctx\n") == [8]


def test_added_line_numbers_ignores_deletions_and_file_headers() -> None:
    diff = "--- a/CHANGELOG.md\n+++ b/CHANGELOG.md\n@@ -6,2 +6,0 @@\n-a\n-b\n"
    assert added_line_numbers(diff) == []


def test_enclosing_heading_reads_the_nearest_heading_above() -> None:
    assert enclosing_heading(MERGED, 7) == "## [Unreleased]"
    assert enclosing_heading(MERGED, 13) == "## [1.27.0] - 2026-11-18"


def test_enclosing_heading_is_none_above_the_first_heading() -> None:
    assert enclosing_heading(MERGED, 1) is None


def test_an_unreleased_entry_is_not_misplaced() -> None:
    assert misplaced("@@ -6,0 +7,1 @@\n+- a pending entry\n", MERGED) == {}


def test_an_entry_under_a_released_heading_is_misplaced() -> None:
    diff = "@@ -12,0 +13,1 @@\n+- a shipped entry\n"
    assert misplaced(diff, MERGED) == {"## [1.27.0] - 2026-11-18": 1}


def test_a_block_straddling_a_heading_is_counted_per_section() -> None:
    # The case a first-line-only reader gets wrong: half the block is fine, half is not.
    merged = MERGED[:7] + ["- added here"] + MERGED[7:] + ["- and here"]
    diff = "@@ -6,0 +8,1 @@\n+- added here\n@@ -13,0 +15,1 @@\n+- and here\n"
    assert misplaced(diff, merged) == {"## [1.27.0] - 2026-11-18": 1}


def test_a_preamble_line_is_not_attributed_to_a_release() -> None:
    assert misplaced("@@ -1,0 +2,1 @@\n+a note\n", MERGED) == {}
