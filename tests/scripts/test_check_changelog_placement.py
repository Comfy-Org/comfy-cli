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


def test_a_hunk_carrying_old_lines_is_not_a_pure_insertion() -> None:
    # -U0 emits no context, so a non-zero old count always means lines were replaced.
    assert added_line_numbers("@@ -6,2 +7,3 @@\n-x\n-y\n+a\n+b\n+c\n") == []


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


def test_a_replacement_hunk_is_not_a_new_entry() -> None:
    # -U0 writes an in-place edit as a -/+ pair. Rewording a shipped entry is legitimate.
    diff = "@@ -13,1 +13,1 @@\n-- a shipped entry\n+- a shipped entry, reworded\n"
    assert added_line_numbers(diff) == []
    assert misplaced(diff, MERGED) == {}


def test_an_omitted_old_count_means_one_line_not_zero() -> None:
    # `@@ -13 +13 @@` is a one-for-one replacement, not an insertion.
    assert added_line_numbers("@@ -13 +13 @@\n-old\n+new\n") == []


def test_an_added_line_beginning_with_plus_plus_is_counted() -> None:
    # It arrives as `+++...`. Dropping it also stops the cursor, shifting every later line.
    diff = "@@ -6,0 +7,2 @@\n+++ a literal line\n+- and the next one\n"
    assert added_line_numbers(diff) == [7, 8]


def test_file_headers_before_the_first_hunk_are_still_skipped() -> None:
    diff = "--- a/CHANGELOG.md\n+++ b/CHANGELOG.md\n@@ -1,0 +2,1 @@\n+a\n"
    assert added_line_numbers(diff) == [2]


def test_a_section_the_pr_introduces_is_not_already_released() -> None:
    # A release PR adds its own `## [X.Y.Z]` and the entries under it. It must not fail its own
    # check for doing that.
    merged = ["# Changelog", "", "## [1.30.0] - 2026-12-01", "", "- moved down"]
    diff = "@@ -2,0 +3,3 @@\n+## [1.30.0] - 2026-12-01\n+\n+- moved down\n"
    assert misplaced(diff, merged) == {}


def test_a_heading_introduced_inside_a_replacement_hunk_still_counts_as_introduced() -> None:
    # The real case: `docs(changelog): roll Unreleased into 1.17.0 through 1.21.0` adds its
    # headings in replacement hunks while the entries arrive in pure insertions. Reading
    # headings from insertions only left that PR failing its own check.
    merged = ["# Changelog", "", "## [1.21.0] - 2026-09-24", "", "- rolled down"]
    diff = "@@ -2,1 +3,1 @@\n-## [Unreleased]\n+## [1.21.0] - 2026-09-24\n@@ -3,0 +5,1 @@\n+- rolled down\n"
    assert added_line_numbers(diff) == [5]
    assert added_line_numbers(diff, insertions_only=False) == [3, 5]
    assert misplaced(diff, merged) == {}


def test_an_entry_added_under_a_heading_the_pr_did_not_add_still_fails() -> None:
    # The exemption above must not extend to a pre-existing release section.
    diff = "@@ -12,0 +13,1 @@\n+- a shipped entry\n"
    assert misplaced(diff, MERGED) == {"## [1.27.0] - 2026-11-18": 1}


def test_a_non_release_level_two_heading_ends_the_release_section() -> None:
    # `## Earlier releases` and the link-reference footer below it are not part of 1.27.0.
    merged = MERGED + ["", "## Earlier releases", "", "[1.27.0]: https://example/compare"]
    diff = f"@@ -{len(MERGED) + 3},0 +{len(MERGED) + 4},1 @@\n+[1.26.0]: https://example/c\n"
    assert misplaced(diff, merged) == {}
