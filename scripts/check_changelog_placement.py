"""Fail a PR whose CHANGELOG.md entry lands inside an already-released section.

CONTRIBUTING.md asks for notable user-facing changes under `## [Unreleased]`. Nothing enforced
it, and the interesting failures are not authors ignoring the rule -- they are authors following
it and the section moving underneath them. When #1189 folded `## [Unreleased]` into 1.27.0,
1.28.0 and 1.29.0, every branch whose entry was the tail of that section kept a correct-looking
diff and acquired a merge result that files the entry inside a shipped release.

So the question has to be asked about the merge result, not the branch head. On a `pull_request`
event, `actions/checkout` checks out `refs/pull/N/merge`, whose first parent is the base commit
the merge was computed against. Diffing `HEAD^1` against `HEAD` is therefore exactly this PR's
effect on that base -- and `origin/main` is the wrong comparison even though it looks like the
right one: GitHub recomputes the merge ref lazily, so when main has moved on, every file main
gained since reads as a file this PR deleted. That is not cosmetic. It is what made #1200, a
changelog-only PR that should be exempt, fail the exemption during this check's own trial run.

A changelog-only diff is exempt. It cannot be describing its own code, so it is a retroactive
correction to notes for something that genuinely shipped -- and filing *that* under
`## [Unreleased]` would be the misleading option.
"""

from __future__ import annotations

import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
RELEASE_HEADING = re.compile(r"^## \[(?!Unreleased\])")


def added_line_numbers(diff: str) -> list[int]:
    """New-file line numbers of every added line in a `-U0` diff."""
    numbers: list[int] = []
    cursor = 0
    for line in diff.splitlines():
        hunk = HUNK.match(line)
        if hunk:
            cursor = int(hunk.group(1))
        elif line.startswith(("+++", "---")):
            continue
        elif line.startswith("+"):
            numbers.append(cursor)
            cursor += 1
        elif line.startswith(" "):
            cursor += 1
    return numbers


def enclosing_heading(lines: list[str], line_number: int) -> str | None:
    """The `## [...]` heading a 1-indexed line sits under, or None above the first one."""
    heading = None
    for index, text in enumerate(lines, start=1):
        if index > line_number:
            break
        if text.startswith("## ["):
            heading = text.rstrip()
    return heading


def misplaced(diff: str, merged: list[str]) -> dict[str, int]:
    """Added lines per released heading. Empty means nothing is misplaced."""
    counts: dict[str, int] = {}
    for line_number in added_line_numbers(diff):
        heading = enclosing_heading(merged, line_number)
        if heading and RELEASE_HEADING.match(heading):
            counts[heading] = counts.get(heading, 0) + 1
    return counts


def _git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True, check=True).stdout


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD^1"
    changed = _git("diff", "--name-only", base, "HEAD").split()
    if CHANGELOG not in changed:
        print(f"{CHANGELOG} untouched; nothing to place.")
        return 0
    if changed == [CHANGELOG]:
        print(f"{CHANGELOG}-only diff; a note-only correction is exempt.")
        return 0
    diff = _git("diff", "-U0", base, "HEAD", "--", CHANGELOG)
    merged = _git("show", f"HEAD:{CHANGELOG}").splitlines()
    counts = misplaced(diff, merged)
    if not counts:
        print(f"{CHANGELOG} entry lands under ## [Unreleased]. OK.")
        return 0
    for heading, count in counts.items():
        print(f"{CHANGELOG}: {count} added line(s) land under {heading}", file=sys.stderr)
    print(
        "\nThat section is already released, so this promises its readers a change it does not\n"
        "carry. The usual cause is a rebase: the entry was the tail of ## [Unreleased] before\n"
        "that section was folded into a release. Move the block up under ## [Unreleased].",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
