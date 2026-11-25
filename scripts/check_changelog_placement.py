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

Deliberately narrow, so that a failure is always worth acting on:

- **Only pure insertions count.** A `-U0` diff encodes an in-place edit as a `-`/`+` pair, so
  rewording an existing released entry would otherwise read as filing a new one. Fixing a typo in
  shipped notes is legitimate; only a hunk that adds and deletes nothing is a new entry.
- **A section this PR introduces is not an already-released one.** A release PR adds
  `## [X.Y.Z] - date` and the entries below it; without this it would fail its own check for
  doing exactly its job. A heading does not sit inside itself, and nothing sits "inside a
  release" when this very diff created the heading.
- **Any `## ` heading ends a release section**, not only the next `## [`. `## Earlier releases`
  and the link-reference footer below it are not part of the last release.
- **A changelog-only diff is exempt.** It cannot be describing its own code, so it is a
  retroactive correction to notes for something that genuinely shipped, and filing *that* under
  `## [Unreleased]` would be the misleading option. Known gap: a follow-up adding a forgotten
  entry for already-merged-but-unreleased code is exempt too, and would pass silently if
  `[Unreleased]` were folded while it was open.
"""

from __future__ import annotations

import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"
HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,\d+)? @@")
RELEASE_HEADING = re.compile(r"^## \[(?!Unreleased\])")


def added_line_numbers(diff: str, insertions_only: bool = True) -> list[int]:
    """New-file line numbers of added lines in a `-U0` diff.

    With `insertions_only` (the default) a hunk whose old-side count is anything but 0 is a
    replacement rather than a new entry and is skipped. Pass False to get every added line,
    which is what heading detection needs: a release PR can introduce `## [X.Y.Z]` inside a
    replacement hunk, and a heading it created is not an already-released section whichever
    hunk carried it.

    `---`/`+++` are file headers only before the first hunk header: inside a hunk, an added
    line whose own text begins `++` arrives as `+++...` and must be counted, or every later
    line in that hunk is numbered one too low and can cross a heading boundary.
    """
    numbers: list[int] = []
    cursor = 0
    in_hunk = False
    counting = False
    for line in diff.splitlines():
        hunk = HUNK.match(line)
        if hunk:
            in_hunk = True
            old_count = 1 if hunk.group(1) is None else int(hunk.group(1))
            counting = old_count == 0 or not insertions_only
            cursor = int(hunk.group(2))
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            if counting:
                numbers.append(cursor)
            cursor += 1
        elif line.startswith(" "):
            cursor += 1
    return numbers


def enclosing_heading(lines: list[str], line_number: int) -> str | None:
    """The `## ` heading a 1-indexed line sits under, or None above the first one.

    The line itself is excluded: a heading does not sit inside the section it opens.
    """
    heading = None
    for index, text in enumerate(lines, start=1):
        if index >= line_number:
            break
        if text.startswith("## "):
            heading = text.rstrip()
    return heading


def misplaced(diff: str, merged: list[str]) -> dict[str, int]:
    """Added lines per already-released heading. Empty means nothing is misplaced."""
    added = added_line_numbers(diff)
    introduced = {
        merged[number - 1].rstrip()
        for number in added_line_numbers(diff, insertions_only=False)
        if 0 < number <= len(merged) and merged[number - 1].startswith("## ")
    }
    counts: dict[str, int] = {}
    for line_number in added:
        heading = enclosing_heading(merged, line_number)
        if not heading or heading in introduced:
            continue
        if RELEASE_HEADING.match(heading):
            counts[heading] = counts.get(heading, 0) + 1
    return counts


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git",) + args, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        # Without this the failure is a bare CalledProcessError and the actual cause -- a
        # shallow checkout with no HEAD^1, a bad base, a deleted CHANGELOG.md -- is invisible.
        raise SystemExit(f"git {' '.join(args)} failed ({error.returncode}): {error.stderr.strip()}") from error


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD^1"
    changed = _git("diff", "--name-only", "-z", base, "HEAD").split("\0")
    changed = [path for path in changed if path]
    if CHANGELOG not in changed:
        print(f"{CHANGELOG} untouched; nothing to place.")
        return 0
    if changed == [CHANGELOG]:
        print(f"{CHANGELOG}-only diff; a note-only correction is exempt.")
        return 0
    diff = _git("diff", "-U0", base, "HEAD", "--", CHANGELOG)
    # split("\n"), not splitlines(): the latter also breaks on U+2028, U+2029 and \x0b-\x1e,
    # which git does not, and any of them would drift this numbering from git's.
    merged = _git("show", f"HEAD:{CHANGELOG}").split("\n")
    counts = misplaced(diff, merged)
    if not counts:
        print(f"{CHANGELOG}: no new entry lands in an already-released section. OK.")
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
