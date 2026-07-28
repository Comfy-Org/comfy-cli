"""Lint the ``comfy …`` command strings we hand users, against the real command tree.

Help text, error hints, the README and the bundled agent skills are full of
copy-pasteable invocations (``run `comfy cloud login```). Nothing binds them to
the Typer tree, so they rot silently: an audit found eight sites telling users to
run an ``auth login`` subcommand that has never existed (the real one is
`comfy cloud login`) — a dead end for every human and agent that followed it.

This module extracts those mentions and resolves each against the live tree
built by :mod:`comfy_cli.help_json`. ``tests/comfy_cli/test_command_mentions.py``
runs it over the repo as a guardrail.

The resolver is deliberately biased toward false NEGATIVES: it stops scanning at
the first token it cannot confidently classify as a subcommand (an option, a
placeholder, a path, a quoted value), because a lint that cries wolf on
``comfy run --prompt 'a fox'`` gets disabled. What it does catch is the exact
failure mode above: a word that *looks* like a subcommand, sitting where the
tree has no such subcommand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# A backtick-quoted span that starts with the program name, e.g. `comfy cloud login`.
# Also matches the bare `run: comfy cloud login` shape used in some hints.
_BACKTICK_SPAN = re.compile(r"`(?P<cmd>comfy\s+[^`]*)`")
_RUN_PREFIX = re.compile(r"(?:^|[\s\"'(])run:?\s+(?P<cmd>comfy\s+[a-z0-9 _-]+)")
# A whole string literal that is nothing but an invocation, e.g. the
# ``label="comfy cloud whoami"`` used by the run-cli tour. Restricted to
# command-shaped words so prose sentences beginning with "comfy" don't match.
_QUOTED_INVOCATION = re.compile(r"""(?P<q>['"])(?P<cmd>comfy(?:\s+-{1,2}[a-z0-9-]+|\s+[a-z][a-z0-9-]*)+)(?P=q)""")

# A token we are willing to treat as a subcommand name. Anything else (options,
# <PLACEHOLDERS>, paths, quoted values, ALLCAPS) ends the scan.
_SUBCOMMAND_TOKEN = re.compile(r"[a-z][a-z0-9-]*\Z")


@dataclass(frozen=True)
class Mention:
    """One ``comfy …`` string found in a source file."""

    text: str
    path: str
    line: int


@dataclass(frozen=True)
class Violation:
    """A mention whose command path does not exist in the CLI."""

    mention: Mention
    resolved: str
    unknown_token: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.mention.path}:{self.mention.line}: `{self.mention.text}` — "
            f"{self.resolved!r} has no subcommand {self.unknown_token!r}"
        )


def extract_mentions(text: str, *, path: str) -> list[Mention]:
    """Return every ``comfy …`` invocation mentioned in ``text``."""
    out: list[Mention] = []
    seen: set[tuple[str, int]] = set()
    for pattern in (_BACKTICK_SPAN, _RUN_PREFIX, _QUOTED_INVOCATION):
        for match in pattern.finditer(text):
            span = " ".join(match.group("cmd").split())
            line = text.count("\n", 0, match.start()) + 1
            if (span, line) in seen:
                continue
            seen.add((span, line))
            out.append(Mention(text=span, path=path, line=line))
    return sorted(out, key=lambda m: (m.line, m.text))


def check_mention(mention: Mention, tree: dict[str, Any]) -> Violation | None:
    """Resolve one mention against ``tree`` (a ``help_json`` command map).

    ``tree`` is the ``commands`` map from :func:`comfy_cli.help_json.build_help_json`
    — ``{name: {"subcommands": {...}}}``. Returns ``None`` when the mention is
    valid or too ambiguous to judge.
    """
    tokens = mention.text.split()[1:]  # drop the "comfy" program name
    node: dict[str, Any] = {"subcommands": tree}
    resolved = "comfy"

    for token in tokens:
        if token.startswith("-") or not _SUBCOMMAND_TOKEN.match(token):
            # An option, a placeholder, a path, a quoted value: from here on we
            # can no longer tell subcommands from argument values. Stop clean.
            return None
        subs = node.get("subcommands") or {}
        if token in subs:
            node = subs[token]
            resolved = f"{resolved} {token}"
            continue
        if subs:
            # A group with no such subcommand: this is the BE-2996 auth-login
            # bug — the word reads as a subcommand but is really an unparsed
            # positional the group will reject.
            return Violation(mention=mention, resolved=resolved, unknown_token=token)
        # A leaf command; the leftover words are its positional arguments.
        return None
    return None


def check_text(text: str, *, path: str, tree: dict[str, Any]) -> list[Violation]:
    """Extract and check every mention in one file's ``text``."""
    return [v for m in extract_mentions(text, path=path) if (v := check_mention(m, tree)) is not None]
