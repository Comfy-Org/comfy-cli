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
``comfy run --prompt 'a fox'`` gets disabled. The one exception is the root
callback's own options, which are legal only *before* the subcommand and are
skipped rather than treated as the end of the scan — otherwise prefixing the
bad command with ``--json`` would hide it from the lint entirely.

What it does catch is the exact failure mode above: a word that *looks* like a
subcommand, sitting where the tree has no such subcommand.
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

# A Markdown fenced code block. The README's primary examples live in ```bash
# fences as bare shell lines with no backticks of their own, so none of the
# patterns above see them — the guardrail would skip the very copy users are
# most likely to paste.
_FENCE_BLOCK = re.compile(r"^```[^\n]*\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
# One shell line inside such a block: an optional `$ ` prompt marker, the
# invocation, and an optional trailing `# comment`.
_FENCED_LINE = re.compile(r"^[ \t]*(?:\$[ \t]+)?(?P<cmd>comfy(?:[ \t]+[^\s#]+)*)[ \t]*(?:#.*)?$", re.MULTILINE)

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


def _iter_spans(text: str):
    """Yield ``(command text, offset of the command itself)`` for every mention.

    The offset is that of the ``cmd`` group, not of the whole match: the
    ``run:`` pattern's leading boundary class can swallow the preceding
    newline, and anchoring the line number on ``match.start()`` would then
    report the violation one line too high.
    """
    for pattern in (_BACKTICK_SPAN, _RUN_PREFIX, _QUOTED_INVOCATION):
        for match in pattern.finditer(text):
            yield match.group("cmd"), match.start("cmd")
    for block in _FENCE_BLOCK.finditer(text):
        body, base = block.group("body"), block.start("body")
        for match in _FENCED_LINE.finditer(body):
            yield match.group("cmd"), base + match.start("cmd")


def extract_mentions(text: str, *, path: str) -> list[Mention]:
    """Return every ``comfy …`` invocation mentioned in ``text``."""
    out: list[Mention] = []
    seen: set[tuple[str, int]] = set()
    for raw, offset in _iter_spans(text):
        span = " ".join(raw.split())
        line = text.count("\n", 0, offset) + 1
        if (span, line) in seen:
            continue
        seen.add((span, line))
        out.append(Mention(text=span, path=path, line=line))
    return sorted(out, key=lambda m: (m.line, m.text))


def global_options(help_doc: dict[str, Any]) -> dict[str, bool]:
    """Map each root-callback option flag to whether it consumes a following value.

    Derived from the live tree rather than hardcoded so the set cannot drift as
    root options are added or renamed. ``help_doc`` is the whole document from
    :func:`comfy_cli.help_json.build_help_json`.
    """
    out: dict[str, bool] = {}
    for param in help_doc.get("root", {}).get("params") or []:
        if param.get("param_kind") != "option":
            continue
        takes_value = not param.get("is_flag")
        for flag in param.get("flags") or []:
            out[flag] = takes_value
    return out


def check_mention(
    mention: Mention,
    tree: dict[str, Any],
    *,
    globals_: dict[str, bool] | None = None,
) -> Violation | None:
    """Resolve one mention against ``tree`` (a ``help_json`` command map).

    ``tree`` is the ``commands`` map from :func:`comfy_cli.help_json.build_help_json`
    — ``{name: {"subcommands": {...}}}``. ``globals_`` is :func:`global_options`;
    without it a leading global option ends the scan and the subcommand after it
    is never checked. Returns ``None`` when the mention is valid or too
    ambiguous to judge.
    """
    tokens = mention.text.split()[1:]  # drop the "comfy" program name
    node: dict[str, Any] = {"subcommands": tree}
    resolved = "comfy"
    known_globals = globals_ or {}

    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        # Root-callback options are only legal before the subcommand, so skip
        # them only while nothing has resolved yet. Bailing at the first `-`
        # would wave through a `--json`-prefixed spelling of the very dead-end
        # this module exists to catch (see the auth-login case below).
        if resolved == "comfy" and token.startswith("-"):
            name, eq, _value = token.partition("=")
            if name in known_globals:
                if known_globals[name] and not eq:
                    i += 1  # the option consumes the next token as its value
                continue
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


def check_text(
    text: str,
    *,
    path: str,
    tree: dict[str, Any],
    globals_: dict[str, bool] | None = None,
) -> list[Violation]:
    """Extract and check every mention in one file's ``text``."""
    return [
        v for m in extract_mentions(text, path=path) if (v := check_mention(m, tree, globals_=globals_)) is not None
    ]
