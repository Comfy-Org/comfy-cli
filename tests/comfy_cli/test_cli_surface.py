"""CLI-surface guards: hint-string lint + command-inventory snapshot.

Two anti-rot tests, mirroring the MCP server's ``EXPECTED_TOOLS`` +
description-budget snapshot pattern for the CLI:

1. **Hint-string lint** (:func:`test_no_invalid_command_references`) — extracts
   every ``comfy …`` command invocation referenced in help text, docstrings,
   hint strings, error messages, examples, and bundled skill docs, then
   validates each against the *registered* command set by walking the Typer app
   tree. Catches the class of rot where a help/hint string points at a command
   that does not exist (e.g. the seven strings that referenced the nonexistent
   ``comfy auth login`` for who knows how long, since fixed to ``comfy cloud
   login``).

2. **Command-inventory snapshot** (:func:`test_command_inventory_snapshot`) —
   snapshots the full command tree so any surface addition/removal/hide shows up
   as an explicit diff in review.

    Regenerate the snapshot after an intentional surface change with::

        UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py

Both run under the normal ``pytest`` invocation, so they gate every PR in CI.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from comfy_cli.cmdline import app
from comfy_cli.help_json import build_help_json

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "comfy_cli"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures" / "cli_command_inventory.txt"

# ``comfy version`` is not an invokable subcommand — it is the output-envelope
# command name emitted by the top-level ``--version`` flag (cmdline.py) and is
# registered as such in discovery.COMMAND_SCHEMAS. Accept it as a real surface
# name even though the tree walk (which only sees invokable subcommands) can't.
ALLOWED_NONINVOKABLE_REFS = frozenset({"comfy version"})

# Known, pre-existing invalid references that this PR intentionally does NOT
# fix — the lint still reports them, but they don't fail the build. Each entry
# is tracked for a separate fix (mirroring how the seven ``auth login`` strings
# were corrected in their own change, separate from adding this guard). Removing
# the rot must also remove its baseline entry — test_baseline_is_not_stale enforces
# that so the baseline can never silently outlive the bug.
BASELINE_INVALID_REFS = frozenset()


# --------------------------------------------------------------------------- #
# command tree                                                                #
# --------------------------------------------------------------------------- #
def _command_tree() -> dict:
    """Nested ``{name: node}`` tree of the registered command surface.

    ``node`` carries a ``subcommands`` dict for groups; leaf commands omit it.
    Wrapped under a synthetic root so the validator can start at ``comfy``.
    """
    doc = build_help_json(app)
    return {"subcommands": doc["commands"]}


def _ref_is_valid(tokens: list[str], tree: dict) -> bool:
    """Is ``comfy <tokens…>`` a real command path?

    Walk the tree consuming subcommand-name tokens. A reference is valid when it
    lands on a real command node — trailing option/argument tokens after a leaf
    are fine (they're stripped before we get here). It is invalid the moment a
    command-name token is not a subcommand of the current group (the ``comfy
    auth login`` / ``comfy auth whoami`` class of rot).
    """
    node = tree
    for tok in tokens:
        subs = node.get("subcommands")
        if subs is None:
            return True  # reached a leaf; the rest are its args
        if tok in subs:
            node = subs[tok]
        else:
            return False
    return True


def _command_inventory() -> list[str]:
    """Every command path (groups + leaves), sorted, with a ``[hidden]`` marker."""
    doc = build_help_json(app)
    out: list[str] = []

    def walk(commands: dict, path: list[str]) -> None:
        for name, entry in sorted(commands.items()):
            fqp = " ".join(path + [name])
            out.append(f"{fqp}\t[hidden]" if entry.get("hidden") else fqp)
            subs = entry.get("subcommands")
            if subs:
                walk(subs, path + [name])

    walk(doc["commands"], ["comfy"])
    return out


# --------------------------------------------------------------------------- #
# reference extraction                                                         #
# --------------------------------------------------------------------------- #
# A command-name token: lowercase, the shape Typer/Click command names take.
_COMMAND_WORD = re.compile(r"[a-z][a-z0-9-]*$")
# Recognizable ``comfy …`` invocations. We only treat a string as a command
# reference when it carries an invocation marker, so prose like "the comfy
# skills into Claude Code" is not mistaken for ``comfy skills into``.
_BACKTICK = re.compile(r"`\s*(?:\$\s*)?comfy\b([^`]*)`")  # `comfy …` / `$ comfy …`
_SHELL_HINT = re.compile(r"(?:^|[\s(])(?:run:?\s+|\$\s+)comfy\b([^\n`.,;:)]*)", re.IGNORECASE)
_WHOLE_LINE = re.compile(r"^comfy\b([a-z0-9 _-]*)$")  # the whole string is a command line


def _leading_command_tokens(rest: str) -> list[str]:
    """Leading subcommand-name tokens of ``comfy <rest>`` (stop at the first
    option/argument/placeholder token)."""
    tokens: list[str] = []
    for tok in rest.strip().split():
        if _COMMAND_WORD.match(tok):
            tokens.append(tok)
        else:
            break
    return tokens


def _refs_in_text(text: str) -> list[list[str]]:
    """Token-lists for every recognizable ``comfy …`` reference in ``text``."""
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    parts = [m.group(1) for m in _BACKTICK.finditer(text)]
    parts += [m.group(1) for m in _SHELL_HINT.finditer(text)]
    whole = _WHOLE_LINE.match(text.strip())
    if whole:
        parts.append(whole.group(1))
    for rest in parts:
        tokens = _leading_command_tokens(rest)
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        out.append(tokens)
    return out


def _iter_source_strings():
    """Yield ``(relpath, lineno, string)`` for every scannable help/hint string.

    Python files contribute their string *literals* (docstrings, ``help=`` /
    ``hint=`` / ``message=`` kwargs, examples) via the AST, so command
    identifiers and comments are never mistaken for references. Bundled skill
    docs (``*.md``) contribute their raw text — they're agent-facing help.
    """
    for py in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - our own sources parse
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield rel, node.lineno, node.value
    for md in sorted(PACKAGE_DIR.rglob("*.md")):
        rel = md.relative_to(REPO_ROOT).as_posix()
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), start=1):
            yield rel, i, line


def _scan_invalid_references() -> list[tuple[str, int, str]]:
    """``(relpath, lineno, "comfy …")`` for every invalid command reference."""
    tree = _command_tree()
    found: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for rel, lineno, text in _iter_source_strings():
        for tokens in _refs_in_text(text):
            ref = " ".join(["comfy", *tokens])
            if ref in ALLOWED_NONINVOKABLE_REFS or _ref_is_valid(tokens, tree):
                continue
            key = (rel, lineno, ref)
            if key not in seen:
                seen.add(key)
                found.append(key)
    return found


# --------------------------------------------------------------------------- #
# tests: hint-string lint                                                      #
# --------------------------------------------------------------------------- #
def test_no_invalid_command_references():
    """No help/hint/error string references a command the CLI doesn't register."""
    invalid = [v for v in _scan_invalid_references() if v[2] not in BASELINE_INVALID_REFS]
    if invalid:
        lines = "\n".join(f"  {rel}:{lineno}: `{ref}`" for rel, lineno, ref in sorted(invalid))
        pytest.fail(
            "Help/hint/error strings reference commands that are not registered "
            "on the CLI (run `comfy <path>` to confirm each is missing):\n"
            f"{lines}\n\n"
            "Fix the string to name a real command, or register the command. "
            "If this is known, tracked rot, add it to BASELINE_INVALID_REFS."
        )


def test_baseline_is_not_stale():
    """Every BASELINE_INVALID_REFS entry must still be a live, invalid reference.

    Once the underlying rot is fixed the reference disappears; this test then
    fails, prompting removal of the now-stale baseline entry so the allowlist
    can never quietly outlive the bug it excuses.
    """
    live_refs = {ref for _, _, ref in _scan_invalid_references()}
    stale = sorted(BASELINE_INVALID_REFS - live_refs)
    assert not stale, f"BASELINE_INVALID_REFS entries no longer found (fixed?) — remove them: {stale}"


def test_lint_catches_auth_login_and_friends():
    """Regression fixture: the extractor+validator flags exactly the bad refs.

    The seven historical `comfy auth login` strings have since been corrected,
    so they no longer live in the tree — this pins the lint's *behaviour* against
    synthetic data (an `auth login` fixture included per the ticket) so a future
    refactor can't silently stop catching them.
    """
    tree = _command_tree()

    def is_flagged(text: str) -> bool:
        return any(
            " ".join(["comfy", *toks]) not in ALLOWED_NONINVOKABLE_REFS and not _ref_is_valid(toks, tree)
            for toks in _refs_in_text(text)
        )

    should_flag = [
        "re-run `comfy auth login` and complete the sign-in",  # the historical rot
        "run: comfy auth login",  # non-backtick shell-hint form
        "check network / `comfy auth whoami`",  # sibling rot (wrong group)
        "comfy auth login",  # whole-string form
        "`comfy nodes bogus`",  # invalid leaf under a real group
    ]
    should_pass = [
        "run `comfy cloud login` to sign in",  # the correct command
        "`comfy run --workflow wf.json`",  # valid leaf + args
        "`comfy workflow`",  # a bare group is a valid reference
        "`comfy generate list`",  # leaf that takes positional subcommands
        "`comfy version`",  # allowlisted --version envelope name
        "Write the comfy skills into Claude Code, Cursor.",  # prose, not an invocation
    ]
    missed = [t for t in should_flag if not is_flagged(t)]
    false_pos = [t for t in should_pass if is_flagged(t)]
    assert not missed, f"lint failed to flag invalid references: {missed}"
    assert not false_pos, f"lint false-positived on valid references / prose: {false_pos}"


# --------------------------------------------------------------------------- #
# tests: command-inventory snapshot                                           #
# --------------------------------------------------------------------------- #
def test_command_inventory_snapshot():
    """The registered command surface matches the committed snapshot."""
    current = _command_inventory()
    rendered = "\n".join(current) + "\n"

    if os.environ.get("UPDATE_CLI_SNAPSHOT"):
        SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
        pytest.skip(f"snapshot regenerated: {SNAPSHOT_PATH}")

    assert SNAPSHOT_PATH.exists(), (
        f"missing snapshot {SNAPSHOT_PATH}; regenerate with "
        "`UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py`"
    )
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()
    if current != expected:
        added = sorted(set(current) - set(expected))
        removed = sorted(set(expected) - set(current))
        detail = ""
        if added:
            detail += "\n  added:\n" + "\n".join(f"    + {p}" for p in added)
        if removed:
            detail += "\n  removed:\n" + "\n".join(f"    - {p}" for p in removed)
        pytest.fail(
            "CLI command inventory changed vs. the committed snapshot."
            f"{detail}\n\n"
            "If intentional, regenerate with "
            "`UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py` "
            "and review the diff."
        )
