"""Guardrail: every ``comfy …`` we print must be a command that exists (BE-2996).

The audit found eight help/hint strings pointing users at ``comfy auth login``,
which has never been a command — the real one is ``comfy cloud login``. Nothing
caught it because help text is just string literals. This test walks the shipped
source, the README and the bundled agent skills, and resolves every mention
against the live Typer tree.

If this fails, the mention is wrong (or the command was renamed) — fix the
string, don't loosen the lint. See :mod:`comfy_cli.command_mentions` for why the
resolver is deliberately conservative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_cli.cmdline import app
from comfy_cli.command_mentions import Mention, check_mention, check_text, extract_mentions
from comfy_cli.help_json import build_help_json

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "comfy_cli"

# ``discovery.py`` holds no prose — it is a ``"comfy <emit name>" -> schema`` map,
# and envelope names are not all subcommands (the root ``--version`` flag emits
# ``command="version"``). Its keys get the stronger structural check below
# instead of the prose scan.
PROSE_SCAN_EXCLUDED = {PACKAGE_ROOT / "discovery.py"}


@pytest.fixture(scope="module")
def tree() -> dict:
    return build_help_json(app)["commands"]


def _scanned_files() -> list[Path]:
    """Python sources plus the user-facing prose that ships with the package."""
    files = [p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts and p not in PROSE_SCAN_EXCLUDED]
    files += sorted(PACKAGE_ROOT.rglob("*.md"))
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    return files


def _all_command_paths() -> set[str]:
    """Every command path in the tree, groups included (``iter_command_paths`` gives leaves only)."""
    paths: set[str] = set()

    def walk(subs: dict, prefix: str) -> None:
        for name, child in subs.items():
            path = f"{prefix} {name}"
            paths.add(path)
            walk(child.get("subcommands") or {}, path)

    walk(build_help_json(app)["commands"], "comfy")
    return paths


def test_scan_covers_the_expected_surface():
    """A silently-empty scan would make this whole guardrail a no-op."""
    files = _scanned_files()
    assert len(files) > 50
    assert any(p.name == "cmdline.py" for p in files)
    assert any(p.name == "SKILL.md" for p in files)


def test_no_mentions_of_nonexistent_commands(tree):
    violations = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        violations += check_text(text, path=str(path.relative_to(REPO_ROOT)), tree=tree)
    assert not violations, "help/hint strings reference commands that do not exist:\n" + "\n".join(
        str(v) for v in violations
    )


# --- the machine-readable catalogs keyed by command path ---


def test_help_examples_keys_are_real_commands():
    """A stale ``HELP_EXAMPLES`` key is silently dropped — the command loses its examples."""
    from comfy_cli.help_json import HELP_EXAMPLES

    stale = sorted(k for k in HELP_EXAMPLES if k not in _all_command_paths())
    assert not stale, f"HELP_EXAMPLES keys that are not commands (their examples never render): {stale}"


def test_command_schema_keys_are_commands_or_emitted_envelopes():
    """``COMMAND_SCHEMAS`` ships verbatim in ``comfy discover``.

    Its keys are ``"comfy <envelope command name>"``. Most are real command
    paths; a few name an envelope emitted by a root flag (``--version``). A key
    that is neither advertises a schema for something no agent can ever invoke.
    """
    import ast

    from comfy_cli.discovery import COMMAND_SCHEMAS, STREAM_EVENT_SCHEMAS

    emitted: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(module):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "emit"):
                continue
            for kw in node.keywords:
                if kw.arg == "command" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    emitted.add(f"comfy {kw.value.value}")

    known = _all_command_paths() | emitted
    for name, catalog in (("COMMAND_SCHEMAS", COMMAND_SCHEMAS), ("STREAM_EVENT_SCHEMAS", STREAM_EVENT_SCHEMAS)):
        stale = sorted(k for k in catalog if k not in known)
        assert not stale, f"{name} keys that are neither a command nor an emitted envelope: {stale}"


# --- resolver unit tests: the lint is only as good as its false-positive rate ---


def _check(text: str, tree: dict):
    return check_mention(Mention(text=text, path="<test>", line=1), tree)


@pytest.mark.parametrize(
    "text",
    [
        "comfy cloud login",
        "comfy install --nvidia",
        "comfy run --prompt 'a red fox in snow'",
        "comfy model download --url https://example.com/x.safetensors",
        "comfy --json env",  # stops at the global option; no false positive
        "comfy node show installed",  # positional argument to a leaf command
        "comfy manager disable-gui",
        "comfy",
        "comfy auth",  # a bare group mention is fine
    ],
)
def test_valid_mentions_pass(text, tree):
    assert _check(text, tree) is None


@pytest.mark.parametrize(
    ("text", "unknown"),
    [
        ("comfy auth login", "login"),  # the BE-2996 regression itself
        ("comfy cloud singin", "singin"),
        ("comfy nope", "nope"),
        ("comfy model downlaod --url x", "downlaod"),
    ],
)
def test_invalid_mentions_are_flagged(text, unknown, tree):
    violation = _check(text, tree)
    assert violation is not None
    assert violation.unknown_token == unknown


def test_extract_finds_backticked_and_run_prefixed_mentions():
    text = "first line\nrun: comfy cloud login\nand `comfy jobs ls` too\n"
    found = {m.text for m in extract_mentions(text, path="<test>")}
    assert "comfy cloud login" in found
    assert "comfy jobs ls" in found


def test_extract_finds_bare_quoted_invocations(tree):
    """The run-cli tour labels invocations as plain string literals, not backticks."""
    text = 'Invocation(argv=[*comfy, "auth", "whoami"], label="comfy auth whoami")\n'
    found = extract_mentions(text, path="<test>")
    assert [m.text for m in found] == ["comfy auth whoami"]
    assert check_mention(found[0], tree).unknown_token == "whoami"


def test_extract_ignores_non_command_comfy_words():
    text = "`comfy-cli` installs things and writes `comfy.yaml`."
    assert extract_mentions(text, path="<test>") == []
