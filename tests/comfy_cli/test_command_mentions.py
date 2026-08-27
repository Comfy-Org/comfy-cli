"""Guardrail: every ``comfy …`` we print must be a command that exists.

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
from comfy_cli.command_mentions import Mention, check_mention, check_text, extract_mentions, global_options
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


@pytest.fixture(scope="module")
def globals_() -> dict:
    return global_options(build_help_json(app))


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


def test_scan_reaches_the_readmes_fenced_examples():
    """The README's copy-pasteable examples are bare lines in ```bash fences.

    They carry no backticks of their own, so they are invisible to the
    backtick/quoted patterns — if fenced extraction regresses, the guardrail
    stops covering the most-copied surface in the repo while still passing.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    found = {m.text for m in extract_mentions(readme, path="README.md")}
    assert "comfy cloud login" in found
    assert "comfy generate list" in found


def test_no_mentions_of_nonexistent_commands(tree, globals_):
    violations = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        violations += check_text(text, path=str(path.relative_to(REPO_ROOT)), tree=tree, globals_=globals_)
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


def _check(text: str, tree: dict, globals_: dict | None = None):
    return check_mention(Mention(text=text, path="<test>", line=1), tree, globals_=globals_)


@pytest.mark.parametrize(
    "text",
    [
        "comfy cloud login",
        "comfy install --nvidia",
        "comfy run --prompt 'a red fox in snow'",
        "comfy model download --url https://example.com/x.safetensors",
        "comfy --json env",  # global option skipped, then a real command
        "comfy --where cloud jobs ls",  # value-taking global; 'cloud' is its value, not a subcommand
        "comfy --where=cloud jobs ls",  # inline value form
        "comfy node show installed",  # positional argument to a leaf command
        "comfy manager disable-gui",
        "comfy",
        "comfy auth",  # a bare group mention is fine
        "comfy --not-a-real-flag auth login",  # unknown option: stop clean rather than guess
    ],
)
def test_valid_mentions_pass(text, tree, globals_):
    assert _check(text, tree, globals_) is None


@pytest.mark.parametrize(
    ("text", "unknown"),
    [
        ("comfy auth login", "login"),  # the original regression itself
        ("comfy cloud singin", "singin"),
        ("comfy nope", "nope"),
        ("comfy model downlaod --url x", "downlaod"),
        # A leading global option must not end the scan before the subcommand.
        ("comfy --json auth login", "login"),
        ("comfy --where cloud cloud singin", "singin"),
    ],
)
def test_invalid_mentions_are_flagged(text, unknown, tree, globals_):
    violation = _check(text, tree, globals_)
    assert violation is not None
    assert violation.unknown_token == unknown


def test_global_options_are_derived_from_the_live_root_callback(globals_):
    """Hardcoding the set would silently rot as root options change."""
    assert globals_["--json"] is False  # a flag: consumes no value
    assert globals_["--where"] is True  # takes a value
    assert "--help-json" in globals_


def test_without_globals_the_resolver_still_stops_at_the_first_option(tree):
    """Default (no ``globals_``) keeps the old conservative behaviour."""
    assert _check("comfy --json auth login", tree) is None


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


def test_extract_reads_markdown_fenced_code_blocks():
    """The README's primary examples are bare shell lines inside ```bash fences."""
    text = "Sign in:\n\n```bash\ncomfy cloud login                   # opens your browser\n$ comfy jobs ls --where cloud\n```\n"
    found = {m.text for m in extract_mentions(text, path="<test>")}
    assert "comfy cloud login" in found
    assert "comfy jobs ls --where cloud" in found


def test_fenced_mentions_report_their_own_line():
    text = "intro\n\n```bash\ncomfy auth login\n```\n"
    (mention,) = extract_mentions(text, path="<test>")
    assert mention.line == 4
    assert text.splitlines()[mention.line - 1] == "comfy auth login"


def test_run_prefixed_mention_at_line_start_reports_its_own_line():
    """``_RUN_PREFIX``'s boundary class eats the preceding newline; the line must not shift."""
    text = "first line\nsecond line\nrun: comfy cloud login\n"
    (mention,) = extract_mentions(text, path="<test>")
    assert mention.line == 3
    assert text.splitlines()[mention.line - 1] == "run: comfy cloud login"
