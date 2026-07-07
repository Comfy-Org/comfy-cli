"""Pin the error-code registry against actual call sites.

These tests are deliberately blunt: they scan the source tree for
``renderer.error(code="X")`` (and the equivalent self-method form),
extract every literal code string, and cross-check against
:mod:`comfy_cli.error_codes`.

Two directions are enforced:

1. Every code raised in source is in the registry. A typo or a fresh code
   added without registry update fails the test and surfaces the typo.
2. Every code in the registry is raised somewhere. A code that's
   deprecated or removed but left in the registry fails the test, forcing
   the dead entry to be deleted.

The shape of the AST scan is conservative (literal strings only). Dynamic
codes (e.g. constructed from variables) are excluded — there are none in
the tree today.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from comfy_cli import error_codes

SRC_ROOT = Path(__file__).resolve().parents[3] / "comfy_cli"


def _iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        # Don't grade the registry against itself.
        if p.name == "error_codes.py":
            continue
        if "__pycache__" in p.parts:
            continue
        # engine.py uses internal validation result codes ("unknown_class_type",
        # "shape_mismatch", etc.) in return-value dicts, not CLI error codes.
        if p.name == "engine.py" and "cql" in p.parts:
            continue
        yield p


def _extract_codes_from_call(call: ast.Call) -> list[str]:
    """Return any string literal passed as ``code=...`` whose shape matches a code.

    Conservative on what counts as a code (must match the snake_case pattern)
    so we don't accept random ``code=1`` ints (e.g. ``typer.Exit(code=1)``) or
    arbitrary string kwargs.
    """
    out: list[str] = []
    for kw in call.keywords:
        if kw.arg != "code":
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            value = kw.value.value
            if error_codes.CODE_PATTERN.match(value):
                out.append(value)
    # Positional first arg on ``.error("code", "msg")`` — keep the heuristic
    # for the small set of callers that use the positional form.
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "error"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
        and error_codes.CODE_PATTERN.match(call.args[0].value)
    ):
        out.append(call.args[0].value)
    return out


def _extract_class_code_attrs(tree: ast.Module) -> list[str]:
    """Find ``code = "some_code"`` class-attribute assignments.

    OAuth error subclasses (e.g. ``class OAuthTokenError(OAuthError): code =
    "oauth_token_failed"``) define their codes as class attributes, not as
    call-site string literals.  These are passed dynamically to
    ``renderer.error(code=e.code, ...)`` so the call-site extractor can't see
    them.  This function picks them up from the class body.
    """
    codes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "code":
                    if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                        value = item.value.value
                        if error_codes.CODE_PATTERN.match(value):
                            codes.append(value)
    return codes


def _extract_dict_code_values(tree: ast.Module) -> list[str]:
    """Find ``"code": "some_code"`` in dict literals.

    The watcher and state-file paths build error dicts inline (e.g.
    ``{"code": "watcher_crashed", ...}``), not via ``renderer.error()``.
    This picks up those codes so the registry stays honest.
    """
    codes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "code"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and error_codes.CODE_PATTERN.match(value.value)
            ):
                codes.append(value.value)
    return codes


def _collect_raised_codes() -> dict[str, list[Path]]:
    """Walk every .py under comfy_cli and collect distinct error codes raised."""
    raised: dict[str, list[Path]] = {}
    for path in _iter_python_files(SRC_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for code in _extract_codes_from_call(node):
                raised.setdefault(code, []).append(path)
        # Also pick up class-level ``code = "..."`` (e.g. OAuthError subclasses).
        for code in _extract_class_code_attrs(tree):
            raised.setdefault(code, []).append(path)
        # Also pick up ``"code": "..."`` in dict literals (inline error dicts).
        for code in _extract_dict_code_values(tree):
            raised.setdefault(code, []).append(path)
    return raised


@pytest.fixture(scope="module")
def raised_codes() -> dict[str, list[Path]]:
    return _collect_raised_codes()


def test_every_raised_code_is_registered(raised_codes):
    """If this fails: you raised a code that isn't in ``error_codes.REGISTRY``.

    Fix: add the code to the registry first, then re-run.
    """
    unregistered = {
        code: [str(p.relative_to(SRC_ROOT.parent)) for p in paths]
        for code, paths in raised_codes.items()
        if not error_codes.is_registered(code)
    }
    assert not unregistered, (
        f"Unregistered error codes raised in source:\n{unregistered}\nAdd each to comfy_cli/error_codes.REGISTRY."
    )


def test_every_error_code_is_a_navigation_signal():
    """An error message must point toward correctness — every registered code
    carries a navigation `hint` (the valid set, the close match, the next
    command). The only exception is a genuinely terminal, user-initiated state
    where there is no "next step" to navigate to.

    If this fails: add a `hint` to the new code saying what to do next.
    """
    # `cancelled` = the user pressed Ctrl-C — nothing to navigate toward.
    TERMINAL_NO_NAVIGATION = {"cancelled"}
    missing = [
        ec.code
        for ec in error_codes.REGISTRY
        if ec.code not in TERMINAL_NO_NAVIGATION and not (ec.hint and ec.hint.strip())
    ]
    assert not missing, (
        f"error codes with no navigation hint: {missing}\n"
        "Every error is a signal toward correctness — add a `hint` telling the agent what to do next "
        "(the valid options, the close match, or the exact command to run)."
    )


def test_every_registered_code_is_raised(raised_codes):
    """If this fails: a code in the registry is no longer raised anywhere.

    Fix: delete the dead entry from the registry, or wire the code up.
    """
    dead = [c for c in error_codes.all_codes() if c not in raised_codes]
    assert not dead, (
        f"Registered but never raised:\n{dead}\nEither delete these from comfy_cli/error_codes.REGISTRY or use them."
    )


def test_codes_match_pattern():
    """Every registered code is snake_case matching the documented pattern."""
    bad = [ec.code for ec in error_codes.REGISTRY if not error_codes.CODE_PATTERN.match(ec.code)]
    assert not bad, f"Codes that don't match {error_codes.CODE_PATTERN.pattern}: {bad}"


def test_no_duplicate_codes():
    """Each entry in the registry is unique by code."""
    seen: set[str] = set()
    dupes: list[str] = []
    for ec in error_codes.REGISTRY:
        if ec.code in seen:
            dupes.append(ec.code)
        seen.add(ec.code)
    assert not dupes, f"Duplicate codes in registry: {dupes}"


def test_discover_includes_all_registered_codes():
    """The discover envelope must surface every registered code so an agent
    that calls `comfy --json discover` sees the full contract.
    """
    from comfy_cli.discovery import load_error_codes

    discovered = {row["code"] for row in load_error_codes()}
    expected = set(error_codes.all_codes())
    missing = expected - discovered
    assert not missing, f"Codes registered but not emitted by discover: {missing}"
