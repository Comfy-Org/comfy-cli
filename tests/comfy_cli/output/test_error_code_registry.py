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


#: ``code=`` is ``renderer.error``'s own kwarg. ``error_code=`` is how
#: ``comfy_cli.interaction``'s ``require_option`` / ``confirm`` take the code:
#: that helper deliberately hardcodes none, so each command names its own code
#: at its own call site (the register-with-first-call-site rule). A command
#: whose only refusal goes through those helpers would otherwise look like it
#: registered an orphan.
_CODE_KWARGS = frozenset({"code", "error_code"})


def _extract_codes_from_call(call: ast.Call) -> list[str]:
    """Return any string literal passed as a code kwarg whose shape matches a code.

    Conservative on what counts as a code (must match the snake_case pattern)
    so we don't accept random ``code=1`` ints (e.g. ``typer.Exit(code=1)``) or
    arbitrary string kwargs.
    """
    out: list[str] = []
    for kw in call.keywords:
        if kw.arg not in _CODE_KWARGS:
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


def test_every_build_code_has_a_first_call_site(raised_codes):
    """The ``build_*`` family is filled in one code at a time, under this rule:

        Each error code is registered in the same change that introduces its
        first call site. No change pre-registers codes for later ones.

    Pre-registering is not a harmless head start. ``comfy discover`` publishes the
    registry verbatim, so a code with no call site advertises a branch to agents
    that nothing can ever take — a documented promise the CLI cannot keep.

    :func:`test_every_registered_code_is_raised` already rejects an orphan anywhere
    in the registry; this narrows that direction to the build family so the failure
    names the rule that was broken instead of dropping a bare code into a flat list
    shared with every other subsystem.

    If this fails: move the registration into the change that raises the code.
    """
    BUILD_PREFIX = "build_"
    build_codes = [c for c in error_codes.all_codes() if c.startswith(BUILD_PREFIX)]
    # Guards the orphan check below against a silently empty enumeration.
    assert build_codes, f"no {BUILD_PREFIX}* codes found in the registry — this guard would pass vacuously"

    orphans = sorted(c for c in build_codes if c not in raised_codes)
    assert not orphans, (
        f"Registered with no call site under comfy_cli/: {orphans}\n"
        "Each error code is registered in the same change that introduces its first call site. "
        "Move the registration into the change that raises it, or wire up the call site now."
    )


def test_every_deferred_build_code_landed_with_a_call_site(raised_codes):
    """The build design deferred seven codes to the commands that would raise them.

    The two generic guards above cannot see a deferral that was simply forgotten:
    a code that was never added, together with the call site that never landed,
    leaves both of them green. Naming the seven is what turns "we meant to add
    this" into a red test.

    If this fails: the command that owes this code was never finished, or its
    only call site was deleted. Wire it up — do not delete the name from here.
    """
    deferred = {
        "build_spec_exists",
        "build_missing_input",
        "build_update_needs_confirm",
        "build_spec_stale",
        "build_pull_needs_confirm",
        "build_id_unknown",
        "build_release_not_found",
    }

    unregistered = sorted(code for code in deferred if not error_codes.is_registered(code))
    assert not unregistered, f"deferred build codes missing from comfy_cli/error_codes.REGISTRY: {unregistered}"

    unraised = sorted(code for code in deferred if code not in raised_codes)
    assert not unraised, f"deferred build codes registered but raised nowhere under comfy_cli/: {unraised}"


def test_deploy_error_codes_are_the_exact_final_set() -> None:
    # Given
    expected = set(
        """deploy_not_signed_in deploy_not_found deploy_build_not_pushed deploy_no_deployable_release
        deploy_missing_input deploy_compute_unavailable deploy_forbidden deploy_conflict deploy_immutable_compute
        deploy_deleted deploy_payment_required deploy_quota_exceeded deploy_delete_needs_confirm deploy_endpoint_unknown
        deploy_not_ready deploy_workflow_invalid deploy_asset_missing deploy_asset_upload_failed deploy_job_failed
        deploy_job_canceled deploy_rate_limited deploy_ambiguous_deployment deploy_job_submit_unknown deploy_bad_request
        deploy_server_error deploy_idempotency_reuse deploy_workflow_format_ui
        deploy_workflow_asset_outside_root deploy_workflow_asset_marker_reserved
        deploy_insecure_url deploy_unrelated_deployment deploy_workflow_empty
        deploy_workflow_not_api_format""".split()
    )

    # When
    actual = {code for code in error_codes.all_codes() if code.startswith("deploy_")}

    # Then
    assert len(actual) == 33
    assert actual == expected


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
