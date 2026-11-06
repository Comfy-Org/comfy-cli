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

Warning objects (``{"code": …, "message": …}`` placed in a ``warnings`` list)
are collected separately from failures. Registered codes are failure
categories (ADR-AUTOMATION-0003), so a code a command only ever emits as a
warning stays out of the registry; the codes registered before that boundary
was enforced are listed in :data:`LEGACY_WARNING_REGISTRATIONS` and stay
(the registry is append-only).

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
#: registered an orphan. ``not_found_code=`` is the same channel in
#: ``comfy_cli.command._cloud_errors.handle_cloud_http_error``: each caller
#: names the 404 code for its own resource (``asset_not_found``, …).
_CODE_KWARGS = frozenset({"code", "error_code", "not_found_code"})


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


def _dict_code(node: ast.Dict) -> str | None:
    """The literal ``"code": "some_code"`` value of a dict literal, if it has one."""
    for key, value in zip(node.keys, node.values):
        if (
            isinstance(key, ast.Constant)
            and key.value == "code"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and error_codes.CODE_PATTERN.match(value.value)
        ):
            return value.value
    return None


def _names_warnings(node: ast.AST) -> bool:
    """An expression naming a warnings list: ``warnings``, ``op.warnings``,
    ``data["warnings"]`` or ``data.setdefault("warnings", [])``."""
    if isinstance(node, ast.Name):
        return "warning" in node.id
    if isinstance(node, ast.Attribute):
        return "warning" in node.attr
    if isinstance(node, ast.Subscript):
        key = node.slice
        return isinstance(key, ast.Constant) and isinstance(key.value, str) and "warning" in key.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("setdefault", "get"):
        first = node.args[0] if node.args else None
        return isinstance(first, ast.Constant) and isinstance(first.value, str) and "warning" in first.value
    return False


#: Nodes a warning dict can sit inside on its way into a warnings list.
_CARRIERS = (ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp, ast.IfExp, ast.Starred)


def _is_warning_dict(node: ast.Dict, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when the dict literal is placed into a warnings list.

    Followed up through list/comprehension wrappers to what holds it: a
    ``"warnings": [...]`` key, an assignment to a warnings target, a
    ``warnings.append/insert/extend(...)`` call, or a ``warnings=`` keyword.
    """
    child: ast.AST = node
    holder = parents.get(child)
    while isinstance(holder, _CARRIERS):
        child, holder = holder, parents.get(holder)
    if isinstance(holder, ast.Dict):
        return any(
            value is child and isinstance(key, ast.Constant) and isinstance(key.value, str) and "warning" in key.value
            for key, value in zip(holder.keys, holder.values)
        )
    if isinstance(holder, ast.Assign):
        return any(_names_warnings(target) for target in holder.targets)
    if isinstance(holder, (ast.AugAssign, ast.AnnAssign)):
        return _names_warnings(holder.target)
    if isinstance(holder, ast.keyword):
        return holder.arg is not None and "warning" in holder.arg
    if isinstance(holder, ast.Call) and isinstance(holder.func, ast.Attribute):
        return holder.func.attr in ("append", "insert", "extend") and _names_warnings(holder.func.value)
    return False


def _extract_dict_code_values(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Find ``"code": "some_code"`` in dict literals, as ``(failures, warnings)``.

    The watcher and state-file paths build error dicts inline (e.g.
    ``{"code": "watcher_crashed", ...}``), not via ``renderer.error()``.
    This picks up those codes so the registry stays honest. A dict placed
    into a warnings list is a warning object, not a failure, and is
    returned separately.
    """
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    failures: list[str] = []
    warnings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        code = _dict_code(node)
        if code is None:
            continue
        (warnings if _is_warning_dict(node, parents) else failures).append(code)
    return failures, warnings


def _collect_codes() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Walk every .py under comfy_cli and collect distinct codes, as ``(raised, warned)``."""
    raised: dict[str, list[Path]] = {}
    warned: dict[str, list[Path]] = {}
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
        # Also pick up ``"code": "..."`` in dict literals: inline error dicts
        # are failures; dicts placed into a warnings list are warnings.
        failures, warnings = _extract_dict_code_values(tree)
        for code in failures:
            raised.setdefault(code, []).append(path)
        for code in warnings:
            warned.setdefault(code, []).append(path)
    return raised, warned


#: Warning-only codes registered before warnings were kept out of the error
#: registry. They stay registered because registered codes are append-only;
#: the ``workflow save``/``delete`` sidebar warnings joined because they were
#: registered just before the rule landed. No other code joins this list. A
#: new warning keeps its code in ``warnings[]`` and is not registered.
LEGACY_WARNING_REGISTRATIONS = frozenset(
    {
        "api_format_saved_as_record",
        "description_ignored",
        "dynamic_group_unsupported",
        "installed_versions_unavailable",
        "pack_read_error",
        "partial_execution",
        "partial_execution_targets_ignored",
        "primitive_feed_unrepresentable",
        "registry_invalid_node_id",
        "registry_no_dependency_metadata",
        "registry_partial_dependency_metadata",
        "select_no_match",
        "sidebar_file_kept",
        "ui_only_link_dropped",
        "ui_only_node_skipped",
        "workflow_content_not_json",
        "workflow_record_unknown",
        "workspace_unknown",
    }
)


@pytest.fixture(scope="module")
def collected_codes() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    return _collect_codes()


@pytest.fixture(scope="module")
def raised_codes(collected_codes) -> dict[str, list[Path]]:
    """Codes emitted as failures (error envelopes, inline error dicts, error classes)."""
    return collected_codes[0]


@pytest.fixture(scope="module")
def warned_codes(collected_codes) -> dict[str, list[Path]]:
    """Codes emitted only inside a ``warnings`` list of an otherwise successful result."""
    return collected_codes[1]


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


def test_warning_only_codes_stay_out_of_the_registry(raised_codes, warned_codes):
    """Registered codes are failure categories (ADR-AUTOMATION-0003), and
    ``comfy discover`` publishes the registry under ``data.error_codes``. A code
    a command only ever emits inside ``warnings[]`` is not a failure, so
    registering it would advertise an error branch nothing can take.

    If this fails: drop the registration and keep the code in ``warnings[]``,
    or raise it as an error somewhere it really is one.
    """
    # Guards the check below against a scanner that silently finds no warnings.
    assert {"workspace_not_found", "desktop_config_unreadable"} <= set(warned_codes)

    warning_only = {code for code in warned_codes if code not in raised_codes}
    registered = sorted(
        code for code in warning_only if error_codes.is_registered(code) and code not in LEGACY_WARNING_REGISTRATIONS
    )
    assert not registered, (
        f"Warning-only codes registered as error codes: {registered}\n"
        "Remove them from comfy_cli/error_codes.REGISTRY; warnings keep their code in warnings[]."
    )

    # The legacy list only shrinks: an entry that is no longer a registered
    # warning-only code is deleted from it.
    stale = sorted(
        code for code in LEGACY_WARNING_REGISTRATIONS if code not in warning_only or not error_codes.is_registered(code)
    )
    assert not stale, f"no longer registered warning-only codes, delete from LEGACY_WARNING_REGISTRATIONS: {stale}"


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


def test_every_registered_code_is_raised(raised_codes, warned_codes):
    """If this fails: a code in the registry is no longer raised anywhere.

    A legacy warning registration (:data:`LEGACY_WARNING_REGISTRATIONS`)
    counts as used while its warning is still emitted.

    Fix: delete the dead entry from the registry, or wire the code up.
    """
    dead = [
        c
        for c in error_codes.all_codes()
        if c not in raised_codes and not (c in LEGACY_WARNING_REGISTRATIONS and c in warned_codes)
    ]
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
        deploy_workflow_not_api_format deploy_status_terminal deploy_workflow_too_large deploy_watch_lost
        deploy_update_failed deploy_update_replaced deploy_updates_unavailable deploy_name_not_found deploy_invalid_name deploy_name_taken
        deploy_names_unavailable deploy_editor_unchanged
        deploy_build_not_found deploy_ambiguous_build""".split()
    )

    # When
    actual = {code for code in error_codes.all_codes() if code.startswith("deploy_")}

    # Then
    assert len(actual) == 46
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
