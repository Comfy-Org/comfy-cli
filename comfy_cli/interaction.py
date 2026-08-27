"""The three interaction tenets, in one place.

Every ``comfy build`` command follows the same three rules (builder-cli-design
lines 73-94):

1. Required input missing **in an interactive context** → a good TUI prompt.
2. Required input missing **in a non-interactive context** → fail with a
   structured error naming the missing options, and print the command's help.
3. All required input provided → run to completion with zero interaction, so
   the command stays scriptable.

``caller.detect_caller()`` already draws the line: ``.agentic is False`` is
tenet 1, ``.agentic is True`` is tenet 2. Two escape hatches force tenet-3
behaviour *even on a TTY* — ``-y`` / ``--yes`` accepts every confirmation, and
the global ``--skip-prompt`` flag suppresses every prompt.

**Output-channel rule.** Tenet 2 has two halves that land on *different*
streams. The structured error is the machine contract and goes through
``renderer.error``, which in JSON mode writes exactly one ``envelope/1`` line
to stdout. The command help is a human adjunct and goes to **stderr, always**;
otherwise a ``--json`` consumer would find help text glued to its envelope.

That hazard is not theoretical. Under Typer's rich markup mode (this app's
mode) ``ctx.get_help()`` returns the *empty string* and prints ~1KB of
formatted help straight to ``sys.stdout`` as a side effect, so the obvious
``renderer.print(ctx.get_help())`` would print nothing useful *and* corrupt the
envelope. ``_write_command_help`` captures that side effect and re-emits it on
stderr. ``tests/comfy_cli/test_interaction.py`` pins both halves.

Nothing here reads the environment: the caller, the skip-prompt flag and the
click context are all injectable, so tests pass a synthetic ``Caller`` instead
of mutating ``os.environ``.

The one piece of global state this module *does* read is the installed
``Renderer``, because a prompt and the envelope share stdout and only the
renderer knows which of the two owns it — see ``_may_prompt``. It is not
injectable alongside ``caller`` / ``skip_prompt``, so a test that wants a
prompt has to give the command a pretty renderer (``--no-json``, or
``COMFY_OUTPUT=pretty``) rather than only a non-agentic ``Caller``.

Tested in ``tests/comfy_cli/test_interaction.py``.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn, TypeVar

import typer

from comfy_cli.caller import Caller, detect_caller
from comfy_cli.output import get_renderer

if TYPE_CHECKING:
    import click

__all__ = ["confirm", "require_option"]

T = TypeVar("T")


def _skip_prompt_flag() -> bool:
    """The global ``--skip-prompt``, read live from the house singleton.

    Read at call time, never cached: the root Typer callback installs the flag
    (``cmdline.setup_workspace_manager``) long after this module is imported,
    so a value captured at import would always be the pre-parse ``None``. The
    lazy import mirrors ``ui.py``'s lazy ``questionary`` import — it keeps
    ``WorkspaceManager``'s ``ConfigManager`` construction off the import path
    of a module that most commands only touch on an error branch.
    """
    from comfy_cli.workspace_manager import WorkspaceManager

    return bool(WorkspaceManager().skip_prompting)


def _may_prompt(caller: Caller, skip_prompt: bool) -> bool:
    """Tenet 1 applies only to a human at a terminal who has not opted out.

    A JSON caller is never prompted, whatever ``detect_caller`` decided. This is
    the output-channel rule above applied to the *prompt* half: questionary draws
    on **stdout**, which in JSON mode carries the ``envelope/1`` contract, so a
    prompt there writes escape sequences into the machine channel and then blocks
    forever on an answer no ``--json`` consumer is present to give.

    The check is needed because ``detect_caller`` reads only the environment and
    stdout's tty-ness, never the *requested* output mode, so a human who types
    ``comfy --json ...`` at a terminal is ``kind="user"``, ``agentic=False``.

    ``is_pretty()`` is the exact predicate rather than an approximation of one.
    A non-tty stdout already yields ``kind="pipe", agentic=True``
    (``caller.detect_caller`` branch 4), so reaching this term at all implies
    stdout *is* a tty; and with a tty, ``Renderer.resolve`` leaves pretty mode
    only for an explicit ``--json`` / ``--json-stream`` / ``COMFY_OUTPUT``. So
    "not pretty" here means precisely "this caller asked for machine output",
    never merely "stdout happens to be redirected".
    """
    return not caller.agentic and not skip_prompt and get_renderer().is_pretty()


def _write_command_help(ctx: click.Context | None) -> None:
    """Render *ctx*'s help onto stderr, never stdout.

    The capture-then-re-emit dance is the whole point — as the module docstring
    records, under Typer's rich markup mode ``get_help()`` prints to
    ``sys.stdout`` and returns ``""``. ``captured or returned`` covers both
    that path and plain click's (which returns the text and prints nothing),
    so this keeps working if the app ever drops rich formatting.

    The help is a human adjunct to the structured error, never a precondition
    for it, so a context whose help fails to render still gets its envelope —
    the same best-effort reasoning as ``_cloud_errors._read_error_body``.
    """
    if ctx is None:
        return
    stream = getattr(sys, "stderr", None)
    if stream is None:
        # `print(file=None)` falls back to *stdout*, which would glue help
        # text onto the JSON envelope — exactly what this function exists to
        # prevent. Under `pythonw` / a detached parent `sys.stderr` really is
        # None, so this is a live branch, not a hypothetical. No stderr, no
        # help; the envelope still goes out.
        return
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            returned = ctx.get_help()
        text = captured.getvalue() or returned
        if text.strip():
            print(text.rstrip("\n"), file=stream, flush=True)
    except Exception:
        return


def _refuse(
    *,
    error_code: str,
    message: str,
    hint: str,
    details: Mapping[str, object],
    ctx: click.Context | None,
) -> NoReturn:
    """Tenet 2, both halves, each on the stream it belongs on.

    The single funnel for every refusal in this module, for the reason
    ``command/_cloud_errors.py`` exists: two call sites emitting their own
    error would eventually drift on the stdout-purity invariant, and that
    invariant is the entire contract here.

    Help first, envelope last — in NDJSON mode the envelope must be the final
    line, and ``renderer.error`` is what writes it.
    """
    _write_command_help(ctx)
    renderer = get_renderer()
    renderer.error(code=error_code, message=message, hint=hint, details=details)
    raise typer.Exit(code=renderer.exit_code or 1)


def _option_names(missing: Sequence[str], name: str) -> list[str]:
    """*missing* in call order, with *name* appended only if it is not there.

    A command that collects its missing options into one list will usually
    include the option it is currently requiring, and the error must not name
    it twice; ``dict.fromkeys`` de-duplicates while preserving call order, so
    the payload reads in the order the command declares its options rather
    than alphabetically.
    """
    return list(dict.fromkeys([*missing, name]))


def require_option(
    name: str,
    value: T | None,
    *,
    prompt_fn: Callable[[], T | None],
    error_code: str,
    missing: Sequence[str] = (),
    caller: Caller | None = None,
    skip_prompt: bool | None = None,
    ctx: click.Context | None = None,
) -> T:
    """Return *value*, or prompt for it, or refuse — the three tenets in order.

    Args:
        name: the option this call requires, spelled the way a user types it
            (``"--name"``), because it lands verbatim in the error payload.
        value: whatever the command parsed. ``None`` means "not supplied";
            every other value — including ``""``, ``0`` and ``False`` — counts
            as given, so a legitimately falsy answer is never re-prompted.
        prompt_fn: the TUI prompt for tenet 1. Called **at most once**, and
            only when a prompt is allowed. ``None`` back (questionary's answer
            when the user aborts with Ctrl-C / EOF) falls through to the
            tenet-2 refusal rather than asking again.
        error_code: the registered ``error.code`` for the refusal. An argument
            rather than a constant so each command registers its own code at
            its own first call site, per the register-with-first-call-site
            rule — this module deliberately hardcodes none.
        missing: every option the command found missing, not just this one, so
            an agent fixes them all in one retry instead of discovering them
            one round-trip at a time. *name* is appended if absent.
        caller: injected for tests; defaults to a live ``detect_caller()``.
        skip_prompt: injected for tests; ``None`` reads the global
            ``--skip-prompt`` flag.
        ctx: the command's click/typer context, whose help accompanies a
            refusal on stderr. ``None`` skips the help half.

    Returns:
        The supplied value, or the prompt's answer.

    Raises:
        typer.Exit: tenet 2 — the value is missing and no prompt is allowed.
    """
    if value is not None:
        return value  # Tenet 3: supplied → run on, zero interaction.

    resolved_caller = caller if caller is not None else detect_caller()
    skip = skip_prompt if skip_prompt is not None else _skip_prompt_flag()

    if _may_prompt(resolved_caller, skip):
        answer = prompt_fn()  # Tenet 1. Exactly once — an abort falls through.
        if answer is not None:
            return answer

    options = _option_names(missing, name)
    joined = ", ".join(options)
    plural = "s" if len(options) > 1 else ""
    _refuse(  # Tenet 2.
        error_code=error_code,
        message=f"Missing required option{plural}: {joined}",
        hint=f"pass {joined}",
        details={"missing": options, "caller": resolved_caller.kind},
        ctx=ctx,
    )


def _ask_confirm(question: str) -> bool | None:
    """The house prompt: lazy-import ``questionary``, ask, hand back its answer.

    ``questionary`` pulls in ``prompt_toolkit`` (~50ms of import time), so it
    is imported here rather than at module scope — the same reason every
    prompting function in ``ui.py`` imports it inline. Returns ``None`` when
    the user aborts.

    A module-level function rather than an inline import so tests can replace
    the prompt without a TTY and count how often it was reached.
    """
    import questionary

    return questionary.confirm(question).ask()


def confirm(
    question: str,
    *,
    yes: bool = False,
    error_code: str,
    details: Mapping[str, object] | None = None,
    caller: Caller | None = None,
    skip_prompt: bool | None = None,
    ctx: click.Context | None = None,
) -> bool:
    """Ask *question*, or take an escape hatch, or refuse.

    Both escape hatches mean *proceed*, because answering ``False`` would abort
    the command instead of completing it, which is the opposite of tenet 3:
    ``--yes`` accepts every confirmation, and the global ``--skip-prompt`` says
    run to completion without asking.

    ``--skip-prompt`` is honoured **only for a non-agentic caller** — design
    line 90 scopes it to forcing tenet-3 behaviour *even on a TTY*. It cannot
    extend to an agent, because the root callback turns it on for every agentic
    caller automatically (``cmdline.py``, "Agentic callers shouldn't get
    interactive prompts"). Reading that derived flag as consent would silently
    accept every destructive confirmation for exactly the callers tenet 2
    exists to protect, and would make each command's ``*_needs_confirm`` code
    unreachable. Suppressing a prompt is not answering it; only ``-y`` answers
    it.

    Note the hatch is keyed on the *caller*, not on ``_may_prompt``: a human who
    passes ``--json`` is unpromptable yet still non-agentic, so an explicit
    ``--skip-prompt`` from them means proceed. That is deliberate — they asked
    for both machine output and no questions — and it is why the check below is
    ``not resolved_caller.agentic`` rather than a second ``_may_prompt`` call.

    A caller who declines at the prompt gets ``False``, and so does one who
    aborts it — questionary answers ``None`` on Ctrl-C / EOF, and the safe
    reading of "no answer" for a confirmation is "not confirmed".

    Args:
        question: the confirmation text, echoed into the refusal payload so an
            agent can see what it was being asked to accept.
        yes: the ``-y`` / ``--yes`` escape hatch.
        error_code: the registered ``error.code`` for the refusal; see
            ``require_option``.
        details: extra payload keys merged into the refusal, for the identifier
            the command was about to act on. Without it a refusal names only the
            question, forcing an agent to parse the subject back out of prose.
        caller: injected for tests; defaults to a live ``detect_caller()``.
        skip_prompt: injected for tests; ``None`` reads the global
            ``--skip-prompt`` flag.
        ctx: the command's click/typer context, whose help accompanies a
            refusal on stderr.

    Returns:
        Whether to proceed.

    Raises:
        typer.Exit: tenet 2 — non-interactive with no ``--yes``.
    """
    if yes:
        return True  # Tenet 3.

    resolved_caller = caller if caller is not None else detect_caller()
    skip = skip_prompt if skip_prompt is not None else _skip_prompt_flag()

    if _may_prompt(resolved_caller, skip):
        return _ask_confirm(question) is True  # Tenet 1; `is True` maps an abort to False.

    if skip and not resolved_caller.agentic:
        return True  # Tenet 3, the other escape hatch: a TTY that opted out of prompts.

    _refuse(  # Tenet 2.
        error_code=error_code,
        message=f"Confirmation required, but nothing can answer it: {question}",
        hint="pass --yes to confirm without prompting",
        details={
            # Call-site keys first: the three below are the contract every
            # consumer reads, so a caller cannot displace them with its payload.
            **(details or {}),
            "missing": ["--yes"],
            "question": question,
            "caller": resolved_caller.kind,
        },
        ctx=ctx,
    )
