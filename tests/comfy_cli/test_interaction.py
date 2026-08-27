"""The three interaction tenets (``comfy_cli.interaction``).

1. Missing required input, interactive caller → a TUI prompt.
2. Missing required input, agentic caller → a structured error naming *every*
   missing option, plus the command's help — on stderr, never stdout.
3. Everything supplied → run to completion with zero interaction.

Two invariants get most of the attention here because they are the ones that
break silently:

- ``prompt_fn`` call *counts*, not just outcomes. A fail-safe helper produces
  the right answer even when it prompts an agent that cannot answer, so only a
  count pins "never" and "exactly once".
- stdout purity under ``--json``. The refusal path writes on two streams, and
  ``ctx.get_help()`` prints to stdout as a side effect (see
  ``test_typer_get_help_pollutes_stdout_on_its_own``), so the envelope contract
  is one bad line away from breaking.

Callers are injected, never derived from ``os.environ``: every
``detect_caller`` call below passes an explicit ``env`` mapping and ``is_tty``.
The one exception is deliberate — ``TestRealTerminalDetection`` drives the real
no-arguments code path in a subprocess attached to a real PTY, because that is
the only branch a stub cannot honestly stand in for.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import click
import pytest
import typer

from comfy_cli import interaction
from comfy_cli.caller import Caller, detect_caller
from comfy_cli.interaction import confirm, require_option
from comfy_cli.output import Renderer, set_renderer

CLI_ROOT = Path(__file__).resolve().parents[2]
# Rich styles an option name by dimming its first dash, which puts an escape
# sequence between the two dashes of `--base-image`. A plain substring search
# for the flag then only matches when color happens to be off.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


HUMAN = Caller(kind="user", agentic=False, source_env=None)
AGENT = Caller(kind="agent", agentic=True, source_env="AI_AGENT")

# (label, env, is_tty, kind, agentic, source_env) — the five branches of
# `detect_caller`, in the priority order the function evaluates them.
BRANCHES = (
    ("explicit_user_agent", {"COMFY_USER_AGENT": "comfy-mcp"}, True, "comfy-mcp", True, "COMFY_USER_AGENT"),
    ("ai_agent", {"AI_AGENT": "1"}, True, "agent", True, "AI_AGENT"),
    ("claudecode", {"CLAUDECODE": "1"}, True, "claude-code", True, "CLAUDECODE"),
    ("non_tty_pipe", {}, False, "pipe", True, None),
    ("human_at_a_terminal", {}, True, "user", False, None),
)
BRANCH_IDS = [row[0] for row in BRANCHES]


class Spy:
    """A prompt that records how often it was reached.

    Outcome assertions cannot distinguish "never prompted" from "prompted and
    got lucky", so every prompt assertion in this file is a count.
    """

    def __init__(self, answer: object = None) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self.answer


def help_context() -> click.Context:
    """A real Typer command context.

    Deliberately not a stub: Typer's rich markup mode is what makes
    ``get_help()`` print to stdout instead of returning text, and a stub with
    a well-behaved ``get_help`` would hide the exact hazard the stdout-purity
    rule exists to catch.
    """
    app = typer.Typer()

    @app.command()
    def create(
        name: str = typer.Option(None, "--name"),
        base_image: str = typer.Option(None, "--base-image"),
    ) -> None:
        """Create a build."""

    return click.Context(typer.main.get_command(app), info_name="comfy build create")


class TestDetectCallerBranches:
    """Each of the five branches, by explicit ``env`` + ``is_tty``."""

    @pytest.mark.parametrize(
        ("env", "is_tty", "kind", "agentic", "source_env"), [r[1:] for r in BRANCHES], ids=BRANCH_IDS
    )
    def test_branch_resolves_to_its_caller(self, env, is_tty, kind, agentic, source_env):
        detected = detect_caller(env=env, is_tty=is_tty)
        assert (detected.kind, detected.agentic, detected.source_env) == (kind, agentic, source_env)

    @pytest.mark.parametrize(("env", "is_tty", "agentic"), [(r[1], r[2], r[4]) for r in BRANCHES], ids=BRANCH_IDS)
    def test_branch_routes_to_the_matching_tenet(self, env, is_tty, agentic):
        """``.agentic`` is the tenet switch, so pin the branch → tenet mapping.

        Detecting the caller correctly is worthless if the helpers then route
        it to the wrong tenet; this is the assertion ``output/test_caller.py``
        cannot make because it does not know about ``require_option``.
        """
        prompt = Spy(answer="prompted")
        caller = detect_caller(env=env, is_tty=is_tty)

        if agentic:
            with pytest.raises(typer.Exit):
                require_option(
                    "--name",
                    None,
                    prompt_fn=prompt,
                    error_code="build_missing_option",
                    caller=caller,
                    skip_prompt=False,
                )
            assert prompt.calls == 0
        else:
            answer = require_option(
                "--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=caller, skip_prompt=False
            )
            assert (answer, prompt.calls) == ("prompted", 1)


class TestRequireOption:
    def test_agentic_caller_never_prompts_and_names_every_missing_option(self, capsys):
        """Tenet 2. Two missing options at once, because "every" is the point:
        an agent must be able to fix the whole command in one retry rather than
        discovering the next missing flag on the next round-trip."""
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build create", version="test"))
        prompt = Spy(answer="never used")

        with pytest.raises(typer.Exit):
            require_option(
                "--name",
                None,
                prompt_fn=prompt,
                error_code="build_missing_option",
                missing=("--name", "--base-image"),
                caller=AGENT,
                skip_prompt=False,
            )

        assert prompt.calls == 0
        envelope = json.loads(capsys.readouterr().out.strip())
        assert envelope["error"]["details"]["missing"] == ["--name", "--base-image"]
        assert "--base-image" in envelope["error"]["message"]

    def test_the_required_option_is_named_even_when_missing_omits_it(self, capsys):
        """``missing`` is the command's whole picture and ``name`` is this
        call's. A refusal that dropped ``--name`` because the caller forgot to
        list it there would send the agent back with the wrong flags."""
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build create", version="test"))

        with pytest.raises(typer.Exit):
            require_option(
                "--name",
                None,
                prompt_fn=Spy(),
                error_code="build_missing_option",
                missing=("--base-image",),
                caller=AGENT,
                skip_prompt=False,
            )

        envelope = json.loads(capsys.readouterr().out.strip())
        assert envelope["error"]["details"]["missing"] == ["--base-image", "--name"]

    def test_interactive_caller_prompts_exactly_once_and_returns_the_answer(self):
        """Tenet 1."""
        prompt = Spy(answer="from-the-prompt")

        answer = require_option(
            "--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN, skip_prompt=False
        )

        assert answer == "from-the-prompt"
        assert prompt.calls == 1

    def test_a_json_caller_at_a_terminal_is_refused_rather_than_prompted(self, capsys):
        """`detect_caller` never reads the requested output mode, so a human who
        types `comfy --json …` at a terminal is `agentic=False` and would be
        prompted — drawing a TUI on the envelope channel, then blocking forever
        on an answer no `--json` consumer is there to give."""
        set_renderer(Renderer.resolve(json_flag=True, caller=HUMAN, command="build create", version="test"))
        prompt = Spy(answer="should not be reached")

        with pytest.raises(typer.Exit):
            require_option(
                "--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN, skip_prompt=False
            )

        assert prompt.calls == 0
        assert json.loads(capsys.readouterr().out.strip())["error"]["code"] == "build_missing_option"

    @pytest.mark.parametrize("caller", [HUMAN, AGENT], ids=["human", "agent"])
    def test_a_supplied_value_is_returned_without_prompting(self, caller):
        """Tenet 3, for both caller kinds — a supplied value short-circuits
        before the caller is even consulted."""
        prompt = Spy(answer="should not be reached")

        answer = require_option(
            "--name", "given", prompt_fn=prompt, error_code="build_missing_option", caller=caller, skip_prompt=False
        )

        assert answer == "given"
        assert prompt.calls == 0

    @pytest.mark.parametrize("value", ["", 0, False, []], ids=["empty_str", "zero", "false", "empty_list"])
    def test_a_falsy_value_still_counts_as_supplied(self, value):
        """ "Missing" is ``None``, not falsiness. ``--replicas 0`` and
        ``--name ""`` are answers; re-prompting for them would ignore the user."""
        prompt = Spy(answer="wrong")

        answer = require_option(
            "--name", value, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN, skip_prompt=False
        )

        assert answer == value
        assert prompt.calls == 0

    def test_skip_prompt_refuses_instead_of_prompting_a_human(self):
        """``--skip-prompt`` forces tenet-3 behaviour even on a TTY, and with
        no value to fall back on that means the tenet-2 refusal."""
        prompt = Spy(answer="should not be reached")

        with pytest.raises(typer.Exit):
            require_option(
                "--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN, skip_prompt=True
            )

        assert prompt.calls == 0

    def test_the_global_skip_prompt_flag_is_honoured_when_not_injected(self, monkeypatch):
        """With ``skip_prompt=None`` the helper reads the global flag the root
        callback installs, so ``comfy --skip-prompt build ...`` works without
        every command threading the flag through by hand."""
        from comfy_cli.workspace_manager import WorkspaceManager

        monkeypatch.setattr(WorkspaceManager(), "skip_prompting", True)
        prompt = Spy(answer="should not be reached")

        with pytest.raises(typer.Exit):
            require_option("--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN)

        assert prompt.calls == 0

    def test_an_aborted_prompt_refuses_instead_of_asking_again(self):
        """questionary answers ``None`` when the user hits Ctrl-C / EOF. That
        is still "no value", so it must fall through to the refusal — and must
        not turn into a re-ask loop."""
        prompt = Spy(answer=None)

        with pytest.raises(typer.Exit):
            require_option(
                "--name", None, prompt_fn=prompt, error_code="build_missing_option", caller=HUMAN, skip_prompt=False
            )

        assert prompt.calls == 1


class TestConfirm:
    @pytest.fixture(autouse=True)
    def _forbid_real_prompts(self, monkeypatch):
        """No test in this class may reach questionary — there is no TTY, so a
        real prompt would hang the suite rather than fail it."""
        self.asked: list[str] = []

        def spy(question: str) -> bool | None:
            self.asked.append(question)
            return self.answer

        self.answer: bool | None = True
        monkeypatch.setattr(interaction, "_ask_confirm", spy)

    def test_yes_returns_true_without_prompting(self):
        assert confirm("Delete it?", yes=True, error_code="build_confirm_required", caller=HUMAN, skip_prompt=False)
        assert self.asked == []

    def test_skip_prompt_returns_true_without_prompting(self):
        """The second escape hatch. ``True`` rather than ``False`` because both
        hatches mean "run to completion" — declining would abort the command,
        which is the opposite of tenet 3."""
        assert confirm("Delete it?", error_code="build_confirm_required", caller=HUMAN, skip_prompt=True)
        assert self.asked == []

    def test_an_interactive_caller_is_asked(self):
        assert confirm("Delete it?", error_code="build_confirm_required", caller=HUMAN, skip_prompt=False)
        assert self.asked == ["Delete it?"]

    def test_a_declined_prompt_is_false(self):
        self.answer = False
        assert confirm("Delete it?", error_code="build_confirm_required", caller=HUMAN, skip_prompt=False) is False

    def test_an_aborted_prompt_is_false(self):
        """``None`` is questionary's Ctrl-C / EOF answer. The safe reading of
        "no answer" for a confirmation is "not confirmed"."""
        self.answer = None
        assert confirm("Delete it?", error_code="build_confirm_required", caller=HUMAN, skip_prompt=False) is False

    def test_a_json_caller_at_a_terminal_is_refused_rather_than_prompted(self, capsys):
        """The confirm half of the same rule: `--json` is a promise that stdout
        carries one envelope, so a destructive confirmation is refused there
        instead of opening questionary on that stream."""
        set_renderer(Renderer.resolve(json_flag=True, caller=HUMAN, command="build delete", version="test"))

        with pytest.raises(typer.Exit):
            confirm("Delete it?", error_code="build_confirm_required", caller=HUMAN, skip_prompt=False)

        assert self.asked == []
        assert json.loads(capsys.readouterr().out.strip())["error"]["code"] == "build_confirm_required"

    def test_extra_details_reach_the_refusal_payload(self, capsys):
        """A refusal has to name the subject it refused to act on; without this
        an agent would have to parse the id back out of the question text."""
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build delete", version="test"))

        with pytest.raises(typer.Exit):
            confirm(
                "Delete it?",
                error_code="build_confirm_required",
                details={"distributionId": "build-1"},
                caller=AGENT,
                skip_prompt=False,
            )

        details = json.loads(capsys.readouterr().out.strip())["error"]["details"]
        assert details["distributionId"] == "build-1"
        assert details["missing"] == ["--yes"]

    def test_an_agentic_caller_without_yes_is_refused(self, capsys):
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build delete", version="test"))

        with pytest.raises(typer.Exit):
            confirm("Delete it?", error_code="build_confirm_required", caller=AGENT, skip_prompt=False)

        assert self.asked == []
        envelope = json.loads(capsys.readouterr().out.strip())
        assert envelope["error"]["code"] == "build_confirm_required"
        assert envelope["error"]["details"]["missing"] == ["--yes"]

    def test_skip_prompt_is_not_consent_for_an_agentic_caller(self, capsys):
        """The root callback turns ``--skip-prompt`` on for every agentic caller
        (``cmdline.py``), so honouring it here would auto-accept every
        destructive confirmation an agent ever reaches and make each command's
        ``*_needs_confirm`` code unreachable. Suppressing a prompt is not
        answering it."""
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build update", version="test"))

        with pytest.raises(typer.Exit):
            confirm("Rewrite it?", error_code="build_confirm_required", caller=AGENT, skip_prompt=True)

        assert self.asked == []
        assert json.loads(capsys.readouterr().out.strip())["error"]["code"] == "build_confirm_required"


class TestOutputChannels:
    """Under ``--json``, stdout stays exactly one ``envelope/1`` line — on the
    refusal path too, which is the one that also has help to print."""

    def test_typer_get_help_pollutes_stdout_on_its_own(self, capsys):
        """Why ``_write_command_help`` captures rather than printing the return
        value. Under Typer's rich markup mode ``ctx.get_help()`` returns ``""``
        and dumps the formatted help straight to ``sys.stdout``, so the obvious
        ``renderer.print(ctx.get_help())`` would print nothing *and* corrupt
        the envelope. If this test ever fails, Typer changed and
        ``_write_command_help`` should be revisited — not deleted."""
        returned = help_context().get_help()

        captured = capsys.readouterr()
        assert returned == ""
        assert "Usage:" in captured.out

    def test_refusal_puts_one_envelope_on_stdout_and_the_help_on_stderr(self, capsys):
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build create", version="test"))

        with pytest.raises(typer.Exit):
            require_option(
                "--name",
                None,
                prompt_fn=Spy(),
                error_code="build_missing_option",
                missing=("--name", "--base-image"),
                caller=AGENT,
                skip_prompt=False,
                ctx=help_context(),
            )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) == 1, f"stdout must be exactly one envelope line, got {lines!r}"
        envelope = json.loads(lines[0])
        assert envelope["schema"] == "envelope/1"
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "build_missing_option"
        assert envelope["error"]["details"]["missing"] == ["--name", "--base-image"]
        # `comfy build create` pins identity: this command's help, not the root
        # app's, which is what `cmdline.py`'s `ctx.find_root().get_help()` gives.
        help_text = _plain(captured.err)
        assert "Usage:" in help_text
        assert "comfy build create" in help_text
        assert "--base-image" in help_text
        assert "Usage:" not in captured.out

    def test_pretty_mode_also_keeps_the_help_off_stdout(self, capsys):
        """The stderr rule is unconditional, not a JSON-mode special case: a
        human's ``comfy build create > out.txt`` should not capture help text,
        and click writes usage-on-error to stderr for the same reason."""
        set_renderer(Renderer.resolve(no_json_flag=True, caller=HUMAN, command="build create", version="test"))

        with pytest.raises(typer.Exit):
            require_option(
                "--name",
                None,
                prompt_fn=Spy(),
                error_code="build_missing_option",
                caller=HUMAN,
                skip_prompt=True,
                ctx=help_context(),
            )

        captured = capsys.readouterr()
        assert "Usage:" in captured.err
        assert "Usage:" not in captured.out

    def test_a_context_whose_help_explodes_still_emits_the_envelope(self, capsys):
        """The help is an adjunct; the envelope is the contract. A command that
        cannot render its own help must not cost an agent its structured
        error."""
        set_renderer(Renderer.resolve(json_flag=True, caller=AGENT, command="build create", version="test"))

        class Exploding:
            def get_help(self) -> str:
                raise RuntimeError("help rendering blew up")

        with pytest.raises(typer.Exit):
            require_option(
                "--name",
                None,
                prompt_fn=Spy(),
                error_code="build_missing_option",
                caller=AGENT,
                skip_prompt=False,
                ctx=Exploding(),
            )

        envelope = json.loads(capsys.readouterr().out.strip())
        assert envelope["error"]["code"] == "build_missing_option"


PTY_PROBE = textwrap.dedent(
    """
    import json, os, sys
    from comfy_cli.caller import detect_caller

    caller = detect_caller()
    sys.stderr.write(
        json.dumps(
            {
                "kind": caller.kind,
                "agentic": caller.agentic,
                "isatty": sys.stdout.isatty(),
                "markers": {
                    k: os.environ.get(k)
                    for k in ("COMFY_USER_AGENT", "AI_AGENT", "CLAUDECODE", "CI")
                },
            }
        )
        + "\\n"
    )
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only")
class TestRealTerminalDetection:
    def test_a_real_pty_with_no_agent_markers_is_a_human(self):
        """The one test that exercises the real TTY probe.

        Everything else here injects a ``Caller``, which proves the routing but
        not the detection. This attaches a genuine PTY to a child's stdout,
        scrubs every agent marker from its environment, and calls
        ``detect_caller()`` with no arguments — the exact call a real
        invocation makes. The child reports the markers it saw so a ``user``
        verdict cannot be explained by anything but the terminal.

        The result travels on stderr (a pipe) rather than the PTY so the parent
        never has to drain a pty master, which would deadlock or raise EIO.
        """
        import pty

        master, slave = pty.openpty()
        env = {k: v for k, v in os.environ.items() if k not in {"COMFY_USER_AGENT", "AI_AGENT", "CLAUDECODE", "CI"}}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", PTY_PROBE],
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=subprocess.PIPE,
                env=env,
                cwd=CLI_ROOT,
                timeout=120,
                check=False,
            )
        finally:
            os.close(slave)
            os.close(master)

        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        report = json.loads(proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1])
        assert report["markers"] == {"COMFY_USER_AGENT": None, "AI_AGENT": None, "CLAUDECODE": None, "CI": None}
        assert report["isatty"] is True
        assert report["kind"] == "user"
        assert report["agentic"] is False

    def test_the_same_probe_without_a_pty_is_a_pipe(self):
        """The control. Same code, same scrubbed environment, stdout on a pipe
        instead of a terminal — if this also said ``user``, the test above
        would be proving nothing."""
        env = {k: v for k, v in os.environ.items() if k not in {"COMFY_USER_AGENT", "AI_AGENT", "CLAUDECODE", "CI"}}
        proc = subprocess.run(
            [sys.executable, "-c", PTY_PROBE],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=env,
            cwd=CLI_ROOT,
            timeout=120,
            check=False,
        )

        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        report = json.loads(proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1])
        assert report["isatty"] is False
        assert report["kind"] == "pipe"
        assert report["agentic"] is True
