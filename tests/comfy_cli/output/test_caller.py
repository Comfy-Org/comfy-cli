"""Caller detection: env vars + TTY → agentic mode.

Priority: COMFY_USER_AGENT > AI_AGENT > CLAUDECODE > non-TTY > user.
All agentic callers flip the same three defaults (JSON, skip prompts,
no banner). The ``kind`` field is for analytics/logging only.
"""

import sys

import pytest

from comfy_cli.caller import detect_caller


def test_tty_no_env_is_user():
    c = detect_caller(env={}, is_tty=True)
    assert c.kind == "user"
    assert c.agentic is False
    assert c.source_env is None


def test_no_tty_is_pipe():
    """Piped into another process or backgrounded → agentic, kind="pipe"."""
    c = detect_caller(env={}, is_tty=False)
    assert c.agentic is True
    assert c.kind == "pipe"
    assert c.source_env is None


def test_ai_agent_env_var_forces_agentic_even_on_tty():
    c = detect_caller(env={"AI_AGENT": "1"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "agent"
    assert c.source_env == "AI_AGENT"


def test_ai_agent_explicit_off_falls_through():
    """A user who set AI_AGENT=0 doesn't want the agentic path."""
    c = detect_caller(env={"AI_AGENT": "0"}, is_tty=True)
    assert c.agentic is False
    assert c.kind == "user"


def test_claude_code_detected_on_tty():
    """CLAUDECODE=1 is set in every shell Claude Code spawns."""
    c = detect_caller(env={"CLAUDECODE": "1"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "claude-code"
    assert c.source_env == "CLAUDECODE"


def test_claude_code_off_falls_through():
    c = detect_caller(env={"CLAUDECODE": "0"}, is_tty=True)
    assert c.agentic is False
    assert c.kind == "user"


def test_ai_agent_wins_over_claudecode():
    """AI_AGENT is more specific (explicitly set by agent frameworks)."""
    c = detect_caller(env={"AI_AGENT": "1", "CLAUDECODE": "1"}, is_tty=True)
    assert c.kind == "agent"
    assert c.source_env == "AI_AGENT"


def test_user_agent_override_carries_label():
    """COMFY_USER_AGENT lets a wrapper self-attribute with a specific label."""
    c = detect_caller(env={"COMFY_USER_AGENT": "my-bot"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "my-bot"
    assert c.source_env == "COMFY_USER_AGENT"


def test_user_agent_override_wins_over_everything():
    c = detect_caller(env={"COMFY_USER_AGENT": "harness", "AI_AGENT": "1", "CLAUDECODE": "1"}, is_tty=True)
    assert c.kind == "harness"
    assert c.source_env == "COMFY_USER_AGENT"


class TestStdoutProbeIsFailSafe:
    """``detect_caller`` runs during CLI startup — renderer construction and the
    module-scope caller kind in ``comfy_cli.tracking``, which is imported for
    every command. Probing stdout must therefore never raise: in detached /
    ``pythonw`` contexts ``sys.stdout`` is ``None`` or already closed, and an
    exception here would break even ``comfy --help`` with tracking disabled.
    """

    def test_missing_stdout_is_pipe_not_attribute_error(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", None)
        c = detect_caller(env={})
        assert c.kind == "pipe"
        assert c.agentic is True

    def test_closed_stdout_is_pipe_not_value_error(self, monkeypatch, tmp_path):
        handle = open(tmp_path / "out.txt", "w")
        handle.close()
        monkeypatch.setattr(sys, "stdout", handle)
        # Sanity: the raw probe really does raise on this object.
        with pytest.raises(ValueError):
            handle.isatty()
        c = detect_caller(env={})
        assert c.kind == "pipe"

    def test_stdout_without_isatty_is_pipe(self, monkeypatch):
        """A replacement stream that doesn't implement isatty at all."""

        class Bare:
            pass

        monkeypatch.setattr(sys, "stdout", Bare())
        assert detect_caller(env={}).kind == "pipe"

    def test_revoked_fd_stdout_is_pipe_not_os_error(self, monkeypatch):
        """A stream over a closed/revoked file descriptor raises OSError
        (EBADF, or WinError 6 on Windows), not ValueError."""

        class Revoked:
            def isatty(self):
                raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(sys, "stdout", Revoked())
        assert detect_caller(env={}).kind == "pipe"

    def test_non_conforming_stdout_is_pipe_not_type_error(self, monkeypatch):
        """`isatty` is just an attribute on an arbitrary object, so a
        replacement stream can raise anything at all — including from a
        non-callable `isatty`. The handler is broad on purpose."""

        class Weird:
            isatty = "not callable"

        monkeypatch.setattr(sys, "stdout", Weird())
        assert detect_caller(env={}).kind == "pipe"

        class Hostile:
            def isatty(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(sys, "stdout", Hostile())
        assert detect_caller(env={}).kind == "pipe"

    def test_explicit_env_signals_never_probe_stdout(self, monkeypatch):
        """The env-var branches return before stdout is probed at all, so an
        explicitly-attributed caller is answered without touching the stream.

        Asserted by counting probes rather than by outcome: a fail-safe probe
        would make the outcome right even if it were called eagerly, so only
        the count actually pins the short-circuit."""
        probes = []

        class Counting:
            def isatty(self):
                probes.append(1)
                return False

        monkeypatch.setattr(sys, "stdout", Counting())
        assert detect_caller(env={"AI_AGENT": "1"}).kind == "agent"
        assert detect_caller(env={"COMFY_USER_AGENT": "my-bot"}).kind == "my-bot"
        assert detect_caller(env={"CLAUDECODE": "1"}).kind == "claude-code"
        assert probes == []
        # Sanity: the probe IS reached once no env signal matches.
        assert detect_caller(env={}).kind == "pipe"
        assert probes == [1]

    def test_a_raising_attribute_lookup_is_pipe(self, monkeypatch):
        """The attribute LOOKUP can raise too, not just the call: on a proxy or
        lazy wrapper `isatty` may be a property or come from `__getattr__`.
        `getattr(..., None)` only swallows AttributeError, so the lookup has to
        sit inside the try — otherwise a ValueError/OSError there escapes a
        function whose entire contract is "never raises", and because
        `comfy_cli.tracking` evaluates `detect_caller()` at import, that is an
        import-time crash for every command."""

        class RaisingProperty:
            @property
            def isatty(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr(sys, "stdout", RaisingProperty())
        assert detect_caller(env={}).kind == "pipe"

        class RaisingGetattr:
            def __getattr__(self, name):
                raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(sys, "stdout", RaisingGetattr())
        assert detect_caller(env={}).kind == "pipe"
