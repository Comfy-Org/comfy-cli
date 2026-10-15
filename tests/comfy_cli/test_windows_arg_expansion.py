"""An agent's arguments reach the CLI as written: no Windows expansion."""

from __future__ import annotations

import pytest

from comfy_cli import cancellation, cmdline
from comfy_cli.caller import Caller


class _Tty:
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("caller", "stdin_tty", "expand"),
    [
        (Caller(kind="user", agentic=False, source_env=None), True, True),
        (Caller(kind="pipe", agentic=True, source_env=None), False, False),
        (Caller(kind="pipe", agentic=True, source_env=None), True, True),
        (Caller(kind="inapp-agent", agentic=True, source_env="COMFY_USER_AGENT"), True, False),
        (Caller(kind="user", agentic=True, source_env="COMFY_USER_AGENT"), True, False),
    ],
    ids=["person", "piped-agent", "person-redirecting-output", "marked-agent", "agent-calling-itself-user"],
)
def test_only_a_person_gets_windows_argument_expansion(monkeypatch, caller, stdin_tty, expand):
    monkeypatch.setattr(cmdline, "detect_caller", lambda: caller)
    monkeypatch.setattr("sys.stdin", _Tty() if stdin_tty else None)
    assert cmdline._windows_expand_args() is expand


@pytest.mark.parametrize(("agentic", "expanded"), [(True, False), (False, True)])
def test_the_real_app_takes_the_decision(monkeypatch, agentic, expanded):
    """Through cmdline.main and comfy's own app, on Windows."""
    from typer import _click

    calls = []
    monkeypatch.setattr(cancellation, "install_sigint_handler", lambda: None)
    monkeypatch.setattr(
        cmdline, "detect_caller", lambda: Caller(kind="agent" if agentic else "user", agentic=agentic, source_env=None)
    )
    import os

    import typer.core

    class _WindowsOS:
        name = "nt"

        def __getattr__(self, attr):
            return getattr(os, attr)

    monkeypatch.setattr(typer.core, "os", _WindowsOS())
    monkeypatch.setattr(_click.utils, "_expand_args", lambda args: calls.append(list(args)) or list(args))
    monkeypatch.setattr("sys.argv", ["comfy", "--help"])
    with pytest.raises(SystemExit):
        cmdline.main()
    assert bool(calls) is expanded
