"""An agent's arguments reach the CLI as written: no Windows expansion."""

from __future__ import annotations

import pytest

from comfy_cli import cmdline
from comfy_cli.caller import Caller


@pytest.mark.parametrize(
    ("caller", "expand"),
    [
        (Caller(kind="user", agentic=False, source_env=None), True),
        (Caller(kind="pipe", agentic=True, source_env=None), False),
        (Caller(kind="inapp-agent", agentic=True, source_env="COMFY_USER_AGENT"), False),
    ],
)
def test_only_a_person_gets_windows_argument_expansion(monkeypatch, caller, expand):
    seen = {}

    def fake_app(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(cmdline, "app", fake_app)
    monkeypatch.setattr(cmdline, "detect_caller", lambda: caller)
    cmdline.main()
    assert seen == {"windows_expand_args": expand}


def test_typer_skips_expansion_when_told(monkeypatch):
    """The flag reaches Typer's parser: off, a glob stays a glob even on Windows."""
    import typer
    from typer import _click

    seen = []
    app = typer.Typer()

    @app.command()
    def cmd(words: list[str]):
        seen.extend(words)

    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr(_click.utils, "_expand_args", lambda args: ["EXPANDED"])
    monkeypatch.setattr("sys.argv", ["comfy", "ag?nt"])
    with pytest.raises(SystemExit):
        app(windows_expand_args=False)
    assert seen == ["ag?nt"]
    seen.clear()
    with pytest.raises(SystemExit):
        app(windows_expand_args=True)
    assert seen == ["EXPANDED"]
