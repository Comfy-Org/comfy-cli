"""Regression tests for the background child's stdout/stderr relay.

`comfy launch --background` re-execs itself with `COMFY_CLI_BACKGROUND` set; that
child pipes ComfyUI's output through `_relay_child_line`, whose writes land in
`comfyui_<port>.log` (the file `comfy logs` reads back).

That relay prints via the Rich-backed `rprint`, which parses markup. ComfyUI's
output is dense with square brackets, so before the escape was added a bracketed
run could be silently swallowed and an unbalanced closing tag raised
`rich.errors.MarkupError` inside the redirector thread, killing it and
truncating the logfile from that line onward.
"""

import io

import pytest
from rich.console import Console

from comfy_cli.command import launch

# Lines ComfyUI (or a custom node) realistically emits. Each one is either
# silently mangled or fatal when passed to a markup-parsing sink unescaped.
HOSTILE_LINES = [
    "[INFO] Loading model checkpoint\n",
    "Progress: [####    ] 50%\n",
    "got prompt [1/4]\n",
    'Traceback (most recent call last): File "[x]"\n',
    "[bold red]not actually styling[/bold red]\n",
    "unbalanced closing tag [/red] mid-line\n",
    "[/]\n",
]


@pytest.mark.parametrize("line", HOSTILE_LINES)
def test_relay_round_trips_line_verbatim(line, monkeypatch):
    """Every byte ComfyUI wrote must reach the log unchanged.

    Escaping is invisible in the output: the markup parser consumes the
    backslashes, so the rendered text equals the original line.
    """
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    monkeypatch.setattr(launch, "print", lambda *a, **kw: console.print(*a, **kw), raising=False)

    launch._relay_child_line(line)

    assert buf.getvalue() == line


def test_relay_does_not_raise_on_unbalanced_markup(monkeypatch):
    """An unbalanced tag must not kill the redirector thread.

    This is the failure that truncated crash logs: `MarkupError` escaping the
    relay stops the child echoing ComfyUI's output for the rest of the run.
    """
    console = Console(file=io.StringIO(), force_terminal=False, width=200)
    monkeypatch.setattr(launch, "print", lambda *a, **kw: console.print(*a, **kw), raising=False)

    # Would raise rich.errors.MarkupError without the escape.
    launch._relay_child_line("ComfyUI failed at [/red] during startup\n")


def test_relay_preserves_ansi_sequences(monkeypatch):
    """ANSI is captured verbatim; sanitizing belongs to the render path.

    `_relay_child_line` writes the logfile, so it records what ComfyUI actually
    emitted. `Renderer` strips escapes when `comfy logs` displays the file.
    """
    line = "\x1b[32mgreen\x1b[0m 100%\n"
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    monkeypatch.setattr(launch, "print", lambda *a, **kw: console.print(*a, **kw), raising=False)

    launch._relay_child_line(line)

    assert buf.getvalue() == line


def test_redirectors_route_through_the_escaping_relay():
    """Both redirector threads must use the relay, not a bare `print`.

    A future edit reintroducing `print(process.stdout.readline())` would restore
    the truncation bug without failing any behavioral test above, since the
    redirectors are nested closures that never run in the suite.
    """
    import inspect

    source = inspect.getsource(launch.launch_comfyui)

    assert source.count("_relay_child_line(") == 2
    assert "print(process.stdout.readline()" not in source
    assert "print(process.stderr.readline()" not in source
