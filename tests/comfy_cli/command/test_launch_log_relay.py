"""Regression tests for the background child's stdout/stderr relay.

`comfy launch --background` re-execs itself with `COMFY_CLI_BACKGROUND` set; that
child pipes ComfyUI's output through `_relay_child_line`, whose writes land in
`comfyui_<port>.log` (the file `comfy logs` reads back).

The relay used to go through the Rich-backed `rprint`, which parses markup,
word-wraps to an auto-detected 80 columns against the non-tty logfile, and
highlights. So a bracketed run could be silently swallowed, an unbalanced
closing tag raised `rich.errors.MarkupError` inside the redirector thread
(killing it and truncating the logfile from that line onward), and long paths
grew injected newlines. The relay now writes to the stream directly.
"""

import io
import threading
import time

import pytest

from comfy_cli.command import launch
from comfy_cli.output.renderer import OutputMode, Renderer


@pytest.fixture
def relay_output(monkeypatch):
    """Capture what `_relay_child_line` writes, via the real renderer path.

    The relay resolves `get_renderer().pretty_stream`, so overriding that
    exercises production's stream selection rather than a stand-in print.
    """
    buf = io.StringIO()
    renderer = Renderer(mode=OutputMode.PRETTY)
    renderer.pretty_stream = buf
    monkeypatch.setattr(launch, "get_renderer", lambda: renderer)
    return buf


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
    # `escape()` is not a clean inverse of the markup parser: `render` rewrites
    # `\[` to `[` unconditionally, so an escaped-then-rendered round trip drops
    # the backslash here.
    "loading C:\\[TEMP]\\model.safetensors\n",
    # A chunk ending in a lone backslash gains a second one under `escape()`.
    "workspace root C:\\models\\",
    # Rich substitutes emoji for colon syntax; escaping does not touch it.
    "status :x: failed, :100: done\n",
    # ANSI is part of what ComfyUI genuinely wrote, so the capture keeps it.
    "\x1b[32mgreen\x1b[0m 100%\n",
]


@pytest.mark.parametrize("line", HOSTILE_LINES)
def test_relay_round_trips_line_verbatim(line, relay_output):
    """Every byte ComfyUI wrote must reach the log unchanged."""
    launch._relay_child_line(line)

    assert relay_output.getvalue() == line


def test_relay_does_not_wrap_long_lines(relay_output):
    """A long line must not gain injected newlines.

    Production reached Rich via `rich.print(file=...)`, which auto-detects width
    to 80 against the non-tty logfile and word-wraps — so absolute model paths,
    traceback frames and tqdm bars were silently reflowed in `comfyui_<port>.log`.
    """
    line = "loading " + "/very/long/model/path" * 20 + " done\n"
    assert len(line) > 80

    launch._relay_child_line(line)

    assert relay_output.getvalue() == line
    assert relay_output.getvalue().count("\n") == 1


def test_relay_does_not_raise_on_unbalanced_markup(relay_output):
    """An unbalanced tag must not kill the redirector thread.

    This is the failure that truncated crash logs: `MarkupError` escaping the
    relay stops the child echoing ComfyUI's output for the rest of the run.
    """
    launch._relay_child_line("ComfyUI failed at [/red] during startup\n")

    assert relay_output.getvalue() == "ComfyUI failed at [/red] during startup\n"


def test_relay_flushes_each_line(monkeypatch):
    """`launch_and_monitor` tails the logfile live, so writes cannot sit in a buffer."""
    flushes = []

    class RecordingStream(io.StringIO):
        def flush(self):
            flushes.append(self.getvalue())

    renderer = Renderer(mode=OutputMode.PRETTY)
    renderer.pretty_stream = RecordingStream()
    monkeypatch.setattr(launch, "get_renderer", lambda: renderer)

    launch._relay_child_line("starting\n")

    assert flushes == ["starting\n"]


def test_redirectors_route_through_the_relay():
    """Both redirector threads must use the relay, not a bare `print`.

    A future edit reintroducing `print(process.stdout.readline())` would restore
    the truncation bug without failing any behavioral test above, since nothing
    else in the suite runs the redirector wiring.
    """
    import inspect

    source = inspect.getsource(launch.launch_comfyui)

    assert source.count("_pump_child_pipe,") == 2
    assert "_relay_child_line(line)" in inspect.getsource(launch._pump_child_pipe)
    assert "print(process.stdout.readline()" not in source
    assert "print(process.stderr.readline()" not in source


def test_child_pipes_decode_leniently():
    """Non-UTF-8 bytes from a custom node must not raise inside the pump.

    Both `Popen` calls in the background branch open text pipes; without
    `errors="replace"` a single undecodable byte raises `UnicodeDecodeError`
    in the redirector thread and truncates the log.
    """
    import inspect

    source = inspect.getsource(launch.launch_comfyui)

    assert source.count('errors="replace"') == 2


class _ClosedPipe:
    """A pipe whose reads behave like the child's after it exits."""

    def __init__(self):
        self.reads = 0

    def readline(self):
        self.reads += 1
        return ""


class _RaisingPipe:
    """The reboot path swaps the pipe out; the old handle raises on read."""

    def __init__(self):
        self.reads = 0

    def readline(self):
        self.reads += 1
        raise ValueError("readline of closed file")


class _UnwritablePipe:
    """Reads fine; the relay's write is what fails (a torn-down stdout)."""

    def __init__(self):
        self.reads = 0

    def readline(self):
        self.reads += 1
        return "output that cannot be written\n"


def _raise_broken_pipe(*_args, **_kwargs):
    raise BrokenPipeError("stdout is gone")


@pytest.fixture
def run_pump():
    """Run `_pump_child_pipe` in a thread for `seconds`, then stop and join it.

    The pump loops forever by design, so a test that just starts a daemon thread
    leaks one that outlives the test — and once the fixtures unwind it writes to
    the *real* stdout, which is fatal at interpreter shutdown. Always stop it.
    """
    stops = []

    def _run(pick_pipe, seconds=0.2):
        stop = threading.Event()
        thread = threading.Thread(target=launch._pump_child_pipe, args=(pick_pipe, stop), daemon=True)
        stops.append((stop, thread))
        thread.start()
        time.sleep(seconds)
        return thread

    yield _run

    for stop, thread in stops:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), "pump did not stop"


@pytest.mark.parametrize("pipe_factory", [_ClosedPipe, _RaisingPipe, _UnwritablePipe, lambda: None])
def test_pump_does_not_spin_on_a_dead_pipe(pipe_factory, monkeypatch, relay_output, run_pump):
    """EOF, a closed handle, a failing write, and not-yet-started all back off.

    The pump cannot break out — `process` is reassigned on every reboot, so an
    empty read is equally "exited" and "restarting". But an unguarded
    `readline()` on a closed pipe returns instantly and forever, burning a core.
    """
    monkeypatch.setattr(launch, "_REDIRECTOR_IDLE_SLEEP", 0.01)
    monkeypatch.setattr(relay_output, "write", _raise_broken_pipe)
    pipe = pipe_factory()

    thread = run_pump(lambda: pipe)

    assert thread.is_alive(), "pump died instead of backing off"
    # ~20 backoff-paced reads, not the tens of thousands a busy loop manages.
    if pipe is not None:
        assert pipe.reads < 200, f"pump spun: {pipe.reads} reads in 0.2s"


def test_pump_relays_lines_then_survives_eof(relay_output, run_pump, monkeypatch):
    """A pump must hand every line to the relay and stay alive past EOF."""
    monkeypatch.setattr(launch, "_REDIRECTOR_IDLE_SLEEP", 0.01)
    lines = ["first [1/2]\n", "second [/red]\n"]

    class Pipe:
        def readline(self):
            return lines.pop(0) if lines else ""

    pipe = Pipe()
    thread = run_pump(lambda: pipe)

    assert relay_output.getvalue() == "first [1/2]\nsecond [/red]\n"
    assert thread.is_alive()


def test_pump_picks_up_a_rebooted_process(relay_output, run_pump, monkeypatch):
    """`process` is reassigned on reboot; the pump must follow it.

    This is why the pump backs off on EOF rather than breaking: the pipe going
    quiet means "restarting" as often as it means "gone".
    """
    monkeypatch.setattr(launch, "_REDIRECTOR_IDLE_SLEEP", 0.01)

    class Pipe:
        def __init__(self, text):
            self.lines = [text]

        def readline(self):
            return self.lines.pop(0) if self.lines else ""

    pipes = [Pipe("before reboot\n"), None, Pipe("after reboot\n")]
    state = {"i": 0}

    def pick():
        # Walk to the next pipe once the current one is drained, mimicking the
        # exit -> no-process -> new-Popen sequence of the reboot path.
        i = state["i"]
        if i < len(pipes) - 1 and (pipes[i] is None or not pipes[i].lines):
            state["i"] = i + 1
        return pipes[state["i"]]

    run_pump(pick, seconds=0.3)

    assert relay_output.getvalue() == "before reboot\nafter reboot\n"
