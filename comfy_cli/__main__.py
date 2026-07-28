"""Console-script entrypoint for `comfy` / `comfy-cli` / `comfycli`.

A thin wrapper around `comfy_cli.cmdline.main` whose only job is to make the
process a well-behaved pipe producer. When the consumer of our stdout goes away
early — `comfy run --wait … | head`, `comfy jobs ls | head -5` — the next write
raises `BrokenPipeError`. The right answer is a silent exit 0: the consumer got
exactly what it asked for. Without this wrapper the process instead exits 1 (or,
when the failing write is the interpreter's own shutdown flush, prints an
"Exception ignored" traceback), which reads as a command failure to every
shell/CI/agent downstream.

Output written before the pipe closed is of course truncated — a half-written
`--json` envelope stays half-written. That is the consumer's problem by
construction; the exit code is ours.
"""

import os
import sys

from comfy_cli.cmdline import main as _cli_main


def _exit_quietly_on_broken_pipe() -> None:
    """Absorb a closed stdout and exit 0. Never returns."""
    # Point fd 1 at /dev/null so that nothing between here and process death —
    # an atexit hook, a `finally` that prints, the interpreter's own shutdown
    # flush — can hit EPIPE a second time and turn this clean exit back into a
    # traceback on stderr.
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        # No usable fd 1 (already closed, or stdout was replaced by something
        # without a real file descriptor). Nothing to protect — os._exit below
        # skips Python-level flushing anyway.
        pass
    # os._exit rather than sys.exit: it skips atexit handlers and stdio
    # flushing entirely, so no buffered-but-unwritable stdout can resurrect the
    # error we just absorbed. The cost is that the telemetry atexit flush in
    # comfy_cli.tracking is skipped for this invocation, which is the intended
    # trade — a truncated pipe is not worth risking a second failure or a
    # shutdown hang on the way out.
    os._exit(0)


def _broken_pipe_swallowed_by_click() -> bool:
    """Report whether the `SystemExit` we just caught is click's EPIPE bail-out.

    click's `BaseCommand.main` handles EPIPE itself: it swaps `sys.stdout` and
    `sys.stderr` for `click.utils.PacifyFlushWrapper` and raises
    `SystemExit(1)`. So on the common path a broken pipe never reaches us as a
    `BrokenPipeError` at all, and the exit 1 is otherwise indistinguishable
    from a genuine command failure. That wrapper — which click installs in this
    branch and nowhere else — is the one reliable in-process signal.
    """
    try:
        from click.utils import PacifyFlushWrapper
    except ImportError:  # pragma: no cover — click ships as a typer dependency
        return False
    return isinstance(sys.stdout, PacifyFlushWrapper)


def _flush_stdout() -> None:
    """Flush stdout, letting only `BrokenPipeError` escape."""
    stream = sys.stdout
    if stream is None:  # stdout detached (pythonw and friends)
        return
    try:
        stream.flush()
    except BrokenPipeError:
        raise
    except (OSError, ValueError, AttributeError):
        # Closed, detached, or non-file-like stdout (`comfy … >&-`) — nothing to
        # flush, and not the broken-pipe case we are guarding against. Swallow
        # it so this `finally`-time flush can never mask the real exception on
        # its way out.
        pass


def main() -> None:
    try:
        try:
            _cli_main()
        finally:
            # stdout is block-buffered when it is a pipe, so the write that
            # actually lands on the closed reader is frequently the
            # interpreter's shutdown flush — long after any handler here could
            # run. Flushing inside the guard pulls that failure forward to
            # where we can absorb it.
            _flush_stdout()
    except BrokenPipeError:
        _exit_quietly_on_broken_pipe()
    except SystemExit:
        if _broken_pipe_swallowed_by_click():
            _exit_quietly_on_broken_pipe()
        raise


if __name__ == "__main__":  # pragma: nocover
    main()
