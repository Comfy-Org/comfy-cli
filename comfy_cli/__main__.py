"""Console-script entrypoint for `comfy` / `comfy-cli` / `comfycli`.

A thin wrapper around `comfy_cli.cmdline.main` whose only job is to make the
process a well-behaved pipe producer. When the consumer of our stdout goes away
early — `comfy run --wait … | head`, `comfy jobs ls | head -5` — the next write
raises `BrokenPipeError`. The right answer is a silent exit 0: the consumer got
exactly what it asked for. Without this wrapper the process instead exits 1 (or,
when the failing write is the interpreter's own shutdown flush, prints an
"Exception ignored" traceback), which reads as a command failure to every
shell/CI/agent downstream.

The exit 0 is only ever granted to a command that *succeeded*. A closed reader
on a command that failed still exits with the failure's code — `comfy --json
<failing-cmd> | head` is a failure that happened to be piped, not a success.
Absorbing the broken pipe there would launder a genuine error into 0, which is
the one outcome worse than the exit 1 this wrapper exists to remove.

Output written before the pipe closed is of course truncated — a half-written
`--json` envelope stays half-written. That is the consumer's problem by
construction; the exit code is ours.
"""

import os
import sys

from comfy_cli.cmdline import main as _cli_main


def _silence_stdout() -> None:
    """Point fd 1 at /dev/null so nothing later can hit EPIPE a second time.

    Between here and process death an atexit hook, a `finally` that prints, or
    the interpreter's own shutdown flush can all write to stdout. Once its
    reader is gone each of those turns into a fresh `BrokenPipeError` — an
    "Exception ignored" traceback on stderr, and (for the shutdown flush) exit
    120 in place of whatever code we meant to report.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:  # pragma: no cover — /dev/null is not openable
        return
    try:
        os.dup2(devnull_fd, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        # No usable fd 1 (already closed, or stdout was replaced by something
        # without a real file descriptor). Nothing to protect.
        pass
    finally:
        # dup2 gives fd 1 its own reference, so this one is ours to release.
        os.close(devnull_fd)


def _exit_quietly_on_broken_pipe() -> None:
    """Absorb a closed stdout and exit 0. Never returns."""
    _silence_stdout()
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


def _flush_stdout(*, unwinding: bool) -> bool:
    """Flush stdout; return False iff the flush hit a broken pipe.

    Reporting the broken pipe by return value instead of raising is what lets
    `main` keep the two cases apart. This flush runs at a point where a real
    exception may already be on its way out, and a `BrokenPipeError` raised
    from here would silently *replace* it — turning a failed command's non-zero
    exit into 0.

    `unwinding` says whether that is the case. While unwinding, every other
    flush failure is swallowed too, for the same reason: a `finally`-time flush
    must never mask the exception it is running underneath. On the clean path
    there is nothing to mask, so a non-broken-pipe `OSError` (a full disk, EIO)
    is allowed to propagate — output was genuinely lost and a real traceback
    beats the interpreter's opaque exit 120. Only the nothing-to-flush shapes
    stay quiet either way.
    """
    stream = sys.stdout
    if stream is None:  # stdout detached (pythonw and friends)
        return True
    try:
        stream.flush()
    except BrokenPipeError:
        return False
    except (ValueError, AttributeError):
        # Closed or non-file-like stdout (`comfy … >&-`) — nothing to flush,
        # and not the broken-pipe case we are guarding against.
        pass
    except OSError:
        if not unwinding:
            raise
    return True


def _is_clean_exit(exc: SystemExit) -> bool:
    """Whether this `SystemExit` reports success (`sys.exit(0)` / `sys.exit()`)."""
    return exc.code is None or exc.code == 0


def main() -> None:
    try:
        _cli_main()
    except BrokenPipeError:
        # click's standalone `main` converts every EPIPE raised inside a command
        # into the pacified `SystemExit(1)` handled just below, so a raw
        # `BrokenPipeError` reaching here is stdout dying outside that guard.
        _exit_quietly_on_broken_pipe()
    except SystemExit as exc:
        if _broken_pipe_swallowed_by_click():
            _exit_quietly_on_broken_pipe()
        # stdout is block-buffered when it is a pipe, so the write that actually
        # lands on the closed reader is frequently the interpreter's shutdown
        # flush — long after any handler here could run. Flushing now pulls that
        # failure forward to where we can still decide what it means.
        clean = _is_clean_exit(exc)
        if _flush_stdout(unwinding=not clean):
            raise
        if clean:
            # The command succeeded and its reader left early: that is the whole
            # story, so exit 0.
            _exit_quietly_on_broken_pipe()
        # The command failed. Its exit code is the truth and outranks the closed
        # reader; only silence fd 1 so the shutdown flush cannot re-raise what
        # we just absorbed (which would replace that code with 120).
        _silence_stdout()
        raise
    except BaseException:
        # A crash is already on its way out. Same rule: flush what we can, let
        # nothing from the flush displace it.
        if not _flush_stdout(unwinding=True):
            _silence_stdout()
        raise
    else:
        # `_cli_main` returned instead of exiting. click always raises in
        # standalone mode, but `cmdline.main` can fall through on rc == 0.
        if not _flush_stdout(unwinding=False):
            _exit_quietly_on_broken_pipe()


if __name__ == "__main__":  # pragma: nocover
    main()
