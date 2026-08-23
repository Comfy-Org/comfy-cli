"""`comfy … | head` must exit 0, not 1.

A CLI whose stdout consumer closes early is not a failing CLI — the consumer got
what it asked for. These cover both halves of the entrypoint guard in
`comfy_cli/__main__.py`: the end-to-end process behaviour (stdout's reader gone)
and the two in-process signals the guard keys off.
"""

import io
import os
import shutil
import subprocess
import sys

import pytest

from comfy_cli import __main__ as entrypoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `--help-json` dumps the whole command surface (>100 KB) and needs no
# workspace, network, or auth — the cheapest command that writes a lot.
BIG_OUTPUT_CMD = [sys.executable, "-m", "comfy_cli", "--help-json"]

# A stand-in for a command that writes a small `--json` envelope and *then*
# fails. The write is small enough to stay in stdout's 8 KiB buffer, so the
# broken pipe surfaces from the entrypoint's own exit-time flush — the one
# window where it could displace the pending `SystemExit` and report success.
# No real subcommand fails offline while writing to stdout, hence the stub.
FAILING_COMMAND_STUB = """
import sys
from comfy_cli import __main__ as entrypoint

def _envelope_then_fail():
    sys.stdout.write('{"schema": "envelope/1", "ok": false}')
    sys.exit(7)

entrypoint._cli_main = _envelope_then_fail
entrypoint.main()
"""


def _run_with_stdout_reader_gone(cmd=None):
    """Run `cmd` with a stdout pipe whose read end is already closed.

    Deterministic by construction: there is no pipe buffer left to absorb the
    output, so the very first write raises `BrokenPipeError`. (Reading a chunk
    first and *then* closing is the more literal `| head` shape, but it races
    the kernel's pipe buffer — see the `head` test below for that variant.)
    """
    # Block-buffered stdout is the whole point of these tests; an inherited
    # PYTHONUNBUFFERED would move the failing write somewhere else entirely.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(cmd or BIG_OUTPUT_CMD, cwd=REPO_ROOT, stdout=write_fd, stderr=subprocess.PIPE, env=env)
    os.close(write_fd)
    os.close(read_fd)
    try:
        assert proc.stderr is not None
        stderr = proc.stderr.read()
        proc.stderr.close()
        returncode = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:  # pragma: no cover — only if the child hangs
            proc.kill()
            proc.wait(timeout=30)
    return returncode, stderr


class TestBrokenPipeExitCode:
    def test_closed_stdout_exits_zero(self):
        returncode, _ = _run_with_stdout_reader_gone()
        assert returncode == 0, f"expected exit 0 for a closed stdout, got {returncode}"

    def test_closed_stdout_is_silent(self):
        _, stderr = _run_with_stdout_reader_gone()
        text = stderr.decode(errors="replace")
        assert "BrokenPipeError" not in text, text
        assert "Traceback" not in text, text
        assert "Exception ignored" not in text, text
        assert "Abort" not in text, text

    @pytest.mark.skipif(
        sys.platform == "win32" or not (shutil.which("bash") and shutil.which("head")),
        reason="needs a POSIX shell with PIPESTATUS and head",
    )
    def test_head_pipeline_exits_zero(self):
        """The literal ticket scenario: `comfy … | head` reads a bit, then leaves."""
        proc = subprocess.run(
            [
                "bash",
                "-c",
                '"$1" -m comfy_cli --help-json | head -c 100 > /dev/null; exit "${PIPESTATUS[0]}"',
                "bash",
                sys.executable,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")

    def test_fully_consumed_output_still_exits_zero(self):
        """Sanity: the guard does not disturb the ordinary success path."""
        proc = subprocess.run(BIG_OUTPUT_CMD, cwd=REPO_ROOT, capture_output=True, timeout=120)
        assert proc.returncode == 0
        assert proc.stdout

    def test_real_failure_still_exits_nonzero(self):
        """The guard must not launder genuine failures into exit 0."""
        proc = subprocess.run(
            [sys.executable, "-m", "comfy_cli", "definitely-not-a-command"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode != 0

    def test_failure_plus_broken_pipe_still_exits_nonzero(self):
        """`comfy --json <failing-cmd> | head`: the failure outranks the closed reader.

        The case `test_real_failure_still_exits_nonzero` cannot reach — it
        drains stdout, so no pipe ever breaks. Here both happen at once, which
        is where an exit-time flush is able to swap a `SystemExit(7)` for a
        `BrokenPipeError` and hand the shell a 0.
        """
        returncode, _ = _run_with_stdout_reader_gone([sys.executable, "-c", FAILING_COMMAND_STUB])
        assert returncode == 7, f"expected the command's exit 7 to survive the broken pipe, got {returncode}"

    def test_failure_plus_broken_pipe_is_silent(self):
        """Preserving the exit code must not cost a shutdown-flush traceback."""
        _, stderr = _run_with_stdout_reader_gone([sys.executable, "-c", FAILING_COMMAND_STUB])
        text = stderr.decode(errors="replace")
        assert "BrokenPipeError" not in text, text
        assert "Exception ignored" not in text, text


class TestClickPacifySignal:
    """click swallows EPIPE itself, so the guard detects it by the stream swap."""

    def test_detects_click_pacified_stdout(self, monkeypatch):
        from click.utils import PacifyFlushWrapper

        monkeypatch.setattr(sys, "stdout", PacifyFlushWrapper(io.StringIO()))
        assert entrypoint._broken_pipe_swallowed_by_click() is True

    def test_ordinary_stdout_is_not_a_broken_pipe(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        assert entrypoint._broken_pipe_swallowed_by_click() is False

    def test_systemexit_is_reraised_untouched_when_pipe_is_healthy(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: sys.exit(3))

        with pytest.raises(SystemExit) as excinfo:
            entrypoint.main()

        assert excinfo.value.code == 3


class BrokenStdout(io.StringIO):
    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


class ExitedZero(BaseException):
    """Stand-in for `_exit_quietly_on_broken_pipe`, which really `os._exit`s.

    A stub that merely *returns* would let `main` run on past the call and
    reach code the real process never does — hiding exactly the mistakes these
    tests are here to catch.
    """


class TestFlushStdout:
    def test_broken_pipe_is_reported_not_raised(self, monkeypatch):
        """Raising here would replace whatever exception is already unwinding."""
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        assert entrypoint._flush_stdout(unwinding=False) is False
        assert entrypoint._flush_stdout(unwinding=True) is False

    def test_closed_stdout_is_tolerated(self, monkeypatch):
        stream = io.StringIO()
        stream.close()
        monkeypatch.setattr(sys, "stdout", stream)
        assert entrypoint._flush_stdout(unwinding=False) is True

    def test_detached_stdout_is_tolerated(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", None)
        assert entrypoint._flush_stdout(unwinding=False) is True

    def test_other_oserror_is_swallowed_while_unwinding(self, monkeypatch):
        """A flush running under a live exception must never mask it."""

        class BadFd(io.StringIO):
            def flush(self):
                raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(sys, "stdout", BadFd())
        assert entrypoint._flush_stdout(unwinding=True) is True

    def test_other_oserror_surfaces_on_the_clean_path(self, monkeypatch):
        """A full disk lost output; with nothing to mask, say so properly."""

        class FullDisk(io.StringIO):
            def flush(self):
                raise OSError(28, "No space left on device")

        monkeypatch.setattr(sys, "stdout", FullDisk())
        with pytest.raises(OSError):
            entrypoint._flush_stdout(unwinding=False)


class TestFailureIsNeverLaundered:
    """A broken pipe from the exit-time flush must not outrank a real failure."""

    @staticmethod
    def _forbid_exit_zero(monkeypatch):
        def _boom():
            raise AssertionError("exit 0 taken for a failing command")

        monkeypatch.setattr(entrypoint, "_exit_quietly_on_broken_pipe", _boom)

    def test_nonzero_systemexit_survives(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: sys.exit(2))
        self._forbid_exit_zero(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            entrypoint.main()

        assert excinfo.value.code == 2

    def test_string_systemexit_survives(self, monkeypatch):
        """`sys.exit("message")` is exit 1, not a success code."""
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: sys.exit("nope"))
        self._forbid_exit_zero(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            entrypoint.main()

        assert excinfo.value.code == "nope"

    def test_crash_survives(self, monkeypatch):
        def _crash():
            raise RuntimeError("real failure")

        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", _crash)
        self._forbid_exit_zero(monkeypatch)

        with pytest.raises(RuntimeError, match="real failure"):
            entrypoint.main()

    @staticmethod
    def _record_exit_zero(monkeypatch):
        def _exit_zero():
            raise ExitedZero

        monkeypatch.setattr(entrypoint, "_exit_quietly_on_broken_pipe", _exit_zero)

    def test_successful_exit_still_becomes_exit_zero(self, monkeypatch):
        """The point of the guard: a succeeded command keeps its exit 0."""
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: sys.exit(0))
        self._record_exit_zero(monkeypatch)

        with pytest.raises(ExitedZero):
            entrypoint.main()

    def test_bare_sys_exit_still_becomes_exit_zero(self, monkeypatch):
        """`sys.exit()` carries `code is None`, which is success."""
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: sys.exit())
        self._record_exit_zero(monkeypatch)

        with pytest.raises(ExitedZero):
            entrypoint.main()

    def test_clean_return_still_becomes_exit_zero(self, monkeypatch):
        """`cmdline.main` falls through without raising when rc == 0."""
        monkeypatch.setattr(sys, "stdout", BrokenStdout())
        monkeypatch.setattr(entrypoint, "_cli_main", lambda: None)
        self._record_exit_zero(monkeypatch)

        with pytest.raises(ExitedZero):
            entrypoint.main()


class TestSilenceStdout:
    def test_closes_its_devnull_descriptor(self, tmp_path, monkeypatch):
        """The /dev/null fd is duplicated onto stdout's fd, not left dangling."""
        with open(tmp_path / "out", "w") as target:
            monkeypatch.setattr(sys, "stdout", target)
            probe = os.open(os.devnull, os.O_WRONLY)
            os.close(probe)

            entrypoint._silence_stdout()

            after = os.open(os.devnull, os.O_WRONLY)
            os.close(after)
        # os.open hands back the lowest free descriptor, so a leak would push
        # this one past `probe`.
        assert after == probe

    def test_writes_go_to_devnull_afterwards(self, tmp_path, monkeypatch):
        path = tmp_path / "out"
        with open(path, "w") as target:
            monkeypatch.setattr(sys, "stdout", target)
            entrypoint._silence_stdout()
            target.write("must not land in the file")
        assert path.read_text() == ""

    def test_stdout_without_a_descriptor_is_tolerated(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        entrypoint._silence_stdout()  # must not raise
