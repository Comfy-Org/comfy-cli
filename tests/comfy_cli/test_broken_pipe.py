"""BE-4795: `comfy … | head` must exit 0, not 1.

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


def _run_with_stdout_reader_gone():
    """Run the CLI with a stdout pipe whose read end is already closed.

    Deterministic by construction: there is no pipe buffer left to absorb the
    output, so the very first write raises `BrokenPipeError`. (Reading a chunk
    first and *then* closing is the more literal `| head` shape, but it races
    the kernel's pipe buffer — see the `head` test below for that variant.)
    """
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(BIG_OUTPUT_CMD, cwd=REPO_ROOT, stdout=write_fd, stderr=subprocess.PIPE)
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


class TestFlushStdout:
    def test_broken_pipe_propagates(self, monkeypatch):
        class Exploding(io.StringIO):
            def flush(self):
                raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(sys, "stdout", Exploding())
        with pytest.raises(BrokenPipeError):
            entrypoint._flush_stdout()

    def test_closed_stdout_is_tolerated(self, monkeypatch):
        stream = io.StringIO()
        stream.close()
        monkeypatch.setattr(sys, "stdout", stream)
        entrypoint._flush_stdout()  # must not raise

    def test_detached_stdout_is_tolerated(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", None)
        entrypoint._flush_stdout()  # must not raise

    def test_other_oserror_is_tolerated(self, monkeypatch):
        """A `finally`-time flush must never mask the exception on its way out."""

        class BadFd(io.StringIO):
            def flush(self):
                raise OSError(9, "Bad file descriptor")

        monkeypatch.setattr(sys, "stdout", BadFd())
        entrypoint._flush_stdout()  # must not raise
