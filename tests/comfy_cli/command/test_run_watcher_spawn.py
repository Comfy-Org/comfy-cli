"""The job watcher must be detached on every platform CI runs.

``start_new_session`` is POSIX-only — CPython silently ignores it on Windows,
which left the watcher in the parent's console and process group despite the
docstring promising otherwise. Windows needs
``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` instead, the same way
``models._spawn_download_worker`` does it.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

from comfy_cli.command.run import watcher


class TestWatcherSpawnFlags:
    def _capture_popen(self, monkeypatch):
        popen = MagicMock()
        monkeypatch.setattr(subprocess, "Popen", popen)
        return popen

    def test_posix_uses_a_new_session(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        popen = self._capture_popen(monkeypatch)

        assert watcher._spawn_watcher("abc123", where="cloud") is True

        kwargs = popen.call_args.kwargs
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs

    def test_windows_uses_detached_process_flags(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        # These constants only exist on Windows builds of the stdlib.
        monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x8, raising=False)
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
        popen = self._capture_popen(monkeypatch)

        assert watcher._spawn_watcher("abc123", where="local", host="127.0.0.1", port=8188) is True

        kwargs = popen.call_args.kwargs
        assert kwargs["creationflags"] == 0x8 | 0x200
        assert "start_new_session" not in kwargs

    def test_stdio_stays_detached_from_the_parent(self, monkeypatch):
        popen = self._capture_popen(monkeypatch)

        watcher._spawn_watcher("abc123", where="cloud")

        kwargs = popen.call_args.kwargs
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["close_fds"] is True

    def test_spawn_failure_is_reported_not_raised(self, monkeypatch):
        popen = self._capture_popen(monkeypatch)
        popen.side_effect = OSError("no fork for you")

        assert watcher._spawn_watcher("abc123", where="cloud") is False

    def test_argument_rejection_is_reported_not_raised(self, monkeypatch):
        # Popen rejects bad *arguments* with ValueError, not OSError — an
        # embedded NUL in host/prompt_id, or creationflags off Windows. The
        # workflow is already submitted by the time we spawn, so neither may
        # escape and abort `comfy run`.
        popen = self._capture_popen(monkeypatch)
        popen.side_effect = ValueError("embedded null byte")

        assert watcher._spawn_watcher("abc123", where="local", host="127.0.0.1\x00") is False
