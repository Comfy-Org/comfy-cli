"""``popen_detached`` must break a Windows child out of the parent's Job Object.

The local Comfy agent runs every comfy-cli invocation inside a kill-on-close Job
Object. A child spawned with only ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``
stays in that job and dies the moment the command returns (a background download
stuck at ``starting``), so Windows adds ``CREATE_BREAKAWAY_FROM_JOB`` — and falls
back to the old flags when the job forbids breakaway (``ERROR_ACCESS_DENIED``).
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, sentinel

import pytest

from comfy_cli import detach

DETACHED = 0x00000008
NEW_GROUP = 0x00000200
BREAKAWAY = 0x01000000


def _access_denied() -> OSError:
    exc = OSError(13, "Access is denied")
    exc.winerror = 5  # ERROR_ACCESS_DENIED; only set natively on Windows
    return exc


@pytest.fixture
def popen(monkeypatch):
    mock = MagicMock(return_value=sentinel.proc)
    monkeypatch.setattr(subprocess, "Popen", mock)
    return mock


@pytest.fixture
def win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    # These constants only exist on Windows builds of the stdlib.
    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", DETACHED, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", NEW_GROUP, raising=False)
    monkeypatch.delattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", raising=False)


class TestWindows:
    def test_passes_breakaway_flag(self, win32, popen):
        proc = detach.popen_detached(["worker"], stdin=subprocess.DEVNULL, close_fds=True)

        assert proc is sentinel.proc
        popen.assert_called_once()
        kwargs = popen.call_args.kwargs
        assert kwargs["creationflags"] == DETACHED | NEW_GROUP | BREAKAWAY
        assert "start_new_session" not in kwargs
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert popen.call_args.args == (["worker"],)

    def test_uses_stdlib_breakaway_constant_when_present(self, win32, popen, monkeypatch):
        monkeypatch.setattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x4000000, raising=False)

        detach.popen_detached(["worker"])

        assert popen.call_args.kwargs["creationflags"] == DETACHED | NEW_GROUP | 0x4000000

    def test_access_denied_retries_without_breakaway(self, win32, popen):
        popen.side_effect = [_access_denied(), sentinel.proc]

        proc = detach.popen_detached(["worker"], stdout=subprocess.DEVNULL)

        assert proc is sentinel.proc
        assert popen.call_count == 2
        first, second = popen.call_args_list
        assert first.kwargs["creationflags"] == DETACHED | NEW_GROUP | BREAKAWAY
        assert second.kwargs["creationflags"] == DETACHED | NEW_GROUP
        assert second.args == (["worker"],)
        assert second.kwargs["stdout"] is subprocess.DEVNULL

    def test_permission_error_retries_without_breakaway(self, win32, popen):
        popen.side_effect = [PermissionError(13, "Access is denied"), sentinel.proc]

        assert detach.popen_detached(["worker"]) is sentinel.proc
        assert popen.call_args.kwargs["creationflags"] == DETACHED | NEW_GROUP

    def test_retry_failure_propagates(self, win32, popen):
        popen.side_effect = [_access_denied(), _access_denied()]

        with pytest.raises(OSError):
            detach.popen_detached(["worker"])
        assert popen.call_count == 2

    def test_unrelated_oserror_is_not_retried(self, win32, popen):
        popen.side_effect = FileNotFoundError(2, "No such file")

        with pytest.raises(FileNotFoundError):
            detach.popen_detached(["worker"])
        popen.assert_called_once()


class TestPosix:
    def test_uses_new_session_and_no_creationflags(self, monkeypatch, popen):
        monkeypatch.setattr(sys, "platform", "linux")

        proc = detach.popen_detached(["worker"], stdin=subprocess.DEVNULL)

        assert proc is sentinel.proc
        popen.assert_called_once()
        kwargs = popen.call_args.kwargs
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs

    def test_oserror_is_not_retried(self, monkeypatch, popen):
        monkeypatch.setattr(sys, "platform", "darwin")
        popen.side_effect = PermissionError(13, "denied")

        with pytest.raises(PermissionError):
            detach.popen_detached(["worker"])
        popen.assert_called_once()
