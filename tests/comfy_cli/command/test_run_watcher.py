"""``COMFY_NO_WATCH`` — the env kill switch that suppresses the detached
watcher subprocess for agentic callers.

The non-wait run path (both local and cloud) spawns a detached, credential-
inheriting watcher via ``subprocess.Popen(..., start_new_session=True)`` that
survives the parent process and polls the jobs API for up to 6h. Agents that
already have their own job-wait loop (e.g. the cloud agent's Redis pub/sub +
reconcile GET) have no use for it — it's a pure-waste orphan process holding
onto COMFY_CLOUD_AUTH_TOKEN / COMFY_CLOUD_API_KEY after the parent exits.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from comfy_cli.command.run.watcher import _no_watch_requested, _spawn_watcher, _tail_state_file


class TestNoWatchRequested:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("COMFY_NO_WATCH", raising=False)
        assert _no_watch_requested() is False

    def test_one_is_true(self, monkeypatch):
        monkeypatch.setenv("COMFY_NO_WATCH", "1")
        assert _no_watch_requested() is True

    def test_false_like_values_are_false(self, monkeypatch):
        for v in ("0", "false", "False", "no", "off", ""):
            monkeypatch.setenv("COMFY_NO_WATCH", v)
            assert _no_watch_requested() is False, f"{v!r} should not suppress the watcher"

    def test_other_truthy_values_are_true(self, monkeypatch):
        for v in ("true", "TRUE", "yes", "1", "on"):
            monkeypatch.setenv("COMFY_NO_WATCH", v)
            assert _no_watch_requested() is True, f"{v!r} should suppress the watcher"


class TestSpawnWatcherHonorsKillSwitch:
    def test_no_watch_env_suppresses_spawn(self, monkeypatch):
        monkeypatch.setenv("COMFY_NO_WATCH", "1")
        with patch("comfy_cli.command.run.watcher.subprocess.Popen") as mock_popen:
            result = _spawn_watcher("prompt-123", where="cloud", notify=False)

        mock_popen.assert_not_called()
        assert result is False

    def test_without_env_spawn_still_happens(self, monkeypatch):
        # Control: with the kill switch unset, the existing spawn behavior is
        # unchanged — Popen IS called.
        monkeypatch.delenv("COMFY_NO_WATCH", raising=False)
        with patch("comfy_cli.command.run.watcher.subprocess.Popen") as mock_popen:
            result = _spawn_watcher("prompt-123", where="cloud", notify=False)

        mock_popen.assert_called_once()
        assert result is True

    def test_no_watch_env_suppresses_local_spawn_too(self, monkeypatch):
        # The same env check gates the local (non-cloud) watcher spawn site
        # in run/__init__.py's execute(), since both route through
        # _spawn_watcher.
        monkeypatch.setenv("COMFY_NO_WATCH", "1")
        with patch("comfy_cli.command.run.watcher.subprocess.Popen") as mock_popen:
            result = _spawn_watcher("prompt-456", where="local", host="127.0.0.1", port=8188, notify=True)

        mock_popen.assert_not_called()
        assert result is False


class TestTailStateFileToleratesAnUnreadableStateDir:
    """A state-file read that can't happen at all must not fail the run.

    ``_tail_state_file`` is called from ``_announce_async_queued`` — inside
    ``execute``'s ``except (WebSocketException, ConnectionError, OSError)``
    block's try, and after the server already accepted the prompt. So an
    ``OSError`` escaping it is reported to the caller as ``ws_disconnected``
    with a "re-run the command" hint for a job that is already queued, and an
    agent caller obeys that hint (OBL-192: under Comfy Desktop on Windows the
    `clishim` broker forwards ``HOME`` but not ``USERPROFILE``, so
    ``expanduser("~")`` put the config root outside the sandbox and every
    touch of the jobs dir raised ENOENT).

    Pretty mode only, because that is the only mode that tails at all — but it
    is the mode a human at a Desktop terminal is in.
    """

    # Factories, not instances: a shared exception instance would pin its
    # `__traceback__` on a class attribute for the whole session.
    FAILURES = [
        lambda: FileNotFoundError(2, "No such file or directory"),
        lambda: PermissionError(13, "Permission denied"),
    ]

    @staticmethod
    def _pretty():
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        set_renderer(Renderer(mode=OutputMode.PRETTY, command="run"))

    @staticmethod
    def _unreadable(monkeypatch, make_exc):
        from comfy_cli import jobs_state

        def _boom(*_a, **_k):
            raise make_exc()

        monkeypatch.setattr(jobs_state, "read", _boom)

    @pytest.mark.parametrize("make_exc", FAILURES, ids=lambda f: type(f()).__name__)
    def test_poll_loop_read_returns_instead_of_raising(self, monkeypatch, capsys, make_exc):
        self._pretty()
        self._unreadable(monkeypatch, make_exc)
        # A non-zero budget so the poll loop runs at least once; the assertion
        # is simply that this call returns.
        _tail_state_file("obl192-tail", seconds=0.05)
        capsys.readouterr()

    @pytest.mark.parametrize("make_exc", FAILURES, ids=lambda f: type(f()).__name__)
    def test_final_read_returns_instead_of_raising(self, monkeypatch, capsys, make_exc):
        """``seconds=0`` skips the poll loop entirely, isolating the second
        read — the one after the ``Live`` block, which is outside its
        ``except``."""
        self._pretty()
        self._unreadable(monkeypatch, make_exc)
        _tail_state_file("obl192-tail", seconds=0.0)
        capsys.readouterr()
