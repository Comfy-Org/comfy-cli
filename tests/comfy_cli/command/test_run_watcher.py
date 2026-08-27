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

from comfy_cli.command.run.watcher import _no_watch_requested, _spawn_watcher


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
