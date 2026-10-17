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
    with a "re-run the command" hint for a job that is already queued, and a
    caller that obeys that hint submits it again. The state dir is reached through
    ``expanduser("~")``, so a launcher that does not pass through the variable
    the platform expands ``~`` from — or a sandboxed, read-only home — makes
    every touch of the jobs dir raise ENOENT.

    Pretty mode only, because that is the only mode that tails at all — but it
    is the mode a human at a terminal is in.
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
        """Make every ``jobs_state.read`` raise. Returns the call counter, so a
        test can prove *which* read site consumed the failure."""
        from comfy_cli import jobs_state

        calls = []

        def _boom(*_a, **_k):
            calls.append(1)
            raise make_exc()

        monkeypatch.setattr(jobs_state, "read", _boom)
        return calls

    @pytest.mark.parametrize("make_exc", FAILURES, ids=lambda f: type(f()).__name__)
    def test_poll_loop_read_returns_instead_of_raising(self, monkeypatch, capsys, make_exc):
        """The read inside the ``Live`` block.

        The budget is deliberately generous rather than near-zero: with a tiny
        one the deadline can pass before the loop body runs even once, and the
        final read would then consume the injected failure — the test would go
        green without ever exercising the guard it is named for. With 5s, one
        read proves the loop body ran, and *exactly* one proves it returned
        from inside the loop instead of polling on at 0.25s intervals.
        """
        self._pretty()
        calls = self._unreadable(monkeypatch, make_exc)
        _tail_state_file("statefail-tail", seconds=5.0)
        capsys.readouterr()
        assert len(calls) == 1, "expected one read inside the poll loop, then an immediate return"

    @pytest.mark.parametrize("make_exc", FAILURES, ids=lambda f: type(f()).__name__)
    def test_final_read_returns_instead_of_raising(self, monkeypatch, capsys, make_exc):
        """``seconds=0`` skips the poll loop entirely, isolating the second
        read — the one after the ``Live`` block, which is outside its
        ``except``."""
        self._pretty()
        calls = self._unreadable(monkeypatch, make_exc)
        _tail_state_file("statefail-tail", seconds=0.0)
        capsys.readouterr()
        assert len(calls) == 1, "the poll loop must not have run; this pins the post-Live read"

    @pytest.mark.parametrize("seconds", [0.0, 5.0], ids=["final_read", "poll_loop"])
    def test_an_unsafe_prompt_id_is_not_a_crash(self, monkeypatch, capsys, seconds):
        """``state_path()`` raises ``ValueError``, not ``OSError``, for a
        prompt_id that fails ``_SAFE_PROMPT_ID``. Unsafe ids used to stop at the
        submit-time `jobs_state.write`; now that `_write_state` swallows that
        rejection they reach here, where neither caller's handler covers a
        ``ValueError`` — so it would be a traceback on an accepted job."""
        self._pretty()
        # The real `jobs_state.read`, against a genuinely unsafe id: no mock,
        # so this also pins that `state_path` still rejects it.
        _tail_state_file("not/a/safe\\id.[x]", seconds=seconds)
        capsys.readouterr()

    def test_a_markup_bearing_prompt_id_renders_inert(self, capsys):
        """`prompt_id` is server-chosen and lands in `Text.from_markup`. Raw
        Rich markup there raises `MarkupError` — after the job was accepted —
        or injects into the terminal."""
        from comfy_cli import jobs_state

        self._pretty()
        evil = "[/]" + "[bold red]x"
        # Readable but empty, so the tail reaches its render path and then its
        # final "still in flight" line.
        with patch.object(jobs_state, "read", return_value=None):
            _tail_state_file(evil, seconds=0.0)
        out = capsys.readouterr()
        assert "[bold red]" not in out.out + out.err
