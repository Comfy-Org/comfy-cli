"""`comfy model download --background` + the download-status / downloads /
download-cancel poll verbs.

Covers the four things that are easy to get wrong and impossible to notice
without a test: state reconciliation against a dead worker, the fail-fast
guarantee (nothing detaches until resolution has succeeded), progress-callback
accounting through the real transfer loop, and the envelope shapes agents parse.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer

from comfy_cli import download_state, file_utils
from comfy_cli.command.models import models
from comfy_cli.file_utils import DownloadException, _download_file_httpx, download_file
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point ``models.get_workspace()`` at a per-test directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(models, "get_workspace", lambda: ws)
    return ws


@pytest.fixture
def json_renderer(capsys):
    """Install a JSON-mode renderer and return a reader for the emitted envelope."""
    set_renderer(Renderer(mode=OutputMode.JSON, version="test"))

    def read_envelope() -> dict:
        out = capsys.readouterr().out.strip().splitlines()
        assert out, "no envelope was written to stdout"
        return json.loads(out[-1])

    return read_envelope


def _state(**overrides) -> download_state.DownloadState:
    """A fresh DownloadState with any field overridden."""
    fields = {"url": "https://example.com/m.safetensors", "dest": "/tmp/m.safetensors"}
    fields.update(overrides)
    state = download_state.new(url=fields.pop("url"), dest=fields.pop("dest"))
    for key, value in fields.items():
        setattr(state, key, value)
    return state


# ---------------------------------------------------------------------------
# 1. state persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_round_trip(self, workspace):
        state = _state(total_bytes=100, completed_bytes=40, status="downloading", pid=1234)
        download_state.write(workspace, state)

        loaded = download_state.read(workspace, state.id)
        assert loaded is not None
        assert loaded.schema == "download-state/1"
        assert (loaded.total_bytes, loaded.completed_bytes, loaded.status, loaded.pid) == (100, 40, "downloading", 1234)

    def test_write_is_atomic_and_leaves_no_tmp_files(self, workspace):
        state = _state()
        path = download_state.write(workspace, state)
        state.completed_bytes = 999
        download_state.write(workspace, state)

        leftovers = [p.name for p in path.parent.iterdir() if p.suffix == ".tmp" or ".tmp" in p.name]
        assert leftovers == []
        assert json.loads(path.read_text())["completed_bytes"] == 999

    def test_read_missing_returns_none(self, workspace):
        assert download_state.read(workspace, "deadbeefcafe") is None

    def test_read_tolerates_unknown_keys(self, workspace):
        state = _state()
        path = download_state.write(workspace, state)
        data = json.loads(path.read_text())
        data["invented_by_a_future_version"] = True
        path.write_text(json.dumps(data))

        assert download_state.read(workspace, state.id) is not None

    def test_read_rejects_corrupt_json(self, workspace):
        state = _state()
        path = download_state.write(workspace, state)
        path.write_text("{not json")
        assert download_state.read(workspace, state.id) is None

    @pytest.mark.parametrize("bad_id", ["../../etc/passwd", "a/b", "", "x" * 65])
    def test_unsafe_ids_are_rejected(self, workspace, bad_id):
        with pytest.raises(ValueError):
            download_state.state_path(workspace, bad_id)
        # read() swallows the rejection rather than exploding on a user-typed id.
        assert download_state.read(workspace, bad_id) is None

    def test_list_all_is_newest_first(self, workspace):
        old = _state(started_at="2020-01-01T00:00:00+00:00")
        new = _state(started_at="2030-01-01T00:00:00+00:00")
        # write() stamps updated_at but leaves started_at alone.
        download_state.write(workspace, old)
        download_state.write(workspace, new)

        assert [s.id for s in download_state.list_all(workspace)] == [new.id, old.id]

    def test_list_all_on_missing_dir(self, tmp_path):
        assert download_state.list_all(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# 1.5 retention: prune() is the only thing that removes a record
# ---------------------------------------------------------------------------

_OLD_S = download_state.PRUNE_MAX_AGE_S + 3600


def _stamp(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(timespec="seconds")


def _record(workspace, *, status="completed", age_s=0.0, dest=None):
    """Persist a state file with an exact age.

    Written by hand rather than through ``write()``, which stamps ``updated_at``
    with *now* and so can't produce the stale records prune is about.
    """
    dest = Path(dest) if dest is not None else workspace / "m.safetensors"
    state = download_state.new(url="https://example.com/m.safetensors", dest=str(dest))
    state.status = status
    state.started_at = state.updated_at = _stamp(age_s)
    path = download_state.state_path(workspace, state.id)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return state


def _make_partial(dest: Path) -> Path:
    """A ``.part`` sibling shaped exactly like the ones the downloader mkstemps."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / f"{dest.name}.ab3d9f01.part"
    partial.write_bytes(b"partial bytes")
    assert file_utils.partial_paths_for(dest) == [partial], "test fixture is not shaped like a real .part"
    return partial


def _ids_on_disk(workspace) -> set[str]:
    return {p.stem for p in download_state.state_dir(workspace).glob("*.json")}


class TestPrune:
    def test_stale_terminal_record_is_removed(self, workspace):
        state = _record(workspace, status="completed", age_s=_OLD_S)

        assert download_state.prune(workspace) == 1
        assert download_state.read(workspace, state.id) is None

    def test_fresh_terminal_record_is_kept(self, workspace):
        state = _record(workspace, status="completed", age_s=3600)

        assert download_state.prune(workspace) == 0
        assert download_state.read(workspace, state.id) is not None

    @pytest.mark.parametrize("status", ["starting", "downloading"])
    def test_in_flight_records_are_never_removed_at_any_age(self, workspace, status):
        """An active worker still owns its record — age says nothing about that."""
        state = _record(workspace, status=status, age_s=_OLD_S * 100)

        assert download_state.prune(workspace) == 0
        assert download_state.read(workspace, state.id) is not None

    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    def test_a_partial_pins_a_failed_or_cancelled_record(self, workspace, status):
        """Those bytes are on disk and this record is the only handle on them."""
        dest = workspace / "models" / "m.safetensors"
        state = _record(workspace, status=status, age_s=_OLD_S, dest=dest)
        partial = _make_partial(dest)

        assert download_state.prune(workspace) == 0
        assert download_state.read(workspace, state.id) is not None

        # Once the disk is reclaimed there is nothing left to point at.
        partial.unlink()
        assert download_state.prune(workspace) == 1
        assert download_state.read(workspace, state.id) is None

    def test_a_partial_does_not_pin_a_completed_record(self, workspace):
        """The carve-out is about unreclaimed bytes; a completed download's are
        at `dest`, and any leftover `.part` is unrelated debris."""
        dest = workspace / "models" / "m.safetensors"
        state = _record(workspace, status="completed", age_s=_OLD_S, dest=dest)
        partial = _make_partial(dest)

        assert download_state.prune(workspace) == 1
        assert download_state.read(workspace, state.id) is None
        assert partial.exists(), "prune must not touch the user's bytes"

    def test_the_cap_keeps_the_newest_records_and_evicts_oldest_first(self, workspace):
        cap = download_state.PRUNE_MAX_TERMINAL_RECORDS
        # All well inside the 7-day window, so only the cap can remove them.
        states = [_record(workspace, status="completed", age_s=i) for i in range(cap + 5)]
        newest = {s.id for s in states[:cap]}

        assert download_state.prune(workspace) == 5
        assert _ids_on_disk(workspace) == newest
        assert len(download_state.list_all(workspace)) == cap

    def test_the_cap_ignores_in_flight_records(self, workspace):
        cap = download_state.PRUNE_MAX_TERMINAL_RECORDS
        terminal = [_record(workspace, status="completed", age_s=i) for i in range(cap)]
        active = [_record(workspace, status="downloading", age_s=1000 + i) for i in range(5)]

        assert download_state.prune(workspace) == 0
        assert _ids_on_disk(workspace) == {s.id for s in terminal + active}

    def test_the_cap_overrides_the_partial_carve_out(self, workspace):
        """The cap is what makes the directory bounded rather than merely
        self-expiring, so unlike the age rule it applies unconditionally."""
        cap = download_state.PRUNE_MAX_TERMINAL_RECORDS
        dest = workspace / "models" / "m.safetensors"
        _make_partial(dest)
        oldest = _record(workspace, status="failed", age_s=10_000, dest=dest)
        [_record(workspace, status="completed", age_s=i) for i in range(cap)]

        assert download_state.prune(workspace) == 1
        assert download_state.read(workspace, oldest.id) is None

    def test_companion_log_and_cancel_files_go_with_the_record(self, workspace):
        state = _record(workspace, status="completed", age_s=_OLD_S)
        log = download_state.log_path(workspace, state.id)
        log.write_text("worker output")
        cancel = download_state.cancel_path(workspace, state.id)
        cancel.touch()

        assert download_state.prune(workspace) == 1
        assert not log.exists()
        assert not cancel.exists()

    def test_a_corrupt_file_is_left_alone(self, workspace):
        """`read_path` reads it as absent, so prune has no status to judge it by
        and must not guess — deleting unparseable state is not its job."""
        path = download_state.state_path(workspace, "deadbeefcafe")
        path.write_text("{not json")

        assert download_state.prune(workspace) == 0
        assert path.exists()

    def test_a_garbage_timestamp_is_not_treated_as_ancient(self, workspace):
        state = _record(workspace, status="completed", age_s=_OLD_S)
        path = download_state.state_path(workspace, state.id)
        data = json.loads(path.read_text())
        data["updated_at"] = "not a timestamp"
        path.write_text(json.dumps(data))

        assert download_state.prune(workspace) == 0
        assert path.exists()

    def test_missing_state_dir_is_a_no_op(self, tmp_path):
        assert download_state.prune(tmp_path / "nope") == 0

    def test_an_undeletable_record_is_a_silent_no_op(self, workspace, monkeypatch):
        """A read-only state directory must never raise into a download."""
        state = _record(workspace, status="completed", age_s=_OLD_S)

        def refuse(*args, **kwargs):
            raise OSError("Read-only file system")

        monkeypatch.setattr(Path, "unlink", refuse)

        assert download_state.prune(workspace) == 0
        assert download_state.read(workspace, state.id) is not None

    def test_a_failing_prune_never_blocks_a_submit(self, workspace, monkeypatch, json_renderer):
        _record(workspace, status="completed", age_s=_OLD_S)
        monkeypatch.setattr(Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("Read-only file system")))
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
        )

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["status"] == "starting"

    def test_an_exploding_prune_never_blocks_a_submit(self, workspace, monkeypatch, json_renderer):
        """Belt and braces for the call site: even a non-OSError out of prune."""

        def boom(_workspace):
            raise RuntimeError("prune blew up")

        monkeypatch.setattr(download_state, "prune", boom)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
        )

        assert json_renderer()["ok"] is True

    def test_submit_prunes(self, workspace, monkeypatch, json_renderer):
        calls = []
        monkeypatch.setattr(download_state, "prune", lambda ws: calls.append(ws))
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
        )

        assert calls == [workspace]
        # The record submit just wrote must survive its own prune.
        assert download_state.read(workspace, json_renderer()["data"]["download_id"]) is not None

    def test_downloads_prunes_before_listing(self, workspace, monkeypatch, json_renderer):
        stale = _record(workspace, status="completed", age_s=_OLD_S)
        fresh = _record(workspace, status="completed", age_s=60)
        calls = []
        real_prune = download_state.prune
        monkeypatch.setattr(download_state, "prune", lambda ws: (calls.append(ws), real_prune(ws))[1])

        models.downloads(None)

        assert calls == [workspace]
        listed = {row["id"] for row in json_renderer()["data"]["downloads"]}
        assert listed == {fresh.id}
        assert download_state.read(workspace, stale.id) is None


# ---------------------------------------------------------------------------
# 2. reconciliation matrix: (pid alive|dead) x (size==total | size<total | total unknown)
# ---------------------------------------------------------------------------


class TestReconcile:
    @staticmethod
    def _with_dest(tmp_path, size: int) -> Path:
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"x" * size)
        return dest

    @pytest.mark.parametrize(
        ("total", "size", "expected"),
        [
            (100, 100, "downloading"),  # complete but the worker hasn't said so yet
            (100, 40, "downloading"),
            (None, 40, "downloading"),
        ],
    )
    def test_pid_alive_keeps_status(self, tmp_path, total, size, expected):
        dest = self._with_dest(tmp_path, size)
        state = _state(dest=str(dest), status="downloading", pid=4242, total_bytes=total, completed_bytes=0)

        fresh = download_state.reconcile(state, pid_alive=lambda pid: True)

        assert fresh.status == expected
        # A live stat() always wins over the last value the worker persisted.
        assert fresh.completed_bytes == size

    def test_pid_dead_and_size_equals_total_is_completed(self, tmp_path):
        dest = self._with_dest(tmp_path, 100)
        state = _state(dest=str(dest), status="downloading", pid=4242, total_bytes=100)

        fresh = download_state.reconcile(state, pid_alive=lambda pid: False)

        assert (fresh.status, fresh.completed_bytes, fresh.error) == ("completed", 100, None)

    def test_pid_dead_and_size_exceeds_total_is_completed(self, tmp_path):
        """A server that under-reports Content-Length must not read as a failure."""
        dest = self._with_dest(tmp_path, 120)
        state = _state(dest=str(dest), status="downloading", pid=4242, total_bytes=100)

        assert download_state.reconcile(state, pid_alive=lambda pid: False).status == "completed"

    def test_pid_dead_and_size_below_total_is_failed(self, tmp_path):
        dest = self._with_dest(tmp_path, 40)
        state = _state(dest=str(dest), status="downloading", pid=4242, total_bytes=100)

        fresh = download_state.reconcile(state, pid_alive=lambda pid: False)

        assert fresh.status == "failed"
        assert "worker died" in fresh.error

    def test_pid_dead_and_total_unknown_is_failed(self, tmp_path):
        """Without a total there is no evidence the transfer finished."""
        dest = self._with_dest(tmp_path, 40)
        state = _state(dest=str(dest), status="downloading", pid=4242, total_bytes=None)

        fresh = download_state.reconcile(state, pid_alive=lambda pid: False)

        assert fresh.status == "failed"

    def test_pid_dead_and_dest_missing_is_failed(self, tmp_path):
        state = _state(dest=str(tmp_path / "never-written"), status="downloading", pid=4242, total_bytes=0)

        assert download_state.reconcile(state, pid_alive=lambda pid: False).status == "failed"

    def test_starting_with_dead_pid_is_failed(self, tmp_path):
        """A worker that died before its first write is still a dead worker."""
        state = _state(dest=str(tmp_path / "nothing"), status="starting", pid=4242)

        assert download_state.reconcile(state, pid_alive=lambda pid: False).status == "failed"

    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    def test_terminal_statuses_are_never_rewritten(self, tmp_path, status):
        dest = self._with_dest(tmp_path, 10)
        state = _state(dest=str(dest), status=status, pid=4242, total_bytes=100)

        assert download_state.reconcile(state, pid_alive=lambda pid: False).status == status

    def test_cancelled_does_not_adopt_a_stale_dest_size(self, tmp_path):
        """`download-cancel` removed the partial; an unrelated file reappearing at
        dest must not resurrect a byte count for a cancelled download."""
        dest = self._with_dest(tmp_path, 10)
        state = _state(dest=str(dest), status="cancelled", completed_bytes=0)

        assert download_state.reconcile(state, pid_alive=lambda pid: False).completed_bytes == 0

    def test_percent_is_none_without_a_total(self):
        assert download_state.percent(_state(total_bytes=None, completed_bytes=5)) is None
        assert download_state.percent(_state(total_bytes=0, completed_bytes=5)) is None
        assert download_state.percent(_state(total_bytes=200, completed_bytes=50)) == 25.0

    def test_percent_is_capped_at_100(self):
        assert download_state.percent(_state(total_bytes=100, completed_bytes=120)) == 100.0

    def test_elapsed_seconds_freezes_at_terminal(self):
        state = _state(
            status="completed",
            started_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:30+00:00",
        )
        assert download_state.elapsed_seconds(state) == pytest.approx(30.0)

    def test_elapsed_seconds_tolerates_a_garbage_timestamp(self):
        assert download_state.elapsed_seconds(_state(started_at="not-a-date")) == 0.0


# ---------------------------------------------------------------------------
# 3. progress callback plumbing
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, total: int | None, status_code: int = 200):
        self.chunks = chunks
        self.status_code = status_code
        self.headers = {} if total is None else {"Content-Length": str(total)}

    def iter_bytes(self):
        yield from self.chunks

    def read(self):
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestProgressCallback:
    def test_counts_bytes_and_reports_the_total(self, tmp_path):
        seen: list[tuple[int, int | None]] = []
        response = _FakeResponse([b"aaaa", b"bb", b"cccccc"], total=12)

        with patch("httpx.stream", return_value=response):
            _download_file_httpx(
                "https://x/y", tmp_path / "out.bin", progress_callback=lambda c, t: seen.append((c, t))
            )

        # Fires once before the first chunk so `total_bytes` stops being null
        # as soon as the headers are read, then once per chunk.
        assert seen == [(0, 12), (4, 12), (6, 12), (12, 12)]
        assert (tmp_path / "out.bin").read_bytes() == b"aaaabbcccccc"

    def test_total_is_none_without_content_length(self, tmp_path):
        seen: list[tuple[int, int | None]] = []
        response = _FakeResponse([b"abc"], total=None)

        with patch("httpx.stream", return_value=response):
            _download_file_httpx(
                "https://x/y", tmp_path / "out.bin", progress_callback=lambda c, t: seen.append((c, t))
            )

        assert seen == [(0, None), (3, None)]

    def test_no_callback_is_fine(self, tmp_path):
        with patch("httpx.stream", return_value=_FakeResponse([b"abc"], total=3)):
            _download_file_httpx("https://x/y", tmp_path / "out.bin")
        assert (tmp_path / "out.bin").read_bytes() == b"abc"

    def test_a_raising_callback_never_breaks_the_download(self, tmp_path):
        def boom(completed, total):
            raise RuntimeError("disk full while persisting state")

        with patch("httpx.stream", return_value=_FakeResponse([b"abc"], total=3)):
            _download_file_httpx("https://x/y", tmp_path / "out.bin", progress_callback=boom)

        assert (tmp_path / "out.bin").read_bytes() == b"abc"

    def test_retry_resets_the_counter_when_the_partial_is_cleaned(self, tmp_path):
        """A retry deletes the partial file, so the observer must be told the
        count went back to zero rather than left holding a stale high-water mark."""
        seen: list[tuple[int, int | None]] = []
        attempts = {"n": 0}

        def stream(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _FakeResponse([b"aaaa"], total=8)
            return _FakeResponse([b"bbbbbbbb"], total=8)

        original = _download_file_httpx

        def flaky(url, path, headers=None, *, state=None, progress_callback=None):
            if attempts["n"] == 0:
                attempts["n"] = 1
                if state is not None:
                    state["file_opened"] = True
                path.write_bytes(b"aaaa")
                progress_callback(0, 8)
                progress_callback(4, 8)
                raise httpx.ReadTimeout("boom")
            return original(url, path, headers, state=state, progress_callback=progress_callback)

        with (
            patch("comfy_cli.file_utils._download_file_httpx", side_effect=flaky),
            patch("comfy_cli.file_utils.time.sleep"),
            patch("httpx.stream", side_effect=stream),
        ):
            download_file("https://x/y", tmp_path / "out.bin", progress_callback=lambda c, t: seen.append((c, t)))

        # (4, 8) from the doomed attempt, then the reset, then the clean run.
        assert (0, None) in seen
        assert seen.index((0, None)) > seen.index((4, 8))
        assert seen[-1] == (8, 8)

    def test_aria2_callback_is_fed_from_the_poll_loop(self):
        from comfy_cli.file_utils import _poll_aria2_download

        seen: list[tuple[int, int | None]] = []
        download = MagicMock()
        # total_length unknown -> known; completed_length advances; then complete.
        states = [(0, 0, False), (100, 40, False), (100, 100, True)]

        def update():
            total, completed, complete = states.pop(0)
            download.total_length = total
            download.completed_length = completed
            download.is_complete = complete
            download.has_failed = False
            download.is_removed = False

        download.update.side_effect = update

        with patch("time.sleep"):
            _poll_aria2_download(download, lambda c, t: seen.append((c, t)))

        assert (0, None) in seen  # size not yet known
        assert (40, 100) in seen
        assert seen[-1] == (100, 100)

    def test_ctrl_c_stops_the_daemon_side_aria2_transfer(self):
        """Ctrl-C has to reach *aria2c*, not just this process.

        With aria2 the bytes move inside the daemon, and the foreground
        `download_file` call passes a progress callback that never raises
        `DownloadCancelled` — so an interrupt lands here, in the poll loop, and
        nothing else would ever tell aria2c to stop. Walking away would leave the
        daemon writing to a destination whose *claim* the interrupted CLI has just
        withdrawn: the unguarded double-writer the claim exists to prevent, via
        the very keystroke `model_download_foreground_cancel` tells users to press.
        """
        from comfy_cli.file_utils import _poll_aria2_download

        download = MagicMock()
        download.total_length = 100
        download.completed_length = 10
        download.is_complete = False
        download.has_failed = False
        download.is_removed = False

        with patch("time.sleep", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                _poll_aria2_download(download)

        download.remove.assert_called_once_with(force=True, files=True)


class TestWorkerThrottle:
    def test_progress_writes_are_throttled_but_terminal_always_lands(self, workspace, monkeypatch, tmp_path):
        """~1 write/s while streaming; the completed transition is unconditional."""
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), total_bytes=9)
        path = download_state.write(workspace, state)

        writes: list[int] = []
        real_write = download_state.write_path
        monkeypatch.setattr(
            download_state,
            "write_path",
            lambda p, s: (writes.append(s.completed_bytes), real_write(p, s))[1],
        )

        clock = {"t": 1000.0}
        monkeypatch.setattr(models.time, "monotonic", lambda: clock["t"])

        def fake_download_file(url, filepath, headers, downloader, progress_callback):
            for completed, tick in [(3, 0.1), (6, 0.2), (9, 5.0)]:
                clock["t"] += tick
                progress_callback(completed, 9)
            dest.write_bytes(b"x" * 9)

        monkeypatch.setattr(models, "download_file", fake_download_file)
        models._download_worker(state_file=str(path))

        # The first callback always lands (it carries the just-learned total);
        # the one 0.2s behind it is dropped; the one 5s later lands again.
        assert writes.count(3) == 1
        assert writes.count(6) == 0
        assert writes.count(9) >= 1

        final = download_state.read(workspace, state.id)
        assert (final.status, final.completed_bytes, final.total_bytes) == ("completed", 9, 9)

    def test_worker_records_a_failure_with_a_friendly_message(self, workspace, monkeypatch, tmp_path):
        state = _state(dest=str(tmp_path / "m.safetensors"))
        path = download_state.write(workspace, state)

        def boom(*args, **kwargs):
            raise DownloadException("Failed to download file.\nFile not found on server (404)")

        monkeypatch.setattr(models, "download_file", boom)
        with pytest.raises(typer.Exit):
            models._download_worker(state_file=str(path))

        final = download_state.read(workspace, state.id)
        assert final.status == "failed"
        assert "404" in final.error

    def test_worker_writes_its_own_pid_on_startup(self, workspace, monkeypatch, tmp_path):
        state = _state(dest=str(tmp_path / "m.safetensors"), pid=None)
        path = download_state.write(workspace, state)

        recorded = {}

        def capture(url, filepath, headers, downloader, progress_callback):
            recorded.update(download_state.read(workspace, state.id).to_dict())
            filepath.write_bytes(b"ok")

        monkeypatch.setattr(models, "download_file", capture)
        models._download_worker(state_file=str(path))

        assert recorded["pid"] == os.getpid()
        assert recorded["status"] == "downloading"

    def test_worker_exits_quietly_when_the_state_file_vanished(self, tmp_path):
        with pytest.raises(typer.Exit):
            models._download_worker(state_file=str(tmp_path / "gone.json"))

    def test_worker_rederives_auth_headers_from_config_not_state(self, workspace, monkeypatch, tmp_path):
        """The state file records *which* credential is needed, never the secret."""
        state = _state(
            url="https://civitai.com/api/download/models/1",
            dest=str(tmp_path / "m.safetensors"),
            needs_civitai_auth=True,
        )
        path = download_state.write(workspace, state)
        assert "Authorization" not in path.read_text()

        monkeypatch.setattr(models, "_civitai_headers", lambda: {"Authorization": "Bearer from-config"})
        seen = {}

        def capture(url, filepath, headers, downloader, progress_callback):
            seen["headers"] = headers
            filepath.write_bytes(b"ok")

        monkeypatch.setattr(models, "download_file", capture)
        models._download_worker(state_file=str(path))

        assert seen["headers"] == {"Authorization": "Bearer from-config"}


class TestWorkerCredentialScoping:
    """A state file is data on disk; `needs_*_auth` and `url` can disagree.

    Whoever can write one must not be able to aim the user's bearer token at a
    host of their choosing.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://attacker.example/m.safetensors",
            "https://civitai.com.attacker.example/m.safetensors",
            "https://notcivitai.com/m.safetensors",
        ],
    )
    def test_civitai_token_is_withheld_from_a_foreign_host(self, monkeypatch, url):
        monkeypatch.setattr(models, "_civitai_headers", lambda: {"Authorization": "Bearer secret"})
        headers = models._worker_headers(_state(url=url, needs_civitai_auth=True))
        assert "Authorization" not in (headers or {})

    @pytest.mark.parametrize("url", ["https://civitai.com/x", "https://cdn.civitai.com/x", "https://civitai.red/x"])
    def test_civitai_token_is_sent_to_civitai(self, monkeypatch, url):
        monkeypatch.setattr(models, "_civitai_headers", lambda: {"Authorization": "Bearer secret"})
        assert models._worker_headers(_state(url=url, needs_civitai_auth=True))["Authorization"] == "Bearer secret"

    def test_hf_token_is_withheld_from_a_foreign_host(self, monkeypatch):
        monkeypatch.setattr(models, "_hf_headers", lambda: {"Authorization": "Bearer secret"})
        assert models._worker_headers(_state(url="https://attacker.example/m", needs_hf_auth=True)) is None

    @pytest.mark.parametrize("url", ["https://huggingface.co/x", "https://cdn-lfs.huggingface.co/x", "https://hf.co/x"])
    def test_hf_token_is_sent_to_hugging_face(self, monkeypatch, url):
        monkeypatch.setattr(models, "_hf_headers", lambda: {"Authorization": "Bearer secret"})
        assert models._worker_headers(_state(url=url, needs_hf_auth=True))["Authorization"] == "Bearer secret"


# ---------------------------------------------------------------------------
# 4. submit: fail fast, then detach
# ---------------------------------------------------------------------------


@pytest.fixture
def no_spawn(monkeypatch):
    """Assert nothing detaches, by making a spawn attempt an outright failure."""
    calls = []

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a worker was spawned but resolution should have failed first")

    monkeypatch.setattr(models, "_spawn_download_worker", spawn)
    return calls


class TestSubmitFailsFast:
    def test_unknown_scheme_never_detaches(self, workspace, no_spawn, monkeypatch, capsys):
        """An unrecognized source can't resolve a filename; under skip_prompting
        `ui.prompt_input` returns "" and the empty-filename guard fires — as an
        `envelope/1` error, not a raw `DownloadException`."""
        monkeypatch.setattr(models.ui, "prompt_input", lambda *a, **k: k.get("default", ""))

        with pytest.raises(typer.Exit) as exc:
            models.download(None, url="ftp://example.com/model.safetensors", background=True)

        assert exc.value.exit_code == 1
        assert "Could not determine a filename" in capsys.readouterr().out

    def test_destination_exists_never_detaches(self, workspace, no_spawn, monkeypatch, capsys):
        dest = workspace / "models" / "loras" / "already.safetensors"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"already here")
        monkeypatch.setattr(models.ui, "prompt_input", lambda *a, **k: k.get("default", ""))

        with pytest.raises(typer.Exit) as exc:
            models.download(
                None,
                url="https://example.com/already.safetensors",
                relative_path="models/loras",
                filename="already.safetensors",
                background=True,
            )

        assert exc.value.exit_code == 1
        assert "already exists" in capsys.readouterr().out

    def test_unresolvable_filename_under_skip_prompting_never_detaches(self, workspace, no_spawn, monkeypatch, capsys):
        monkeypatch.setattr(models.ui, "prompt_input", lambda *a, **k: k.get("default", ""))

        with pytest.raises(typer.Exit) as exc:
            models.download(None, url="https://example.com/", background=True)

        assert exc.value.exit_code == 1
        assert "Could not determine a filename" in capsys.readouterr().out

    def test_missing_hf_token_never_detaches(self, workspace, no_spawn, monkeypatch, capsys):
        monkeypatch.setattr(models, "check_unauthorized", lambda url, headers: True)
        monkeypatch.setattr(models.config_manager, "get_or_override", lambda *a, **k: None)

        with pytest.raises(typer.Exit) as exc:
            models.download(
                None,
                url="https://huggingface.co/org/repo/resolve/main/model.safetensors",
                relative_path="models/loras",
                filename="model.safetensors",
                background=True,
            )

        assert exc.value.exit_code == 1
        assert "Hugging Face API token" in capsys.readouterr().out


class TestSubmitRefusesAClaimedDestination:
    """A destination already claimed by a *live* background download is refused.

    `local_filepath.exists()` cannot catch this: a background transfer streams
    into a `.part` sibling and only renames onto `dest` at the very end, so the
    destination is absent for the whole transfer and a second submission would
    otherwise sail through, stream a full second copy, and silently overwrite the
    first at rename time.
    """

    DEST = ("models/loras", "m.safetensors")

    def _dest(self, workspace) -> Path:
        return workspace / self.DEST[0] / self.DEST[1]

    def _download(self, **kwargs):
        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path=self.DEST[0],
            filename=self.DEST[1],
            **kwargs,
        )

    def test_a_second_submission_is_refused(self, workspace, no_spawn, json_renderer):
        """The classic double-submit: a live worker owns the destination."""
        live = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234, total_bytes=4096)
        download_state.write(workspace, live)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download(background=True)

        assert exc.value.exit_code == 1
        env = json_renderer()
        assert env["ok"] is False
        assert env["error"]["code"] == "model_download_in_flight"
        assert env["error"]["details"]["download_id"] == live.id
        assert env["error"]["details"]["status"] == "downloading"
        assert env["error"]["details"]["path"] == str(self._dest(workspace))
        # The refusal is read-only: the in-flight record is left exactly as it was.
        assert download_state.read(workspace, live.id).status == "downloading"
        assert len(download_state.list_all(workspace)) == 1

    def test_a_pidless_starting_record_still_blocks(self, workspace, no_spawn, json_renderer):
        """The regression test for reconcile-vs-`worker_alive`.

        A just-submitted download sits in `starting` with no pid for up to
        STARTUP_GRACE_S while its worker's interpreter boots, and `worker_alive`
        reports False for a pidless record — so a `worker_alive` predicate would
        wave through exactly the near-simultaneous double-submit this guard is
        for. `_state()` stamps `started_at` now, i.e. inside the grace window.
        """
        live = _state(dest=str(self._dest(workspace)), status="starting", pid=None)
        download_state.write(workspace, live)

        with pytest.raises(typer.Exit) as exc:
            self._download(background=True)

        assert exc.value.exit_code == 1
        assert json_renderer()["error"]["code"] == "model_download_in_flight"

    def test_a_dead_workers_record_self_clears(self, workspace, monkeypatch, json_renderer):
        """A SIGKILLed worker's stale record must never wedge the path: the scan
        reconciles first, so the record demotes to `failed` and the submit runs."""
        stale = _state(dest=str(self._dest(workspace)), status="downloading", pid=4242, total_bytes=4096)
        download_state.write(workspace, stale)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        with patch("comfy_cli.utils.is_running", return_value=False):
            self._download(background=True)

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["download_id"] != stale.id
        # ...and the demotion was persisted, not merely computed in memory.
        assert download_state.read(workspace, stale.id).status == "failed"

    def test_a_live_download_to_another_destination_does_not_block(self, workspace, monkeypatch, json_renderer):
        other = _state(dest=str(workspace / "models" / "loras" / "other.safetensors"), status="downloading", pid=1234)
        download_state.write(workspace, other)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        with patch("comfy_cli.utils.is_running", return_value=True):
            self._download(background=True)

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["dest"] == str(self._dest(workspace))

    def test_an_unnormalized_relative_path_does_not_slip_past(self, workspace, no_spawn, json_renderer):
        """`--relative-path` is only `expanduser`-ed, never rejected for `..`, so
        both sides of the comparison have to be normalized or a caller could
        spell the same destination differently and defeat the guard."""
        live = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234)
        download_state.write(workspace, live)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit):
                models.download(
                    None,
                    url="https://example.com/m.safetensors",
                    relative_path="models/loras/../loras",
                    filename=self.DEST[1],
                    background=True,
                )

        assert json_renderer()["error"]["details"]["download_id"] == live.id

    def test_an_unreadable_state_directory_does_not_break_the_download(self, workspace, monkeypatch, json_renderer):
        """The scan is advisory. It is now on the foreground path too, which never
        read the state directory before, so a failure there must degrade to the
        pre-guard behavior rather than become a traceback."""
        monkeypatch.setattr(download_state, "list_all", MagicMock(side_effect=OSError("state dir is not readable")))
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        self._download(background=True)

        assert json_renderer()["ok"] is True

    def test_a_foreground_download_is_refused_too(self, workspace, monkeypatch, json_renderer):
        """The guard sits before the `--background` split, so a foreground
        transfer into a claimed destination is refused before any bytes move."""
        live = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234)
        download_state.write(workspace, live)
        monkeypatch.setattr(models, "download_file", MagicMock(side_effect=AssertionError("a transfer started")))

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download()

        assert exc.value.exit_code == 1
        env = json_renderer()
        assert env["error"]["code"] == "model_download_in_flight"
        assert env["error"]["details"]["download_id"] == live.id

    def test_a_live_transfer_beats_the_exists_check(self, workspace, no_spawn, json_renderer):
        """`--downloader aria2` writes straight to the destination (it owns its
        own `.aria2` resume file), so a live aria2 transfer makes `exists()` true.
        Ordered the other way the caller would get `model_file_exists` and its
        hint to "remove the existing file" — advice that deletes the output a
        running worker is still writing."""
        dest = self._dest(workspace)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial aria2 output")
        live = _state(dest=str(dest), status="downloading", pid=1234, downloader="aria2", total_bytes=4096)
        download_state.write(workspace, live)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download(background=True)

        assert exc.value.exit_code == 1
        env = json_renderer()
        assert env["error"]["code"] == "model_download_in_flight"
        assert env["error"]["details"]["download_id"] == live.id
        # ...and the bytes the live worker is still writing were not implicated.
        assert dest.exists()

    def test_a_record_for_another_destination_is_not_reconciled(self, workspace, monkeypatch, json_renderer):
        """The scan filters by destination *before* reconciling.

        `_reconciled` persists a status correction, so reconciling every record
        would make a plain `comfy model download` rewrite bookkeeping for
        unrelated downloads — off `list_all`'s stale snapshot, so a worker that
        completes during the scan window could have its `completed` record
        overwritten with `failed`.
        """
        other = _state(dest=str(workspace / "models" / "loras" / "other.safetensors"), status="downloading", pid=4242)
        download_state.write(workspace, other)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        # A dead worker: reconcile *would* demote this record to `failed`. It is
        # not this command's record to touch.
        with patch("comfy_cli.utils.is_running", return_value=False):
            self._download(background=True)

        assert json_renderer()["ok"] is True
        assert download_state.read(workspace, other.id).status == "downloading"

    def test_a_corrupt_record_does_not_break_the_download(self, workspace, monkeypatch, json_renderer):
        """The advisory scan has to degrade on a bad record, not traceback.

        Only `list_all` used to sit inside the `try`, so a record that tripped a
        lookup *inside* the loop — a tampered negative pid reaching
        `psutil.Process`, say — turned every `comfy model download` in the
        workspace, foreground included, into a bare traceback.
        """
        corrupt = _state(dest=str(self._dest(workspace)), status="downloading", pid=-1)
        download_state.write(workspace, corrupt)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        self._download(background=True)

        assert json_renderer()["ok"] is True
        # No live worker can be behind a pid that cannot exist, so it demotes.
        assert download_state.read(workspace, corrupt.id).status == "failed"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
    def test_a_symlinked_model_directory_is_the_same_destination(self, workspace, no_spawn, json_renderer):
        """ComfyUI model directories are routinely symlinks (`models/loras` at
        `/data/loras`) and get addressed both ways. A lexical normalization
        leaves the two spellings unequal, so both submissions would pass and
        their transfers would rename onto the same inode."""
        real_dir = workspace.parent / "data" / "loras"
        real_dir.mkdir(parents=True)
        link_dir = workspace / "models" / "loras"
        link_dir.parent.mkdir(parents=True, exist_ok=True)
        link_dir.symlink_to(real_dir, target_is_directory=True)

        # The live record names the resolved path; the submission names the link.
        live = _state(dest=str(real_dir / self.DEST[1]), status="downloading", pid=1234)
        download_state.write(workspace, live)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit):
                self._download(background=True)

        assert json_renderer()["error"]["details"]["download_id"] == live.id

    def test_a_claim_that_landed_first_wins_the_post_write_recheck(
        self, workspace, no_spawn, monkeypatch, json_renderer
    ):
        """The pre-flight scan is check-then-act: filename resolution and, for a
        Hugging Face url, a whole `check_unauthorized` round trip sit between it
        and the state write, so two near-simultaneous submissions can both pass
        it, both stream a full copy, and the later `os.replace` can silently
        overwrite the earlier. The record each one writes is its claim; the
        re-scan after that write is what turns two winners into one.

        Scope: this plants the competitor's claim *before* the re-scan, so the
        re-scan can see it. The interleaving where a competitor's write lands
        after our scan is not covered here because it is not covered by the code
        either — see `_submit_background_download`, which narrows that race
        rather than closing it.
        """
        dest = self._dest(workspace)
        competitor = _state(dest=str(dest), status="downloading", pid=1234)
        competitor.started_at = "2000-01-01T00:00:00+00:00"  # claimed first

        real_write = download_state.write
        planted: list = []

        def write_then_race(ws, state):
            path = real_write(ws, state)
            if not planted:
                # The competitor's claim lands in exactly the window between our
                # own claim and the re-check — the race this exists to lose.
                planted.append(real_write(ws, competitor))
            return path

        monkeypatch.setattr(download_state, "write", write_then_race)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download(background=True)

        assert exc.value.exit_code == 1
        env = json_renderer()
        assert env["error"]["code"] == "model_download_in_flight"
        assert env["error"]["details"]["download_id"] == competitor.id
        # The loser withdrew its own claim, so the destination is left owned by
        # exactly one record — a phantom would refuse every later submission.
        assert [s.id for s in download_state.list_all(workspace)] == [competitor.id]

    def test_a_claim_that_landed_second_does_not_take_the_destination(self, workspace, monkeypatch, json_renderer):
        """The other side of the same race: both racers compute `_claim_order`
        over the same two records, so the one that claimed first proceeds rather
        than both backing off and neither download happening."""
        dest = self._dest(workspace)
        latecomer = _state(dest=str(dest), status="starting", pid=None)
        latecomer.started_at = "2999-01-01T00:00:00+00:00"  # claimed second

        real_write = download_state.write
        planted: list = []

        def write_then_race(ws, state):
            path = real_write(ws, state)
            if not planted:
                planted.append(real_write(ws, latecomer))
            return path

        monkeypatch.setattr(download_state, "write", write_then_race)
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)

        self._download(background=True)

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["download_id"] != latecomer.id


class TestForegroundClaimsItsDestination:
    """A foreground `comfy model download` writes a claim record too.

    Before it did, the foreground path was a blind spot: it wrote no state at all,
    so a second foreground run — or a `--background` submit started during one —
    passed both the destination scan (which only saw background records) and
    `exists()` (the httpx downloader streams into a `.part` sibling, so nothing is
    at `dest` until the rename), and the two transfers landed on the same file.
    With `--downloader aria2`, which writes straight to the destination, they
    interleave into it byte by byte.
    """

    DEST = ("models/loras", "m.safetensors")

    def _dest(self, workspace) -> Path:
        return workspace / self.DEST[0] / self.DEST[1]

    def _download(self, **kwargs):
        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path=self.DEST[0],
            filename=self.DEST[1],
            **kwargs,
        )

    def _transfer(self, monkeypatch, fn=None):
        """Patch the byte transfer; returns the list of calls it recorded."""
        calls: list = []

        def download_file(*args, **kwargs):
            calls.append((args, kwargs))
            if fn is not None:
                return fn(*args, **kwargs)
            return None

        monkeypatch.setattr(models, "download_file", download_file)
        return calls

    # -- the hole this closes -------------------------------------------------

    def test_a_second_foreground_run_is_refused(self, workspace, monkeypatch, json_renderer):
        """Ticket case 1: a live *foreground* record now blocks the next run."""
        live = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234, kind="foreground")
        download_state.write(workspace, live)
        monkeypatch.setattr(models, "download_file", MagicMock(side_effect=AssertionError("a transfer started")))

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download()

        assert exc.value.exit_code == 1
        env = json_renderer()
        assert env["error"]["code"] == "model_download_in_flight"
        assert env["error"]["details"]["download_id"] == live.id
        assert env["error"]["details"]["kind"] == "foreground"
        # The refusal names Ctrl-C, not `download-cancel` — which would itself
        # refuse a live foreground record, so pointing at it would be dead advice.
        assert "Ctrl-C" in env["error"]["hint"]
        assert "download-cancel" not in env["error"]["hint"]

    def test_a_background_submit_during_a_foreground_run_is_refused(self, workspace, no_spawn, json_renderer):
        """The cross-kind direction: `--background` must see the foreground claim."""
        live = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234, kind="foreground")
        download_state.write(workspace, live)

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download(background=True)

        assert exc.value.exit_code == 1
        assert json_renderer()["error"]["details"]["download_id"] == live.id

    def test_the_record_is_written_before_any_bytes_move(self, workspace, monkeypatch, capsys):
        """Ticket case 6. Order is the whole point: `--downloader aria2` writes
        straight to the destination, so a claim published *after* the transfer
        starts leaves the window it exists to close wide open."""
        seen: list = []

        def download_file(*args, **kwargs):
            seen.append([(s.kind, s.status) for s in download_state.list_all(workspace)])

        monkeypatch.setattr(models, "download_file", download_file)

        self._download(downloader="aria2")

        assert seen == [[("foreground", "downloading")]]

    # -- terminal bookkeeping -------------------------------------------------

    def test_success_marks_the_record_completed(self, workspace, monkeypatch, capsys):
        """Ticket case 7a."""

        def land_the_file(url, dest, headers, **kwargs):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * 17)

        self._transfer(monkeypatch, land_the_file)

        self._download()

        (record,) = download_state.list_all(workspace)
        assert record.kind == "foreground"
        assert record.status == "completed"
        # The size comes off the finished file, so `download-status` reports 100%
        # rather than a bare `completed` with no bytes.
        assert record.completed_bytes == 17
        assert download_state.percent(record) == 100.0

    def test_a_failed_transfer_marks_the_record_failed(self, workspace, monkeypatch, json_renderer):
        """Ticket case 7b: the failure text is persisted, not just rendered."""
        self._transfer(monkeypatch, MagicMock(side_effect=DownloadException("connection reset by peer")))

        with pytest.raises(typer.Exit):
            self._download()

        assert json_renderer()["error"]["code"] == "download_failed"
        (record,) = download_state.list_all(workspace)
        assert record.status == "failed"
        assert "connection reset by peer" in record.error

    def test_a_keyboard_interrupt_marks_the_record_failed(self, workspace, monkeypatch, capsys):
        """Ctrl-C is a `BaseException`, and it is precisely what the foreground
        cancel hint tells users to send — so an `except Exception` here would
        leave the record pinned at `downloading` after the documented way out."""
        self._transfer(monkeypatch, MagicMock(side_effect=KeyboardInterrupt))

        with pytest.raises(KeyboardInterrupt):
            self._download()

        (record,) = download_state.list_all(workspace)
        assert record.status == "failed"

    def test_a_completed_record_does_not_block_the_next_download(self, workspace, monkeypatch, capsys):
        """The records accumulate, so they must be inert once terminal."""
        self._transfer(monkeypatch)
        self._download()
        self._download()

        assert [s.status for s in download_state.list_all(workspace)] == ["completed", "completed"]

    def test_a_dead_foreground_record_self_clears(self, workspace, monkeypatch, capsys):
        """Ticket case 5, the mirror of `test_a_dead_workers_record_self_clears`.

        A SIGKILLed foreground run never reaches its `finally`, so its record is
        left claiming `downloading` forever. Nothing else would ever clear it —
        reconcile has to, off the pid it recorded.
        """
        stale = _state(
            dest=str(self._dest(workspace)), status="downloading", pid=4242, kind="foreground", total_bytes=4096
        )
        download_state.write(workspace, stale)
        self._transfer(monkeypatch)

        with patch("comfy_cli.utils.is_running", return_value=False):
            self._download()

        assert download_state.read(workspace, stale.id).status == "failed"
        assert len(download_state.list_all(workspace)) == 2

    # -- claim-then-check, foreground side ------------------------------------

    def _race(self, monkeypatch, rival):
        """Plant `rival`'s claim in the window between our write and the re-scan."""
        real_write = download_state.write
        planted: list = []

        def write_then_race(ws, state):
            path = real_write(ws, state)
            if not planted:
                planted.append(real_write(ws, rival))
            return path

        monkeypatch.setattr(download_state, "write", write_then_race)

    def test_an_earlier_rival_takes_the_destination(self, workspace, monkeypatch, json_renderer):
        """Ticket case 2: we lose, and we take our own claim back off disk."""
        rival = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234)
        rival.started_at = "2000-01-01T00:00:00+00:00"
        self._race(monkeypatch, rival)
        monkeypatch.setattr(models, "download_file", MagicMock(side_effect=AssertionError("a transfer started")))

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit) as exc:
                self._download()

        assert exc.value.exit_code == 1
        assert json_renderer()["error"]["details"]["download_id"] == rival.id
        # Our record is gone and the rival's is untouched. A withdrawn claim left
        # behind would refuse every later submission to this destination.
        assert [s.id for s in download_state.list_all(workspace)] == [rival.id]
        assert download_state.read(workspace, rival.id).status == "downloading"

    def test_a_later_rival_does_not_take_the_destination(self, workspace, monkeypatch, capsys):
        """Ticket case 3: we win and the transfer proceeds."""
        rival = _state(dest=str(self._dest(workspace)), status="starting", pid=None)
        rival.started_at = "2999-01-01T00:00:00+00:00"
        self._race(monkeypatch, rival)
        calls = self._transfer(monkeypatch)

        self._download()

        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("rival_id", "we_win"),
        [("zzzzzzzzzzzz", True), ("000000000000", False)],
        ids=["our-id-sorts-first", "rival-id-sorts-first"],
    )
    def test_identical_timestamps_produce_exactly_one_winner(
        self, workspace, monkeypatch, json_renderer, capsys, rival_id, we_win
    ):
        """Ticket case 4, the mutual-refusal regression.

        `started_at` is second-resolution, so two racers colliding inside the same
        second is the *common* tie, not an exotic one. Without the `id` term in
        `_claim_order` each would see the other as an equally-ranked live claim,
        both would back off, and the destination would be wedged for the user with
        no download running at all — a worse outcome than the race itself. The
        order is total, so the tie resolves the same way from both sides: exactly
        one proceeds, and it is the lower id.
        """
        rival = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234)
        rival.id = rival_id
        monkeypatch.setattr(download_state, "new_id", lambda: "mmmmmmmmmmmm")

        real_write = download_state.write
        planted: list = []

        def write_then_race(ws, state):
            path = real_write(ws, state)
            if not planted:
                # Same second, differing id — the tie the `id` term breaks.
                rival.started_at = state.started_at
                planted.append(real_write(ws, rival))
            return path

        monkeypatch.setattr(download_state, "write", write_then_race)
        calls = self._transfer(monkeypatch)

        with patch("comfy_cli.utils.is_running", return_value=True):
            if we_win:
                self._download()
            else:
                with pytest.raises(typer.Exit) as exc:
                    self._download()

        if we_win:
            assert len(calls) == 1
            assert download_state.read(workspace, "mmmmmmmmmmmm").status == "completed"
        else:
            assert calls == []
            assert exc.value.exit_code == 1
            assert json_renderer()["error"]["details"]["download_id"] == rival.id
            assert download_state.read(workspace, "mmmmmmmmmmmm") is None

    def test_a_claim_we_cannot_unlink_is_made_inert_instead(self, workspace, monkeypatch, json_renderer):
        """Withdrawing the claim is the point of losing; the unlink is only how.

        If the unlink fails, leaving our `downloading` record behind is worse than
        never having written it: it reads as a live claim to `_active_download_for`
        and would refuse every later submission to this destination until something
        reconciled it away. A terminal status is inert to the same readers, so fall
        back to that.
        """
        rival = _state(dest=str(self._dest(workspace)), status="downloading", pid=1234)
        rival.started_at = "2000-01-01T00:00:00+00:00"
        self._race(monkeypatch, rival)
        monkeypatch.setattr(download_state, "delete", lambda workspace, download_id: False)
        monkeypatch.setattr(models, "download_file", MagicMock(side_effect=AssertionError("a transfer started")))

        with patch("comfy_cli.utils.is_running", return_value=True):
            with pytest.raises(typer.Exit):
                self._download()

        assert json_renderer()["error"]["details"]["download_id"] == rival.id
        ours = [s for s in download_state.list_all(workspace) if s.id != rival.id]
        assert [s.status for s in ours] == ["failed"]
        assert download_state.read(workspace, rival.id).status == "downloading"

    def test_an_unwritable_state_directory_still_downloads(self, workspace, monkeypatch, capsys):
        """The claim is bookkeeping. An unwritable workspace must degrade to the
        old behavior — no claim, transfer still runs — not turn a download that
        used to work into an error. (`--background` *does* fail there, because a
        detached worker has nowhere else to report from.)"""
        monkeypatch.setattr(download_state, "write", MagicMock(side_effect=OSError("read-only file system")))
        calls = self._transfer(monkeypatch)

        self._download()

        assert len(calls) == 1

    def test_an_unverifiable_pid_claims_nothing(self, workspace, monkeypatch, capsys):
        """No `pid_create_time` means the claim could never retract itself.

        `worker_alive` falls back to bare pid liveness when the start time is
        missing, so once this process exits and the OS recycles its number the
        record reads *live* forever: `reconcile` never demotes it, every later
        download to that destination is refused, and `download-cancel` refuses it
        as foreground. A permanently wedged destination is worse than the race the
        claim narrows, so the claim is simply not written — the same degradation
        an unwritable state directory gets.
        """
        monkeypatch.setattr(download_state, "process_create_time", lambda pid: None)
        calls = self._transfer(monkeypatch)

        self._download()

        assert len(calls) == 1
        assert download_state.list_all(workspace) == []

    def test_the_claim_does_not_persist_the_url_query(self, workspace, monkeypatch, capsys):
        """A resolved download url carries credentials — a presigned S3/SAS
        signature, or CivitAI's `?token=`. Nothing reads a *foreground* record's
        url back (only the detached worker needs the real one), so persisting it
        verbatim would write a secret into `<workspace>/.comfy-downloads/`, which
        nothing gitignores, for no reader at all."""
        signed = "https://example.com/m.safetensors?X-Amz-Signature=hunter2"
        seen: list = []
        monkeypatch.setattr(models, "download_file", lambda *a, **k: seen.append(a[0]))

        models.download(None, url=signed, relative_path=self.DEST[0], filename=self.DEST[1])

        # The transfer itself still gets the real url...
        assert seen == [signed]
        # ...but the record on disk does not.
        (record,) = download_state.list_all(workspace)
        assert record.url == "https://example.com/m.safetensors"
        assert "hunter2" not in json.dumps(record.to_dict())

    def test_progress_is_persisted_so_a_killed_run_reconciles_honestly(self, workspace, monkeypatch, capsys):
        """The record has to learn `total_bytes` *during* the transfer.

        `reconcile` resolves a dead `downloading` record to `completed` only when
        the total is known and the file reached it. With no progress callback the
        field stayed None for the whole foreground transfer, so a run SIGKILLed in
        the window between the rename and its `finally` left a *complete* model
        under a record reading `failed` forever — and `download-cancel` then
        deleted that finished file as an aria2 partial.
        """
        snapshot: list = []

        def land_the_file(url, path, headers, downloader=None, progress_callback=None):
            # The size arrives before the first chunk, as it does off Content-Length.
            progress_callback(0, 4096)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * 4096)
            # A SIGKILL landing *here* runs no Python cleanup: no terminal status,
            # no `finally`. Whatever is on disk at this instant is all a later
            # reader ever sees, so that is what this asserts on.
            snapshot.extend(download_state.list_all(workspace))

        monkeypatch.setattr(models, "download_file", land_the_file)

        self._download()

        (record,) = snapshot
        assert record.status == "downloading"
        assert record.total_bytes == 4096

        # And that is enough for the next reader to call it what it is: the file
        # reached the known total, so this is a finished download, not a corpse.
        fresh = download_state.reconcile(record, pid_alive=lambda pid: False)
        assert fresh.status == "completed"

    def test_progress_writes_are_throttled(self, workspace, monkeypatch, capsys):
        """Once the total is known the record must not be rewritten per chunk — a
        multi-GB transfer would otherwise be thousands of state writes."""
        writes: list = []
        real_write = download_state.write

        def counting_write(ws, state):
            writes.append(state.completed_bytes)
            return real_write(ws, state)

        monkeypatch.setattr(download_state, "write", counting_write)

        def chunked(url, path, headers, downloader=None, progress_callback=None):
            for completed in range(0, 500):
                progress_callback(completed, 4096)

        monkeypatch.setattr(models, "download_file", chunked)

        self._download()

        # The claim, the newly-learned total, and the terminal write — not 500.
        assert len(writes) <= 4


class TestSubmitEnvelope:
    def _submit(self, workspace, monkeypatch, **kwargs):
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)
        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
            **kwargs,
        )

    def test_emits_the_submit_envelope(self, workspace, monkeypatch, json_renderer):
        self._submit(workspace, monkeypatch)
        env = json_renderer()

        assert env["ok"] is True
        assert env["command"] == "model download"
        data = env["data"]
        assert set(data) == {"download_id", "pid", "dest", "total_bytes", "status"}
        assert data["pid"] == 31337
        assert data["total_bytes"] is None
        assert data["status"] == "starting"
        assert data["dest"] == str(workspace / "models" / "loras" / "m.safetensors")
        assert Path(data["dest"]).is_absolute()

    def test_writes_a_state_file_the_worker_has_not_claimed_yet(self, workspace, monkeypatch, json_renderer):
        """The submitter never writes a pid — only the worker does, together
        with the start time that proves the pid is still that worker."""
        self._submit(workspace, monkeypatch)
        download_id = json_renderer()["data"]["download_id"]

        state = download_state.read(workspace, download_id)
        assert (state.pid, state.pid_create_time) == (None, None)
        assert (state.status, state.schema) == ("starting", "download-state/1")
        assert state.url == "https://example.com/m.safetensors"

    def test_a_pidless_starting_record_is_not_declared_dead_immediately(self, workspace, json_renderer, tmp_path):
        """Reconcile has to tolerate the worker's interpreter startup, or every
        poll issued in the first moments would report a phantom failure."""
        state = _state(dest=str(tmp_path / "m.safetensors"), status="starting", pid=None)
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)
        assert json_renderer()["data"]["status"] == "starting"

    def test_a_starting_record_that_never_claims_a_pid_eventually_fails(self, workspace, json_renderer, tmp_path):
        state = _state(dest=str(tmp_path / "m.safetensors"), status="starting", pid=None)
        state.started_at = "2020-01-01T00:00:00+00:00"
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)
        assert json_renderer()["data"]["status"] == "failed"

    def test_creates_the_destination_directory_up_front(self, workspace, monkeypatch, json_renderer):
        self._submit(workspace, monkeypatch)
        assert (workspace / "models" / "loras").is_dir()

    def test_a_fast_worker_is_not_clobbered_by_the_submit_writeback(
        self, workspace, monkeypatch, json_renderer, tmp_path
    ):
        """The submitter records the Popen pid after spawning. A small file can
        already be finished by then, so that write-back must not roll the state
        file back to `starting` and lose the worker's progress."""

        def spawn_and_finish(state_file, log_file):
            state = download_state.read_path(state_file)
            state.pid = 999
            state.status = "completed"
            state.total_bytes = state.completed_bytes = 4096
            download_state.write_path(state_file, state)
            return 999

        monkeypatch.setattr(models, "_spawn_download_worker", spawn_and_finish)
        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
        )

        data = json_renderer()["data"]
        assert data["status"] == "completed"
        assert data["total_bytes"] == 4096

        persisted = download_state.read(workspace, data["download_id"])
        assert (persisted.status, persisted.completed_bytes) == ("completed", 4096)

    def test_spawn_failure_is_a_structured_error(self, workspace, monkeypatch, json_renderer):
        def boom(state_file, log_file):
            raise OSError("fork: resource temporarily unavailable")

        monkeypatch.setattr(models, "_spawn_download_worker", boom)
        with pytest.raises(typer.Exit):
            models.download(
                None,
                url="https://example.com/m.safetensors",
                relative_path="models/loras",
                filename="m.safetensors",
                background=True,
            )

        env = json_renderer()
        assert env["ok"] is False
        assert env["error"]["code"] == "download_worker_spawn_failed"
        # ...and the download is recorded as failed rather than left dangling.
        assert download_state.list_all(workspace)[0].status == "failed"

    def test_foreground_is_unchanged_by_the_new_flag(self, workspace, monkeypatch, capsys):
        """Without --background the transfer still runs inline and nothing detaches."""
        monkeypatch.setattr(models, "_spawn_download_worker", MagicMock(side_effect=AssertionError))
        calls = []
        monkeypatch.setattr(models, "download_file", lambda *a, **k: calls.append((a, k)))

        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
        )

        assert len(calls) == 1
        # The foreground call site *does* pass a progress callback now — it used
        # not to, because there was no record to feed. There is one now, and
        # `total_bytes` on it is what tells `reconcile` a run killed just after
        # the rename finished rather than died mid-transfer.
        assert callable(calls[0][1]["progress_callback"])
        # It *does* now leave a record — a foreground transfer that claims nothing
        # is invisible to every other invocation, which is how two of them ended up
        # writing the same file. The record is a foreground one and it is terminal,
        # so it blocks nothing once this run is over.
        records = download_state.list_all(workspace)
        assert [(r.kind, r.status) for r in records] == [("foreground", "completed")]


class TestSpawnFlags:
    """The detach flags are platform-specific; CI runs both POSIX and Windows."""

    def _capture_popen(self, monkeypatch):
        popen = MagicMock()
        popen.return_value.pid = 4242
        monkeypatch.setattr(subprocess, "Popen", popen)
        return popen

    def test_posix_uses_a_new_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        popen = self._capture_popen(monkeypatch)

        pid = models._spawn_download_worker(tmp_path / "s.json", tmp_path / "s.log")

        assert pid == 4242
        kwargs = popen.call_args.kwargs
        assert kwargs["start_new_session"] is True
        assert "creationflags" not in kwargs

    def test_windows_uses_detached_process_flags(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        # These constants only exist on Windows builds of the stdlib.
        monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x8, raising=False)
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
        popen = self._capture_popen(monkeypatch)

        models._spawn_download_worker(tmp_path / "s.json", tmp_path / "s.log")

        kwargs = popen.call_args.kwargs
        assert kwargs["creationflags"] == 0x8 | 0x200
        assert "start_new_session" not in kwargs

    def test_worker_argv_targets_the_hidden_command(self, tmp_path, monkeypatch):
        popen = self._capture_popen(monkeypatch)
        models._spawn_download_worker(tmp_path / "s.json", tmp_path / "s.log")

        argv = popen.call_args.args[0]
        assert argv[:1] == [sys.executable]
        assert argv[1:5] == ["-m", "comfy_cli", "model", "_download-worker"]
        assert argv[-2:] == ["--state", str(tmp_path / "s.json")]

    def test_stdio_is_detached_and_the_log_is_appended(self, tmp_path, monkeypatch):
        log = tmp_path / "s.log"
        log.write_text("previous run\n")
        popen = self._capture_popen(monkeypatch)

        models._spawn_download_worker(tmp_path / "s.json", log)

        kwargs = popen.call_args.kwargs
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["stdout"].mode == "ab"
        # opened for append -> the prior run's output survives
        assert log.read_text() == "previous run\n"


# ---------------------------------------------------------------------------
# 5. poll verbs
# ---------------------------------------------------------------------------


class TestPollVerbs:
    def test_status_envelope_shape(self, workspace, json_renderer, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 50)
        state = _state(dest=str(dest), status="downloading", pid=os.getpid(), total_bytes=200)
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)
        env = json_renderer()

        assert env["ok"] is True
        assert env["command"] == "model download-status"
        assert env["data"] == {
            "id": state.id,
            "status": "downloading",
            "kind": "background",
            "completed_bytes": 50,
            "total_bytes": 200,
            "percent": 25.0,
            "elapsed_seconds": env["data"]["elapsed_seconds"],
            "dest": str(dest),
            "error": None,
        }

    def test_status_percent_is_null_without_a_total(self, workspace, json_renderer, tmp_path):
        state = _state(dest=str(tmp_path / "m"), status="downloading", pid=os.getpid(), total_bytes=None)
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)
        assert json_renderer()["data"]["percent"] is None

    def test_status_reconciles_a_dead_worker_and_persists_it(self, workspace, json_renderer, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 200)
        state = _state(dest=str(dest), status="downloading", pid=2**22 - 1, total_bytes=200)
        download_state.write(workspace, state)

        with patch("comfy_cli.utils.is_running", return_value=False):
            models.download_status(None, download_id=state.id)

        assert json_renderer()["data"]["status"] == "completed"
        # The correction is written back so the next poll is cheap.
        assert download_state.read(workspace, state.id).status == "completed"

    def test_polling_a_live_worker_does_not_rewind_its_state_file(self, workspace, json_renderer, tmp_path):
        """A poll re-derives byte counts from stat() every time, so it must not
        write its (already stale) in-memory copy back over a live worker."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(dest=str(dest), status="downloading", pid=os.getpid(), total_bytes=200, completed_bytes=10)
        path = download_state.write(workspace, state)

        # The worker races ahead between our read and any write-back.
        state.completed_bytes = 150
        download_state.write_path(path, state)
        dest.write_bytes(b"x" * 150)

        models.download_status(None, download_id=state.id)

        assert json_renderer()["data"]["completed_bytes"] == 150
        assert download_state.read(workspace, state.id).completed_bytes == 150

    def test_unknown_id_is_a_structured_error(self, workspace, json_renderer):
        with pytest.raises(typer.Exit):
            models.download_status(None, download_id="nosuchid1234")

        env = json_renderer()
        assert env["ok"] is False
        assert env["error"]["code"] == "download_not_found"
        assert env["error"]["hint"]

    def test_downloads_lists_newest_first(self, workspace, json_renderer, tmp_path):
        old = _state(dest=str(tmp_path / "a"), status="completed", started_at="2020-01-01T00:00:00+00:00")
        new = _state(dest=str(tmp_path / "b"), status="completed", started_at="2030-01-01T00:00:00+00:00")
        download_state.write(workspace, old)
        download_state.write(workspace, new)

        models.downloads(None)
        env = json_renderer()

        assert env["command"] == "model downloads"
        assert env["data"]["total"] == 2
        assert [row["id"] for row in env["data"]["downloads"]] == [new.id, old.id]
        assert set(env["data"]["downloads"][0]) == {
            "id",
            "status",
            "kind",
            "completed_bytes",
            "total_bytes",
            "percent",
            "elapsed_seconds",
            "dest",
            "error",
        }

    def test_downloads_on_an_empty_workspace(self, workspace, json_renderer):
        models.downloads(None)
        env = json_renderer()
        assert env["data"] == {"total": 0, "downloads": []}

    def test_cancel_kills_the_worker_and_removes_the_partial(self, workspace, json_renderer, tmp_path):
        # aria2 is the downloader that writes straight to the destination, so it
        # is the one whose unfinished bytes are found *at* `dest`.
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(
            dest=str(dest),
            status="downloading",
            downloader="aria2",
            pid=5150,
            pid_create_time=1.0,
            total_bytes=100,
            completed_bytes=10,
        )
        download_state.write(workspace, state)

        # Alive for the identity check, then gone once it has been signalled.
        alive = [True, True, False]
        with patch.object(download_state, "is_worker_process", side_effect=lambda *a: alive.pop(0) if alive else False):
            with patch.object(download_state, "kill_worker", return_value=True) as kill:
                models.download_cancel(None, download_id=state.id)

        kill.assert_called_once_with(5150, 1.0)
        assert not dest.exists()

        env = json_renderer()
        assert env["ok"] is True
        assert env["changed"] is True
        assert env["data"]["status"] == "cancelled"
        assert env["data"]["completed_bytes"] == 0
        assert download_state.read(workspace, state.id).status == "cancelled"

    def test_cancel_refuses_a_live_foreground_download(self, workspace, json_renderer, tmp_path):
        """Ticket case 8, and the hazard this whole guard exists for.

        `kill_worker` does `os.killpg(os.getpgid(pid), ...)`. That is safe for a
        background worker because `_spawn_download_worker` gives it its own
        session, so the group holds only it and its children. A *foreground*
        record's pid is the user's own CLI process, sharing the terminal's
        foreground process group — the same killpg would SIGTERM the surrounding
        shell job. So nothing may be signalled, and no sentinel written either.
        """
        state = _state(
            dest=str(tmp_path / "m.safetensors"),
            status="downloading",
            kind="foreground",
            pid=os.getpid(),
            pid_create_time=download_state.process_create_time(os.getpid()),
        )
        download_state.write(workspace, state)

        with patch.object(download_state, "kill_worker") as kill:
            with patch("os.killpg") as killpg:
                with pytest.raises(typer.Exit) as exc:
                    models.download_cancel(None, download_id=state.id)

        assert exc.value.exit_code == 1
        kill.assert_not_called()
        killpg.assert_not_called()
        assert not download_state.cancel_path(workspace, state.id).exists()

        env = json_renderer()
        assert env["ok"] is False
        assert env["error"]["code"] == "model_download_foreground_cancel"
        assert env["error"]["details"]["kind"] == "foreground"
        assert env["error"]["details"]["pid"] == os.getpid()
        assert "Ctrl-C" in env["error"]["hint"]
        # The refusal changed nothing.
        assert download_state.read(workspace, state.id).status == "downloading"

    def test_cancel_of_a_dead_foreground_record_still_sweeps(self, workspace, json_renderer, tmp_path):
        """The other half: once the foreground process is gone there is no group
        left to signal, and its partial file is exactly what the user is trying to
        reclaim — so a dead foreground record takes the normal path.

        Under `--downloader aria2`, which writes straight to the destination, that
        partial *is* the destination — the interleaving hazard that made the
        foreground claim necessary in the first place.
        """
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(
            dest=str(dest),
            status="downloading",
            kind="foreground",
            downloader="aria2",
            pid=5150,
            pid_create_time=1.0,
            total_bytes=100,
        )
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=False):
            with patch("comfy_cli.utils.is_running", return_value=False):
                models.download_cancel(None, download_id=state.id)

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["status"] == "cancelled"
        assert not dest.exists()

    def test_cancel_still_signals_a_live_background_worker(self, workspace, json_renderer, tmp_path):
        """The guard must key on `kind`, not merely on liveness — an unconditional
        refusal would take `download-cancel` away from the workers it is for."""
        state = _state(dest=str(tmp_path / "m.safetensors"), status="downloading", pid=5150, pid_create_time=1.0)
        download_state.write(workspace, state)
        assert state.kind == "background"

        alive = [True, True, False]
        with patch.object(download_state, "is_worker_process", side_effect=lambda *a: alive.pop(0) if alive else False):
            with patch.object(download_state, "kill_worker", return_value=True) as kill:
                models.download_cancel(None, download_id=state.id)

        kill.assert_called_once_with(5150, 1.0)
        assert json_renderer()["data"]["status"] == "cancelled"

    def test_cancel_writes_the_sentinel_before_signalling(self, workspace, json_renderer, tmp_path):
        """A worker still in interpreter startup has no pid to signal; the
        sentinel is the only thing that reaches it."""
        state = _state(dest=str(tmp_path / "m.safetensors"), status="starting", pid=None)
        download_state.write(workspace, state)

        models.download_cancel(None, download_id=state.id)

        assert download_state.cancel_path(workspace, state.id).exists()
        assert json_renderer()["data"]["status"] == "cancelled"

    def test_cancel_keeps_a_file_that_finished_during_the_kill_window(self, workspace, json_renderer, tmp_path):
        """A worker SIGKILLed after the last byte landed but before it could
        persist `completed` still reads as `downloading`. Deleting its file
        would be silent data loss."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 100)
        state = _state(dest=str(dest), status="downloading", pid=5150, total_bytes=100, completed_bytes=10)
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=False):
            models.download_cancel(None, download_id=state.id)

        assert dest.exists(), "a fully-downloaded file must never be deleted by cancel"
        assert json_renderer()["data"]["status"] == "completed"

    def test_cancel_of_a_dead_worker_still_clears_the_partial(self, workspace, json_renderer, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(
            dest=str(dest), status="downloading", downloader="aria2", pid=5150, total_bytes=100, completed_bytes=10
        )
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=False):
            models.download_cancel(None, download_id=state.id)

        assert not dest.exists()
        assert json_renderer()["data"]["status"] == "cancelled"

    def test_cancel_of_an_httpx_download_never_deletes_the_destination(self, workspace, json_renderer, tmp_path):
        """The httpx downloader only ever writes `dest` via the final rename, so
        a file sitting there mid-transfer is one this download did not create —
        `download_file` promises such a file survives either way — or a rename
        that just landed. Deleting it is data loss for a file we never wrote."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"someone else's model")
        state = _state(dest=str(dest), status="downloading", pid=5150, total_bytes=100, completed_bytes=10)
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=False):
            models.download_cancel(None, download_id=state.id)

        assert dest.read_bytes() == b"someone else's model"
        assert json_renderer()["data"]["status"] == "cancelled"

    def test_cancel_keeps_an_unknown_length_file_that_the_rename_just_landed(
        self, workspace, json_renderer, monkeypatch, tmp_path
    ):
        """With no Content-Length there is no `total_bytes`, so the `finished`
        check cannot recognise a complete download — and an httpx worker killed
        between its rename and persisting `completed` leaves exactly that. The
        file at `dest` is the finished model; deleting it is silent data loss."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"the whole model")
        state = _state(dest=str(dest), status="downloading", pid=None, total_bytes=None)
        download_state.write(workspace, state)

        monkeypatch.setattr(download_state, "stop_worker", lambda *_a, **_k: True)
        models.download_cancel(None, download_id=state.id)

        assert dest.read_bytes() == b"the whole model"

    def test_cancel_of_a_terminal_download_is_a_no_op(self, workspace, json_renderer, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 100)
        state = _state(dest=str(dest), status="completed", total_bytes=100, completed_bytes=100)
        download_state.write(workspace, state)

        with patch.object(download_state, "kill_worker") as kill:
            models.download_cancel(None, download_id=state.id)

        kill.assert_not_called()
        assert dest.exists(), "a completed download's file must never be deleted"
        env = json_renderer()
        assert env["data"]["status"] == "completed"
        assert env["changed"] is False

    def test_cancel_of_an_unknown_id_is_a_structured_error(self, workspace, json_renderer):
        with pytest.raises(typer.Exit):
            models.download_cancel(None, download_id="nosuchid1234")
        assert json_renderer()["error"]["code"] == "download_not_found"


class TestHumanRendering:
    """Pretty mode writes a table to stdout and no envelope."""

    def test_status_renders_a_table(self, workspace, capsys, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 50)
        state = _state(dest=str(dest), status="downloading", pid=os.getpid(), total_bytes=200)
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)

        out = capsys.readouterr().out
        assert state.id in out
        assert "downloading" in out
        assert "25.0%" in out

    def test_status_renders_an_error(self, workspace, capsys, tmp_path):
        state = _state(dest=str(tmp_path / "gone"), status="failed", error="the server returned HTTP 503")
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)
        assert "503" in capsys.readouterr().out

    def test_empty_downloads_renders_a_message(self, workspace, capsys):
        models.downloads(None)
        assert "No downloads found" in capsys.readouterr().out

    def test_unknown_total_renders_without_crashing(self, workspace, capsys, tmp_path):
        state = _state(dest=str(tmp_path / "m"), status="starting", total_bytes=None)
        download_state.write(workspace, state)

        models.downloads(None)
        assert state.id in capsys.readouterr().out


class TestKillWorker:
    def test_no_pid_is_a_no_op(self):
        assert download_state.kill_worker(None) is False
        assert download_state.kill_worker(0) is False
        assert download_state.kill_worker(-1) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="killpg is POSIX-only")
    def test_posix_signals_the_process_group(self, monkeypatch):
        import signal

        killed = []
        monkeypatch.setattr(download_state, "is_worker_process", lambda *a: True)
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))

        assert download_state.kill_worker(4242, 1.0) is True
        assert killed == [(4242, signal.SIGTERM)]

        assert download_state.kill_worker(4242, 1.0, force=True) is True
        assert killed[-1] == (4242, signal.SIGKILL)

    @pytest.mark.skipif(sys.platform == "win32", reason="killpg is POSIX-only")
    def test_an_already_dead_worker_reports_false(self, monkeypatch):
        monkeypatch.setattr(os, "getpgid", MagicMock(side_effect=ProcessLookupError))
        monkeypatch.setattr("psutil.Process", MagicMock(side_effect=Exception("no such process")))

        assert download_state.kill_worker(4242) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="killpg is POSIX-only")
    def test_a_recycled_pid_is_never_signalled(self, monkeypatch):
        """The worker died and the OS handed its number to something else.
        Signalling it would kill an unrelated process group."""
        killed = []
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
        monkeypatch.setattr("psutil.Process", MagicMock(return_value=MagicMock(create_time=lambda: 9999.0)))

        assert download_state.kill_worker(4242, 1.0) is False
        assert killed == []

    def test_a_pid_with_no_recorded_start_time_falls_back_to_argv(self, monkeypatch):
        stranger = MagicMock(cmdline=lambda: ["/usr/bin/vim", "notes.txt"])
        monkeypatch.setattr("psutil.Process", MagicMock(return_value=stranger))
        assert download_state.is_worker_process(4242, None) is False

        worker = MagicMock(cmdline=lambda: [sys.executable, "-m", "comfy_cli", "model", "_download-worker", "--state"])
        monkeypatch.setattr("psutil.Process", MagicMock(return_value=worker))
        assert download_state.is_worker_process(4242, None) is True


class TestMachineOutputIsParseable:
    """The whole point of the envelope is that `jq` / `json.loads` can read it.

    `ui.display_table` writes to its own Rich console on stdout, so a table
    rendered in JSON mode would be prepended to the envelope and break every
    strict consumer.
    """

    def test_status_emits_only_the_envelope(self, workspace, capsys, tmp_path):
        set_renderer(Renderer(mode=OutputMode.JSON, version="test"))
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 50)
        state = _state(dest=str(dest), status="downloading", pid=os.getpid(), total_bytes=200)
        download_state.write(workspace, state)

        models.download_status(None, download_id=state.id)

        out = capsys.readouterr().out
        assert json.loads(out)["data"]["id"] == state.id

    def test_downloads_emits_only_the_envelope(self, workspace, capsys, tmp_path):
        set_renderer(Renderer(mode=OutputMode.JSON, version="test"))
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 50)
        download_state.write(workspace, _state(dest=str(dest), status="downloading", pid=os.getpid(), total_bytes=200))

        models.downloads(None)

        assert json.loads(capsys.readouterr().out)["data"]["total"] == 1


class TestCancellationReachesTheWorker:
    """SIGTERM alone doesn't cancel: a worker that is still starting up has no
    pid to signal, and one wedged in a syscall can outlive the grace period.
    Either way it must not go on downloading — or resurrect a cancelled record.
    """

    def test_worker_exits_without_downloading_when_the_sentinel_is_already_there(
        self, workspace, monkeypatch, tmp_path
    ):
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), status="starting", pid=None)
        path = download_state.write(workspace, state)
        download_state.request_cancel(download_state.cancel_path(workspace, state.id))

        monkeypatch.setattr(
            models,
            "download_file",
            MagicMock(side_effect=AssertionError("a cancelled download must never transfer bytes")),
        )
        with pytest.raises(typer.Exit) as exc:
            models._download_worker(state_file=str(path))

        assert exc.value.exit_code == 0
        assert download_state.read(workspace, state.id).status == "cancelled"
        assert not dest.exists()

    def test_worker_aborts_mid_transfer_and_clears_the_partial(self, workspace, monkeypatch, tmp_path):
        # aria2 writes through the destination, so that is where its abandoned
        # bytes are; see the httpx counterpart below.
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), status="starting", downloader="aria2", pid=None)
        path = download_state.write(workspace, state)

        def transfer(url, filepath, headers, downloader, progress_callback):
            filepath.write_bytes(b"partial")
            # The cancel lands after the transfer is already under way.
            download_state.request_cancel(download_state.cancel_path(workspace, state.id))
            progress_callback(7, 4096)

        monkeypatch.setattr(models, "download_file", transfer)
        monkeypatch.setattr(download_state, "PROGRESS_THROTTLE_S", 0.0)

        with pytest.raises(typer.Exit) as exc:
            models._download_worker(state_file=str(path))

        assert exc.value.exit_code == 0
        final = download_state.read(workspace, state.id)
        assert (final.status, final.completed_bytes) == ("cancelled", 0)
        assert not dest.exists(), "the partial file must not survive the cancel"

    def test_worker_cancel_on_httpx_leaves_the_destination_alone(self, workspace, monkeypatch, tmp_path):
        """`DownloadCancelled` unwinds out of the httpx transfer *before* the
        rename, and `_download_file_httpx` reclaims its own `.part` on the way —
        so the worker has nothing at `dest` to delete, and whatever is there
        belongs to someone else."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"a neighbour's file")
        state = _state(dest=str(dest), status="starting", pid=None)
        path = download_state.write(workspace, state)

        def transfer(url, filepath, headers, downloader, progress_callback):
            download_state.request_cancel(download_state.cancel_path(workspace, state.id))
            progress_callback(7, 4096)

        monkeypatch.setattr(models, "download_file", transfer)
        monkeypatch.setattr(download_state, "PROGRESS_THROTTLE_S", 0.0)

        with pytest.raises(typer.Exit) as exc:
            models._download_worker(state_file=str(path))

        assert exc.value.exit_code == 0
        assert download_state.read(workspace, state.id).status == "cancelled"
        assert dest.read_bytes() == b"a neighbour's file"

    def test_a_transfer_that_beat_the_cancel_keeps_its_file(self, workspace, json_renderer, monkeypatch, tmp_path):
        """Both sides have to agree on this one, or cancel deletes a model that
        the state file calls `completed`."""
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), status="starting", pid=None)
        path = download_state.write(workspace, state)

        def transfer(url, filepath, headers, downloader, progress_callback):
            filepath.write_bytes(b"done")
            download_state.request_cancel(download_state.cancel_path(workspace, state.id))

        monkeypatch.setattr(models, "download_file", transfer)
        models._download_worker(state_file=str(path))
        assert download_state.read(workspace, state.id).status == "completed"

        with patch.object(download_state, "is_worker_process", return_value=False):
            models.download_cancel(None, download_id=state.id)

        assert dest.exists()
        assert json_renderer()["data"]["status"] == "completed"

    def test_cancel_reclaims_the_part_file_a_killed_worker_left(self, workspace, json_renderer, monkeypatch, tmp_path):
        """The httpx downloader streams into a `.part` sibling, so a SIGKILLed
        worker's gigabytes are *there*, not at `dest`. Cancel has to sweep them or
        it reports success while reclaiming nothing — the by-hand cleanup this
        command exists to spare the user.
        """
        dest = tmp_path / "m.safetensors"
        part = tmp_path / "m.safetensors.a1b2c3d4.part"
        part.write_bytes(b"y" * 3600)
        state = _state(dest=str(dest), status="downloading", pid=None, total_bytes=13000, completed_bytes=3600)
        download_state.write(workspace, state)

        monkeypatch.setattr(download_state, "stop_worker", lambda *_a, **_k: True)
        models.download_cancel(None, download_id=state.id)

        assert not part.exists(), "the killed worker's partial must be reclaimed"
        assert not dest.exists()
        payload = json_renderer()["data"]
        assert (payload["status"], payload["completed_bytes"]) == ("cancelled", 0)

    def test_cancel_reclaims_the_part_of_an_already_failed_record(self, workspace, json_renderer, tmp_path):
        """The realistic ordering: the SIGKILLed worker's *first* `download-status`
        poll persists reconcile's `failed` verdict, so by the time the user runs
        `download-cancel` the record is already terminal. If that short-circuits
        without sweeping, the multi-GB `.part` is unreclaimable — it is invisible
        to `model list` and `model remove`, which only know about `dest`."""
        dest = tmp_path / "m.safetensors"
        part = tmp_path / "m.safetensors.a1b2c3d4.part"
        part.write_bytes(b"y" * 3600)
        state = _state(
            dest=str(dest),
            status="failed",
            error="worker died before the download finished",
            pid=5150,
            total_bytes=13000,
            completed_bytes=3600,
        )
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=False):
            models.download_cancel(None, download_id=state.id)

        assert not part.exists(), "a terminal record's orphaned partial is still the user's disk"
        env = json_renderer()
        assert env["changed"] is True
        assert env["data"]["completed_bytes"] == 0
        assert download_state.read(workspace, state.id).completed_bytes == 0

    def test_cancel_of_a_failed_record_whose_worker_is_alive_sweeps_nothing(self, workspace, json_renderer, tmp_path):
        """A `.part` under an live worker's pen is not ours to delete."""
        dest = tmp_path / "m.safetensors"
        part = tmp_path / "m.safetensors.a1b2c3d4.part"
        part.write_bytes(b"y" * 3600)
        state = _state(dest=str(dest), status="failed", pid=5150, pid_create_time=1.0, total_bytes=13000)
        download_state.write(workspace, state)

        with patch.object(download_state, "is_worker_process", return_value=True):
            with patch("comfy_cli.utils.is_running", return_value=True):
                models.download_cancel(None, download_id=state.id)

        assert part.exists()
        assert json_renderer()["changed"] is False

    def test_cancel_leaves_an_unrelated_neighbour_alone(self, workspace, json_renderer, monkeypatch, tmp_path):
        dest = tmp_path / "m.safetensors"
        neighbour = tmp_path / "m.safetensors.notes.part"
        neighbour.write_bytes(b"a user's own file")
        state = _state(dest=str(dest), status="downloading", pid=None, total_bytes=13000)
        download_state.write(workspace, state)

        monkeypatch.setattr(download_state, "stop_worker", lambda *_a, **_k: True)
        models.download_cancel(None, download_id=state.id)

        assert neighbour.read_bytes() == b"a user's own file"

    def test_cancel_does_not_delete_a_finished_file_when_the_total_was_unknown(
        self, workspace, json_renderer, monkeypatch, tmp_path
    ):
        """The canceller's in-memory copy predates the worker learning the size
        from the response headers; only the file on disk is up to date."""
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), status="downloading", pid=None, total_bytes=None)
        download_state.write(workspace, state)

        # The worker finishes (and records the total) between our read and the kill.
        def stop(_state, **_kwargs):
            done = download_state.read(workspace, state.id)
            dest.write_bytes(b"x" * 4096)
            done.status, done.total_bytes, done.completed_bytes = "completed", 4096, 4096
            download_state.write(workspace, done)
            return True

        monkeypatch.setattr(download_state, "stop_worker", stop)
        models.download_cancel(None, download_id=state.id)

        assert dest.exists(), "cancel must not delete a model the worker had already finished"
        assert json_renderer()["data"]["status"] == "completed"


class TestStateFilePermissions:
    """A resolved url can carry a presigned/SAS query token, so these files are
    secrets on a shared host — and a writable one is an attack surface."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_directory_is_owner_only(self, workspace):
        base = download_state.state_dir(workspace)
        assert base.stat().st_mode & 0o777 == 0o700

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_state_file_is_owner_only(self, workspace):
        path = download_state.write(workspace, _state())
        assert path.stat().st_mode & 0o777 == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
    def test_a_preexisting_loose_directory_is_tightened(self, workspace):
        base = workspace / download_state.STATE_DIRNAME
        base.mkdir(mode=0o777)
        assert download_state.state_dir(workspace).stat().st_mode & 0o777 == 0o700


class TestCorruptStateFiles:
    """A tampered or truncated file must read as *absent*, not construct a
    DownloadState that raises a TypeError deep inside kill_worker/reconcile and
    takes `downloads` down with it."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("pid", "not-a-pid"),
            ("pid", True),
            ("total_bytes", "1024"),
            ("completed_bytes", None),
            ("needs_civitai_auth", "yes"),
            ("error", 42),
            ("dest", ["/tmp/x"]),
            ("pid_create_time", "soon"),
        ],
    )
    def test_a_wrongly_typed_field_reads_as_absent(self, workspace, tmp_path, field, value):
        path = tmp_path / "corrupt.json"
        payload = _state().to_dict()
        payload[field] = value
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert download_state.read_path(path) is None

    def test_a_record_without_kind_reads_as_background(self, workspace, tmp_path):
        """Ticket case 9. Every record written before `kind` existed was a
        detached worker, which is what the dataclass default says."""
        path = tmp_path / "legacy.json"
        payload = _state().to_dict()
        del payload["kind"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        state = download_state.read_path(path)
        assert state is not None
        assert state.kind == "background"
        assert state.is_foreground is False

    @pytest.mark.parametrize("value", ["worker", "", None, 3])
    def test_an_unrecognized_kind_reads_as_the_non_cancellable_one(self, workspace, tmp_path, value):
        """`kind` is the one *tolerant* field: a bad value replaces the field
        rather than rejecting the whole record — and it fails *closed*.

        Rejecting the record is the far more dangerous outcome. A record that
        reads as absent is invisible to the destination-claim scan too, so one
        unrecognized `kind` on a *live* download would un-claim its destination
        and let a second writer into the same file — the exact corruption the
        claim exists to prevent.

        But the substitute cannot be the `background` default, because `kind` is
        what gates a destructive action: `background` reaches `kill_worker`'s
        `os.killpg`. A value we could not parse is no evidence that the pid is a
        detached worker, so it reads as `foreground` — refused by
        `download-cancel`, which is the recoverable way to be wrong.
        """
        path = tmp_path / "odd-kind.json"
        payload = _state().to_dict()
        payload["kind"] = value
        path.write_text(json.dumps(payload), encoding="utf-8")

        state = download_state.read_path(path)
        assert state is not None
        assert state.kind == "foreground"
        assert state.is_foreground is True

    @pytest.mark.parametrize("kind", ["background", "foreground"])
    def test_the_status_row_reports_the_kind(self, kind):
        """`comfy model downloads` is now the only place a foreground download is
        visible, and the two kinds differ in a way a consumer has to act on: a
        live foreground row cannot be cancelled. Without this field the only way
        to find that out is to try `download-cancel` and read the refusal."""
        state = _state(status="downloading")
        state.kind = kind

        assert download_state.status_payload(state)["kind"] == kind

    def test_list_all_skips_a_corrupt_file_instead_of_crashing(self, workspace):
        good = _state()
        download_state.write(workspace, good)
        (download_state.state_dir(workspace) / "bad.json").write_text('{"pid": "nope"}', encoding="utf-8")

        assert [s.id for s in download_state.list_all(workspace)] == [good.id]

    @pytest.mark.parametrize("pid", [-1, 0])
    def test_a_nonpositive_pid_is_screened_before_psutil(self, pid):
        """The field validator only rejects non-ints, so a tampered record can
        carry a negative pid — and `psutil.Process(-1)` raises ValueError, which
        `utils.is_running` does not catch (it catches only NoSuchProcess). Every
        command that reconciles would traceback on one bad file, so screen it
        here the way `is_worker_process`/`kill_worker` already do."""
        state = _state(status="downloading", pid=pid)

        # No `pid_alive` injection: the real liveness helper must never be reached.
        assert download_state.worker_alive(state) is False


class TestWorkerIdentity:
    def test_reconcile_ignores_liveness_when_the_start_time_disagrees(self, tmp_path):
        """A live-but-recycled pid must not pin a dead transfer at `downloading`
        forever — pollers would loop on it indefinitely."""
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(dest=str(dest), status="downloading", pid=4242, pid_create_time=1.0, total_bytes=100)

        with patch("psutil.Process", MagicMock(return_value=MagicMock(create_time=lambda: 9999.0))):
            fresh = download_state.reconcile(state, pid_alive=lambda pid: True)

        assert fresh.status == "failed"

    def test_reconcile_trusts_a_matching_start_time(self, tmp_path):
        dest = tmp_path / "m.safetensors"
        dest.write_bytes(b"x" * 10)
        state = _state(dest=str(dest), status="downloading", pid=4242, pid_create_time=1.0, total_bytes=100)

        with patch("psutil.Process", MagicMock(return_value=MagicMock(create_time=lambda: 1.0))):
            fresh = download_state.reconcile(state, pid_alive=lambda pid: True)

        assert fresh.status == "downloading"
