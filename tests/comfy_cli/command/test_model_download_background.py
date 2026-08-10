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
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer

from comfy_cli import download_state
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
# 1.5 pruning: nothing else ever removes a record, so the state dir would grow
#     without bound in a long-lived workspace.
# ---------------------------------------------------------------------------


def _backdate(path: Path, days_ago: float) -> None:
    """Push ``path``'s mtime into the past — the only age the junk sweep has."""
    when = time.time() - days_ago * 86400
    os.utime(path, (when, when))


def _persist_aged(
    workspace,
    *,
    status: str = "completed",
    days_ago: float = 8.0,
    started_days_ago: float | None = None,
) -> download_state.DownloadState:
    """Persist one record whose timestamps are backdated on disk.

    A plain ``write`` always stamps ``updated_at`` with *now*, so the ages this
    module's prune window keys off have to be rewritten into the JSON directly.
    """
    state = _state(dest=str(Path(workspace) / "m.safetensors"), status=status)
    path = download_state.write(workspace, state)

    now = datetime.now(timezone.utc)
    state.updated_at = (now - timedelta(days=days_ago)).isoformat(timespec="seconds")
    started = days_ago if started_days_ago is None else started_days_ago
    state.started_at = (now - timedelta(days=started)).isoformat(timespec="seconds")

    data = json.loads(path.read_text())
    data["updated_at"] = state.updated_at
    data["started_at"] = state.started_at
    path.write_text(json.dumps(data))
    return state


class TestPrune:
    def test_removes_an_old_terminal_record_and_its_sidecars(self, workspace):
        state = _persist_aged(workspace, status="completed", days_ago=8)
        log = download_state.log_path(workspace, state.id)
        cancel = download_state.cancel_path(workspace, state.id)
        log.write_text("worker output\n")
        cancel.touch()

        assert download_state.prune(workspace, keep=0) == 1

        assert not download_state.state_path(workspace, state.id).exists()
        assert not log.exists()
        assert not cancel.exists()
        assert download_state.list_all(workspace) == []

    @pytest.mark.parametrize("status", ["starting", "downloading"])
    def test_active_records_survive_the_terminal_window(self, workspace, status):
        """An in-flight record has no business being deleted, and prune reads the
        on-disk status rather than reconciling — a worker that died mid-transfer
        is collected on a later sweep, once a poll verb has marked it failed."""
        state = _persist_aged(workspace, status=status, days_ago=20)

        assert download_state.prune(workspace, keep=0) == 0
        assert download_state.state_path(workspace, state.id).exists()

    @pytest.mark.parametrize("status", ["starting", "downloading"])
    def test_a_long_dead_active_record_is_eventually_collected(self, workspace, status):
        """Otherwise the directory is not bounded at all: a workflow that only
        ever submits never runs the poll verb that would reconcile these to a
        terminal status, so they would accumulate forever. A worker writes at
        least every PROGRESS_THROTTLE_S, so nothing live is this stale."""
        state = _persist_aged(workspace, status=status, days_ago=40)

        assert download_state.prune(workspace, keep=0) == 1
        assert not download_state.state_path(workspace, state.id).exists()

    def test_an_unrecognised_status_ages_out_on_the_normal_window(self, workspace):
        """Neither terminal nor active: junk, and it must not pin a file forever."""
        state = _persist_aged(workspace, status="wat", days_ago=8)

        assert download_state.prune(workspace, keep=0) == 1
        assert not download_state.state_path(workspace, state.id).exists()

    def test_deletion_targets_come_from_the_filename_not_the_id_field(self, workspace):
        """A record's `id` is untrusted content read back off disk. Rebuilding
        the unlink targets from it lets a corrupt or hand-copied `<a>.json`
        carrying `"id": "<b>"` delete *b*'s state and cancel sentinel — possibly
        an in-flight download's — while surviving every sweep itself."""
        victim = _state(status="downloading")
        download_state.write(workspace, victim)
        victim_cancel = download_state.cancel_path(workspace, victim.id)
        victim_cancel.touch()

        impostor = _persist_aged(workspace, status="completed", days_ago=8)
        impostor_path = download_state.state_path(workspace, impostor.id)
        data = json.loads(impostor_path.read_text())
        data["id"] = victim.id
        impostor_path.write_text(json.dumps(data))

        assert download_state.prune(workspace, keep=0) == 1

        assert not impostor_path.exists(), "the file actually enumerated is the one that goes"
        assert download_state.state_path(workspace, victim.id).exists()
        assert victim_cancel.exists()

    def test_sidecars_survive_a_state_file_that_could_not_be_removed(self, workspace, monkeypatch):
        """A record still listed by `comfy model downloads` must keep its
        `<id>.log` — the only diagnostic for why that download failed."""
        state = _persist_aged(workspace, status="failed", days_ago=8)
        log = download_state.log_path(workspace, state.id)
        log.write_text("traceback\n")
        real_unlink = Path.unlink

        def selective(self, missing_ok=False):
            if self.suffix == ".json":
                raise OSError("read-only file system")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", selective)

        assert download_state.prune(workspace, keep=0) == 0
        assert download_state.state_path(workspace, state.id).exists()
        assert log.read_text() == "traceback\n"

    def test_keep_floor_wins_over_age(self, workspace):
        # started_at descending, so "newest" is unambiguous.
        states = [_persist_aged(workspace, days_ago=30 + i, started_days_ago=30 + i) for i in range(5)]

        assert download_state.prune(workspace, keep=3) == 2

        survivors = {s.id for s in download_state.list_all(workspace)}
        assert survivors == {s.id for s in states[:3]}

    def test_a_young_terminal_record_survives_the_default_window(self, workspace):
        state = _persist_aged(workspace, status="completed", days_ago=1)

        assert download_state.prune(workspace, keep=0) == 0
        assert download_state.state_path(workspace, state.id).exists()

    def test_unparsable_timestamps_are_treated_as_old(self, workspace):
        """A terminal record that carries no usable age is junk once it is past
        the keep floor — there is nothing left that could make it recent."""
        state = _persist_aged(workspace, status="failed")
        path = download_state.state_path(workspace, state.id)
        data = json.loads(path.read_text())
        data["updated_at"] = data["started_at"] = ""
        path.write_text(json.dumps(data))

        assert download_state.prune(workspace, keep=0) == 1
        assert not path.exists()

    def test_an_unlink_failure_is_not_fatal(self, workspace, monkeypatch):
        state = _persist_aged(workspace, days_ago=30)

        def boom(self, missing_ok=False):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "unlink", boom)

        # Best effort: the sweep neither raises nor claims a record it failed to
        # remove.
        assert download_state.prune(workspace, keep=0) == 0
        assert download_state.state_path(workspace, state.id).exists()

    def test_prune_leaves_a_workspace_under_the_floor_alone(self, workspace):
        states = [_persist_aged(workspace, days_ago=30 + i, started_days_ago=30 + i) for i in range(3)]

        assert download_state.prune(workspace, keep=len(states)) == 0
        assert len(download_state.list_all(workspace)) == len(states)


class TestPruneCollectsJunk:
    """`list_all` can only see what parses, so exactly the debris most likely to
    accumulate — unreadable records, tmp files a killed writer leaked, sidecars
    whose record is gone — is what a record-only sweep would leak forever. None
    of it has a record timestamp, so mtime stands in for one.
    """

    def _junk(self, workspace, name: str, *, body: bytes = b"junk", days_ago: float) -> Path:
        path = download_state.state_dir(workspace) / name
        path.write_bytes(body)
        _backdate(path, days_ago)
        return path

    @pytest.mark.parametrize("body", [b"{not json", b"\xff\xfe not utf-8"], ids=["bad-json", "bad-utf8"])
    def test_an_old_unreadable_record_is_collected_with_its_sidecars(self, workspace, body):
        path = self._junk(workspace, "deadbeefcafe.json", body=body, days_ago=8)
        log = self._junk(workspace, "deadbeefcafe.log", days_ago=8)

        assert download_state.prune(workspace, keep=0) == 1
        assert not path.exists()
        assert not log.exists()

    def test_a_fresh_unreadable_record_is_left_alone(self, workspace):
        """It could be a file something is still mid-way through creating."""
        path = self._junk(workspace, "deadbeefcafe.json", body=b"{not json", days_ago=0)

        assert download_state.prune(workspace, keep=0) == 0
        assert path.exists()

    def test_an_old_leaked_tmp_file_is_collected_but_a_fresh_one_is_not(self, workspace):
        stale = self._junk(workspace, "deadbeefcafe.1234.ab12cd34.tmp", days_ago=8)
        live = self._junk(workspace, "cafedeadbeef.1234.ef56ab78.tmp", days_ago=0)

        # tmp files are not records, so they never count toward the return value.
        assert download_state.prune(workspace, keep=0) == 0
        assert not stale.exists()
        assert live.exists()

    def test_orphaned_sidecars_are_collected_but_paired_ones_are_kept(self, workspace):
        kept = _state(status="downloading")
        download_state.write(workspace, kept)
        paired_log = download_state.log_path(workspace, kept.id)
        paired_log.write_text("still in flight\n")
        _backdate(paired_log, 8)

        orphan_log = self._junk(workspace, "deadbeefcafe.log", days_ago=8)
        orphan_cancel = self._junk(workspace, "deadbeefcafe.cancel", body=b"", days_ago=8)

        assert download_state.prune(workspace, keep=0) == 0
        assert not orphan_log.exists()
        assert not orphan_cancel.exists()
        assert paired_log.exists()

    def test_an_unreadable_record_never_fails_the_sweep_or_the_listing(self, workspace):
        """`read_text` raises UnicodeDecodeError — a ValueError, not an OSError —
        on invalid UTF-8, and the submit path's sweep must survive it."""
        good = _persist_aged(workspace, status="completed", days_ago=1)
        self._junk(workspace, "deadbeefcafe.json", body=b"\xff\xfe", days_ago=0)

        assert download_state.read_path(download_state.state_dir(workspace) / "deadbeefcafe.json") is None
        assert [s.id for s in download_state.list_all(workspace)] == [good.id]
        assert download_state.prune(workspace, keep=0) == 0


class TestPruneOnSubmit:
    """`--background` is the one command that grows the state dir, so it sweeps."""

    def _submit(self, workspace, monkeypatch):
        monkeypatch.setattr(models, "_spawn_download_worker", lambda state_file, log_file: 31337)
        models.download(
            None,
            url="https://example.com/m.safetensors",
            relative_path="models/loras",
            filename="m.safetensors",
            background=True,
        )

    def test_submit_prunes_records_past_the_keep_floor(self, workspace, monkeypatch, json_renderer):
        # One more than the floor, so exactly the oldest falls outside it once
        # the submit's own record is written.
        seeded = [
            _persist_aged(workspace, days_ago=30 + i, started_days_ago=30 + i)
            for i in range(download_state.PRUNE_KEEP + 1)
        ]

        self._submit(workspace, monkeypatch)
        env = json_renderer()
        assert env["ok"] is True

        survivors = {s.id for s in download_state.list_all(workspace)}
        assert seeded[-1].id not in survivors  # the oldest went
        assert seeded[0].id in survivors  # recent history is untouched
        assert len(survivors) == download_state.PRUNE_KEEP

    def test_a_pruning_failure_never_fails_the_submit(self, workspace, monkeypatch, json_renderer):
        def boom(*args, **kwargs):
            raise OSError("state dir is unreadable")

        monkeypatch.setattr(download_state, "prune", boom)
        self._submit(workspace, monkeypatch)

        env = json_renderer()
        assert env["ok"] is True
        assert env["data"]["status"] == "starting"


class TestDownloadsPruneFlag:
    def _seed(self, workspace) -> list[download_state.DownloadState]:
        """One over the keep floor plus a fresh record — so exactly the two
        oldest are prunable, and everything else is recent history to retain."""
        old = [
            _persist_aged(workspace, days_ago=30 + i, started_days_ago=30 + i)
            for i in range(download_state.PRUNE_KEEP + 1)
        ]
        fresh = _persist_aged(workspace, days_ago=1, started_days_ago=1)
        return [fresh, *old]

    def test_prune_flag_removes_and_reports(self, workspace, json_renderer):
        records = self._seed(workspace)

        models.downloads(None, prune=True)
        env = json_renderer()

        assert env["command"] == "model downloads"
        assert env["changed"] is True
        assert env["data"]["pruned"] == 2
        assert env["data"]["total"] == download_state.PRUNE_KEEP
        listed = {row["id"] for row in env["data"]["downloads"]}
        assert listed == {r.id for r in records[:-2]}

    def test_without_the_flag_nothing_is_removed(self, workspace, json_renderer):
        records = self._seed(workspace)

        models.downloads(None)
        env = json_renderer()

        assert env["data"]["total"] == len(records)
        # The envelope shape agents already parse is unchanged without the flag.
        assert "pruned" not in env["data"]
        assert "changed" not in env

    def test_prune_flag_on_an_empty_workspace(self, workspace, json_renderer):
        models.downloads(None, prune=True)
        env = json_renderer()
        assert env["data"] == {"total": 0, "downloads": [], "pruned": 0}
        assert env["changed"] is False

    def test_a_reconciliation_write_back_counts_as_changed(self, workspace, json_renderer, tmp_path):
        """Nothing was deleted, but the listing still rewrote a record on disk —
        reporting `changed: false` would tell an agent this call was a no-op."""
        state = _state(dest=str(tmp_path / "missing.bin"), status="downloading", pid=424242, total_bytes=10)
        download_state.write(workspace, state)

        models.downloads(None, prune=True)
        env = json_renderer()

        assert env["data"]["pruned"] == 0
        assert env["changed"] is True
        assert download_state.read(workspace, state.id).status == "failed"

    def test_a_failed_sweep_is_reported_rather_than_read_as_a_no_op(self, workspace, json_renderer, monkeypatch):
        """`prune` swallows OSError per file and on the directory scan, so an
        escape is unexpected — and may have deleted records before it happened.
        The user asked for a destructive operation; `pruned: 0, changed: false`
        would be a lie about it."""

        def boom(*args, **kwargs):
            raise OSError("state dir is unreadable")

        monkeypatch.setattr(download_state, "prune", boom)

        models.downloads(None, prune=True)
        env = json_renderer()

        assert env["ok"] is True
        assert env["data"]["pruned"] == 0
        assert env["changed"] is True


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
        `envelope/1` error (BE-4217), not a raw `DownloadException`."""
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
        # The foreground call site does not pass a progress callback.
        assert "progress_callback" not in calls[0][1]
        assert download_state.list_all(workspace) == []


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
        assert "No background downloads" in capsys.readouterr().out

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

    def test_worker_stops_when_its_record_is_pruned_out_from_under_it(self, workspace, monkeypatch, tmp_path):
        """`download-cancel` writes a terminal `cancelled` record and warns that
        the worker may still be running when the kill fails; such a worker writes
        nothing, so the record ages out and `prune` takes it — sentinel included.
        Without treating a vanished record as a cancellation, a worker that woke
        after that sweep would no longer see the cancel and would resume writing
        to `dest`, untracked and uncancellable.
        """
        dest = tmp_path / "m.safetensors"
        state = _state(dest=str(dest), status="starting", downloader="aria2", pid=None)
        path = download_state.write(workspace, state)

        def transfer(url, filepath, headers, downloader, progress_callback):
            filepath.write_bytes(b"partial")
            path.unlink()  # the sweep collects the record mid-transfer
            progress_callback(7, 4096)

        monkeypatch.setattr(models, "download_file", transfer)
        monkeypatch.setattr(download_state, "PROGRESS_THROTTLE_S", 0.0)

        with pytest.raises(typer.Exit) as exc:
            models._download_worker(state_file=str(path))

        assert exc.value.exit_code == 0
        assert not dest.exists(), "the abandoned partial must not survive"
        assert not path.exists(), "a pruned record must not be resurrected by the write-back"

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

    def test_list_all_skips_a_corrupt_file_instead_of_crashing(self, workspace):
        good = _state()
        download_state.write(workspace, good)
        (download_state.state_dir(workspace) / "bad.json").write_text('{"pid": "nope"}', encoding="utf-8")

        assert [s.id for s in download_state.list_all(workspace)] == [good.id]


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
