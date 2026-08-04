"""Unit tests for `comfy jobs` — ls, status, watch.

The WebSocket and HTTP calls are mocked. The live round-trip against a real
ComfyUI server is a separate manual demo step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import typer

from comfy_cli.command import jobs as jobs_mod

# ---------------------------------------------------------------------------
# Pure data shaping
# ---------------------------------------------------------------------------


_HISTORY_FIXTURE = {
    "abc-1": {
        "prompt": [
            0,
            "abc-1",
            {"1": {"class_type": "KSampler", "inputs": {}}, "2": {"class_type": "VAEDecode", "inputs": {}}},
        ],
        "status": {"completed": True, "messages": []},
        "outputs": {
            "9": {
                "images": [
                    {"filename": "out.png", "subfolder": "", "type": "output"},
                    {"filename": "out_1.png", "subfolder": "", "type": "output"},
                ]
            },
        },
    },
    "abc-2": {
        "prompt": [0, "abc-2", {"1": {"class_type": "X", "inputs": {}}}],
        "status": {"completed": False, "messages": [["execution_error", {"node_id": "1"}]]},
        "outputs": {},
    },
}


def test_gather_jobs_combines_queue_and_history(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url, timeout=10.0):
        if url.endswith("/queue"):
            return {
                "queue_running": [[0, "running-id", {"a": {}, "b": {}, "c": {}}, {}, {}]],
                "queue_pending": [[1, "pending-id", {"a": {}}, {}, {}]],
            }
        if url.endswith("/history"):
            return _HISTORY_FIXTURE
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    rows = jobs_mod._gather_jobs("h", 8188, limit=10)

    assert any(r.prompt_id == "running-id" and r.status == "running" for r in rows)
    assert any(r.prompt_id == "pending-id" and r.status == "pending" and r.queue_position == 1 for r in rows)
    completed = [r for r in rows if r.prompt_id == "abc-1"]
    assert completed and completed[0].status == "completed"
    assert completed[0].outputs == 2  # two images
    errored = [r for r in rows if r.prompt_id == "abc-2"]
    assert errored and errored[0].status == "error"


def test_snapshot_finds_running_in_queue(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url, timeout=10.0):
        if url.endswith("/queue"):
            return {
                "queue_running": [[0, "live-id", {"a": {}, "b": {}}, {}, {}]],
                "queue_pending": [],
            }
        if url.endswith("/history/live-id"):
            return {}
        raise AssertionError(url)

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    snap = jobs_mod._snapshot("h", 8188, "live-id")
    assert snap is not None
    assert snap["status"] == "running"
    assert snap["workflow_size"] == 2


def test_snapshot_finds_completed_in_history(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url, timeout=10.0):
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        if url.endswith("/history/abc-1"):
            return {"abc-1": _HISTORY_FIXTURE["abc-1"]}
        raise AssertionError(url)

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    snap = jobs_mod._snapshot("h", 8188, "abc-1")
    assert snap is not None
    assert snap["status"] == "completed"
    assert len(snap["outputs"]) == 2
    assert "filename=out.png" in snap["outputs"][0]


def test_snapshot_missing_returns_none(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url, timeout=10.0):
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        return {}

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    assert jobs_mod._snapshot("h", 8188, "ghost") is None


class TestLocalSnapshotGroupedOutputs:
    """Local-path parity with the cloud snapshot: `_snapshot` exposes the
    node-keyed /history outputs grouped by producing node and — when the
    state file carries a compose item_map — by blueprint foreach item."""

    _HISTORY_BODY = {
        "status": {"completed": True, "messages": []},
        "outputs": {
            "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "12": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
        },
    }
    URL_A = "http://h:8188/view?filename=a.png&subfolder=&type=output"
    URL_V = "http://h:8188/view?filename=v.mp4&subfolder=&type=output"

    def _patch_history(self, monkeypatch, prompt_id):
        def fake_get(url, timeout=10.0):
            if url.endswith("/queue"):
                return {"queue_running": [], "queue_pending": []}
            if url.endswith(f"/history/{prompt_id}"):
                return {prompt_id: self._HISTORY_BODY}
            raise AssertionError(url)

        monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)

    def test_history_snapshot_groups_by_node_and_item(self, monkeypatch):
        from comfy_cli import jobs_state

        state = jobs_state.new(prompt_id="grp-local", client_id="c", workflow="w", where="local", host="h", port=8188)
        state.item_map = {
            "s1": {"nodes": ["9"], "save_node": "9", "prefix": "outputs/s1"},
            "s2": {"nodes": ["12"], "save_node": "12", "prefix": "outputs/s2"},
        }
        jobs_state.write(state)
        self._patch_history(monkeypatch, "grp-local")

        snap = jobs_mod._snapshot("h", 8188, "grp-local")
        assert snap is not None
        assert snap["status"] == "completed"
        assert snap["outputs"] == [self.URL_A, self.URL_V]  # flat list untouched
        assert snap["outputs_by_node"] == {"9": [self.URL_A], "12": [self.URL_V]}
        assert snap["outputs_by_item"] == {"s1": [self.URL_A], "s2": [self.URL_V]}

    def test_history_snapshot_without_item_map_emits_empty_by_item(self, monkeypatch):
        self._patch_history(monkeypatch, "grp-nomap")

        snap = jobs_mod._snapshot("h", 8188, "grp-nomap")
        assert snap is not None
        assert snap["outputs_by_node"] == {"9": [self.URL_A], "12": [self.URL_V]}
        assert snap["outputs_by_item"] == {}

    def test_queue_snapshot_keeps_empty_groupings(self, monkeypatch):
        """In-flight jobs have nothing to group — keys present, empty dicts
        (same shape as the cloud snapshot)."""

        def fake_get(url, timeout=10.0):
            if url.endswith("/queue"):
                return {"queue_running": [[0, "grp-live", {"a": {}}, {}, {}]], "queue_pending": []}
            raise AssertionError(url)

        monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
        snap = jobs_mod._snapshot("h", 8188, "grp-live")
        assert snap is not None
        assert snap["status"] == "running"
        assert snap["outputs_by_node"] == {}
        assert snap["outputs_by_item"] == {}


def test_safe_queue_entry_handles_short_rows():
    assert jobs_mod._safe_queue_entry([0, "id", {"node": {}}]) == ("id", {"node": {}})
    assert jobs_mod._safe_queue_entry([])[0] == "?"
    assert jobs_mod._safe_queue_entry("not-a-list")[0] == "?"


# ---------------------------------------------------------------------------
# CLI integration — error envelope when no server
# ---------------------------------------------------------------------------


def _run(args, env=None):
    cli_env = os.environ.copy()
    cli_env["NO_COLOR"] = "1"
    # Pin subprocess routing to local so tests don't depend on whatever
    # `where_default` the developer has persisted in their real config.
    # Individual tests can still override via env={"COMFY_WHERE": "cloud"}.
    cli_env.setdefault("COMFY_WHERE", "local")
    if env:
        cli_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=cli_env,
        check=False,
    )


def _last_json(stdout: str) -> dict:
    last = [line for line in stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


def test_jobs_ls_no_server_degrades_to_local_state():
    """When the server is unreachable, ``jobs ls`` no longer errors — it
    falls back to the local state-dir view so async submits remain visible.
    The user can pass ``--local-only`` to skip the server probe entirely.
    """
    res = _run(["--json", "jobs", "ls", "--local-only", "--host", "127.0.0.1", "--port", "65431"])
    assert res.returncode == 0
    env = _last_json(res.stdout)
    assert env["ok"] is True
    # Count may be 0 (clean machine) or more (state dir has files); the
    # contract is that we got a successful envelope shape.
    assert "jobs" in env["data"]


def test_jobs_status_no_server_emits_error_envelope():
    res = _run(["--json", "jobs", "status", "some-id", "--host", "127.0.0.1", "--port", "65431"])
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "server_not_running"


# ---------------------------------------------------------------------------
# `jobs status` with the server down — fall back to the on-disk state file
# ---------------------------------------------------------------------------


def _invoke_status(prompt_id: str, *extra: str):
    """Run `jobs status <id>` in-process with an NDJSON renderer installed.

    Returns the CliRunner result; parse its stdout with ``_last_json``.
    """
    from typer.testing import CliRunner

    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs status"))
    return CliRunner().invoke(jobs_mod.app, ["status", prompt_id, "--where", "local", *extra])


def test_jobs_status_server_down_no_state_file_keeps_bare_error(monkeypatch: pytest.MonkeyPatch):
    """No state file for the prompt -> the pre-existing `server_not_running`
    envelope, unchanged (backward compatibility for untracked prompts)."""
    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)

    result = _invoke_status("untracked-id", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 1, result.output
    env = _last_json(result.stdout)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "server_not_running"
    assert err["message"] == "ComfyUI not running on 127.0.0.1:65431"
    assert err["hint"] == "run: comfy launch"
    assert err["details"] == {"host": "127.0.0.1", "port": 65431}


def test_jobs_status_server_down_non_terminal_state_attributes_the_death(monkeypatch: pytest.MonkeyPatch):
    """A job recorded as `running` when the server was last seen: same error
    code, but the message/details say what the job was doing."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    st = jobs_state.new(prompt_id="dead-run", client_id="c", workflow="/tmp/wf.json", where="local")
    st.status = "running"
    jobs_state.write(st)

    result = _invoke_status("dead-run", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 1, result.output
    env = _last_json(result.stdout)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "server_not_running"
    assert "dead-run" in err["message"]
    assert "was 'running'" in err["message"]
    assert "out-of-memory" in err["message"]
    details = err["details"]
    assert details["last_known_status"] == "running"
    assert details["prompt_id"] == "dead-run"
    assert details["workflow"] == "/tmp/wf.json"
    assert details["submitted_at"] == st.submitted_at
    assert details["updated_at"] == st.updated_at


def test_jobs_status_server_down_terminal_state_emits_ok_envelope(monkeypatch: pytest.MonkeyPatch):
    """A job that completed before the server stopped is a normal result —
    OK envelope sourced from the state file, exit 0."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    st = jobs_state.new(prompt_id="done-run", client_id="c", workflow="/tmp/wf.json", where="local")
    st.status = "completed"
    st.outputs = ["http://127.0.0.1:8188/view?filename=out.png"]
    jobs_state.write(st)

    result = _invoke_status("done-run", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 0, result.output
    env = _last_json(result.stdout)
    assert env["ok"] is True
    data = env["data"]
    assert data["prompt_id"] == "done-run"
    assert data["status"] == "completed"
    assert data["server_running"] is False
    assert data["source"] == "state_file"
    assert data["outputs"] == ["http://127.0.0.1:8188/view?filename=out.png"]
    assert data["error"] is None
    assert data["workflow"] == "/tmp/wf.json"
    assert data["submitted_at"] == st.submitted_at


def test_jobs_status_server_up_still_uses_live_snapshot(monkeypatch: pytest.MonkeyPatch):
    """Server up *and it has a record for the prompt*: the live `/queue`
    `/history` snapshot wins — the state file is not consulted and the payload
    carries no state-file markers.

    The state file is only a fallback, never an override: it is read when the
    live server has nothing to say about the prompt (see the
    `TestStatusServerUpNoRecord` cases below), not when it does.
    """
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: True)
    monkeypatch.setattr(
        jobs_mod,
        "_snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "running", "outputs": [], "host": h, "port": p},
    )
    # A stale terminal state file must NOT shadow the live answer.
    st = jobs_state.new(prompt_id="live-run", client_id="c", workflow="/tmp/wf.json", where="local")
    st.status = "completed"
    jobs_state.write(st)

    reads: list[str] = []
    real_read = jobs_state.read

    def spy_read(pid):
        reads.append(pid)
        return real_read(pid)

    monkeypatch.setattr(jobs_state, "read", spy_read)

    result = _invoke_status("live-run", "--host", "127.0.0.1", "--port", "8188")
    assert result.exit_code == 0, result.output
    data = _last_json(result.stdout)["data"]
    assert data["status"] == "running"
    assert "source" not in data
    assert "server_running" not in data
    # The live snapshot answered, so the fallback never ran.
    assert reads == []


def test_jobs_status_server_down_surfaces_server_died_verdict(monkeypatch: pytest.MonkeyPatch):
    """The composite the epic exists for: the watcher recorded `server_died`,
    the server is still down, and `jobs status` hands that verdict back.

    The two halves — a terminal state file being emitted, and `server_died`
    being written — are each covered elsewhere; this pins the seam between
    them, which is the only thing a caller actually sees.
    """
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    st = jobs_state.new(prompt_id="oom-run", client_id="c", workflow="/tmp/oom.json", where="local")
    st.status = "error"
    st.error = {
        "code": "server_died",
        "message": "ComfyUI server 127.0.0.1:8188 stopped answering while job oom-run was 'running'.",
        "details": {"host": "127.0.0.1", "port": 8188, "last_status": "running"},
    }
    jobs_state.write(st)

    result = _invoke_status("oom-run", "--host", "127.0.0.1", "--port", "8188")
    assert result.exit_code == 0, result.output
    env = _last_json(result.stdout)
    assert env["ok"] is True
    data = env["data"]
    assert data["status"] == "error"
    assert data["error"]["code"] == "server_died"
    assert data["source"] == "state_file"
    assert data["server_running"] is False
    assert data["prompt_id"] == "oom-run"
    assert data["workflow"] == "/tmp/oom.json"


class TestStatusServerUpNoRecord:
    """Server UP but neither `/queue` nor `/history` knows the prompt.

    This is what the documented crash recovery ("relaunch, then check")
    produces: a fresh ComfyUI answers the port with an empty history, so the
    live snapshot comes back None. Before BE-4749 that unconditionally emitted
    `prompt_not_found` and discarded the `server_died` verdict already sitting
    on disk.
    """

    @staticmethod
    def _server_up_with_no_record(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: True)
        monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: None)

    def test_terminal_state_file_wins_over_prompt_not_found(self, monkeypatch: pytest.MonkeyPatch):
        """Terminal record -> emit it as a normal result, exit 0. This is the
        relaunch-then-check path that used to throw the attribution away."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(prompt_id="relaunched-run", client_id="c", workflow="/tmp/oom.json", where="local")
        st.status = "error"
        st.error = {"code": "server_died", "message": "the server died under this job", "details": {}}
        jobs_state.write(st)

        result = _invoke_status("relaunched-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 0, result.output
        env = _last_json(result.stdout)
        assert env["ok"] is True
        data = env["data"]
        assert data["status"] == "error"
        assert data["error"]["code"] == "server_died"
        assert data["source"] == "state_file"
        # The distinguishing field: the port answers, it just has no record.
        assert data["server_running"] is True
        assert data["prompt_id"] == "relaunched-run"
        assert data["workflow"] == "/tmp/oom.json"
        assert data["submitted_at"] == st.submitted_at

    def test_terminal_completed_state_file_is_also_returned(self, monkeypatch: pytest.MonkeyPatch):
        """Not just failures: a completed job pruned from `/history` still has
        its outputs on disk, and they are worth more than `prompt_not_found`."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(prompt_id="pruned-run", client_id="c", workflow="/tmp/wf.json", where="local")
        st.status = "completed"
        st.outputs = ["http://127.0.0.1:8188/view?filename=out.png"]
        jobs_state.write(st)

        result = _invoke_status("pruned-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 0, result.output
        data = _last_json(result.stdout)["data"]
        assert data["status"] == "completed"
        assert data["outputs"] == ["http://127.0.0.1:8188/view?filename=out.png"]
        assert data["error"] is None
        assert data["source"] == "state_file"
        assert data["server_running"] is True

    def test_non_terminal_state_file_keeps_the_code_but_enriches_details(self, monkeypatch: pytest.MonkeyPatch):
        """Non-terminal record -> still `prompt_not_found` (callers key on the
        code) but carrying the last-known state, like the server-down twin."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(prompt_id="inflight-run", client_id="c", workflow="/tmp/wf.json", where="local")
        st.status = "running"
        jobs_state.write(st)

        result = _invoke_status("inflight-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        env = _last_json(result.stdout)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "prompt_not_found"
        assert "inflight-run" in err["message"]
        assert "'running'" in err["message"]
        details = err["details"]
        assert details["prompt_id"] == "inflight-run"
        assert details["last_known_status"] == "running"
        assert details["workflow"] == "/tmp/wf.json"
        assert details["submitted_at"] == st.submitted_at
        assert details["updated_at"] == st.updated_at

    def test_no_state_file_keeps_the_bare_envelope(self, monkeypatch: pytest.MonkeyPatch):
        """No record anywhere -> today's envelope, byte for byte. An untracked
        prompt id must not start reporting on jobs that never existed."""
        self._server_up_with_no_record(monkeypatch)

        result = _invoke_status("never-existed", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        env = _last_json(result.stdout)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "prompt_not_found"
        assert err["message"] == "No prompt with id 'never-existed' on 127.0.0.1:8188."
        assert err["hint"] == "check `comfy jobs ls`; very old prompts may have been pruned from /history"
        assert err["details"] == {"prompt_id": "never-existed", "host": "127.0.0.1", "port": 8188}


# ---------------------------------------------------------------------------
# `jobs ls --orphaned` — surface watcher_crashed jobs for cleanup
# ---------------------------------------------------------------------------


def _write_state(tmp_dir: Path, prompt_id: str, **fields) -> None:
    """Helper: write a state file shaped like jobs_state.JobState."""
    base = {
        "prompt_id": prompt_id,
        "client_id": "c-" + prompt_id,
        "workflow": f"/tmp/{prompt_id}.json",
        "where": "local",
        "host": "127.0.0.1",
        "port": 8188,
        "base_url": None,
        "submitted_at": "2026-05-19T00:00:00+00:00",
        "updated_at": "2026-05-19T00:00:00+00:00",
        "completed_at": None,
        "status": "queued",
        "outputs": [],
        "error": None,
        "watcher_pid": None,
    }
    base.update(fields)
    (tmp_dir / f"{prompt_id}.json").write_text(json.dumps(base))


def test_orphaned_flag_filters_to_watcher_crashed(monkeypatch):
    """``jobs ls --orphaned`` shows only jobs whose state file records a
    crashed/reaped watcher. Regular ``jobs ls`` includes them alongside
    everything else."""
    # The autouse ``_isolate_jobs_state_dir`` from conftest already
    # repointed ``jobs_state.state_dir`` at a per-test tmp dir — write
    # state files into whatever it returns.
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()

    _write_state(state_dir, "healthy-completed", status="completed")
    _write_state(
        state_dir,
        "orphan-crashed",
        status="error",
        error={
            "code": "watcher_crashed",
            "message": "Background watcher (pid 99999) is no longer running.",
            "hint": "re-submit the workflow, or check `comfy jobs status <id>`",
        },
    )
    _write_state(state_dir, "other-error", status="error", error={"code": "prompt_rejected", "message": "..."})

    all_rows = jobs_mod._gather_local_state_files(limit=100)
    ids = {r.prompt_id for r in all_rows}
    assert {"healthy-completed", "orphan-crashed", "other-error"} <= ids

    orphans = jobs_mod._gather_local_state_files(limit=100, orphaned_only=True)
    orphan_ids = {r.prompt_id for r in orphans}
    assert orphan_ids == {"orphan-crashed"}, f"--orphaned should select only watcher_crashed rows; got {orphan_ids}"


def _command_flags(*path: str) -> list[str]:
    """Flags exposed for a command path via the machine-readable help contract.

    This is the surface agents actually consume (``comfy --help-json``), and it
    is render-independent — unlike scraping the rich-formatted ``--help`` text,
    whose wrapping/styling varies with the CI terminal and silently hid flags.
    """
    from comfy_cli.cmdline import app
    from comfy_cli.help_json import build_help_json

    node: dict = {"subcommands": build_help_json(app)["commands"]}
    for part in path:
        node = node["subcommands"][part]
    return [flag for param in node.get("params", []) for flag in (param.get("flags") or [])]


def test_orphaned_flag_visible_in_help():
    """The flag must be documented on `jobs ls` so agents can
    discover it without reading source."""
    assert "--orphaned" in _command_flags("jobs", "ls")


def test_all_flag_visible_in_help():
    """``--all`` is the escape hatch back to the union view — agents must be
    able to discover it from `comfy --help-json`."""
    assert "--all" in _command_flags("jobs", "ls")


# ---------------------------------------------------------------------------
# `jobs ls` state-file rows are scoped to the resolved --where target
# ---------------------------------------------------------------------------


def _seed_three_targets() -> Path:
    """Seed the (already isolated) state dir with one local, one cloud and
    one legacy (``where`` present but null) job. Returns the dir."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(state_dir, "job-local", where="local", status="completed")
    _write_state(state_dir, "job-cloud", where="cloud", status="completed")
    # Legacy files predate the cloud target. ``jobs_state.read`` requires the
    # key to be present (it's a non-defaulted dataclass field), so the
    # reachable legacy shape is a null/empty value, not an absent one — those
    # must read as "local".
    _write_state(state_dir, "job-legacy", where=None, status="completed")
    return state_dir


def test_gather_local_state_files_filters_by_where():
    """The ``where`` kwarg scopes state-file rows; ``None`` keeps the union."""
    _seed_three_targets()

    local_ids = {r.prompt_id for r in jobs_mod._gather_local_state_files(limit=100, where="local")}
    assert local_ids == {"job-local", "job-legacy"}, f"missing/empty where must count as local; got {local_ids}"

    cloud_ids = {r.prompt_id for r in jobs_mod._gather_local_state_files(limit=100, where="cloud")}
    assert cloud_ids == {"job-cloud"}

    union_ids = {r.prompt_id for r in jobs_mod._gather_local_state_files(limit=100)}
    assert union_ids == {"job-local", "job-cloud", "job-legacy"}


def test_gather_local_state_files_drops_file_with_no_where_key():
    """Belt-and-braces on the legacy shape: a state file that omits ``where``
    entirely never reaches the filter — ``jobs_state.read`` already rejects it
    as truncated — so it can't leak into a scoped *or* an --all listing."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    (state_dir / "job-nowhere.json").write_text(
        json.dumps(
            {
                "prompt_id": "job-nowhere",
                "client_id": "c",
                "workflow": "/tmp/x.json",
                "status": "completed",
            }
        )
    )
    assert jobs_state.read("job-nowhere") is None
    assert jobs_mod._gather_local_state_files(limit=100) == []


def _ls_payload(capsys, **kwargs) -> dict:
    """Run ``jobs ls`` in-process under a JSON renderer, return its data dict."""
    from comfy_cli.caller import Caller
    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    # 65431 is a port nothing listens on — the server query degrades to the
    # state-file view, which is exactly the leak this scoping guards.
    kwargs.setdefault("host", "127.0.0.1")
    kwargs.setdefault("port", 65431)
    jobs_mod.ls_cmd(**kwargs)
    env = _last_json(capsys.readouterr().out)
    assert env["ok"] is True
    return env["data"]


def _ls_ids(payload: dict) -> set[str]:
    return {j["prompt_id"] for j in payload["jobs"]}


def test_ls_default_scopes_state_rows_to_local(capsys, monkeypatch):
    """Acceptance: with a cloud job on disk, a local `jobs ls` has no cloud rows."""
    _seed_three_targets()
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    data = _ls_payload(capsys, limit=100)
    assert data["where"] == "local"
    assert data["scope"] == "local"
    assert _ls_ids(data) == {"job-local", "job-legacy"}
    # Acceptance: no cloud rows at all — and the legacy row reports the target
    # it was scoped under rather than a bare null.
    assert {j["where"] for j in data["jobs"]} == {"local"}


def test_ls_where_cloud_scopes_state_rows_to_cloud(capsys, monkeypatch):
    """Acceptance: `jobs ls --where cloud` shows no local state rows."""
    _seed_three_targets()
    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)

    def _preflight_fails():
        raise typer.Exit(code=1)

    # Cloud unreachable/unauthed: the command falls through to the state-file
    # view, which must still be cloud-scoped.
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", _preflight_fails)

    data = _ls_payload(capsys, limit=100, where="cloud")
    assert data["where"] == "cloud"
    assert data["scope"] == "cloud"
    assert _ls_ids(data) == {"job-cloud"}


def test_ls_all_restores_the_union_view(capsys, monkeypatch):
    """Acceptance: ``--all`` brings every target's state rows back."""
    _seed_three_targets()
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    data = _ls_payload(capsys, limit=100, all_wheres=True)
    assert data["where"] == "local", "--all widens the state-file scope, not the server query"
    assert data["scope"] == "all"
    assert _ls_ids(data) == {"job-local", "job-cloud", "job-legacy"}


def test_ls_orphaned_stays_unfiltered(capsys, monkeypatch):
    """``--orphaned`` keeps the union view — watcher cleanup is where-agnostic,
    so a crashed cloud watcher is still reapable from a default (local) ls."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    crashed = {
        "code": "watcher_crashed",
        "message": "Background watcher (pid 99999) is no longer running.",
        "hint": "re-submit the workflow",
    }
    _write_state(state_dir, "orphan-local", where="local", status="error", error=crashed)
    _write_state(state_dir, "orphan-cloud", where="cloud", status="error", error=crashed)
    _write_state(state_dir, "healthy-cloud", where="cloud", status="completed")
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    data = _ls_payload(capsys, limit=100, orphaned=True)
    assert data["scope"] == "all"
    assert _ls_ids(data) == {"orphan-local", "orphan-cloud"}


def _watch_kwargs(monkeypatch, **kwargs) -> dict:
    """Run ``jobs ls --watch`` under a pretty renderer with ``_watch_ls``
    stubbed, and return the kwargs the live path would have been driven with."""
    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    set_renderer(Renderer(mode=OutputMode.PRETTY, command="jobs ls"))
    seen: dict = {}
    monkeypatch.setattr(jobs_mod, "_watch_ls", lambda **kw: seen.update(kw))
    jobs_mod.ls_cmd(host="127.0.0.1", port=65431, watch=True, **kwargs)
    return seen


def test_ls_watch_mirrors_one_shot_scope(monkeypatch):
    """``--watch`` must apply the *same* state-file filters as the one-shot
    listing: the resolved target by default, the union under ``--all``."""
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    assert _watch_kwargs(monkeypatch)["state_where"] == "local"
    assert _watch_kwargs(monkeypatch, all_wheres=True)["state_where"] is None

    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)
    assert _watch_kwargs(monkeypatch, where="cloud")["state_where"] == "cloud"


def test_ls_watch_threads_orphaned(monkeypatch):
    """``jobs ls --watch --orphaned`` must restrict to crashed-watcher rows,
    keep the union scope, and skip the server query — exactly like one-shot.
    The `if orphaned` handling used to sit *below* the --watch early return."""
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    seen = _watch_kwargs(monkeypatch, orphaned=True)
    assert seen["orphaned_only"] is True
    assert seen["state_where"] is None, "--orphaned is where-agnostic in watch too"
    assert seen["local_only"] is True, "the server can't know a watcher crashed"


def test_watch_build_table_applies_orphaned_and_scope(monkeypatch):
    """The stubbed kwargs above are only useful if ``_watch_ls`` actually
    forwards them to the gatherer."""
    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    class _Stop(Exception):
        pass

    set_renderer(Renderer(mode=OutputMode.PRETTY, command="jobs ls"))
    seen: dict = {}
    monkeypatch.setattr(jobs_mod, "_gather_local_state_files", lambda **kw: seen.update(kw) or [])
    # Bail out of the first table build rather than looping forever. `_Stop` is
    # not KeyboardInterrupt on purpose — `_watch_ls` swallows that one.
    monkeypatch.setattr(jobs_mod, "_merge_jobs", lambda *_a, **_k: (_ for _ in ()).throw(_Stop()))

    with pytest.raises(_Stop):
        jobs_mod._watch_ls(
            host="127.0.0.1",
            port=65431,
            limit=100,
            where="local",
            local_only=True,
            state_where=None,
            orphaned_only=True,
        )
    assert seen["orphaned_only"] is True
    assert seen["where"] is None


# ---------------------------------------------------------------------------
# --where routing — top-level flag must be honored, not just per-command
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# `jobs cancel` — local + cloud paths
# ---------------------------------------------------------------------------


class _Seq:
    """Route payload that changes per call: element i answers the i-th matching
    request, and the last element repeats. Lets a test model a queue whose
    contents change between two GET /queue reads."""

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self._i = 0

    def next(self):
        payload = self._payloads[min(self._i, len(self._payloads) - 1)]
        self._i += 1
        return payload


def _capture_urlopen(monkeypatch: pytest.MonkeyPatch, routes: dict):
    """Capture calls to urlopen and return a list of (url, method, headers) per call."""
    calls: list[dict] = []

    class _Resp:
        def __init__(self, body: bytes = b"{}"):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            # Mirror http.client.HTTPResponse.read(amt) — `_http_get_json`
            # reads with a byte cap, so a no-arg-only fake would not match the
            # real API.
            return self.body if n is None else self.body[:n]

    def _fake(req, timeout=None):
        url = req.full_url
        method = req.get_method()
        calls.append({"url": url, "method": method, "headers": dict(req.headers)})
        for needle, payload in routes.items():
            # A needle may be "<METHOD> <substring>" to match on verb too;
            # plain substrings (no space) match any method.
            want_method = None
            sub = needle
            if " " in needle:
                want_method, sub = needle.split(" ", 1)
            if sub in url and (want_method is None or want_method == method):
                if isinstance(payload, _Seq):
                    payload = payload.next()
                if isinstance(payload, Exception):
                    raise payload
                return _Resp(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL: {url}")

    # Both paths now open through a shared opener in ``comfy_cli.http``: the
    # local queue/interrupt calls via the redirect-following ``_PLAIN_OPENER``,
    # the cloud cancel path via the no-redirect ``_AUTHED_OPENER``. Both receive
    # a ``Request`` object, so route the same fake through both.
    import comfy_cli.http as http_mod

    monkeypatch.setattr(http_mod._PLAIN_OPENER, "open", _fake)
    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", _fake)
    return calls


def test_jobs_cancel_local_hits_queue_and_interrupt(monkeypatch: pytest.MonkeyPatch):
    """`comfy jobs cancel <id>` on local POSTs the queue delete (for pending),
    then GETs /queue and — because this prompt is the running one — POSTs
    /interrupt. /interrupt is gated on queue_running so cancelling a pending
    job never kills an unrelated running job."""
    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            # GET /queue reports prompt-abc as the running job.
            "GET /queue": {"queue_running": [[0, "prompt-abc", {}, {}, {}]], "queue_pending": []},
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "local"])
    assert result.exit_code == 0, result.output

    # Queue delete (POST), queue status (GET), and interrupt (POST) all hit.
    urls = [c["url"] for c in calls]
    assert any("/queue" in u for u in urls), urls
    assert any("/interrupt" in u for u in urls), urls
    methods = {c["method"] for c in calls}
    assert methods == {"POST", "GET"}

    # /queue delete payload carries the prompt_id.
    queue_call = next(c for c in calls if "/queue" in c["url"] and c["method"] == "POST")
    # The body is on the Request, not in our captured dict — re-derive from headers.
    assert queue_call["headers"].get("Content-type") == "application/json"


def test_jobs_cancel_local_tolerates_one_failure(monkeypatch: pytest.MonkeyPatch):
    """If the queue delete 404s but the job is running (queue_running lists it)
    and /interrupt 200s, the cancel still succeeds. Mirrors the real ComfyUI
    server's behavior for a running-not-pending job."""
    import urllib.error

    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "POST /queue": urllib.error.HTTPError("http://x/queue", 404, "Not Found", {}, None),
            "GET /queue": {"queue_running": [[0, "prompt-abc", {}, {}, {}]], "queue_pending": []},
            "/interrupt": b"{}",
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "local"])
    assert result.exit_code == 0, result.output


def test_jobs_cancel_local_pending_job_does_not_interrupt(monkeypatch: pytest.MonkeyPatch):
    """Cancelling a *pending* job (a different prompt is running) must NOT POST
    /interrupt — otherwise 'cancel B' would also kill the running 'A'."""
    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            # prompt-pending is queued; a *different* prompt is running.
            "GET /queue": {
                "queue_running": [[0, "prompt-running", {}, {}, {}]],
                "queue_pending": [[1, "prompt-pending", {}, {}, {}]],
            },
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-pending", "--where", "local"])
    assert result.exit_code == 0, result.output

    urls = [c["url"] for c in calls]
    assert any("/queue" in u for u in urls), urls
    assert not any("/interrupt" in u for u in urls), f"must not interrupt a pending job: {urls}"


def test_jobs_cancel_local_both_fail_returns_error(monkeypatch: pytest.MonkeyPatch):
    """If both /queue and /interrupt fail, surface cancel_failed."""
    import urllib.error

    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "/queue": urllib.error.URLError("connection refused"),
            "/interrupt": urllib.error.URLError("connection refused"),
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "local"])
    assert result.exit_code == 1, result.output


# ---------------------------------------------------------------------------
# `jobs cancel` local — existence probe (prompt_not_found for ids nowhere)
# ---------------------------------------------------------------------------


def _cancel_local_json(prompt_id: str):
    """Run ``comfy --json jobs cancel <id> --where local`` in-process and
    return (result, envelope) — the envelope being the last stdout line."""
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    result = CliRunner().invoke(app, ["--json", "jobs", "cancel", prompt_id, "--where", "local"])
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, f"no stdout envelope: {result.output!r}"
    return result, json.loads(lines[-1])


def test_jobs_cancel_local_unknown_id_emits_prompt_not_found(monkeypatch: pytest.MonkeyPatch):
    """An id the server and the state store have never seen is an error, not a
    silent idempotent ok — otherwise a typo'd id is indistinguishable from a
    real cancel (`POST /queue {"delete": [id]}` 200s for unknown ids)."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": {"queue_running": [], "queue_pending": []},
            # ComfyUI returns {} from /history/<id> for an unknown id.
            "GET /history/": {},
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )

    result, env = _cancel_local_json("not-a-real-id")
    assert result.exit_code == 1, result.output
    assert env["ok"] is False
    assert env["error"]["code"] == "prompt_not_found"
    assert env["error"]["details"]["prompt_id"] == "not-a-real-id"

    # The probe must run BEFORE any mutation — nothing was deleted or
    # interrupted on the way to deciding the id doesn't exist.
    assert not any(c["method"] == "POST" for c in calls), calls


def test_jobs_cancel_local_known_only_in_history_is_idempotent_ok(monkeypatch: pytest.MonkeyPatch):
    """A completed (terminal) prompt is still a KNOWN prompt — the documented
    idempotent contract keeps returning ok for it."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": {"queue_running": [], "queue_pending": []},
            "GET /history/abc-1": {"abc-1": _HISTORY_FIXTURE["abc-1"]},
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )

    result, env = _cancel_local_json("abc-1")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True
    assert env["data"]["found"] is True
    # Terminal, not running → no /interrupt (that would kill an unrelated job).
    assert not any("/interrupt" in c["url"] for c in calls), calls


def test_jobs_cancel_local_known_only_in_state_file_is_ok(monkeypatch: pytest.MonkeyPatch):
    """A state file means WE submitted the prompt, so it existed — even if the
    server has since forgotten it (restart, trimmed history). No history probe
    is needed; the routes below would AssertionError if one were made."""
    from comfy_cli import jobs_state

    jobs_state.write(jobs_state.new(prompt_id="pid-local", client_id="c", workflow="w", where="local"))

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": {"queue_running": [], "queue_pending": []},
            "POST /queue": b"{}",
        },
    )

    result, env = _cancel_local_json("pid-local")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True and env["data"]["found"] is True
    assert not any("/history" in c["url"] for c in calls), calls


def test_jobs_cancel_local_unreachable_queue_is_not_prompt_not_found(monkeypatch: pytest.MonkeyPatch):
    """Absence of evidence isn't evidence of absence: when the existence probe
    can't reach the server, keep the old idempotent behavior rather than
    claiming the id doesn't exist."""
    import urllib.error

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": urllib.error.URLError("connection refused"),
            "POST /queue": b"{}",
        },
    )

    result, env = _cancel_local_json("ghost-id")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True
    assert env["data"]["found"] is False


def test_jobs_cancel_local_unreachable_history_is_not_prompt_not_found(monkeypatch: pytest.MonkeyPatch):
    """Same guard one probe further in: /queue answered but /history failed, so
    existence is unproven — don't report prompt_not_found."""
    import urllib.error

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": {"queue_running": [], "queue_pending": []},
            "GET /history/": urllib.error.URLError("connection reset"),
            "POST /queue": b"{}",
        },
    )

    result, env = _cancel_local_json("ghost-id")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True
    assert env["data"]["found"] is False


def test_jobs_cancel_local_empty_id_is_prompt_not_found(monkeypatch: pytest.MonkeyPatch):
    """An empty/whitespace id must be rejected before any probe: quoting it into
    the history probe would produce `GET /history/` — the list-ALL endpoint —
    whose non-empty body would read as 'found' and wave a garbage id through."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(monkeypatch, {})  # any HTTP call at all is a failure

    result, env = _cancel_local_json("   ")
    assert result.exit_code == 1, result.output
    assert env["error"]["code"] == "prompt_not_found"
    assert calls == [], f"empty id must not touch the server: {calls}"


def test_jobs_cancel_local_non_json_body_is_not_a_traceback(monkeypatch: pytest.MonkeyPatch):
    """A 200 with a non-JSON body (proxy error page, captive portal) must be
    treated like any other probe failure, not crash with a JSONDecodeError."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": b"<html>gateway timeout</html>",
            "POST /queue": b"{}",
        },
    )

    result, env = _cancel_local_json("ghost-id")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True and env["data"]["found"] is False


def test_jobs_cancel_local_rereads_running_set_before_interrupt(monkeypatch: pytest.MonkeyPatch):
    """A job that goes pending→running while the queue delete is in flight must
    still be interrupted. Gating on the pre-delete snapshot would skip the
    interrupt and report a successful cancel of a job that keeps running."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": _Seq(
                # Before the delete: ours is pending, another job is running.
                {"queue_running": [[0, "other", {}, {}, {}]], "queue_pending": [[1, "mine", {}, {}, {}]]},
                # After the delete: 'other' finished and ours started.
                {"queue_running": [[0, "mine", {}, {}, {}]], "queue_pending": []},
            ),
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )

    result, env = _cancel_local_json("mine")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True
    assert any("/interrupt" in c["url"] for c in calls), f"must interrupt the now-running job: {calls}"


def test_jobs_cancel_local_does_not_interrupt_a_job_that_took_over(monkeypatch: pytest.MonkeyPatch):
    """The mirror case: our job finished during the delete round-trip and a
    DIFFERENT job now holds the running slot. /interrupt takes no prompt_id, so
    firing it off the stale snapshot would cancel that unrelated job."""
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": _Seq(
                {"queue_running": [[0, "mine", {}, {}, {}]], "queue_pending": []},
                {"queue_running": [[0, "other", {}, {}, {}]], "queue_pending": []},
            ),
            "POST /queue": b"{}",
            "/interrupt": b"{}",
        },
    )

    result, _env = _cancel_local_json("mine")
    assert result.exit_code == 0, result.output
    assert not any("/interrupt" in c["url"] for c in calls), f"must not kill an unrelated job: {calls}"


def test_jobs_cancel_local_server_dying_mid_cancel_is_not_a_success(monkeypatch: pytest.MonkeyPatch):
    """Reachability for the final gate must be judged AFTER the delete. If the
    server dies between the existence probe and the delete, the delete fails and
    the command must surface cancel_failed rather than report a clean cancel."""
    import urllib.error

    from comfy_cli import jobs_state

    jobs_state.write(jobs_state.new(prompt_id="pid-dying", client_id="c", workflow="w", where="local"))

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": _Seq(
                {"queue_running": [], "queue_pending": []},
                urllib.error.URLError("connection refused"),
            ),
            "POST /queue": urllib.error.URLError("connection refused"),
        },
    )

    result, env = _cancel_local_json("pid-dying")
    assert result.exit_code == 1, result.output
    assert env["error"]["code"] == "cancel_failed"


def test_jobs_cancel_local_keeps_terminal_status(monkeypatch: pytest.MonkeyPatch):
    """Cancelling an already-completed job is an idempotent ok — it must NOT
    rewrite the recorded outcome, or `jobs ls` reports a completed run as
    cancelled."""
    from comfy_cli import jobs_state

    st = jobs_state.new(prompt_id="pid-done", client_id="c", workflow="w", where="local")
    st.status = "completed"
    jobs_state.write(st)

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "GET /queue": {"queue_running": [], "queue_pending": []},
            "POST /queue": b"{}",
        },
    )

    result, env = _cancel_local_json("pid-done")
    assert result.exit_code == 0, result.output
    assert env["ok"] is True and env["data"]["found"] is True
    assert jobs_state.read("pid-done").status == "completed"


def test_jobs_cancel_cloud_posts_to_jobs_cancel_endpoint(monkeypatch: pytest.MonkeyPatch):
    """Cloud cancel POSTs to /api/jobs/<id>/cancel with the auth header."""
    from typer.testing import CliRunner

    from comfy_cli.target import Target

    fake_target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake_target)
    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)

    calls = _capture_urlopen(
        monkeypatch,
        {"/api/jobs/prompt-abc/cancel": b'{"status":"cancelling"}'},
    )

    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "cloud"])
    assert result.exit_code == 0, result.output

    assert len(calls) == 1
    assert calls[0]["method"] == "POST"
    assert "/api/jobs/prompt-abc/cancel" in calls[0]["url"]
    # Auth header (urllib title-cases X-API-Key → X-api-key).
    h = {k.lower(): v for k, v in calls[0]["headers"].items()}
    assert h.get("x-api-key") == "test-key"


def test_jobs_cancel_cloud_404_surfaces_prompt_not_found(monkeypatch: pytest.MonkeyPatch):
    """404 on cloud cancel is the 'unknown prompt_id' signal — surface it as prompt_not_found."""
    import io
    import urllib.error

    from typer.testing import CliRunner

    from comfy_cli.target import Target

    fake_target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake_target)
    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)

    err = urllib.error.HTTPError("https://x/cancel", 404, "Not Found", {}, io.BytesIO(b'{"error":"no such job"}'))
    _capture_urlopen(monkeypatch, {"/api/jobs/missing/cancel": err})

    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "missing", "--where", "cloud"])
    assert result.exit_code == 1
    # Output contains the error code marker.
    assert "prompt_not_found" in result.output


@pytest.mark.parametrize("code", [401, 403])
def test_jobs_cancel_cloud_auth_failure_surfaces_cloud_unauthorized(monkeypatch: pytest.MonkeyPatch, code: int):
    """An expired/insufficient session cancelling a cloud job surfaces the
    actionable ``cloud_unauthorized`` code (shared envelope handler, BE-3266) —
    not the generic ``cloud_http_error`` it produced before the two call sites
    were unified."""
    import io
    import urllib.error

    from typer.testing import CliRunner

    from comfy_cli.target import Target

    fake_target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake_target)
    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)

    err = urllib.error.HTTPError("https://x/cancel", code, "Unauthorized", {}, io.BytesIO(b'{"error":"expired"}'))
    _capture_urlopen(monkeypatch, {"/api/jobs/prompt-abc/cancel": err})

    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "cloud"])
    assert result.exit_code == 1
    assert "cloud_unauthorized" in result.output


def test_is_cloud_honors_env_var(monkeypatch: pytest.MonkeyPatch):
    """``comfy --where cloud jobs status X`` sets COMFY_WHERE in the env.
    ``_is_cloud(None)`` must return True so the cloud path is taken.

    Without this, the top-level ``--where cloud`` flag is silently dropped
    by every ``jobs`` subcommand and the call falls through to local
    routing — the bug observed during the Veo3 video run.
    """
    monkeypatch.setenv("COMFY_WHERE", "cloud")
    assert jobs_mod._is_cloud(None) is True


def test_is_cloud_per_command_flag_still_wins(monkeypatch: pytest.MonkeyPatch):
    """An explicit ``jobs status X --where local`` must override
    ``COMFY_WHERE=cloud`` (flag > env > config > default precedence)."""
    monkeypatch.setenv("COMFY_WHERE", "cloud")
    assert jobs_mod._is_cloud("local") is False
    assert jobs_mod._is_cloud("cloud") is True


def test_is_cloud_default_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COMFY_WHERE", raising=False)
    # Ensure no persisted config interferes with the default-local assumption.
    from comfy_cli.config_manager import ConfigManager

    monkeypatch.setattr(ConfigManager(), "get", lambda key: None)
    assert jobs_mod._is_cloud(None) is False


def test_top_level_where_cloud_reaches_preflight(monkeypatch: pytest.MonkeyPatch):
    """Integration: with COMFY_WHERE=cloud and no auth, ``jobs status``
    must surface ``cloud_not_configured`` (the preflight error), not
    ``server_not_running`` (which would mean it routed to local)."""
    import typer.testing

    monkeypatch.setenv("COMFY_WHERE", "cloud")
    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)

    # Force-empty the auth store so preflight reports not-configured even
    # if the developer running the suite is signed in.
    from comfy_cli.auth import store as auth_store

    monkeypatch.setattr(auth_store, "get", lambda _: None)
    monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)

    from comfy_cli.cmdline import app

    runner = typer.testing.CliRunner()
    result = runner.invoke(app, ["--json", "jobs", "status", "some-id"])

    # The last non-empty line is the envelope (intermediate messages → stderr).
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, f"no stdout: stderr={result.stderr!r}"
    env = json.loads(lines[-1])
    assert env["ok"] is False
    assert env["error"]["code"] == "cloud_not_configured", (
        f"top-level --where cloud was dropped — got {env['error']['code']!r}; this is the routing-flag-position bug"
    )


# ---------------------------------------------------------------------------
# Discover surface — make sure jobs commands are advertised
# ---------------------------------------------------------------------------


def test_jobs_commands_in_discover():
    res = _run(["--json", "discover"])
    env = _last_json(res.stdout)
    cs = env["data"]["command_schemas"]
    for k in ("comfy jobs ls", "comfy jobs status", "comfy jobs watch"):
        assert k in cs, f"{k!r} missing from command_schemas"
    # Stream event schema for watch
    assert "comfy jobs watch" in env["data"]["stream_event_schemas"]


# ---------------------------------------------------------------------------
# Run async-by-default
# ---------------------------------------------------------------------------


def test_run_wait_flag_visible_in_help():
    """Async is the default; --wait is the documented opt-in for blocking."""
    assert "--wait" in _command_flags("run")


def test_run_default_async_emits_clean_server_not_running(tmp_path):
    """The default (no --wait) still validates the server before submitting,
    so a missing server emits the structured envelope. Confirms the path
    is wired through whether async or wait."""
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "Anything", "inputs": {}}}))
    res = _run(
        [
            "--json",
            "run",
            "--workflow",
            str(wf),
            "--host",
            "127.0.0.1",
            "--port",
            "65431",
        ]
    )
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["error"]["code"] == "server_not_running"


# ---------------------------------------------------------------------------
# Cancelled / interrupted job terminal-state fixes
# ---------------------------------------------------------------------------


def test_snapshot_maps_interrupted_to_cancelled(monkeypatch):
    """_snapshot must return status='cancelled' when the history record has
    completed=False and an execution_interrupted message (not execution_error)."""
    body = {
        "pid": {
            "status": {
                "completed": False,
                "messages": [["execution_interrupted", {}]],
            },
            "outputs": {},
        }
    }
    monkeypatch.setattr(
        jobs_mod,
        "_http_get_json",
        lambda url, **kw: {} if "/queue" in url else body,
    )
    snap = jobs_mod._snapshot("127.0.0.1", 8188, "pid")
    assert snap is not None
    assert snap["status"] == "cancelled"


def test_poll_local_once_treats_cancelled_as_terminal(monkeypatch):
    """_poll_local_once must report terminal (and a record it saw) and set
    state.status='cancelled' when _snapshot reports status='cancelled'."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "cancelled", "outputs": []},
    )
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="local")
    assert job_watcher._poll_local_once(state, host=None, port=None) == (True, True)
    assert state.status == "cancelled"


def test_watcher_timeout_preserves_prior_status(monkeypatch):
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="local")
    state.status = "running"
    # First time() call (start) = 0.0, second (loop check) is past the ceiling.
    times = iter([0.0, job_watcher._MAX_RUNTIME_S + 1])
    monkeypatch.setattr(job_watcher.time, "time", lambda: next(times))
    monkeypatch.setattr(jobs_state, "write", lambda s: None)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)
    monkeypatch.setattr(jobs_state, "read", lambda pid: state)
    job_watcher.watch_job("pid", where="local")
    assert state.error["details"]["last_status"] == "running"


class _FakeCloudClient:
    """Minimal stand-in for comfy_client.Client used by cloud status paths."""

    def __init__(self, status_payload):
        self._status_payload = status_payload
        self.target = type("T", (), {"base_url": "https://cloud.example"})()

    def get_job_status(self, prompt_id):
        return dict(self._status_payload)

    def get_history(self, prompt_id):  # pragma: no cover — error paths never fetch
        raise AssertionError("get_history must not be called for failed jobs")

    def extract_output_urls(self, record):  # pragma: no cover
        return []


@pytest.mark.parametrize("raw_status", ["non_retryable_error", "lost"])
def test_cloud_status_snapshot_maps_fatal_statuses_to_error(monkeypatch, raw_status):
    """Cloud statuses like non_retryable_error/lost must snapshot to 'error',
    not leak through raw (which makes `jobs watch` poll forever)."""
    payload = {"status": raw_status, "error_message": "RIP to the server"}
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(payload))
    snap = jobs_mod._cloud_status_snapshot("pid-1")
    assert snap is not None
    assert snap["status"] == "error"
    assert snap["error_message"] == "RIP to the server"


@pytest.mark.parametrize("raw_status", ["non_retryable_error", "lost"])
def test_poll_cloud_once_treats_fatal_statuses_as_terminal(raw_status):
    """The watcher must treat non_retryable_error/lost as terminal errors and
    stop polling, recording state.error."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    client = _FakeCloudClient({"status": raw_status, "error_message": "RIP to the server"})
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    assert job_watcher._poll_cloud_once(state, client=client) is True
    assert state.status == "error"
    assert state.error is not None
    assert state.error["message"] == "RIP to the server"


_CLOUD_RECORD = {
    "status": {"completed": True, "status_str": "success"},
    "outputs": {
        "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
        "12": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
    },
}


class _CompletedCloudClient:
    """Fake cloud client for a job that finished successfully."""

    target = type("T", (), {"base_url": "https://cloud.example"})()

    def __init__(self, record=None):
        self._record = record if record is not None else _CLOUD_RECORD

    def get_job_status(self, prompt_id):
        return {"status": "success"}

    def get_history(self, prompt_id):
        return dict(self._record)

    def extract_outputs(self, record):
        # Mirrors Client.extract_outputs' shape; URL plumbing is the real
        # client's concern (covered in tests/comfy_cli/cloud/test_client.py).
        out = []
        for node_id, node_output in (record.get("outputs") or {}).items():
            for key in ("images", "gifs", "videos", "audio", "files"):
                for item in node_output.get(key) or []:
                    out.append(
                        {
                            "node_id": str(node_id),
                            "url": f"https://cloud.example/view/{item['filename']}",
                            "filename": item["filename"],
                            "type": item.get("type", "output"),
                        }
                    )
        return out


def test_cloud_status_snapshot_groups_outputs_by_node_and_item(monkeypatch):
    """With a state file carrying an item_map, the cloud snapshot exposes
    outputs grouped by producing node and by blueprint foreach item."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _CompletedCloudClient())
    state = jobs_state.new(prompt_id="pid-grouped", client_id="c", workflow="w", where="cloud")
    state.item_map = {
        "s1": {"nodes": ["9"], "save_node": "9", "prefix": "outputs/s1"},
        "s2": {"nodes": ["12"], "save_node": "12", "prefix": "outputs/s2"},
    }
    jobs_state.write(state)

    snap = jobs_mod._cloud_status_snapshot("pid-grouped")
    assert snap is not None
    assert snap["status"] == "completed"
    assert snap["outputs"] == ["https://cloud.example/view/a.png", "https://cloud.example/view/v.mp4"]
    assert snap["outputs_by_node"] == {
        "9": ["https://cloud.example/view/a.png"],
        "12": ["https://cloud.example/view/v.mp4"],
    }
    assert snap["outputs_by_item"] == {
        "s1": ["https://cloud.example/view/a.png"],
        "s2": ["https://cloud.example/view/v.mp4"],
    }


def test_cloud_status_snapshot_without_item_map_emits_empty_by_item(monkeypatch):
    """No state file (or no item_map) → outputs_by_item stays {} while
    outputs_by_node is still grouped from the history record."""
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _CompletedCloudClient())

    snap = jobs_mod._cloud_status_snapshot("pid-no-map")
    assert snap is not None
    assert snap["outputs_by_node"] == {
        "9": ["https://cloud.example/view/a.png"],
        "12": ["https://cloud.example/view/v.mp4"],
    }
    assert snap["outputs_by_item"] == {}


def test_cloud_status_snapshot_non_terminal_keeps_empty_groupings(monkeypatch):
    """In-flight jobs have no record to group — keys present, empty dicts."""

    class _RunningClient(_CompletedCloudClient):
        def get_job_status(self, prompt_id):
            return {"status": "running"}

        def get_history(self, prompt_id):  # pragma: no cover — must not be called
            raise AssertionError("history must not be fetched for in-flight jobs")

    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _RunningClient())
    snap = jobs_mod._cloud_status_snapshot("pid-running")
    assert snap is not None
    assert snap["outputs_by_node"] == {}
    assert snap["outputs_by_item"] == {}


def test_jobs_status_cloud_envelope_carries_grouped_outputs(monkeypatch, capsys):
    """End-to-end through `jobs status --where cloud`: the envelope data
    carries outputs_by_node / outputs_by_item."""
    from comfy_cli import jobs_state
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _CompletedCloudClient())
    state = jobs_state.new(prompt_id="pid-env", client_id="c", workflow="w", where="cloud")
    state.item_map = {"s1": {"nodes": ["9", "12"], "save_node": "12", "prefix": "outputs/s1"}}
    jobs_state.write(state)

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs status"))
    jobs_mod._cloud_status("pid-env")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    env = json.loads(lines[-1])
    assert env["type"] == "envelope" and env["ok"] is True
    assert env["data"]["outputs_by_node"]["9"] == ["https://cloud.example/view/a.png"]
    assert env["data"]["outputs_by_item"]["s1"] == [
        "https://cloud.example/view/a.png",
        "https://cloud.example/view/v.mp4",
    ]


def test_jobs_watch_cloud_terminal_envelope_carries_grouped_outputs(monkeypatch, capsys):
    """`jobs watch --where cloud` reaches terminal via the same snapshot —
    the grouped keys must flow through to the terminal envelope."""
    from comfy_cli import jobs_state
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _CompletedCloudClient())
    state = jobs_state.new(prompt_id="pid-watch", client_id="c", workflow="w", where="cloud")
    state.item_map = {"s1": {"nodes": ["9"], "save_node": "9", "prefix": "outputs/s1"}}
    jobs_state.write(state)

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    jobs_mod._cloud_watch("pid-watch", poll_interval=0.01, max_wait=5)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    env = json.loads(lines[-1])
    assert env["type"] == "envelope" and env["ok"] is True
    assert env["data"]["status"] == "completed"
    assert env["data"]["outputs_by_node"] == {
        "9": ["https://cloud.example/view/a.png"],
        "12": ["https://cloud.example/view/v.mp4"],
    }
    assert env["data"]["outputs_by_item"] == {"s1": ["https://cloud.example/view/a.png"]}


def test_jobs_schema_documents_grouped_outputs():
    """schemas/jobs.json carries the additive grouped-output keys."""
    schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "jobs.json"
    schema = json.loads(schema_path.read_text())
    for key in ("outputs_by_node", "outputs_by_item"):
        prop = schema["properties"][key]
        assert prop["type"] == "object"
        assert prop["additionalProperties"] == {"type": "array", "items": {"type": "string"}}


def test_poll_cloud_once_stashes_history_record_on_completion():
    """When the watcher fetches history at terminal, the full node-keyed
    record must be stashed on state.record so later consumers (grouped
    outputs, item-named downloads) don't need a second API call."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    history = {
        "status": {"completed": True, "status_str": "success"},
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
    }

    class _DoneClient:
        target = type("T", (), {"base_url": "https://cloud.example"})()

        def get_job_status(self, prompt_id):
            return {"status": "success"}  # no inline outputs → history fetch

        def get_history(self, prompt_id):
            return dict(history)

        def extract_output_urls(self, record):
            return ["https://cloud.example/api/view?filename=a.png&subfolder=&type=output"]

    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    assert job_watcher._poll_cloud_once(state, client=_DoneClient()) is True
    assert state.status == "completed"
    assert state.record == history
    assert state.outputs == ["https://cloud.example/api/view?filename=a.png&subfolder=&type=output"]


def test_watcher_unknown_status_stall_writes_error(monkeypatch):
    """A cloud status the CLI does not recognize (and that never changes) must
    not hang the watcher for the full 6h ceiling — after _UNKNOWN_STALL_S it
    writes terminal status='error' with code 'unknown_status_stall'."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    class _WeirdClient:
        target = type("T", (), {"base_url": "https://cloud.example"})()

        def get_job_status(self, prompt_id):
            return {"status": "weird_new_state"}

    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    monkeypatch.setattr(jobs_state, "read", lambda pid: state)
    monkeypatch.setattr(jobs_state, "write", lambda s: None)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda where: object())
    monkeypatch.setattr("comfy_cli.comfy_client.Client", lambda target, **kw: _WeirdClient())
    # Fake clock: each time() call advances 150s; sleep is a no-op. The guard
    # window (300s) elapses after a couple of polls instead of for real.
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 150.0
        return clock["t"]

    monkeypatch.setattr(job_watcher.time, "time", fake_time)
    monkeypatch.setattr(job_watcher.time, "sleep", lambda s: None)

    job_watcher.watch_job("pid", where="cloud")

    assert state.status == "error"
    assert state.error is not None
    assert state.error["code"] == "unknown_status_stall"
    assert "weird_new_state" in state.error["message"]


def test_watcher_known_inflight_status_never_stalls(monkeypatch):
    """Known in-flight statuses (queued/running/...) must not trip the
    unknown-status stall guard even when unchanged past the window."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    statuses = iter(["running"] * 5 + ["success"])

    class _SlowClient:
        target = type("T", (), {"base_url": "https://cloud.example"})()

        def get_job_status(self, prompt_id):
            return {"status": next(statuses)}

        def get_history(self, prompt_id):
            return None

    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    monkeypatch.setattr(jobs_state, "read", lambda pid: state)
    monkeypatch.setattr(jobs_state, "write", lambda s: None)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda where: object())
    monkeypatch.setattr("comfy_cli.comfy_client.Client", lambda target, **kw: _SlowClient())
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 150.0
        return clock["t"]

    monkeypatch.setattr(job_watcher.time, "time", fake_time)
    monkeypatch.setattr(job_watcher.time, "sleep", lambda s: None)

    job_watcher.watch_job("pid", where="cloud")

    assert state.status == "completed"
    assert state.error is None


def test_emit_terminal_verdicts():
    import typer

    from comfy_cli.command import jobs
    from comfy_cli.output.renderer import get_renderer, reset_renderer_for_testing

    def verdict(payload):
        reset_renderer_for_testing()
        r = get_renderer()
        try:
            jobs._emit_terminal(r, dict(payload), command="jobs watch")
        except typer.Exit as e:
            return e.exit_code
        return 0

    assert verdict({"prompt_id": "p", "status": "error"}) == 1
    assert verdict({"prompt_id": "p", "status": "cancelled"}) == 130
    assert verdict({"prompt_id": "p", "status": "completed", "outputs": []}) == 0


def test_emit_terminal_falls_back_to_top_level_error_message(capsys):
    """Cloud snapshots carry failure text at top-level `error_message`, not in
    an `error` dict — _emit_terminal must surface it in the error envelope."""
    import typer

    from comfy_cli.command import jobs
    from comfy_cli.output.renderer import OutputMode, Renderer

    renderer = Renderer(mode=OutputMode.JSON)
    payload = {"prompt_id": "p", "status": "error", "error_message": "OOM on worker"}
    with pytest.raises(typer.Exit) as exc_info:
        jobs._emit_terminal(renderer, payload, command="jobs watch")
    assert exc_info.value.exit_code == 1
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["ok"] is False
    assert "OOM on worker" in env["error"]["message"]


def test_emit_terminal_prefers_error_dict_message(capsys):
    """When both are present, the structured error dict's message wins."""
    import typer

    from comfy_cli.command import jobs
    from comfy_cli.output.renderer import OutputMode, Renderer

    renderer = Renderer(mode=OutputMode.JSON)
    payload = {
        "prompt_id": "p",
        "status": "error",
        "error": {"code": "execution_error", "message": "node 5 exploded"},
        "error_message": "OOM on worker",
    }
    with pytest.raises(typer.Exit):
        jobs._emit_terminal(renderer, payload, command="jobs watch")
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert env["error"]["message"] == "node 5 exploded"


def test_local_cancel_writes_cancelled_state(monkeypatch: pytest.MonkeyPatch):
    """_local_cancel must persist status='cancelled' to the on-disk state file
    after successfully POSTing to /queue and /interrupt."""
    from typer.testing import CliRunner

    from comfy_cli import jobs_state

    # Pre-write a state file so _local_cancel has something to update.
    st = jobs_state.new(prompt_id="pidX", client_id="c", workflow="w", where="local")
    jobs_state.write(st)
    assert jobs_state.read("pidX") is not None

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "/queue": b"{}",
            "/interrupt": b"{}",
        },
    )

    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "pidX", "--where", "local"])
    assert result.exit_code == 0, result.output

    # The on-disk state file must now carry status='cancelled'.
    persisted = jobs_state.read("pidX")
    assert persisted is not None, "state file was deleted instead of updated"
    assert persisted.status == "cancelled", f"expected 'cancelled', got {persisted.status!r}"


def test_watch_already_cancelled_job_exits_130(monkeypatch):
    """An already-cancelled local job must short-circuit to exit 130, not hang
    in the WS loop. Regression for the watch gate omitting 'cancelled'."""
    import typer  # noqa: F401 — ensures typer.Exit is raised, not SystemExit
    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    monkeypatch.setattr(
        jobs_mod,
        "_snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "cancelled", "outputs": []},
    )

    # If the gate is broken, watch would try to open a WebSocket. Make
    # WebSocket construction explode so a fall-through is unmistakable (not a hang).
    def _boom(*a, **k):
        raise AssertionError("watch fell through to WebSocket instead of short-circuiting on 'cancelled'")

    monkeypatch.setattr(jobs_mod, "WebSocket", _boom)

    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["watch", "pidX", "--where", "local"])
    assert result.exit_code == 130, (result.exit_code, result.output)


# ---------------------------------------------------------------------------
# `jobs wait` — block until N prompt_ids are all terminal (multi-job wait)
# ---------------------------------------------------------------------------


def test_wait_loop_settles_all_jobs():
    """_wait_loop polls each id until terminal; returns snapshots + empty pending."""
    import time as _t

    from comfy_cli.output import get_renderer

    bcalls = {"n": 0}

    def fake_fetch(pid):
        if pid == "a":
            return {"prompt_id": "a", "status": "completed", "outputs": ["u"]}
        bcalls["n"] += 1
        if bcalls["n"] < 2:
            return {"prompt_id": "b", "status": "running"}
        return {"prompt_id": "b", "status": "error", "error_message": "boom"}

    snaps, pending = jobs_mod._wait_loop(
        ["a", "b"], fake_fetch, poll_interval=0.0, deadline=_t.time() + 5, renderer=get_renderer()
    )
    assert pending == []
    assert snaps["a"]["status"] == "completed"
    assert snaps["b"]["status"] == "error"


def test_wait_loop_times_out_on_stuck_job():
    import time as _t

    from comfy_cli.output import get_renderer

    snaps, pending = jobs_mod._wait_loop(
        ["stuck"],
        lambda pid: {"prompt_id": pid, "status": "running"},
        poll_interval=0.0,
        deadline=_t.time() + 0.05,
        renderer=get_renderer(),
    )
    assert pending == ["stuck"]
    assert "stuck" not in snaps


def test_wait_cmd_all_completed_exit_zero(monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setattr(
        jobs_mod,
        "_wait_fetch_snapshot",
        lambda pid, **kw: {"prompt_id": pid, "status": "completed", "outputs": []},
    )
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    r = CliRunner().invoke(jobs_mod.app, ["wait", "a", "b", "--where", "local", "--poll-interval", "0"])
    assert r.exit_code == 0, r.output


def test_wait_cmd_any_error_exits_one(monkeypatch):
    from typer.testing import CliRunner

    def fetch(pid, **kw):
        status = "error" if pid == "b" else "completed"
        return {"prompt_id": pid, "status": status, "error_message": "boom"}

    monkeypatch.setattr(jobs_mod, "_wait_fetch_snapshot", fetch)
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    r = CliRunner().invoke(jobs_mod.app, ["wait", "a", "b", "--where", "local", "--poll-interval", "0"])
    assert r.exit_code == 1, r.output


def test_wait_cmd_no_ids_errors():
    from typer.testing import CliRunner

    r = CliRunner().invoke(jobs_mod.app, ["wait", "--where", "local"])
    assert r.exit_code != 0


def test_wait_summary_validates_against_jobs_wait_schema():
    """The `jobs wait` summary payload must validate against its declared schema."""
    import json as _json
    from pathlib import Path

    import jsonschema

    schema_path = Path(jobs_mod.__file__).resolve().parents[1] / "schemas" / "jobs_wait.json"
    schema = _json.loads(schema_path.read_text())
    summary = {
        "total": 2,
        "completed": 1,
        "failed": 1,
        "cancelled": 0,
        "timed_out": 0,
        "elapsed_seconds": 1.5,
        "jobs": [
            {"prompt_id": "a", "status": "completed", "ok": True, "outputs": ["u"]},
            {"prompt_id": "b", "status": "error", "ok": False, "error_message": "boom"},
        ],
    }
    jsonschema.Draft202012Validator(schema).validate(summary)


# ---------------------------------------------------------------------------
# `jobs watch` local WS dispatch — per-type handlers over _WatchState
# ---------------------------------------------------------------------------


class _RecordingRenderer:
    """Minimal non-pretty renderer that records emitted events."""

    def __init__(self):
        self.events: list[tuple] = []
        self.throttled: list[tuple] = []

    def is_pretty(self):
        return False

    def event(self, name, **kwargs):
        self.events.append((name, kwargs))

    def throttled_event(self, key, name, **kwargs):
        self.throttled.append((key, name, kwargs))


def _watch_state(**kw):
    r = _RecordingRenderer()
    st = jobs_mod._WatchState(renderer=r, prompt_id="pid", host="127.0.0.1", port=8188, **kw)
    return st, r


def test_watch_handlers_registry_covers_the_protocol_types():
    """The dispatch table maps exactly the WS event types watch handles."""
    assert set(jobs_mod._WATCH_HANDLERS) == {
        "executing",
        "execution_cached",
        "progress",
        "executed",
        "execution_error",
    }
    assert jobs_mod._WATCH_HANDLERS.get("unknown_type") is None


def test_watch_executing_null_node_is_terminal_completed():
    st, r = _watch_state()
    jobs_mod._watch_executing(st, {"node": None, "prompt_id": "pid"})
    assert st.terminal is True
    assert st.end_reason == "completed"
    assert r.events == []  # terminal sentinel emits nothing


def test_watch_executing_node_emits_event_and_is_not_terminal():
    st, r = _watch_state()
    jobs_mod._watch_executing(st, {"node": "5", "prompt_id": "pid"})
    assert st.terminal is False
    assert r.events == [("executing", {"node": "5", "prompt_id": "pid"})]


def test_watch_execution_cached_accumulates_completed_nodes():
    st, r = _watch_state()
    jobs_mod._watch_execution_cached(st, {"nodes": [1, 2]})
    assert st.completed_nodes == {"1", "2"}
    assert r.events == [("execution_cached", {"nodes": ["1", "2"], "prompt_id": "pid"})]


def test_watch_progress_uses_throttled_event():
    st, r = _watch_state()
    jobs_mod._watch_progress(st, {"node": "3", "value": 2, "max": 10})
    assert r.throttled == [
        ("progress:3", "progress", {"max_hz": 10, "node": "3", "completed": 2, "total": 10, "prompt_id": "pid"})
    ]


def test_watch_executed_collects_output_urls_with_host_port():
    st, r = _watch_state()
    data = {
        "node": "9",
        "output": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]},
    }
    jobs_mod._watch_executed(st, data)
    assert st.completed_nodes == {"9"}
    assert st.outputs == ["http://127.0.0.1:8188/view?filename=out.png&subfolder=&type=output"]
    assert ("output", {"url": st.outputs[0], "prompt_id": "pid"}) in r.events
    assert ("executed", {"node": "9", "prompt_id": "pid"}) in r.events


def test_watch_execution_error_is_terminal_and_carries_details():
    st, _ = _watch_state()
    data = {"node_id": "5", "exception_message": "boom"}
    jobs_mod._watch_execution_error(st, data)
    assert st.terminal is True
    assert st.end_reason == "error"
    assert st.end_details == data


class _PrettyRecordingRenderer(_RecordingRenderer):
    """Pretty renderer that records what would be printed to the console."""

    def __init__(self):
        super().__init__()
        self.printed: list[str] = []

    def is_pretty(self):
        return True

    def console(self):
        outer = self

        class _C:
            def print(self, msg):
                outer.printed.append(msg)

        return _C()


def test_watch_executing_escapes_server_controlled_node_markup():
    """A server-controlled node id can't inject Rich markup into pretty output."""
    r = _PrettyRecordingRenderer()
    st = jobs_mod._WatchState(renderer=r, prompt_id="pid", host="127.0.0.1", port=8188)
    jobs_mod._watch_executing(st, {"node": "[red]evil[/red]", "prompt_id": "pid"})
    assert len(r.printed) == 1
    # The injected markup must be escaped, not left as a live tag.
    assert "[bold][red]" not in r.printed[0]
    assert r"\[red]evil\[/red]" in r.printed[0]
    # The event stream still carries the raw node id.
    assert ("executing", {"node": "[red]evil[/red]", "prompt_id": "pid"}) in r.events


# ---------------------------------------------------------------------------
# watcher: local server death detection (server_died)
# ---------------------------------------------------------------------------


def _watched_local_job(monkeypatch, *, status="queued"):
    """A real on-disk state file for a local job, plus a no-op sleep.

    The autouse ``_isolate_jobs_state_dir`` fixture pins the state dir to a tmp
    path, so ``watch_job`` reads/writes real files and the assertions can be
    made against the file rather than an in-memory object.
    """
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)
    monkeypatch.setattr(job_watcher.time, "sleep", lambda s: None)
    state = jobs_state.new(
        prompt_id="pid-dead",
        client_id="c",
        workflow="/w.json",
        where="local",
        host="127.0.0.1",
        port=8188,
    )
    state.status = status
    jobs_state.write(state)
    return state


def _fake_probe(monkeypatch, results):
    """Point the watcher's liveness probe at a scripted sequence of verdicts.

    Each entry is a ``_PROBE_*`` constant, or a bool as shorthand for
    alive/unreachable. The last verdict repeats forever so a test can't hang on
    a short script.
    """
    from comfy_cli.command import job_watcher

    seq = [
        (job_watcher._PROBE_ALIVE if r else job_watcher._PROBE_UNREACHABLE) if isinstance(r, bool) else r
        for r in results
    ]
    calls = []

    def probe(host, port):
        calls.append((host, port))
        return seq[min(len(calls) - 1, len(seq) - 1)]

    monkeypatch.setattr(job_watcher, "_probe_local_server", probe)
    return calls


def test_watcher_records_server_died_after_consecutive_failed_probes(monkeypatch):
    """A local server that dies mid-job (OOM kill) must be recorded as a
    terminal ``server_died`` error instead of leaving a forever-'queued' ghost:
    ``_snapshot`` swallows the connection failure, so only an explicit liveness
    probe can see the death."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    calls = _fake_probe(monkeypatch, [False])
    # Only the one last-chance poll at the limit runs; the earlier cycles must
    # not poll a port that just refused them.
    polls = []
    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: polls.append((h, p)) or None,
    )
    notified = []
    monkeypatch.setattr(job_watcher, "_notify", notified.append)

    job_watcher.watch_job("pid-dead", where="local")

    assert len(calls) == job_watcher._SERVER_DOWN_CONSECUTIVE_LIMIT
    assert polls == [("127.0.0.1", 8188)]
    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "error"
    assert on_disk.error["code"] == "server_died"
    assert on_disk.error["details"]["last_status"] == "queued"
    assert on_disk.error["details"]["host"] == "127.0.0.1"
    assert on_disk.error["details"]["port"] == 8188
    assert on_disk.error["details"]["consecutive_failed_probes"] == job_watcher._SERVER_DOWN_CONSECUTIVE_LIMIT
    assert "pid-dead" in on_disk.error["message"]
    # ...and the user is told exactly once, with the final state.
    assert len(notified) == 1
    assert notified[0].status == "error"
    assert notified[0].error["code"] == "server_died"


def test_watcher_down_probe_streak_resets_on_recovery(monkeypatch):
    """A transient blip (or a quick restart) must not be mistaken for death:
    any successful probe resets the streak and the watcher keeps going."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    _fake_probe(monkeypatch, [False, False, True])
    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "completed", "outputs": ["a.png"]},
    )
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "completed"
    assert on_disk.error is None
    assert on_disk.outputs == ["a.png"]


def test_watcher_does_not_overwrite_a_terminal_state_when_server_dies(monkeypatch):
    """A dead server can't invalidate a verdict the job already reached — the
    watcher stops without rewriting a terminal state."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    state = _watched_local_job(monkeypatch, status="completed")
    state.outputs = ["done.png"]
    jobs_state.write(state)
    _fake_probe(monkeypatch, [False])
    # The dead server has nothing left to report to the last-chance poll.
    monkeypatch.setattr("comfy_cli.command.jobs._snapshot", lambda h, p, pid: None)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "completed"
    assert on_disk.error is None
    assert on_disk.outputs == ["done.png"]


def test_watcher_probe_targets_the_same_address_the_poll_uses(monkeypatch):
    """The liveness probe and the poll resolve their target through the same
    helper, so they can never disagree about which server is being watched."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    state = jobs_state.new(
        prompt_id="pid-v6",
        client_id="c",
        workflow="/w.json",
        where="local",
        host="::1",
        port=9999,
    )
    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)
    jobs_state.write(state)
    monkeypatch.setattr(job_watcher.time, "sleep", lambda s: None)
    calls = _fake_probe(monkeypatch, [False])
    polls = []
    monkeypatch.setattr("comfy_cli.command.jobs._snapshot", lambda h, p, pid: polls.append((h, p)) or None)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-v6", where="local")

    assert job_watcher._resolve_watch_target(state, None, None) == ("[::1]", 9999)
    assert set(calls) == {("[::1]", 9999)}
    assert set(polls) == {("[::1]", 9999)}


@pytest.mark.parametrize(
    "exc, expected",
    [
        # Nothing listening — the only signal that means "dead".
        (requests.exceptions.ConnectionError("refused"), "unreachable"),
        # Slow, not dead. ConnectTimeout subclasses ConnectionError as well as
        # Timeout, so it would be misread as a death if the except order ever
        # regressed — which is exactly the false server_died this guards.
        (requests.exceptions.ReadTimeout("slow"), "unresponsive"),
        (requests.exceptions.ConnectTimeout("slow"), "unresponsive"),
        # A probe must never crash the watcher, and never invent a death.
        (RuntimeError("probe exploded"), "unresponsive"),
    ],
)
def test_probe_classifies_failures_without_inventing_a_death(monkeypatch, exc, expected):
    """Only a refused connection counts as unreachable — a busy server (loading
    a model into VRAM) times out and must stay merely 'unresponsive'."""
    from comfy_cli.command import job_watcher

    urls = []

    def fake_get(url, **kw):
        urls.append(url)
        raise exc

    monkeypatch.setattr(requests, "get", fake_get)

    assert job_watcher._probe_local_server("127.0.0.1", 8188) == expected
    # ...and the probe stays cheap: the unbounded /history body grows without
    # limit on a long-lived server, which would time the probe out by itself.
    assert urls == ["http://127.0.0.1:8188/history?max_items=1"]


@pytest.mark.parametrize("status, expected", [(200, "alive"), (404, "unresponsive"), (500, "unresponsive")])
def test_probe_treats_only_http_200_as_alive(monkeypatch, status, expected):
    from comfy_cli.command import job_watcher

    monkeypatch.setattr(requests, "get", lambda url, **kw: SimpleNamespace(status_code=status))
    assert job_watcher._probe_local_server("127.0.0.1", 8188) == expected


def test_watcher_keeps_polling_an_alive_but_unresponsive_server(monkeypatch):
    """A server that is up but too slow to answer the probe must never be
    declared dead — it is still polled, and its job still completes."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    # Unresponsive forever: if it counted toward the death streak, the watcher
    # would file server_died instead of reading the completion below.
    _fake_probe(monkeypatch, [job_watcher._PROBE_UNRESPONSIVE])
    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "completed", "outputs": ["slow.png"]},
    )
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "completed"
    assert on_disk.error is None


def test_watcher_recovers_a_job_that_finished_during_the_down_streak(monkeypatch):
    """The last-chance poll at the limit wins over the server_died verdict: a
    job that completed as the probes started failing must not be reported as a
    failure with empty outputs."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    _fake_probe(monkeypatch, [False])
    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "completed", "outputs": ["late.png"]},
    )
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "completed"
    assert on_disk.outputs == ["late.png"]
    assert on_disk.error is None


def test_watcher_records_server_died_when_a_restart_loses_the_job(monkeypatch):
    """A server OOM-killed and restarted inside the detection window makes the
    next probe succeed, but the fresh process has no record of the prompt. That
    must be recorded as a death rather than polled until the 6h ceiling."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch, status="running")
    # One outage, then back up — the streak resets well short of the limit.
    _fake_probe(monkeypatch, [False, True])
    # ...but the restarted server never heard of this prompt.
    monkeypatch.setattr("comfy_cli.command.jobs._snapshot", lambda h, p, pid: None)
    monkeypatch.setattr(job_watcher, "_LOST_AFTER_RESTART_S", 0.0)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "error"
    assert on_disk.error["code"] == "server_died"
    assert on_disk.error["details"]["restarted"] is True
    assert on_disk.error["details"]["last_status"] == "running"


def test_watcher_does_not_call_a_missing_record_a_restart_without_an_outage(monkeypatch):
    """No outage seen → a record the server hasn't published yet is just a slow
    start, not a death. The restart guard must stay latched off."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    _fake_probe(monkeypatch, [True])
    monkeypatch.setattr(job_watcher, "_LOST_AFTER_RESTART_S", 0.0)
    monkeypatch.setattr(job_watcher, "_notify", lambda s: None)

    seen = []

    def snapshot(h, p, pid):
        seen.append(pid)
        # Missing for a while, then it shows up and completes.
        if len(seen) < 4:
            return None
        return {"prompt_id": pid, "status": "completed", "outputs": ["ok.png"]}

    monkeypatch.setattr("comfy_cli.command.jobs._snapshot", snapshot)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "completed"
    assert on_disk.error is None


def test_watcher_does_not_clobber_a_concurrent_cancel_with_server_died(monkeypatch):
    """`comfy jobs cancel` writing a verdict while the watcher is failing probes
    must survive: the verdict is re-read from disk, not taken from the
    watcher's stale in-memory copy."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    _watched_local_job(monkeypatch)
    _fake_probe(monkeypatch, [False])
    monkeypatch.setattr("comfy_cli.command.jobs._snapshot", lambda h, p, pid: None)

    def poll_and_cancel_out_of_band(state, **kw):
        """The last-chance poll, racing a concurrent `comfy jobs cancel`.

        The cancel lands on disk while the watcher still holds its stale
        pre-cancel copy of the state — the exact window the verdict re-read
        has to cover.
        """
        other = jobs_state.read("pid-dead")
        other.status = "cancelled"
        other.error = {"code": "cancelled", "message": "user cancelled", "details": {}}
        jobs_state.write(other)
        return False, False

    monkeypatch.setattr(job_watcher, "_poll_local_once", poll_and_cancel_out_of_band)
    notified = []
    monkeypatch.setattr(job_watcher, "_notify", notified.append)

    job_watcher.watch_job("pid-dead", where="local")

    on_disk = jobs_state.read("pid-dead")
    assert on_disk is not None
    assert on_disk.status == "cancelled"
    assert on_disk.error["code"] == "cancelled"
    # The notification reports the verdict that actually stands.
    assert notified[0].status == "cancelled"


# ---------------------------------------------------------------------------
# Bounded reads — a ComfyUI server must not be able to OOM the CLI
# ---------------------------------------------------------------------------


class _CapRecordingResp:
    """A urlopen response that records the byte cap it was read with.

    ``read(None)`` fails outright: an unbounded read is the bug being guarded
    against, and a lenient fake would let it back in.
    """

    def __init__(self, body: bytes):
        self._body = body
        self.status = 200
        self.requested: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=None):
        if n is None:
            raise AssertionError("unbounded read() — the body must be read with a cap")
        self.requested.append(n)
        return self._body[:n]


def test_http_get_json_reads_with_a_cap_not_unbounded(monkeypatch: pytest.MonkeyPatch):
    import comfy_cli.http as http_mod

    resp = _CapRecordingResp(b'{"queue_running": []}')
    monkeypatch.setattr(http_mod._PLAIN_OPENER, "open", lambda req, timeout=None: resp)

    assert jobs_mod._http_get_json("http://127.0.0.1:8188/queue") == {"queue_running": []}
    # One byte past the cap, so a body that exactly fills it is still complete.
    assert resp.requested == [http_mod.MAX_RESPONSE_BYTES + 1]


def test_http_get_json_oversize_body_is_a_runtime_error_not_a_truncated_parse(monkeypatch: pytest.MonkeyPatch):
    # RuntimeError is the single failure family every `_http_get_json` call site
    # already catches; an oversize body must join it rather than escaping as a
    # new exception type or degrading into a confusing JSON parse error.
    import comfy_cli.http as http_mod

    resp = _CapRecordingResp(b"x" * 4096)
    monkeypatch.setattr(http_mod._PLAIN_OPENER, "open", lambda req, timeout=None: resp)
    monkeypatch.setattr(
        jobs_mod,
        "read_capped",
        lambda r, url, max_bytes=8: http_mod.read_capped(r, url, max_bytes=max_bytes),
    )

    with pytest.raises(RuntimeError) as exc_info:
        jobs_mod._http_get_json("http://127.0.0.1:8188/history")
    assert "http://127.0.0.1:8188/history" in str(exc_info.value)


def test_jobs_ls_survives_an_oversize_queue_response(monkeypatch: pytest.MonkeyPatch):
    # End to end: `jobs ls` catches RuntimeError from `_http_get_json`, so an
    # oversize /queue degrades to the on-disk fallback instead of a traceback.
    import comfy_cli.http as http_mod

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    monkeypatch.setattr(http_mod._PLAIN_OPENER, "open", lambda req, timeout=None: _CapRecordingResp(b"x" * 4096))
    monkeypatch.setattr(
        jobs_mod,
        "read_capped",
        lambda r, url, max_bytes=8: http_mod.read_capped(r, url, max_bytes=max_bytes),
    )

    from typer.testing import CliRunner

    result = CliRunner().invoke(jobs_mod.app, ["ls", "--where", "local"])
    # `_gather_jobs` treats an unreadable /queue and /history as empty, so the
    # command still succeeds — what matters is that ResponseTooLarge never
    # escapes as an uncaught exception. A non-zero exit here would mean it did.
    assert result.exit_code == 0, result.output
