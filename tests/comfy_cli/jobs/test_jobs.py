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


class TestLocalSnapshotTextOutputs:
    """`_snapshot` surfaces text/STRING node outputs under an always-present
    additive `text_outputs` key: grouped-by-node full strings on the history
    branch, `{}` when there's no text or the job is still queued/running."""

    def _patch_history(self, monkeypatch, prompt_id, body):
        def fake_get(url, timeout=10.0):
            if url.endswith("/queue"):
                return {"queue_running": [], "queue_pending": []}
            if url.endswith(f"/history/{prompt_id}"):
                return {prompt_id: body}
            raise AssertionError(url)

        monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)

    def test_history_snapshot_groups_text_by_node(self, monkeypatch):
        body = {
            "status": {"completed": True, "messages": []},
            "outputs": {
                "7": {"text": ["a detailed image description that stays untruncated"]},
                "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            },
        }
        self._patch_history(monkeypatch, "txt-1", body)

        snap = jobs_mod._snapshot("h", 8188, "txt-1")
        assert snap is not None
        assert snap["status"] == "completed"
        # Full, untruncated strings grouped by producing node.
        assert snap["text_outputs"] == {"7": ["a detailed image description that stays untruncated"]}
        # Existing media keys are byte-identical to before — additive only.
        assert snap["outputs"] == ["http://h:8188/view?filename=a.png&subfolder=&type=output"]

    def test_history_snapshot_without_text_emits_empty(self, monkeypatch):
        body = {
            "status": {"completed": True, "messages": []},
            "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
        }
        self._patch_history(monkeypatch, "txt-none", body)

        snap = jobs_mod._snapshot("h", 8188, "txt-none")
        assert snap is not None
        assert snap["text_outputs"] == {}

    def test_queue_snapshot_emits_empty_text_outputs(self, monkeypatch):
        def fake_get(url, timeout=10.0):
            if url.endswith("/queue"):
                return {"queue_running": [[0, "txt-live", {"a": {}}, {}, {}]], "queue_pending": []}
            raise AssertionError(url)

        monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
        snap = jobs_mod._snapshot("h", 8188, "txt-live")
        assert snap is not None
        assert snap["status"] == "running"
        assert snap["text_outputs"] == {}


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
        """Server answers, empty `/queue` and `/history` — a fresh relaunch."""
        monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: True)
        monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: None)

        # `_server_confirms_no_record` re-queries to prove the absence is real
        # rather than a swallowed fetch error, so it needs a live-looking server.
        def fake_get(url, timeout=10.0):
            if url.endswith("/queue"):
                return {"queue_running": [], "queue_pending": []}
            return {}

        monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)

    @staticmethod
    def _server_up_but_flaky(monkeypatch: pytest.MonkeyPatch) -> None:
        """Port answers the health probe, but `/queue` and `/history` fail."""
        monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: True)
        monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: None)

        def boom(url, timeout=10.0):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(jobs_mod, "_http_get_json", boom)

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
        # The confirmed side of the fork: the server *did* disown the prompt —
        # only the record is non-terminal — so the message may say the job died
        # with the previous process. The unconfirmed twin below asserts the
        # opposite flag and the hedged tail.
        assert details["server_confirmed_no_record"] is True
        assert "died with the previous process" in err["message"]

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

    def test_unconfirmed_absence_does_not_manufacture_a_death_verdict(self, monkeypatch: pytest.MonkeyPatch):
        """A busy server whose `/queue` and `/history` fetches fail must NOT be
        read as 'the job died'. `_snapshot` returns None for a swallowed fetch
        error too, so the terminal state file stays unpublished and the message
        claims no death."""
        from comfy_cli import jobs_state

        self._server_up_but_flaky(monkeypatch)
        st = jobs_state.new(prompt_id="flaky-run", client_id="c", workflow="/tmp/wf.json", where="local")
        st.status = "error"
        st.error = {"code": "server_died", "message": "died", "details": {}}
        jobs_state.write(st)

        result = _invoke_status("flaky-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        env = _last_json(result.stdout)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "prompt_not_found"
        # The honest phrasing: unknown, not dead.
        assert "unknown" in err["message"]
        assert "died with the previous process" not in err["message"]
        assert err["details"]["server_confirmed_no_record"] is False

    def test_a_cloud_state_file_is_not_an_answer_about_a_local_target(self, monkeypatch: pytest.MonkeyPatch):
        """State files are keyed by prompt_id alone. A cloud run must not be
        returned as an authoritative local result with local output URLs."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="cloud-run", client_id="c", workflow="/tmp/wf.json", where="cloud", base_url="https://example"
        )
        st.status = "completed"
        st.outputs = ["https://cloud.example/out.png"]
        jobs_state.write(st)

        result = _invoke_status("cloud-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        err = _last_json(result.stdout)["error"]
        assert err["code"] == "prompt_not_found"
        # Falls all the way through to the bare envelope — no cloud data leaks.
        assert err["details"] == {"prompt_id": "cloud-run", "host": "127.0.0.1", "port": 8188}
        # ...but not into a dead end: `comfy jobs ls` follows the same resolved
        # target and would not list this job either, so name the query that works.
        assert (
            err["hint"] == "this prompt_id is tracked as a cloud job — try: comfy jobs status cloud-run --where cloud"
        )

    def test_a_job_from_another_local_port_is_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """Two local ComfyUI instances: port 8189's job is not port 8188's
        answer, and its output URLs must not be restamped with 8188."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="other-port", client_id="c", workflow="/tmp/wf.json", where="local", host="127.0.0.1", port=8189
        )
        st.status = "completed"
        jobs_state.write(st)

        result = _invoke_status("other-port", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        err = _last_json(result.stdout)["error"]
        assert err["code"] == "prompt_not_found"
        assert err["details"] == {"prompt_id": "other-port", "host": "127.0.0.1", "port": 8188}

    def test_a_record_whose_host_and_port_match_is_still_used(self, monkeypatch: pytest.MonkeyPatch):
        """The scoping must not break the case it exists to protect: a record
        naming this exact host:port is still the answer."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="same-port", client_id="c", workflow="/tmp/wf.json", where="local", host="127.0.0.1", port=8188
        )
        st.status = "error"
        st.error = {"code": "server_died", "message": "died", "details": {}}
        jobs_state.write(st)

        result = _invoke_status("same-port", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 0, result.output
        data = _last_json(result.stdout)["data"]
        assert data["error"]["code"] == "server_died"
        assert data["source"] == "state_file"
        assert data["server_running"] is True

    def test_a_slash_bearing_id_does_not_borrow_another_jobs_record(self, monkeypatch: pytest.MonkeyPatch):
        """`state_path` maps "/" to "_" before validating, so read("a/b") lands
        on the file for the distinct prompt "a_b". That record must not be
        reported under the queried id."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(prompt_id="a_b", client_id="c", workflow="/tmp/wf.json", where="local")
        st.status = "completed"
        st.outputs = ["http://127.0.0.1:8188/view?filename=ab.png"]
        jobs_state.write(st)

        result = _invoke_status("a/b", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        env = _last_json(result.stdout)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "prompt_not_found"
        # Crucially: not a'b's outputs reported as a/b's.
        assert "ab.png" not in json.dumps(env)

    def test_a_non_list_outputs_field_does_not_crash(self, monkeypatch: pytest.MonkeyPatch):
        """`jobs_state.read` type-checks nothing it keeps, so a mangled
        `outputs` must degrade to [] rather than shred a string into characters
        or raise inside the envelope builder."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(prompt_id="bad-outputs", client_id="c", workflow="/tmp/wf.json", where="local")
        st.status = "completed"
        jobs_state.write(st)
        # Corrupt the file the way a hand-edit would.
        path = jobs_state.state_path("bad-outputs")
        blob = json.loads(path.read_text())
        blob["outputs"] = "not-a-list"
        path.write_text(json.dumps(blob))

        result = _invoke_status("bad-outputs", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 0, result.output
        data = _last_json(result.stdout)["data"]
        assert data["outputs"] == []

    @pytest.mark.parametrize(
        ("recorded_host", "queried_host"),
        [
            ("127.0.0.1", "localhost"),  # `run --host localhost`, status with defaults
            ("localhost", "127.0.0.1"),  # and the reverse
            ("127.0.0.1", "0.0.0.0"),  # COMFY_LOCAL_URL=http://0.0.0.0:8188 — see below
            ("127.0.0.1", "LOCALHOST"),  # hostnames are case-insensitive
            ("[::1]", "::1"),  # brackets are a URL encoding, not the address
        ],
    )
    def test_a_loopback_alias_is_not_treated_as_a_different_server(
        self, monkeypatch: pytest.MonkeyPatch, recorded_host: str, queried_host: str
    ):
        """Scoping the read to the queried target must not reject the SAME
        server spelled differently — that is a silent false negative that
        discards the `server_died` attribution this fallback exists to keep.

        These spellings genuinely diverge in practice: `comfy run`'s `execute()`
        rewrites the wildcard bind `0.0.0.0` to `127.0.0.1` before writing the
        state file, while `resolve_host_port` only canonicalizes a wildcard that
        came from `config.background` — so one `COMFY_LOCAL_URL=http://0.0.0.0:8188`
        makes `run` store `127.0.0.1` and `jobs status` ask about `0.0.0.0`.
        """
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="alias-run",
            client_id="c",
            workflow="/tmp/wf.json",
            where="local",
            host=recorded_host,
            port=8188,
        )
        st.status = "error"
        st.error = {"code": "server_died", "message": "died", "details": {}}
        jobs_state.write(st)

        result = _invoke_status("alias-run", "--host", queried_host, "--port", "8188")
        assert result.exit_code == 0, result.output
        data = _last_json(result.stdout)["data"]
        assert data["error"]["code"] == "server_died"
        assert data["source"] == "state_file"

    def test_a_genuinely_different_host_is_still_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """The alias folding must not over-reach: a non-loopback host is a
        different server and its record is still not this query's answer."""
        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="remote-run",
            client_id="c",
            workflow="/tmp/wf.json",
            where="local",
            host="192.168.1.50",
            port=8188,
        )
        st.status = "completed"
        st.outputs = ["http://192.168.1.50:8188/view?filename=remote.png"]
        jobs_state.write(st)

        result = _invoke_status("remote-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 1, result.output
        env = _last_json(result.stdout)
        assert env["error"]["code"] == "prompt_not_found"
        assert env["error"]["details"] == {"prompt_id": "remote-run", "host": "127.0.0.1", "port": 8188}
        # The other server's artifact URLs must not leak into this answer.
        assert "remote.png" not in json.dumps(env)

    def test_a_cancelled_state_file_payload_is_schema_valid(self, monkeypatch: pytest.MonkeyPatch):
        """`cancelled` is terminal in `jobs_state` and is copied verbatim into
        the payload, so this path can emit it — `schemas/jobs.json` must accept
        it. Without the enum entry the published contract rejects a payload the
        command legitimately produces."""
        import jsonschema

        from comfy_cli import jobs_state

        self._server_up_with_no_record(monkeypatch)
        st = jobs_state.new(
            prompt_id="cancelled-run",
            client_id="c",
            workflow="/tmp/wf.json",
            where="local",
            host="127.0.0.1",
            port=8188,
        )
        st.status = "cancelled"
        jobs_state.write(st)

        result = _invoke_status("cancelled-run", "--host", "127.0.0.1", "--port", "8188")
        assert result.exit_code == 0, result.output
        data = _last_json(result.stdout)["data"]
        assert data["status"] == "cancelled"

        schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "jobs.json"
        jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(data)

    def test_jobs_schema_status_enum_covers_every_terminal_state(self):
        """Every `jobs_state.TERMINAL_STATUSES` value reaches a `jobs status`
        payload verbatim, so each must be a legal `status` in the schema."""
        from comfy_cli import jobs_state

        schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "jobs.json"
        enum = set(json.loads(schema_path.read_text())["properties"]["status"]["enum"])
        assert jobs_state.TERMINAL_STATUSES <= enum, sorted(jobs_state.TERMINAL_STATUSES - enum)


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


def test_reap_finalizes_nonterminal_record_with_dead_watcher_pid(monkeypatch):
    """A `running` record carrying a dead pid + that pid's REAL create_time —
    what a `comfy run --wait` killed from outside now leaves behind (BE-6641)
    — is flipped by the reap to `error`/`watcher_crashed`, surfaces under
    `--orphaned`, and the pid pair is cleared in the rewritten file."""
    import psutil

    from comfy_cli import jobs_state

    # A real process that is provably gone: spawn, capture its create_time
    # while alive, then terminate and reap it.
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        create_time = psutil.Process(p.pid).create_time()
    finally:
        # In `finally` so a create_time() failure can't leave a 30s sleeper
        # behind for the rest of the suite.
        if p.poll() is None:
            p.terminate()
        p.wait(timeout=10)

    state_dir = jobs_state.state_dir()
    _write_state(
        state_dir,
        "wait-killed",
        status="running",
        watcher_pid=p.pid,
        watcher_pid_create_time=create_time,
    )

    rows = jobs_mod._gather_local_state_files(limit=100)
    row = next(r for r in rows if r.prompt_id == "wait-killed")
    assert row.status == "error"
    assert row.error_code == "watcher_crashed"

    orphans = jobs_mod._gather_local_state_files(limit=100, orphaned_only=True)
    assert "wait-killed" in {r.prompt_id for r in orphans}

    rewritten = jobs_state.read("wait-killed")
    assert rewritten.status == "error"
    assert rewritten.error["code"] == "watcher_crashed"
    assert rewritten.watcher_pid is None
    assert rewritten.watcher_pid_create_time is None


def test_reap_leaves_nonterminal_record_without_pid_alone(monkeypatch):
    """A `running` record with NO recorded pid — what a local `--wait` that
    hit its own `--timeout` deliberately leaves (the job may still be running
    server-side) — must never be reaped or listed as an orphan."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(state_dir, "wait-timed-out", status="running", watcher_pid=None)

    rows = jobs_mod._gather_local_state_files(limit=100)
    row = next(r for r in rows if r.prompt_id == "wait-timed-out")
    assert row.status == "running"
    assert row.error_code is None

    orphans = jobs_mod._gather_local_state_files(limit=100, orphaned_only=True)
    assert "wait-timed-out" not in {r.prompt_id for r in orphans}
    assert jobs_state.read("wait-timed-out").status == "running"


def test_reap_reread_yields_to_a_writer_that_finished_first(monkeypatch):
    """The reap's read → liveness-probe → write is not atomic, and `--wait`
    runs now stamp their foreground pid, so an ordinary `comfy run --wait`
    finishing normally can land its terminal `completed` write (outputs and
    all) inside that window — every `jobs ls --watch` refresh tick re-runs the
    scan. The write must re-derive its verdict from a re-read under the lock,
    or it clobbers that result with a generic `watcher_crashed`."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(state_dir, "raced-record", status="running", watcher_pid=999_999)

    # No live process behind the pid, so the snapshot check says "reap it".
    monkeypatch.setattr(jobs_mod, "_is_watcher_alive", lambda state: False)

    real_locked = jobs_state.locked

    def _locked_after_writer_wins(prompt_id):
        # Stand-in for the racing `--wait` process: its terminal write lands
        # between our liveness probe and our own write.
        if prompt_id == "raced-record":
            winner = jobs_state.read("raced-record")
            winner.status = "completed"
            winner.outputs = ["out.png"]
            winner.watcher_pid = None
            jobs_state.write(winner)
        return real_locked(prompt_id)

    monkeypatch.setattr(jobs_state, "locked", _locked_after_writer_wins)

    rows = jobs_mod._gather_local_state_files(limit=100)
    row = next(r for r in rows if r.prompt_id == "raced-record")
    assert row.status == "completed", "the reap overwrote a verdict that landed first"
    assert row.error_code is None
    assert row.outputs == 1

    final = jobs_state.read("raced-record")
    assert final.status == "completed"
    assert final.outputs == ["out.png"]


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


# ---------------------------------------------------------------------------
# `/api/jobs/<id>` (JobDetailResponse) field names — execution_error + *_time
# ---------------------------------------------------------------------------


# A failed job exactly as the plural jobs-detail endpoint serves it: the cause
# is a structured `execution_error` object (never a top-level `error_message`)
# and the timestamps are Unix-millisecond ints (never `created_at` strings).
_FAILED_DETAIL = {
    "status": "failed",
    "execution_error": {
        "node_id": "5",
        "node_type": "KSampler",
        "exception_message": "Allocation on device 0 would exceed allowed memory",
        "exception_type": "torch.cuda.OutOfMemoryError",
        "traceback": ["  File a.py, line 1", "  File b.py, line 2"],
        "current_inputs": {"seed": [42]},
    },
    "create_time": 1_735_689_600_000,
    "update_time": 1_735_689_660_500,
}


def test_cloud_status_snapshot_reads_execution_error_and_ms_timestamps(monkeypatch):
    """The fields `/api/jobs/<id>` actually serves must land on the snapshot:
    a compact `error_message` line, the structured record, ISO timestamps."""
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(_FAILED_DETAIL))

    snap = jobs_mod._cloud_status_snapshot("pid-failed")
    assert snap is not None
    assert snap["status"] == "error"
    assert snap["error_message"] == (
        "torch.cuda.OutOfMemoryError: Allocation on device 0 would exceed allowed memory (node 5 KSampler)"
    )
    # The record rides along for `--json` consumers — minus `current_inputs`,
    # whose widget values can carry API keys.
    assert snap["execution_error"] == {
        k: v for k, v in _FAILED_DETAIL["execution_error"].items() if k != "current_inputs"
    }
    assert "current_inputs" not in snap["execution_error"]
    # `jobs status` is the documented home of the full traceback.
    assert snap["execution_error"]["traceback"] == _FAILED_DETAIL["execution_error"]["traceback"]
    assert snap["created_at"] == "2025-01-01T00:00:00+00:00"
    assert snap["updated_at"] == "2025-01-01T00:01:00.500000+00:00"
    # Not served by this endpoint — never fabricated.
    assert snap["assigned_inference"] is None


def test_cloud_status_snapshot_payload_validates_against_schema(monkeypatch):
    """The emitted payload — `execution_error` included — must satisfy the
    published `jobs status` contract."""
    import jsonschema

    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(_FAILED_DETAIL))
    snap = jobs_mod._cloud_status_snapshot("pid-failed")

    schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "jobs.json"
    schema = json.loads(schema_path.read_text())
    # `host`/`port` are stamped by the renderer, not the snapshot.
    jsonschema.Draft202012Validator(schema).validate({**snap, "host": "cloud.example", "port": 443})


def test_cloud_status_snapshot_keeps_old_shape_fallback(monkeypatch):
    """A deployment still serving the deprecated dialect (top-level
    `error_message`, ready-made `created_at`) populates the same fields."""
    payload = {
        "status": "failed",
        "error_message": "RIP to the server",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:01:00Z",
        "assigned_inference": "inf-7",
    }
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(payload))

    snap = jobs_mod._cloud_status_snapshot("pid-old")
    assert snap is not None
    assert snap["error_message"] == "RIP to the server"
    assert snap["execution_error"] is None
    assert snap["created_at"] == "2025-01-01T00:00:00Z"
    assert snap["updated_at"] == "2025-01-01T00:01:00Z"
    assert snap["assigned_inference"] == "inf-7"


def test_cloud_status_snapshot_tolerates_unusable_timestamps(monkeypatch):
    """A malformed `create_time` degrades to None — it must not raise out of
    `jobs status`."""
    payload = {"status": "running", "create_time": "not-a-number", "update_time": None}
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(payload))

    snap = jobs_mod._cloud_status_snapshot("pid-bad-ts")
    assert snap is not None
    assert snap["created_at"] is None
    assert snap["updated_at"] is None


def test_jobs_watch_cloud_failed_job_reports_the_real_cause(monkeypatch, capsys):
    """`jobs watch --where cloud` on a failed job exits 1 with ok:false and the
    server's own exception text — not the generic "ended in status 'error'"."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(_FAILED_DETAIL))

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    with pytest.raises(typer.Exit) as exc:
        jobs_mod._cloud_watch("pid-failed", poll_interval=0.01, max_wait=5)
    assert exc.value.exit_code == 1

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    env = json.loads(lines[-1])
    assert env["type"] == "envelope" and env["ok"] is False
    assert "Allocation on device 0 would exceed allowed memory" in env["error"]["message"]
    assert "ended in status" not in env["error"]["message"]
    assert env["error"]["details"]["execution_error"]["node_type"] == "KSampler"
    # The whole payload becomes `renderer.error(details=...)`, so the nested
    # record must be as trimmed as the top-level one: no widget values, and a
    # traceback capped at the same two-frame budget as `traceback_tail`.
    record = env["error"]["details"]["execution_error"]
    assert "current_inputs" not in record
    assert len(record["traceback"]) <= 2


def test_jobs_watch_cloud_terminal_envelope_trims_a_long_traceback(monkeypatch, capsys):
    """A server traceback longer than the tail budget is capped in the watch
    envelope — JSON mode auto-engages off a TTY, so an unbounded blob would
    land in CI and agent logs."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    frames = [f"  File f{i}.py, line {i}" for i in range(12)]
    detail = {
        **_FAILED_DETAIL,
        "execution_error": {**_FAILED_DETAIL["execution_error"], "traceback": frames},
    }
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(detail))

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    with pytest.raises(typer.Exit):
        jobs_mod._cloud_watch("pid-failed", poll_interval=0.01, max_wait=5)

    env = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
    assert env["error"]["details"]["execution_error"]["traceback"] == frames[-2:]
    # ...and the classified tail agrees with it, so the envelope carries the
    # same two frames twice over rather than the full blob once.
    assert env["error"]["details"]["error_message"] == env["error"]["message"]


def test_jobs_watch_cloud_prefers_the_structured_record_over_a_json_message(monkeypatch, capsys):
    """An `exception_message` that is itself JSON (API nodes raise with raw
    JSON bodies) must not be re-decoded into a fields-less dict: classify the
    structured record, not the flattened one-liner."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    detail = {
        "status": "failed",
        "execution_error": {
            "node_id": "9",
            "node_type": "ApiNode",
            "exception_message": '{"detail": "upstream refused the request"}',
        },
    }
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(detail))

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    with pytest.raises(typer.Exit):
        jobs_mod._cloud_watch("pid-json-msg", poll_interval=0.01, max_wait=5)

    env = json.loads([ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1])
    assert env["error"]["message"] == 'ApiNode (node 9): {"detail": "upstream refused the request"}'
    assert env["error"]["details"]["execution_error"]["node_type"] == "ApiNode"


def test_cloud_status_snapshot_prefers_structured_over_generic_error_message(monkeypatch):
    """A deployment serving both a terse `error_message` and a detailed
    `execution_error` must surface the detailed one."""
    detail = {**_FAILED_DETAIL, "error_message": "job failed"}
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(detail))

    snap = jobs_mod._cloud_status_snapshot("pid-both")
    assert snap["error_message"].startswith("torch.cuda.OutOfMemoryError: Allocation on device 0")


def test_cloud_status_snapshot_parses_a_string_execution_error(monkeypatch):
    """A non-dict `execution_error` (the deprecated dialect's JSON-encoded
    record) must still produce a cause rather than being discarded to None."""
    payload = {
        "status": "failed",
        "execution_error": json.dumps({"exception_message": "boom", "node_id": 3, "node_type": "VAEDecode"}),
    }
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(payload))

    snap = jobs_mod._cloud_status_snapshot("pid-str-err")
    assert snap["error_message"] == "boom (node 3 VAEDecode)"
    # The published key stays object-or-null, so a string shape lands as null.
    assert snap["execution_error"] is None


def test_cloud_status_snapshot_ignores_a_stale_error_on_a_succeeded_job(monkeypatch):
    """A non-failed job never has a cause synthesized for it — a stale record
    on a retried-then-succeeded job would fabricate an `error` row."""
    payload = {"status": "success", "execution_error": _FAILED_DETAIL["execution_error"]}

    class _SucceededClient(_FakeCloudClient):
        # The shared fake refuses `get_history` to guard the error paths; a
        # completed job legitimately fetches it (and has no outputs here).
        def get_history(self, prompt_id):
            return None

    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _SucceededClient(payload))

    snap = jobs_mod._cloud_status_snapshot("pid-ok")
    assert snap["status"] == "completed"
    assert snap["error_message"] is None


def test_poll_cloud_once_survives_a_malformed_traceback():
    """A `traceback` served as an object slices to a TypeError that only
    `_poll_cloud_once`'s *fetch* is guarded against — it would kill the
    detached watcher and strand the state file mid-flight."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    client = _FakeCloudClient(
        {"status": "failed", "execution_error": {"exception_message": "boom", "traceback": {"frame": 1}}}
    )
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    assert job_watcher._poll_cloud_once(state, client=client) is True
    assert state.status == "error"
    assert state.error["message"] == "boom"


def test_cloud_status_pretty_renders_the_execution_error_row(monkeypatch, capsys):
    """The pretty `jobs status` table shows an `error` row for a failed cloud
    job — the row that has been blank since the endpoint moved."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(jobs_mod, "_cloud_client", lambda: _FakeCloudClient(_FAILED_DETAIL))

    set_renderer(Renderer(mode=OutputMode.PRETTY, command="jobs status"))
    jobs_mod._cloud_status("pid-failed")
    out = capsys.readouterr().out
    assert "error" in out
    assert "OutOfMemoryError" in out


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        ({}, None),
        (None, None),
        ({"exception_message": "boom"}, "boom"),
        ({"exception_message": "boom", "exception_type": "ValueError"}, "ValueError: boom"),
        ({"exception_type": "ValueError"}, "ValueError"),
        ({"exception_message": "boom", "node_id": 5}, "boom (node 5)"),
        # Node 0 is a real node id — it must not be dropped as falsy.
        ({"exception_message": "boom", "node_id": 0}, "boom (node 0)"),
        ({"exception_message": "boom", "node_type": "KSampler"}, "boom (KSampler)"),
        # Node fields but no cause: state one rather than emitting the bare
        # parenthetical `(node 5 KSampler)` as the whole error line.
        ({"node_id": 5, "node_type": "KSampler"}, "ComfyUI reported an execution error. (node 5 KSampler)"),
        # Internal newlines collapse — this helper renders *one* line, and the
        # pretty `error` cell is a single Rich row.
        ({"exception_message": "boom\n  at frame\n  at frame2"}, "boom at frame at frame2"),
        # Non-dict shapes route through `execution_errors.parse_error_message`
        # rather than being discarded, so the watcher and this path agree.
        ("a string, not an object", "a string, not an object"),
        ('{"exception_message": "boom", "node_type": "KSampler"}', "boom (KSampler)"),
    ],
)
def test_execution_error_line_partial_records(err, expected):
    """The endpoint marks every `ExecutionError` field required, but the line
    builder degrades field-by-field rather than emitting `None: None (node
    None None)` if one ever goes missing."""
    assert jobs_mod._execution_error_line(err) == expected


def test_poll_cloud_once_classifies_the_structured_execution_error():
    """The background watcher polls the same `/api/jobs/<id>`, so it must read
    the same fields — otherwise every failed cloud job's state file records the
    generic "ComfyUI reported an execution error." with null timestamps."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    client = _FakeCloudClient(_FAILED_DETAIL)
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    assert job_watcher._poll_cloud_once(state, client=client) is True

    assert state.status == "error"
    assert state.error is not None
    # `classify` parses the object shape directly, so the verdict keeps the
    # node prefix and the structured fields — not just a flattened line.
    assert state.error["message"] == "KSampler (node 5): Allocation on device 0 would exceed allowed memory"
    assert state.error["details"]["exception_type"] == "torch.cuda.OutOfMemoryError"
    assert state.error["details"]["node_id"] == "5"
    assert state.error["details"]["traceback_tail"] == ["  File a.py, line 1", "  File b.py, line 2"]
    assert state.error["details"]["created_at"] == "2025-01-01T00:00:00+00:00"
    assert state.error["details"]["updated_at"] == "2025-01-01T00:01:00.500000+00:00"


def test_poll_cloud_once_cancelled_carries_iso_timestamps():
    """The cancelled branch reads the same timestamps through the same helper."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    client = _FakeCloudClient({"status": "cancelled", "create_time": 1_735_689_600_000})
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="cloud")
    assert job_watcher._poll_cloud_once(state, client=client) is True

    assert state.error["code"] == "cancelled"
    assert state.error["message"] == "Cloud job was cancelled."
    assert state.error["details"]["created_at"] == "2025-01-01T00:00:00+00:00"
    assert state.error["details"]["updated_at"] is None


def test_jobs_schema_documents_execution_error():
    """schemas/jobs.json carries the additive structured-cause key."""
    schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "jobs.json"
    schema = json.loads(schema_path.read_text())
    prop = schema["properties"]["execution_error"]
    assert prop["type"] == ["object", "null"]
    assert prop["additionalProperties"] is True


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


def test_render_status_pretty_previews_text_truncated(monkeypatch, capsys):
    """Pretty render shows a bounded per-entry text preview (first line, ~120
    chars); the full untruncated string is reserved for the `--json` path."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    set_renderer(Renderer(mode=OutputMode.PRETTY))
    tail = "TAILMARKER" + "X" * 400
    snap = {
        "prompt_id": "p",
        "status": "completed",
        "outputs": [],
        # First line is long enough to truncate; a second line must be dropped.
        "text_outputs": {"7": [f"HEADMARKER {'Y' * 130}\ndropped second line {tail}"]},
    }
    jobs_mod._render_status_pretty(snap, host="h", port=8188)
    out = capsys.readouterr().out
    assert "HEADMARKER" in out  # first line previewed
    assert "…" in out  # truncated past ~120 chars
    assert tail not in out  # second line never rendered


def test_render_status_pretty_text_preview_skips_leading_blank_lines(monkeypatch, capsys):
    """A leading blank line must not blank out the preview — the first
    *non-blank* line is what should surface, not the empty string before it."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    set_renderer(Renderer(mode=OutputMode.PRETTY))
    snap = {
        "prompt_id": "p",
        "status": "completed",
        "outputs": [],
        "text_outputs": {"7": ["\n\nactual content"]},
    }
    jobs_mod._render_status_pretty(snap, host="h", port=8188)
    out = capsys.readouterr().out
    assert "actual content" in out


def test_render_status_pretty_text_preview_is_not_rich_markup(monkeypatch, capsys):
    """Node ids/text are server-supplied and may contain `[...]` — the preview
    must render it literally instead of letting Rich interpret it as markup
    (which would otherwise corrupt output or raise on unmatched tags)."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    set_renderer(Renderer(mode=OutputMode.PRETTY))
    snap = {
        "prompt_id": "p",
        "status": "completed",
        "outputs": [],
        "text_outputs": {"7": ["[bold red]not a style tag[/] and an unmatched ]"]},
    }
    jobs_mod._render_status_pretty(snap, host="h", port=8188)
    out = capsys.readouterr().out
    assert "[bold red]not a style tag[/] and an unmatched ]" in out


def test_render_status_pretty_text_preview_bounds_entry_count(monkeypatch, capsys):
    """Many text-output entries must not blow up the pretty table — the
    preview caps the number of lines shown and notes how many were dropped."""
    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    set_renderer(Renderer(mode=OutputMode.PRETTY))
    snap = {
        "prompt_id": "p",
        "status": "completed",
        "outputs": [],
        "text_outputs": {"7": [f"entry {i}" for i in range(jobs_mod._TEXT_PREVIEW_LIMIT + 5)]},
    }
    jobs_mod._render_status_pretty(snap, host="h", port=8188)
    out = capsys.readouterr().out
    assert f"entry {jobs_mod._TEXT_PREVIEW_LIMIT - 1}" in out
    assert f"entry {jobs_mod._TEXT_PREVIEW_LIMIT}" not in out
    assert "5 more" in out


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
        # Current ComfyUI's per-step channel; `progress` is the legacy name and
        # both stay mapped so old and new servers both stream.
        "progress_state",
        "executed",
        "execution_error",
        # Current ComfyUI's end-of-prompt signals — without them a successful
        # watch only ends when a recv times out (a full `--timeout` of silence).
        "execution_success",
        "execution_interrupted",
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
    data = {"node_id": "5", "exception_message": "boom", "executed": ["1", 2]}
    jobs_mod._watch_execution_error(st, data)
    assert st.terminal is True
    assert st.end_reason == "error"
    assert st.end_details == data
    # The `executed` list is the only record of what ran before the failure.
    assert st.completed_nodes == {"1", "2"}


def test_watch_progress_state_streams_per_node_and_marks_finished_nodes():
    """`progress_state` (current ComfyUI) must produce per-step progress events
    and count finished nodes — including compute nodes that never fire
    `executed`. Regression: watch only understood the legacy `progress` type."""
    st, r = _watch_state()
    jobs_mod._watch_progress_state(
        st,
        {
            "prompt_id": "pid",
            "nodes": {
                "3": {"value": 2.0, "max": 8.0, "state": "running"},
                "4": {"value": 8.0, "max": 8.0, "state": "finished"},
            },
        },
    )
    assert [t[0] for t in r.throttled] == ["progress:3"]
    # Floats coerced to the integer|null the event schema declares.
    assert r.throttled[0][2] == {"max_hz": 10, "node": "3", "completed": 2, "total": 8, "prompt_id": "pid"}
    # The 100% line is emitted unthrottled so it can never be swallowed.
    assert r.events == [("progress", {"node": "4", "completed": 8, "total": 8, "prompt_id": "pid"})]
    assert st.completed_nodes == {"4"}


def test_watch_progress_state_reports_each_finished_node_once():
    """`progress_state` repeats every non-pending node on every message, so the
    un-throttled final flush must not re-fire for an already-finished node."""
    st, r = _watch_state()
    msg = {"prompt_id": "pid", "nodes": {"4": {"value": 8, "max": 8, "state": "finished"}}}
    jobs_mod._watch_progress_state(st, msg)
    jobs_mod._watch_progress_state(st, dict(msg))
    assert len(r.events) == 1 and r.throttled == []
    assert st.completed_nodes == {"4"}


def test_watch_progress_state_errored_node_is_final_but_not_completed():
    """An `error` node is final — it must flush its progress line once and never
    fire again — but it did NOT complete, so it stays out of `completed_nodes`."""
    st, r = _watch_state()
    msg = {"prompt_id": "pid", "nodes": {"5": {"value": 1, "max": 8, "state": "error"}}}
    jobs_mod._watch_progress_state(st, msg)
    jobs_mod._watch_progress_state(st, dict(msg))
    assert st.completed_nodes == set()
    assert "5" in st.progress_final
    assert r.throttled == []
    assert r.events == [("progress", {"node": "5", "completed": 1, "total": 8, "prompt_id": "pid"})]


def test_watch_progress_state_ignores_malformed_payloads():
    st, r = _watch_state()
    jobs_mod._watch_progress_state(st, {"prompt_id": "pid", "nodes": "not-a-dict"})
    jobs_mod._watch_progress_state(st, {"prompt_id": "pid"})
    jobs_mod._watch_progress_state(st, {"prompt_id": "pid", "nodes": {"3": "nope"}})
    assert r.throttled == [] and r.events == []
    assert st.completed_nodes == set()


def test_watch_execution_success_is_terminal_completed():
    st, r = _watch_state()
    jobs_mod._watch_execution_success(st, {"prompt_id": "pid"})
    assert st.terminal is True
    assert st.end_reason == "completed"
    assert r.events == []


def test_watch_execution_interrupted_is_terminal_cancelled_and_keeps_nodes():
    st, _ = _watch_state()
    data = {"prompt_id": "pid", "node_id": "7", "executed": ["1", "2"]}
    jobs_mod._watch_execution_interrupted(st, data)
    assert st.terminal is True
    assert st.end_reason == "cancelled"
    assert st.end_details == data
    assert st.completed_nodes == {"1", "2"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3), (3.7, 3), (None, None), ("8", None), (True, None), (float("nan"), None), (float("inf"), None)],
)
def test_progress_int_coerces_to_the_event_schema_type(value, expected):
    assert jobs_mod._progress_int(value) == expected


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
# `jobs watch` — attaching as the submitting client_id (the reason the stream
# was silent) and the history-derived completed_nodes backfill
# ---------------------------------------------------------------------------


def test_resolve_watch_client_id_prefers_the_job_state_file(monkeypatch):
    """`comfy run` records the submitting client_id on disk — cheapest source,
    and no HTTP call should be needed when it is there."""
    from comfy_cli import jobs_state

    jobs_state.write(jobs_state.new(prompt_id="pid-a", client_id="cid-from-state", workflow="w", where="local"))

    def _no_http(url, **kw):
        raise AssertionError(f"should not have queried the server: {url}")

    monkeypatch.setattr(jobs_mod, "_http_get_json", _no_http)
    assert jobs_mod._resolve_watch_client_id("127.0.0.1", 8188, "pid-a") == "cid-from-state"


def test_resolve_watch_client_id_falls_back_to_queue_extra_data(monkeypatch):
    """A prompt submitted by something else (browser, older CLI) has no state
    file — /queue's extra_data still carries the submitting client_id."""

    def fake_get(url, **kw):
        if url.endswith("/queue"):
            return {
                "queue_running": [[0, "other", {}, {"client_id": "nope"}, {}]],
                "queue_pending": [[1, "pid-b", {}, {"client_id": "cid-from-queue"}, {}]],
            }
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    assert jobs_mod._resolve_watch_client_id("127.0.0.1", 8188, "pid-b") == "cid-from-queue"


def test_resolve_watch_client_id_falls_back_to_history_then_none(monkeypatch):
    def fake_get(url, **kw):
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        if url.endswith("/history/pid-c"):
            return {"pid-c": {"prompt": [0, "pid-c", {}, {"client_id": "cid-from-history"}, {}]}}
        return {}

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    assert jobs_mod._resolve_watch_client_id("127.0.0.1", 8188, "pid-c") == "cid-from-history"
    # Nothing anywhere -> None, so the caller can warn instead of pretending.
    assert jobs_mod._resolve_watch_client_id("127.0.0.1", 8188, "pid-missing") is None


def test_resolve_watch_client_id_survives_an_unreachable_server(monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(jobs_mod, "_http_get_json", boom)
    assert jobs_mod._resolve_watch_client_id("127.0.0.1", 8188, "pid-x") is None


def test_history_completed_nodes_unions_cached_executed_and_output_nodes(monkeypatch):
    """The end-state node list must not depend on the live stream at all."""

    def fake_get(url, **kw):
        assert url.endswith("/history/pid-h")
        return {
            "pid-h": {
                "status": {
                    "completed": True,
                    "messages": [
                        ["execution_start", {"prompt_id": "pid-h"}],
                        ["execution_cached", {"nodes": [1, "2"]}],
                        ["execution_interrupted", {"executed": ["3"]}],
                    ],
                },
                "outputs": {"9": {"images": []}},
            }
        }

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    assert jobs_mod._history_completed_nodes("127.0.0.1", 8188, "pid-h") == {"1", "2", "3", "9"}


def test_history_completed_nodes_tolerates_junk(monkeypatch):
    def fake_get(url, **kw):
        return {"pid-j": {"status": {"messages": [["execution_cached"], "junk", ["x", 1]]}, "outputs": "nope"}}

    monkeypatch.setattr(jobs_mod, "_http_get_json", fake_get)
    assert jobs_mod._history_completed_nodes("127.0.0.1", 8188, "pid-j") == set()
    monkeypatch.setattr(jobs_mod, "_http_get_json", lambda url, **kw: (_ for _ in ()).throw(RuntimeError("down")))
    assert jobs_mod._history_completed_nodes("127.0.0.1", 8188, "pid-j") == set()


class _ScriptedWS:
    """A `websocket.WebSocket` stand-in that replays a scripted message list.

    Once the script is exhausted every `recv` raises the same timeout the real
    socket raises, which is how the watch loop is driven to its snapshot check.
    """

    def __init__(self, messages):
        self._messages = [m if isinstance(m, str) else json.dumps(m) for m in messages]
        self.url = None
        self.closed = False

    def connect(self, url):
        self.url = url

    def settimeout(self, _t):
        pass

    def recv(self):
        if not self._messages:
            raise jobs_mod.WebSocketTimeoutException("timed out")
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def _run_local_watch(monkeypatch, capsys, *, messages, prompt_id="pid-w", argv_extra=()):
    """Drive `jobs watch` (local, NDJSON) over a scripted WS; return the lines."""
    from typer.testing import CliRunner

    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    ws = _ScriptedWS(messages)
    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    monkeypatch.setattr(jobs_mod, "WebSocket", lambda *a, **k: ws)
    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    result = CliRunner().invoke(
        jobs_mod.app,
        ["watch", prompt_id, "--where", "local", "--timeout", "1", *argv_extra],
    )
    # NDJSON goes to the CliRunner's captured stdout, not capsys — the renderer
    # resolves its machine stream lazily, i.e. after the runner swapped it in.
    capsys.readouterr()
    lines = [json.loads(ln) for ln in result.output.splitlines() if ln.startswith("{")]
    return result, ws, lines


def test_watch_streams_events_and_reports_completed_nodes(monkeypatch, capsys):
    """The BE-6856 regression, end to end: a multi-node job must produce MORE
    THAN ONE NDJSON line during the watch, and the terminal envelope's
    `completed_nodes` must list the nodes that ran."""
    from comfy_cli import jobs_state

    jobs_state.write(jobs_state.new(prompt_id="pid-w", client_id="cid-sub", workflow="w", where="local"))
    monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: {"prompt_id": pid, "status": "running", "outputs": []})
    monkeypatch.setattr(jobs_mod, "_history_completed_nodes", lambda h, p, pid: {"1"})

    messages = [
        {"type": "status", "data": {"status": {}, "sid": "cid-sub"}},  # no prompt_id -> ignored
        {"type": "execution_cached", "data": {"prompt_id": "pid-w", "nodes": ["1"]}},
        {"type": "executing", "data": {"prompt_id": "pid-w", "node": "3"}},
        {
            "type": "progress_state",
            "data": {"prompt_id": "pid-w", "nodes": {"3": {"value": 4, "max": 8, "state": "running"}}},
        },
        {
            "type": "progress_state",
            "data": {"prompt_id": "pid-w", "nodes": {"3": {"value": 8, "max": 8, "state": "finished"}}},
        },
        {"type": "executed", "data": {"prompt_id": "pid-w", "node": "9", "output": {}}},
        {"type": "execution_success", "data": {"prompt_id": "pid-w"}},
    ]
    result, ws, lines = _run_local_watch(monkeypatch, capsys, messages=messages)

    assert result.exit_code == 0, result.output
    # 1. The watch attached as the submitting session — the whole reason events
    #    reach us at all (ComfyUI addresses them to that sid only).
    assert "clientId=cid-sub" in ws.url
    # 2. Intermediate events actually reached the stream.
    assert len(lines) > 1, f"expected a stream, got a single envelope: {lines}"
    types = [ln.get("type") for ln in lines[:-1]]
    assert "execution_cached" in types and "executing" in types and "executed" in types
    assert types.count("progress") >= 2, types
    # 3. The terminal envelope is last, ok, and names the nodes that ran.
    env = lines[-1]
    assert env["type"] == "envelope" and env["ok"] is True
    assert env["data"]["status"] == "completed"
    assert env["data"]["completed_nodes"] == ["1", "3", "9"]
    assert env["data"]["attached"] is True and env["data"]["client_id"] == "cid-sub"


def test_watch_terminal_envelope_backfills_completed_nodes_without_events(monkeypatch, capsys):
    """Symptom 2 is independent of streaming: even with zero WS events, the
    envelope's `completed_nodes` comes from the server's own /history record."""
    snapshots = iter(
        [
            {"prompt_id": "pid-w", "status": "running", "outputs": []},
            {"prompt_id": "pid-w", "status": "completed", "outputs": ["http://x/view?f=1"]},
        ]
    )
    monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: next(snapshots, None))
    monkeypatch.setattr(jobs_mod, "_resolve_watch_client_id", lambda h, p, pid: None)
    monkeypatch.setattr(jobs_mod, "_history_completed_nodes", lambda h, p, pid: {"4", "1"})

    result, ws, lines = _run_local_watch(monkeypatch, capsys, messages=[])
    assert result.exit_code == 0, result.output
    env = lines[-1]
    assert env["data"]["status"] == "completed"
    assert env["data"]["completed_nodes"] == ["1", "4"]
    # Unresolvable id -> a fresh one, flagged so a caller knows why it saw no
    # events rather than guessing the job was silent.
    assert env["data"]["attached"] is False
    assert "clientId=" in ws.url


def test_watch_already_terminal_job_still_lists_completed_nodes(monkeypatch, capsys):
    """The short-circuit path (job already finished) also carries the node list —
    and the `client_id`/`attached` pair, so a consumer reading `data.attached`
    never hits a missing key on this exit path."""
    import jsonschema

    monkeypatch.setattr(
        jobs_mod,
        "_snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "completed", "outputs": [], "host": h, "port": p},
    )
    monkeypatch.setattr(jobs_mod, "_history_completed_nodes", lambda h, p, pid: {"2", "1"})
    result, _ws, lines = _run_local_watch(monkeypatch, capsys, messages=[])
    assert result.exit_code == 0, result.output
    assert lines[-1]["data"]["completed_nodes"] == ["1", "2"]
    # Nothing was attached: no socket was ever opened on this path.
    assert lines[-1]["data"]["client_id"] is None
    assert lines[-1]["data"]["attached"] is False

    schema_path = Path(jobs_mod.__file__).parent.parent / "schemas" / "jobs.json"
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(lines[-1]["data"])


def test_watch_client_id_flag_overrides_resolution(monkeypatch, capsys):
    monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: {"prompt_id": pid, "status": "running", "outputs": []})
    monkeypatch.setattr(jobs_mod, "_resolve_watch_client_id", lambda h, p, pid: "resolved")
    monkeypatch.setattr(jobs_mod, "_history_completed_nodes", lambda h, p, pid: set())
    messages = [{"type": "execution_success", "data": {"prompt_id": "pid-w"}}]
    _result, ws, lines = _run_local_watch(
        monkeypatch, capsys, messages=messages, argv_extra=("--client-id", "forced id")
    )
    # Percent-encoded into the query string, never interpolated raw.
    assert "clientId=forced%20id" in ws.url
    assert lines[-1]["data"]["client_id"] == "forced id"


def test_watch_terminal_envelope_validates_against_the_jobs_schema(monkeypatch, capsys):
    """The additive `client_id`/`attached` keys are a published contract."""
    import jsonschema

    monkeypatch.setattr(jobs_mod, "_snapshot", lambda h, p, pid: {"prompt_id": pid, "status": "running", "outputs": []})
    monkeypatch.setattr(jobs_mod, "_resolve_watch_client_id", lambda h, p, pid: "cid")
    monkeypatch.setattr(jobs_mod, "_history_completed_nodes", lambda h, p, pid: {"1"})
    messages = [{"type": "execution_success", "data": {"prompt_id": "pid-w"}}]
    _result, _ws, lines = _run_local_watch(monkeypatch, capsys, messages=messages)

    schema_path = Path(jobs_mod.__file__).parent.parent / "schemas" / "jobs.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(lines[-1]["data"])
    assert schema["properties"]["attached"]["type"] == "boolean"
    assert schema["properties"]["client_id"]["type"] == ["string", "null"]


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


def test_state_file_payload_carries_the_grouped_output_keys(monkeypatch: pytest.MonkeyPatch):
    """Shape parity with the live snapshot. `_snapshot` always emits
    `outputs_by_node`, `outputs_by_item` and `workflow_size`; a consumer that
    indexes them on a `jobs status` success payload would hit a `KeyError` on
    the state-file source alone. They are present but empty — the file records
    output URLs flat, so the grouping genuinely cannot be reconstructed."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    st = jobs_state.new(prompt_id="shape-run", client_id="c", workflow="/tmp/wf.json", where="local")
    st.status = "completed"
    st.outputs = ["http://127.0.0.1:8188/view?filename=out.png"]
    jobs_state.write(st)

    result = _invoke_status("shape-run", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 0, result.output
    data = _last_json(result.stdout)["data"]
    assert data["outputs"] == ["http://127.0.0.1:8188/view?filename=out.png"]
    assert data["outputs_by_node"] == {}
    assert data["outputs_by_item"] == {}
    assert data["workflow_size"] is None


def test_server_down_cloud_job_hint_points_at_the_cloud_query(monkeypatch: pytest.MonkeyPatch):
    """Server-down twin of the same redirect: a cloud-tracked prompt asked
    about locally is told to use `--where cloud`, not "run: comfy launch"."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    st = jobs_state.new(
        prompt_id="cloud-down", client_id="c", workflow="/tmp/wf.json", where="cloud", base_url="https://example"
    )
    st.status = "completed"
    jobs_state.write(st)

    result = _invoke_status("cloud-down", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 1, result.output
    err = _last_json(result.stdout)["error"]
    assert err["code"] == "server_not_running"
    assert "--where cloud" in err["hint"]


def test_untracked_prompt_keeps_the_default_hints(monkeypatch: pytest.MonkeyPatch):
    """The redirect is conditional: with no state file at all, both bare
    envelopes keep the hint they have always carried."""
    monkeypatch.setattr(jobs_mod, "check_comfy_server_running", lambda port, host: False)
    result = _invoke_status("no-such-id", "--host", "127.0.0.1", "--port", "65431")
    assert result.exit_code == 1, result.output
    assert _last_json(result.stdout)["error"]["hint"] == "run: comfy launch"


# ---------------------------------------------------------------------------
# `jobs ls` rows carry the server-death attribution (`error_code`)
# ---------------------------------------------------------------------------


def test_state_error_code_extraction_is_defensive():
    """Only a non-empty string `error.code` survives — a state file is
    hand-editable, and `error_code` is typed `string | null` in the schema."""
    assert jobs_mod._state_error_code({"code": "server_died"}, "error") == "server_died"
    assert jobs_mod._state_error_code(None, "error") is None
    assert jobs_mod._state_error_code({}, "error") is None
    assert jobs_mod._state_error_code({"code": ""}, "error") is None
    assert jobs_mod._state_error_code({"code": 500}, "error") is None
    assert jobs_mod._state_error_code("server_died", "error") is None
    assert jobs_mod._state_error_code(["server_died"], "error") is None


@pytest.mark.parametrize("status", ["queued", "running", "pending", "executing", "completed"])
def test_state_error_code_ignores_a_code_on_a_job_that_has_not_failed(status):
    """`job_watcher._poll_local_once`/`_poll_cloud_once` park a transient
    `watcher_poll_error` on the state file *without* moving the status off
    `queued`/`running`, and only clear it on a later poll that returns a
    snapshot — so a healthy in-flight job holds that code for many cycles.
    Surfacing it would advertise a failure cause for a job that hasn't failed
    (and contradict the schema, which promises null on every non-failed row)."""
    assert jobs_mod._state_error_code({"code": "watcher_poll_error"}, status) is None


def test_gather_local_state_files_ignores_a_poll_error_on_a_running_job():
    """End to end through the gather, not just the helper: a running job whose
    last poll blipped must still list as running with no cause attached."""
    from comfy_cli import jobs_state

    _write_state(
        jobs_state.state_dir(),
        "blipped",
        status="running",
        error={"code": "watcher_poll_error", "message": "Connection reset by peer"},
    )

    (row,) = jobs_mod._gather_local_state_files(limit=100)
    assert row.status == "running"
    assert row.error_code is None


def test_state_scalar_narrowing_keeps_a_bad_state_file_from_breaking_the_listing():
    """`jobs_state.read` type-checks nothing it keeps, so `where`,
    `workflow_path`, and `updated_at` — all published with strict types — need
    the same defensiveness `error_code` gets. A numeric `updated_at` is the
    sharp case: it reaches `_merge_jobs`'s `sort_key` and raises `TypeError`
    comparing int against str, aborting the whole listing rather than one row."""
    assert jobs_mod._state_str("2026-08-04T00:00:00+00:00") == "2026-08-04T00:00:00+00:00"
    assert jobs_mod._state_str(None) is None
    assert jobs_mod._state_str("") is None
    assert jobs_mod._state_str(1754265600) is None
    assert jobs_mod._state_where("cloud") == "cloud"
    assert jobs_mod._state_where("local") == "local"
    # Missing, empty, or a legacy/hand-edited value outside the published enum.
    assert jobs_mod._state_where(None) == "local"
    assert jobs_mod._state_where("") == "local"
    assert jobs_mod._state_where("remote") == "local"


def test_gather_local_state_files_carries_error_code():
    """A state-file row reports why the job failed, not just that it did."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(
        state_dir,
        "died-job",
        status="error",
        error={"code": "server_died", "message": "Lost connection while job died-job was running"},
    )
    _write_state(state_dir, "ok-job", status="completed")

    rows = {r.prompt_id: r for r in jobs_mod._gather_local_state_files(limit=100)}
    assert rows["died-job"].error_code == "server_died"
    assert rows["ok-job"].error_code is None, "a healthy row must not invent an error code"


def test_gather_local_state_files_reports_the_reaped_watcher_code(monkeypatch):
    """The stale-watcher reap rewrites the file to `watcher_crashed`; the row it
    then builds must carry that code, or `--orphaned` still can't say why."""
    from comfy_cli import jobs_state

    monkeypatch.setattr(jobs_mod, "_is_watcher_alive", lambda state: False)
    _write_state(jobs_state.state_dir(), "reaped", status="running", watcher_pid=999999)

    (row,) = jobs_mod._gather_local_state_files(limit=100)
    assert row.status == "error"
    assert row.error_code == "watcher_crashed"


def test_jobs_ls_sweeps_stranded_atomic_write_temps():
    """The command that reaps crashed watchers also sweeps the temps those same
    unclean deaths strand — without disturbing the state files it lists."""
    import time as _time

    from typer.testing import CliRunner

    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(state_dir, "job-1", status="completed")
    corpse = state_dir / "job-1.json.abcd1234.tmp"
    corpse.write_text("half a write")
    old = _time.time() - 7200
    os.utime(corpse, (old, old))
    # A coincidental temp whose stem is not a state file: same mkstemp shape,
    # not ours, must survive.
    bystander = state_dir / "notes.abcd1234.tmp"
    bystander.write_text("mine, not yours")
    os.utime(bystander, (old, old))

    result = CliRunner().invoke(jobs_mod.app, ["ls", "--local-only", "--where", "local"])

    assert result.exit_code == 0, result.output
    assert not corpse.exists(), "the stranded atomic-write temp should have been swept"
    assert bystander.exists(), "a temp with no state-file stem is not ours to delete"
    assert (state_dir / "job-1.json").exists()


def test_gather_local_state_files_does_not_mutate_temps():
    """The listing helper is read-only: the sweep belongs to the ``ls`` command
    so ``--watch``'s 2s refresh doesn't re-run it on every table build."""
    import time as _time

    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(state_dir, "job-1", status="completed")
    corpse = state_dir / "job-1.json.abcd1234.tmp"
    corpse.write_text("half a write")
    old = _time.time() - 7200
    os.utime(corpse, (old, old))

    (row,) = jobs_mod._gather_local_state_files(limit=10)
    assert row.prompt_id == "job-1"
    assert corpse.exists()


def _row(prompt_id: str, status: str, **kw) -> jobs_mod.JobRow:
    return jobs_mod.JobRow(
        prompt_id=prompt_id,
        status=status,
        queue_position=None,
        elapsed_seconds=None,
        workflow_size=None,
        outputs=0,
        **kw,
    )


def test_merge_carries_error_code_onto_a_superseding_server_row():
    """`/queue` and `/history` carry no error code, so a server row that wins
    the merge would otherwise drop the cause the state file recorded."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("job-1", "error", error_code="execution_error", updated_at="2026-08-04T00:00:00+00:00")],
            [_row("job-1", "error")],
        )
    }
    assert merged["job-1"].error_code == "execution_error"


def test_merge_drops_a_stale_error_code_when_the_server_says_completed():
    """The carry-over is scoped to failure statuses: a server that reports the
    prompt as completed must not be annotated with a stale `server_died`."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("job-2", "error", error_code="server_died")],
            [_row("job-2", "completed")],
        )
    }
    assert merged["job-2"].status == "completed"
    assert merged["job-2"].error_code is None


def test_merge_prefers_the_server_rows_own_error_code():
    """Carry-over only fills a gap — it never overwrites a code the server row
    already has."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("job-3", "error", error_code="server_died")],
            [_row("job-3", "error", error_code="execution_error")],
        )
    }
    assert merged["job-3"].error_code == "execution_error"


def test_merge_reads_the_prior_row_from_the_state_snapshot_not_the_running_map():
    """`/queue` and `/history` are fetched separately, so one gather can yield
    two rows for a prompt caught mid-transition. Looking the prior code up from
    the accumulating map lets the `running` row clobber the state row first,
    and the `error` row that follows then finds nothing to inherit."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("job-4", "error", error_code="execution_error")],
            [_row("job-4", "running"), _row("job-4", "error")],
        )
    }
    assert merged["job-4"].status == "error"
    assert merged["job-4"].error_code == "execution_error"


def test_merge_ignores_a_code_recorded_next_to_a_healthy_state_status():
    """The gate is on both sides: a code sitting next to a non-failure state
    status is a watcher poll blip, not this job's cause, so a server row that
    genuinely failed must not be attributed to it."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("job-5", "running", error_code="watcher_poll_error")],
            [_row("job-5", "error")],
        )
    }
    assert merged["job-5"].status == "error"
    assert merged["job-5"].error_code is None


def test_merge_carries_the_other_state_only_fields_onto_a_server_row():
    """`workflow_path` and `updated_at` are state-file-only too. Blanking the
    first drops the only field that says which workflow a prompt_id was;
    blanking the second sorts a terminal row to epoch 0, below every dated row,
    where the caller's `[:limit]` slice can drop a fresh completion."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [
                _row(
                    "job-6",
                    "completed",
                    workflow_path="/tmp/wf.json",
                    updated_at="2026-08-04T00:00:00+00:00",
                )
            ],
            [_row("job-6", "completed")],
        )
    }
    assert merged["job-6"].workflow_path == "/tmp/wf.json"
    assert merged["job-6"].updated_at == "2026-08-04T00:00:00+00:00"


def test_merge_keeps_fresh_server_side_completions_within_the_limit():
    """The consequence the carry-over above prevents, end to end: without an
    `updated_at`, the freshly completed job the server knows about sorts below
    a week-old one and falls outside a caller's `[:1]` slice."""
    fresh = _row("fresh", "completed", updated_at="2026-08-04T00:00:00+00:00")
    stale = _row("stale", "completed", updated_at="2026-07-28T00:00:00+00:00")
    merged = jobs_mod._merge_jobs([fresh, stale], [_row("fresh", "completed")])
    assert [r.prompt_id for r in merged][:1] == ["fresh"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("non_retryable_error", "error"),
        ("lost", "error"),
        ("canceled", "cancelled"),
        ("cancelled", "cancelled"),
        ("failed", "error"),
        ("success", "completed"),
    ],
)
def test_cloud_row_normalizes_the_failure_spellings(raw, expected):
    """Cloud's other failure spellings used to pass through raw, missing
    `_ERROR_STATUSES` — so the merge dropped the state file's `error_code` for
    exactly the jobs that had failed."""
    assert jobs_mod._cloud_job_to_row({"id": "j", "status": raw}).status == expected


def test_cloud_row_is_marked_as_a_cloud_row():
    """`where` is published with an enum as a per-row discriminator for a
    follow-up `jobs status`/`jobs cancel`. Taking JobRow's "local" default here
    made every row under `jobs ls --where cloud` claim `local` while the
    envelope said `cloud`, superseding the state row that had it right."""
    assert jobs_mod._cloud_job_to_row({"id": "j", "status": "running"}).where == "cloud"


def test_merge_keeps_the_state_error_code_for_a_cloud_failure_spelling():
    """The two fixes above together: a cloud `non_retryable_error` row now
    normalizes to `error`, so it clears the gate and keeps the cause."""
    merged = {
        r.prompt_id: r
        for r in jobs_mod._merge_jobs(
            [_row("cj", "error", where="cloud", error_code="server_died")],
            [jobs_mod._cloud_job_to_row({"id": "cj", "status": "non_retryable_error"})],
        )
    }
    assert merged["cj"].status == "error"
    assert merged["cj"].where == "cloud"
    assert merged["cj"].error_code == "server_died"


def test_ls_payload_names_the_server_death(capsys, monkeypatch):
    """Acceptance: after a server death, `comfy jobs ls` — the escape hatch
    `jobs status` points at — reports both the failure and its cause."""
    from comfy_cli import jobs_state

    state_dir = jobs_state.state_dir()
    _write_state(
        state_dir,
        "oom-job",
        status="error",
        error={"code": "server_died", "message": "Lost connection to ComfyUI while job oom-job was running"},
    )
    _write_state(state_dir, "fine-job", status="completed")
    monkeypatch.delenv("COMFY_WHERE", raising=False)

    data = _ls_payload(capsys, limit=100)
    by_id = {j["prompt_id"]: j for j in data["jobs"]}
    assert by_id["oom-job"]["status"] == "error"
    assert by_id["oom-job"]["error_code"] == "server_died"
    assert by_id["oom-job"]["workflow_path"] == "/tmp/oom-job.json"
    # Present on every row (agents can index it unconditionally), null when
    # there is nothing to attribute.
    assert by_id["fine-job"]["error_code"] is None


def test_ls_payload_validates_against_the_jobs_schema(capsys, monkeypatch):
    """`error_code` is a published contract field, not just an emitted one."""
    import jsonschema

    from comfy_cli import jobs_state

    _write_state(
        jobs_state.state_dir(),
        "sch-job",
        status="cancelled",
        error={"code": "cancelled", "message": "Cancelled by user"},
    )
    monkeypatch.delenv("COMFY_WHERE", raising=False)
    data = _ls_payload(capsys, limit=100)

    schema_path = Path(jobs_mod.__file__).parent.parent / "schemas" / "jobs.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(data)
    assert data["jobs"][0]["error_code"] == "cancelled"
    # The row `status` this asserts on is validated by `jobs.items`, whose
    # `status` is a bare string — not by the top-level `status` enum, which
    # describes the single-job `jobs status` envelope. Row statuses are
    # deliberately unconstrained (an unrecognized cloud status passes through
    # raw rather than being dropped), so validate the field that *is* the
    # contract here — `error_code` alongside a `cancelled` row — rather than
    # asserting against an enum that never sees this value.
    assert data["jobs"][0]["status"] == "cancelled"
    assert schema["properties"]["jobs"]["items"]["properties"]["error_code"]["type"] == ["string", "null"]


# ---------------------------------------------------------------------------
# schemas/jobs.json — host/port is required only for host/port-shaped payloads
# ---------------------------------------------------------------------------


def _jobs_schema() -> dict:
    return json.loads((Path(jobs_mod.__file__).parent.parent / "schemas" / "jobs.json").read_text())


def _fake_cloud_client(raw_status: str, outputs: list[dict] | None = None):
    """Stand-in for `comfy_cli.api.Client` covering everything the cloud
    status/watch path touches: the three calls `_cloud_status_snapshot` makes
    and the `target.base_url` it stamps onto every snapshot."""
    return SimpleNamespace(
        target=SimpleNamespace(base_url="https://api.comfy.example"),
        get_job_status=lambda pid: {
            "status": raw_status,
            "assigned_inference": "inference-1",
            "error_message": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        },
        get_history=lambda pid: {"outputs": {}},
        extract_outputs=lambda record: list(outputs or []),
    )


def test_jobs_watch_cloud_terminal_envelope_is_schema_conformant(monkeypatch: pytest.MonkeyPatch):
    """`comfy --json jobs watch --where cloud <id>` emits a terminal payload
    that validates against `schemas/jobs.json` with ZERO tolerated errors.

    Cloud has no host/port to report — `_cloud_status_snapshot` reports the
    `base_url` it polled instead — so the schema's top-level requirement is
    conditional: a payload carrying `base_url` is exempt from `host` + `port`.
    """
    import jsonschema
    from typer.testing import CliRunner

    from comfy_cli.output import Renderer, set_renderer
    from comfy_cli.output.renderer import OutputMode

    monkeypatch.setattr(jobs_mod, "_is_cloud", lambda w: True)
    monkeypatch.setattr(jobs_mod, "cloud_preflight_or_exit", lambda: None)
    monkeypatch.setattr(
        jobs_mod,
        "_cloud_client",
        lambda: _fake_cloud_client("success", outputs=[{"url": "https://cdn.example/out.png", "node_id": "9"}]),
    )

    set_renderer(Renderer(mode=OutputMode.NDJSON, command="jobs watch"))
    result = CliRunner().invoke(jobs_mod.app, ["watch", "cloud-p1", "--where", "cloud"])
    assert result.exit_code == 0, result.output

    data = _last_json(result.stdout)["data"]
    # The shape the conditional exists for: base_url present, host/port absent.
    assert data["base_url"] == "https://api.comfy.example"
    assert "host" not in data and "port" not in data
    assert data["status"] == "completed"
    assert data["outputs"] == ["https://cdn.example/out.png"]

    errors = list(jsonschema.Draft202012Validator(_jobs_schema()).iter_errors(data))
    assert errors == [], [e.message for e in errors]


def test_jobs_schema_still_requires_host_and_port_without_base_url():
    """The `if base_url / else host+port` conditional must not weaken the local
    guarantee: a payload with neither `base_url` nor `host`/`port` is still a
    contract violation. Without this, a future edit to the conditional could
    silently drop the requirement for every payload, not just cloud ones."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(_jobs_schema())

    # Local-shaped payload missing both — the `else` branch must reject it.
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"prompt_id": "p", "status": "completed"})

    # A partial local payload is still short of the requirement.
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"prompt_id": "p", "status": "completed", "host": "127.0.0.1"})

    # Both legitimate shapes still validate.
    validator.validate({"prompt_id": "p", "status": "completed", "host": "127.0.0.1", "port": 8188})
    validator.validate({"prompt_id": "p", "status": "completed", "base_url": "https://api.comfy.example"})


def test_jobs_schema_empty_base_url_does_not_buy_the_host_port_exemption():
    """An empty `base_url` names no source URL, so it must not be the key that
    unlocks the cloud exemption. Guarded twice on purpose: `minLength` on the
    property rejects the empty string outright, and the same `minLength` inside
    the `if` keeps such a payload in the `else` branch, where it still owes
    `host` + `port` — so neither guard alone is load-bearing."""
    import jsonschema

    schema = _jobs_schema()
    validator = jsonschema.Draft202012Validator(schema)

    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"prompt_id": "p", "status": "completed", "base_url": ""})

    # The `if` branch alone: strip the property-level `minLength` and the empty
    # `base_url` must STILL be rejected, now for missing `host` + `port`.
    del schema["properties"]["base_url"]["minLength"]
    errors = list(
        jsonschema.Draft202012Validator(schema).iter_errors({"prompt_id": "p", "status": "completed", "base_url": ""})
    )
    assert errors, "empty base_url must fall to the `else` branch and be held to host+port"
    assert any("host" in e.message for e in errors), [e.message for e in errors]

    # A real cloud payload is untouched by either guard.
    validator.validate({"prompt_id": "p", "status": "completed", "base_url": "https://api.comfy.example"})
