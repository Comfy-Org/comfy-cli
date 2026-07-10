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

import pytest

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


# ---------------------------------------------------------------------------
# --where routing — top-level flag must be honored, not just per-command
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# `jobs cancel` — local + cloud paths
# ---------------------------------------------------------------------------


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

        def read(self):
            return self.body

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
                if isinstance(payload, Exception):
                    raise payload
                return _Resp(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
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
            "GET /queue": {"queue_running": [[0, "prompt-running", {}, {}, {}]], "queue_pending": []},
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
    """_poll_local_once must return True (terminal) and set state.status='cancelled'
    when _snapshot reports status='cancelled'."""
    from comfy_cli import jobs_state
    from comfy_cli.command import job_watcher

    monkeypatch.setattr(
        "comfy_cli.command.jobs._snapshot",
        lambda h, p, pid: {"prompt_id": pid, "status": "cancelled", "outputs": []},
    )
    state = jobs_state.new(prompt_id="pid", client_id="c", workflow="w", where="local")
    assert job_watcher._poll_local_once(state, host=None, port=None) is True
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
