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


def test_orphaned_flag_visible_in_help():
    """The flag must be documented on `jobs ls --help` so agents can
    discover it without reading source."""
    res = _run(["jobs", "ls", "--help"])
    assert res.returncode == 0
    assert "--orphaned" in res.stdout


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
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                return _Resp(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


def test_jobs_cancel_local_hits_queue_and_interrupt(monkeypatch: pytest.MonkeyPatch):
    """`comfy jobs cancel <id>` on local must POST to both /queue (for
    pending) and /interrupt (for running). Both are best-effort — one
    failing doesn't abort the other."""
    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    calls = _capture_urlopen(
        monkeypatch,
        {
            "/queue": b"{}",
            "/interrupt": b"{}",
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "local"])
    assert result.exit_code == 0, result.output

    # Both endpoints called, POST.
    urls = [c["url"] for c in calls]
    assert any("/queue" in u for u in urls), urls
    assert any("/interrupt" in u for u in urls), urls
    methods = {c["method"] for c in calls}
    assert methods == {"POST"}

    # /queue payload carries the prompt_id.
    queue_call = next(c for c in calls if "/queue" in c["url"])
    # The body is on the Request, not in our captured dict — re-derive from headers.
    assert queue_call["headers"].get("Content-type") == "application/json"


def test_jobs_cancel_local_tolerates_one_failure(monkeypatch: pytest.MonkeyPatch):
    """If /queue 404s but /interrupt 200s (job is running not pending), the
    cancel still succeeds. Mirrors the real ComfyUI server's behavior."""
    import urllib.error

    from typer.testing import CliRunner

    monkeypatch.setattr(jobs_mod, "_server_or_error", lambda h, p, **kw: True)
    _capture_urlopen(
        monkeypatch,
        {
            "/queue": urllib.error.HTTPError("http://x/queue", 404, "Not Found", {}, None),
            "/interrupt": b"{}",
        },
    )
    runner = CliRunner()
    result = runner.invoke(jobs_mod.app, ["cancel", "prompt-abc", "--where", "local"])
    assert result.exit_code == 0, result.output


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
    monkeypatch.setattr(jobs_mod, "_cloud_preflight_or_exit", lambda: None)

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
    monkeypatch.setattr(jobs_mod, "_cloud_preflight_or_exit", lambda: None)

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
    res = _run(["run", "--help"])
    assert "--wait" in res.stdout


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
        lambda url, **kw: ({} if "/queue" in url else body),
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
    from comfy_cli.command import job_watcher
    from comfy_cli import jobs_state
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
