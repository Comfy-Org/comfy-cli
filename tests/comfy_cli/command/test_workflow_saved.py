"""Tests for `comfy workflow list/get/save/delete` — cloud-saved workflows.

These commands wrap the `/api/workflows` surface documented at
docs.comfy.org/api-reference/cloud/workflow. All HTTP is mocked; the
live round-trip is verified manually.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    return json.loads(captured.strip().splitlines()[-1])


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch):
    """Pin resolve_target to a known cloud Target — same shape used by `models search` tests."""
    from comfy_cli.target import Target

    fake = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    return fake


@pytest.fixture
def local_target(monkeypatch: pytest.MonkeyPatch):
    from comfy_cli.target import Target

    fake = Target(
        kind="local",
        base_url="http://127.0.0.1:8188",
        path_prefix="",
        history_path="history",
        host="127.0.0.1",
        port=8188,
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    return fake


def _fake_resp(body: bytes, status: int = 200):
    class _R:
        def __init__(self):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n: int | None = None):
            return body if n is None else body[:n]

    return _R()


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, routes: dict):
    calls: list[dict] = []

    def _fake(req, timeout=None):
        url = req.full_url
        method = req.get_method()
        body = req.data
        calls.append({"url": url, "method": method, "body": body, "headers": dict(req.headers)})
        for needle, payload in routes.items():
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, tuple):
                    body_bytes, status = payload
                    return _fake_resp(body_bytes, status)
                return _fake_resp(json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


_WORKFLOW_LIST_RESPONSE = {
    "data": [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "name": "Hello flux",
            "description": "first test",
            "default_view": "workflow",
            "latest_version": 1,
            "created_at": "2026-05-23T01:00:00Z",
            "updated_at": "2026-05-23T01:00:00Z",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "name": "Wan i2v",
            "default_view": "workflow",
            "latest_version": 3,
            "created_at": "2026-05-23T02:00:00Z",
            "updated_at": "2026-05-23T03:00:00Z",
        },
    ],
    "pagination": {"total": 2, "limit": 20, "offset": 0},
}


_WORKFLOW_CONTENT_RESPONSE = {
    "id": "version-uuid",
    "version": 3,
    "workflow_json": {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "VAEDecode", "inputs": {}},
    },
    "created_by": "user-uuid",
    "created_at": "2026-05-23T02:00:00Z",
}


# ---------------------------------------------------------------------------
# Local route — all four subcommands must reject with workflow_saved_local_unsupported
# ---------------------------------------------------------------------------


class TestLocalIsRejectedCleanly:
    def test_list_local_unsupported(self, local_target, capsys):
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_saved_local_unsupported"

    def test_get_local_unsupported(self, local_target, capsys):
        env = _run(["get", "any-id", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_saved_local_unsupported"

    def test_delete_local_unsupported(self, local_target, capsys):
        env = _run(["delete", "any-id", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_saved_local_unsupported"

    def test_save_local_unsupported(self, local_target, tmp_path, capsys):
        wf = tmp_path / "wf.json"
        wf.write_text(json.dumps({"1": {"class_type": "X"}}))
        env = _run(["save", str(wf), "--name", "test", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_saved_local_unsupported"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_returns_paginated_rows(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/workflows": _WORKFLOW_LIST_RESPONSE})
        env = _run(["list", "--where", "cloud", "--limit", "5"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 2
        names = [r["name"] for r in env["data"]["workflows"]]
        assert names == ["Hello flux", "Wan i2v"]
        # Auth header present.
        h = {k.lower(): v for k, v in calls[0]["headers"].items()}
        assert h.get("x-api-key") == "test-key"

    def test_passes_name_filter_through(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/workflows": _WORKFLOW_LIST_RESPONSE})
        _run(["list", "--name", "wan", "--where", "cloud"], capsys)
        assert "name=wan" in calls[0]["url"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_writes_to_stdout_or_file(self, cloud_target, tmp_path, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid/content": _WORKFLOW_CONTENT_RESPONSE})
        out = tmp_path / "out.json"
        env = _run(["get", "wf-uuid", "--out", str(out), "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "wf-uuid"
        assert env["data"]["version"] == 3
        assert env["data"]["node_count"] == 2
        assert out.exists()
        on_disk = json.loads(out.read_text())
        assert on_disk["1"]["class_type"] == "KSampler"

    def test_404_surfaces_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError("https://x/content", 404, "Not Found", {}, io.BytesIO(b"{}"))
        _patch_urlopen(monkeypatch, {"/api/workflows/ghost/content": err})
        env = _run(["get", "ghost", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestSave:
    def test_posts_workflow_payload(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path = tmp_path / "wf.json"
        wf_data = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
        wf_path.write_text(json.dumps(wf_data))

        created = {
            "id": "new-uuid",
            "name": "my workflow",
            "latest_version": 1,
            "created_by": "user",
            "created_at": "2026-05-23T00:00:00Z",
            "updated_at": "2026-05-23T00:00:00Z",
        }
        calls = _patch_urlopen(monkeypatch, {"/api/workflows": created})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "new-uuid"
        # POSTed JSON carries the workflow_json + name.
        assert calls[0]["method"] == "POST"
        sent = json.loads(calls[0]["body"])
        assert sent["name"] == "my workflow"
        assert sent["workflow_json"] == wf_data

    def test_invalid_json_file_surfaces_workflow_invalid_json(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path = tmp_path / "bad.json"
        wf_path.write_text("not valid json")
        # urlopen should never fire since we fail before HTTP.
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf_path), "--name", "x", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_invalid_json"

    def test_missing_file_surfaces_workflow_not_found(self, cloud_target, tmp_path, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(tmp_path / "missing.json"), "--name", "x", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_sends_delete_request(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid": (b"", 204)})
        env = _run(["delete", "wf-uuid", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["deleted"] is True
        assert calls[0]["method"] == "DELETE"

    def test_404_surfaces_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError("https://x/workflows/g", 404, "Not Found", {}, io.BytesIO(b"{}"))
        _patch_urlopen(monkeypatch, {"/api/workflows/ghost": err})
        env = _run(["delete", "ghost", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"
