"""Tests for `comfy workflow list/get/save/delete` — saved workflows.

The cloud path wraps the `/api/workflows` surface documented at
docs.comfy.org/api-reference/cloud/workflow. The local path (`--where local`)
wraps the running ComfyUI's `/userdata` file store under `workflows/`, mirroring
the ComfyUI frontend's layout. All HTTP is mocked; the live round-trip is
verified manually.
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


# A frontend-format workflow (nodes[] list) — what the local /userdata store holds.
_LOCAL_WORKFLOW = {
    "last_node_id": 2,
    "nodes": [
        {"id": 1, "type": "KSampler"},
        {"id": 2, "type": "VAEDecode"},
    ],
    "links": [],
}

# ComfyUI's /userdata?dir=workflows&full_info=true response: FileInfo dicts with
# `path` relative to the workflows/ dir.
_USERDATA_LIST_RESPONSE = [
    {"path": "flux.json", "size": 120, "modified": 1700000002000, "created": 1700000000000},
    {"path": "sub/wan.json", "size": 340, "modified": 1700000001000, "created": 1700000001000},
]


def _http_error(code: int):
    return urllib.error.HTTPError("http://127.0.0.1:8188/userdata/x", code, "err", {}, io.BytesIO(b""))


# ---------------------------------------------------------------------------
# Local route — /userdata-backed CRUD
# ---------------------------------------------------------------------------


class TestLocalList:
    def test_lists_userdata_workflows(self, local_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/userdata": _USERDATA_LIST_RESPONSE})
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 2
        # id == name == the path relative to workflows/ (what get/delete take).
        ids = {r["id"] for r in env["data"]["workflows"]}
        assert ids == {"flux.json", "sub/wan.json"}
        # Requests the recursive full-info listing of the workflows/ dir.
        assert "dir=workflows" in calls[0]["url"]
        assert "full_info=true" in calls[0]["url"]

    def test_name_filter_and_limit(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata": _USERDATA_LIST_RESPONSE})
        env = _run(["list", "--where", "local", "--name", "wan", "--limit", "10"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["workflows"][0]["id"] == "sub/wan.json"

    def test_missing_workflows_dir_is_empty_not_error(self, local_target, monkeypatch, capsys):
        # ComfyUI 404s when the workflows/ dir doesn't exist yet — that's "none saved".
        _patch_urlopen(monkeypatch, {"/userdata": _http_error(404)})
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 0

    def test_server_not_running_envelope(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata": urllib.error.URLError("Connection refused")})
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"

    def test_reachable_server_error_is_not_server_not_running(self, local_target, monkeypatch, capsys):
        # A reachable server that 500s must not be mislabeled "run comfy launch".
        _patch_urlopen(monkeypatch, {"/userdata": _http_error(500)})
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_error"

    def test_unparseable_body_surfaces_invalid_response(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata": (b"<html>not json</html>", 200)})
        env = _run(["list", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_response"

    def test_invalid_order_rejected(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})  # must fail before any HTTP
        env = _run(["list", "--where", "local", "--order", "sideways"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"

    def test_order_normalized_case_insensitively(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata": _USERDATA_LIST_RESPONSE})
        env = _run(["list", "--where", "local", "--order", "ASC"], capsys)
        assert env["ok"] is True

    def test_invalid_sort_rejected(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})
        env = _run(["list", "--where", "local", "--sort", "bogus"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"


class TestLocalGet:
    def test_writes_content_to_file(self, local_target, tmp_path, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata/": _LOCAL_WORKFLOW})
        out = tmp_path / "got.json"
        env = _run(["get", "flux.json", "--out", str(out), "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "flux.json"
        assert env["data"]["node_count"] == 2  # frontend format → len(nodes)
        assert json.loads(out.read_text()) == _LOCAL_WORKFLOW

    def test_encodes_subdir_key_as_single_segment(self, local_target, tmp_path, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/userdata/": _LOCAL_WORKFLOW})
        _run(["get", "sub/wan.json", "--out", str(tmp_path / "o.json"), "--where", "local"], capsys)
        # workflows/sub/wan.json is percent-encoded whole (slashes → %2F).
        assert "workflows%2Fsub%2Fwan.json" in calls[0]["url"]

    def test_404_surfaces_workflow_not_found(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata/": _http_error(404)})
        env = _run(["get", "ghost.json", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_rejects_path_traversal(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})  # must fail before any HTTP
        env = _run(["get", "../../etc/passwd", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"

    def test_rejects_windows_trailing_dot_traversal(self, local_target, monkeypatch, capsys):
        # A Windows server strips trailing dots/spaces, so "sub/.. /x" collapses to
        # "sub/../x" and escapes workflows/ — reject it before any HTTP.
        _patch_urlopen(monkeypatch, {})
        env = _run(["get", "sub/.. /secret", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"

    def test_non_json_body_warns_and_writes_raw(self, local_target, tmp_path, monkeypatch, capsys):
        # A 200 with a non-JSON body (e.g. an HTML error page) still gets written,
        # but a warning surfaces so the corrupt fetch isn't silent.
        _patch_urlopen(monkeypatch, {"/userdata/": (b"<html>nope</html>", 200)})
        out = tmp_path / "got.json"
        env = _run(["get", "flux.json", "--out", str(out), "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["node_count"] is None
        assert any(w["code"] == "workflow_content_not_json" for w in env["data"].get("warnings", []))
        assert out.read_bytes() == b"<html>nope</html>"

    def test_non_utf8_body_does_not_crash(self, local_target, tmp_path, monkeypatch, capsys):
        # json.loads on non-UTF-8 bytes raises UnicodeDecodeError, not JSONDecodeError.
        _patch_urlopen(monkeypatch, {"/userdata/": (b"\xff\xfe\x00bad", 200)})
        out = tmp_path / "got.json"
        env = _run(["get", "flux.json", "--out", str(out), "--where", "local"], capsys)
        assert env["ok"] is True
        assert any(w["code"] == "workflow_content_not_json" for w in env["data"].get("warnings", []))
        assert out.read_bytes() == b"\xff\xfe\x00bad"

    def test_write_error_surfaces_envelope(self, local_target, tmp_path, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata/": _LOCAL_WORKFLOW})

        def _boom(self, data):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.write_bytes", _boom)
        env = _run(["get", "flux.json", "--out", str(tmp_path / "o.json"), "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_write_error"

    def test_response_over_cap_refuses_to_truncate(self, local_target, tmp_path, monkeypatch, capsys):
        # Shrink the cap so the mocked body exceeds it; the CLI must fail loudly
        # rather than silently writing a truncated file.
        monkeypatch.setattr(workflow_cmd, "_USERDATA_MAX_BYTES", 4)
        _patch_urlopen(monkeypatch, {"/userdata/": (b'{"nodes": []}', 200)})
        env = _run(["get", "flux.json", "--out", str(tmp_path / "o.json"), "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_too_large"


class TestLocalSave:
    def test_posts_file_bytes_and_appends_json_ext(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        stored = {"path": "flux.json", "size": 120, "modified": 1700000002000, "created": 1700000000000}
        calls = _patch_urlopen(monkeypatch, {"/userdata/": stored})
        env = _run(["save", str(wf), "--name", "flux", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "flux.json"
        assert calls[0]["method"] == "POST"
        # '.json' appended to the bare --name; encoded into the userdata path.
        assert "workflows%2Fflux.json" in calls[0]["url"]
        assert "overwrite=true" in calls[0]["url"]
        assert json.loads(calls[0]["body"]) == _LOCAL_WORKFLOW

    def test_description_ignored_with_warning(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        _patch_urlopen(monkeypatch, {"/userdata/": {"path": "flux.json"}})
        env = _run(["save", str(wf), "--name", "flux.json", "--description", "hi", "--where", "local"], capsys)
        assert env["ok"] is True
        assert any(w["code"] == "description_ignored" for w in env["data"].get("warnings", []))

    def test_invalid_json_file(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "bad.json"
        wf.write_text("not json")
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf), "--name", "x", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_invalid_json"

    def test_non_utf8_file_surfaces_read_error(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "bin.json"
        wf.write_bytes(b"\xff\xfe\x00")  # not valid UTF-8 → UnicodeDecodeError on read
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf), "--name", "x", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_read_error"

    def test_json_ext_appended_case_insensitively(self, local_target, tmp_path, monkeypatch, capsys):
        # `--name flux.JSON` already ends in .json (case-insensitive) → don't double the ext.
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        calls = _patch_urlopen(monkeypatch, {"/userdata/": {"path": "flux.JSON"}})
        env = _run(["save", str(wf), "--name", "flux.JSON", "--where", "local"], capsys)
        assert env["ok"] is True
        assert "workflows%2Fflux.JSON" in calls[0]["url"]
        assert "flux.JSON.json" not in calls[0]["url"]

    def test_missing_file(self, local_target, tmp_path, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(tmp_path / "nope.json"), "--name", "x", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_server_not_running_envelope(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        _patch_urlopen(monkeypatch, {"/userdata/": urllib.error.URLError("refused")})
        env = _run(["save", str(wf), "--name", "flux", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"


class TestLocalDelete:
    def test_sends_delete(self, local_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/userdata/": (b"", 204)})
        env = _run(["delete", "flux.json", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["deleted"] is True
        assert calls[0]["method"] == "DELETE"
        assert "workflows%2Fflux.json" in calls[0]["url"]

    def test_404_surfaces_workflow_not_found(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata/": _http_error(404)})
        env = _run(["delete", "ghost.json", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_rejects_absolute_path(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {})
        env = _run(["delete", "/etc/passwd", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"


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
