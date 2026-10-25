"""Tests for `comfy workflow list/get/save/delete` — saved workflows.

The cloud path wraps the `/api/workflows` surface documented at
docs.comfy.org/api-reference/cloud/workflow, except `save`, which writes the
`/api/userdata` file the Cloud Workflows sidebar lists. The local path (`--where local`)
wraps the running ComfyUI's `/userdata` file store under `workflows/`, mirroring
the ComfyUI frontend's layout. All HTTP is mocked; the live round-trip is
verified manually.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import typer
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


_WORKFLOW_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas" / "workflow.json").read_text()
)


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
                if callable(payload):
                    payload = payload(req)
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, tuple):
                    body_bytes, status = payload
                    return _fake_resp(body_bytes, status)
                return _fake_resp(json.dumps(payload).encode())
        raise AssertionError(f"unexpected URL: {url}")

    # Both the local ``/userdata`` and cloud workflow paths now open through the
    # shared no-redirect opener in ``comfy_cli.http``; the fake still receives a
    # ``Request`` object, so the route-matching above is unchanged.
    import comfy_cli.http as http_mod

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", _fake)
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

    def test_total_counts_every_match_and_pages_by_offset(self, local_target, monkeypatch, capsys):
        # A page smaller than the store must not read as the whole store:
        # `count` is this page, `total` is every saved workflow.
        rows = [
            {"path": f"t{i:02}.json", "size": 1, "modified": 1700000000000 + i, "created": 1700000000000 + i}
            for i in range(30)
        ]
        _patch_urlopen(monkeypatch, {"/userdata": rows})
        first = _run(["list", "--where", "local", "--limit", "20"], capsys)["data"]
        assert (first["count"], first["total"], first["has_more"], first["next_offset"]) == (20, 30, True, 20)
        assert first["workflows"][0]["id"] == "t29.json"  # newest first

        last = _run(["list", "--where", "local", "--limit", "20", "--offset", "20"], capsys)["data"]
        assert (last["count"], last["total"], last["has_more"]) == (10, 30, False)
        assert "next_offset" not in last
        assert [r["id"] for r in last["workflows"]] == [f"t{i:02}.json" for i in range(9, -1, -1)]

    def test_offset_pages_are_stable_when_sort_keys_tie(self, local_target, monkeypatch, capsys):
        # Same timestamp on every row: page 2 must not depend on /userdata's order.
        rows = [{"path": f"t{i}.json", "size": 1, "modified": 1, "created": 1} for i in range(4)]
        pages = []
        for listing in (rows, rows[::-1]):
            _patch_urlopen(monkeypatch, {"/userdata": listing})
            data = _run(["list", "--where", "local", "--limit", "2", "--offset", "2"], capsys)["data"]
            pages.append([r["id"] for r in data["workflows"]])
        assert pages[0] == pages[1] == ["t1.json", "t0.json"]

    def test_total_is_over_name_filter_matches(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/userdata": _USERDATA_LIST_RESPONSE})
        env = _run(["list", "--where", "local", "--name", "wan"], capsys)
        assert (env["data"]["count"], env["data"]["total"], env["data"]["has_more"]) == (1, 1, False)

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
        # node_count: null must validate against the shared "workflow" schema
        # (comfy workflow print resolves to the same schema and can also emit
        # null) — regression guard for a schema that only allowed "integer".
        jsonschema.Draft202012Validator(_WORKFLOW_SCHEMA).validate(env["data"])

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

    def test_no_overwrite_sends_overwrite_false(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        calls = _patch_urlopen(monkeypatch, {"/userdata/": {"path": "workflows/flux.json"}})
        env = _run(["save", str(wf), "--name", "flux", "--no-overwrite", "--where", "local"], capsys)
        assert env["ok"] is True
        assert "overwrite=false" in calls[0]["url"]

    def test_no_overwrite_collision_is_workflow_exists(self, local_target, tmp_path, monkeypatch, capsys):
        # ComfyUI answers 409 when `overwrite=false` meets an existing file.
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        _patch_urlopen(monkeypatch, {"/userdata/": _http_error(409)})
        env = _run(["save", str(wf), "--name", "flux", "--no-overwrite", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["where"] == "local"
        assert env["error"]["code"] == "workflow_exists"
        assert env["error"]["details"]["filename"] == "flux.json"

    def test_refuses_the_favorites_index(self, local_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "src.json"
        wf.write_text(json.dumps(_LOCAL_WORKFLOW))
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf), "--name", ".index.json", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []


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


class TestLocalTooLargeHints:
    """An oversize /userdata response means something different per verb, so the
    hint must too — a `list` overflow is too many workflows, and save/delete have
    already sent their write by the time the body is read."""

    def _args(self, verb: str, tmp_path) -> list[str]:
        if verb == "list":
            return ["list", "--where", "local"]
        if verb == "get":
            return ["get", "flux.json", "--out", str(tmp_path / "o.json"), "--where", "local"]
        if verb == "save":
            src = tmp_path / "src.json"
            src.write_text(json.dumps(_LOCAL_WORKFLOW))
            return ["save", str(src), "--name", "flux", "--where", "local"]
        return ["delete", "flux.json", "--where", "local"]

    @pytest.mark.parametrize("verb", ["list", "get", "save", "delete"])
    def test_hint_is_per_operation(self, verb, local_target, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(workflow_cmd, "_USERDATA_MAX_BYTES", 4)
        _patch_urlopen(monkeypatch, {"/userdata": (b'[{"path": "flux.json"}]', 200)})
        env = _run(self._args(verb, tmp_path), capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_too_large"
        assert env["error"]["details"]["operation"] == verb
        assert env["error"]["hint"] == workflow_cmd._LOCAL_TOO_LARGE_HINTS[verb]

    def test_list_hint_does_not_name_a_single_workflow(self):
        # The listing is the whole workflows/ dir — "the saved workflow" (the old
        # hardcoded hint) names a thing that doesn't exist for this verb.
        assert "the saved workflow is" not in workflow_cmd._LOCAL_TOO_LARGE_HINTS["list"]

    @pytest.mark.parametrize("verb", ["save", "delete"])
    def test_write_verb_hints_do_not_imply_the_write_was_rejected(self, verb):
        # The request is already on the wire before the body is read, so the
        # mutation may have landed. The hint must send the user to confirm, not
        # imply a clean failure they should retry.
        hint = workflow_cmd._LOCAL_TOO_LARGE_HINTS[verb]
        assert "may still have been" in hint
        assert "workflow list" in hint

    def test_unknown_operation_falls_back(self, capsys):
        # Defensive: a future call site with a new operation still gets a hint,
        # and one that claims nothing about what did or didn't happen.
        renderer = _force_json_renderer()
        exit_exc = workflow_cmd._handle_local_http_error(renderer, workflow_cmd._ResponseTooLarge(), operation="purge")
        assert isinstance(exit_exc, typer.Exit)
        env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert env["error"]["code"] == "workflow_too_large"
        assert env["error"]["hint"] == "the local response was unexpectedly large"


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

    def test_forwards_server_total_and_pages_by_offset(self, cloud_target, monkeypatch, capsys):
        # The server knows the user has 30; the default page is 20. Reporting
        # only `count` is how the agent told a user "You have 20".
        def page(start, n, has_more):
            return {
                "data": [{"id": f"wf-{i}", "name": f"T{i:02}"} for i in range(start, start + n)],
                "pagination": {"total": 30, "limit": 20, "offset": start, "has_more": has_more},
            }

        calls = _patch_urlopen(monkeypatch, {"/api/workflows": page(0, 20, True)})
        first = _run(["list", "--where", "cloud"], capsys)["data"]
        assert (first["count"], first["total"], first["has_more"], first["next_offset"]) == (20, 30, True, 20)
        assert "offset=" not in calls[0]["url"]

        calls = _patch_urlopen(monkeypatch, {"/api/workflows": page(20, 10, False)})
        last = _run(["list", "--where", "cloud", "--offset", "20"], capsys)["data"]
        assert "offset=20" in calls[0]["url"]
        assert (last["count"], last["total"], last["has_more"]) == (10, 30, False)
        assert "next_offset" not in last

    def test_omits_a_total_that_cannot_be_true(self, cloud_target, monkeypatch, capsys):
        # Cursor-paged PaginationInfo may report total 0 beside a full page.
        # That form may also ignore `?offset=`, so no `next_offset` is offered.
        body = {"data": _WORKFLOW_LIST_RESPONSE["data"], "pagination": {"total": 0, "offset": 0, "has_more": True}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud"], capsys)["data"]
        assert "total" not in env
        assert env["has_more"] is True
        assert "next_offset" not in env

    def test_next_offset_only_when_paging_advances(self, cloud_target, monkeypatch, capsys):
        rows = _WORKFLOW_LIST_RESPONSE["data"]
        # The server answered offset 0 to a request for offset 20: it ignored it.
        body = {"data": rows, "pagination": {"total": 30, "offset": 0, "has_more": True}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud", "--offset", "20"], capsys)["data"]
        assert env["has_more"] is True
        assert "next_offset" not in env

        # An empty page cannot advance, whatever `has_more` says.
        body = {"data": [], "pagination": {"total": 30, "offset": 5, "has_more": True}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud", "--offset", "5"], capsys)["data"]
        assert "next_offset" not in env

    def test_derives_has_more_from_total_when_server_omits_it(self, cloud_target, monkeypatch, capsys):
        rows = _WORKFLOW_LIST_RESPONSE["data"]
        body = {"data": rows, "pagination": {"total": 30, "offset": 0}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud"], capsys)["data"]
        assert (env["total"], env["has_more"], env["next_offset"]) == (30, True, len(rows))

    def test_keeps_total_on_an_empty_page_past_the_end(self, cloud_target, monkeypatch, capsys):
        # An offset beyond the last row still learns how many exist; a zero
        # total there is the cursor sentinel and stays omitted.
        body = {"data": [], "pagination": {"total": 30, "has_more": False}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud", "--offset", "40"], capsys)["data"]
        assert (env["count"], env["total"], env["has_more"]) == (0, 30, False)

        body = {"data": [], "pagination": {"total": 0, "has_more": False}}
        _patch_urlopen(monkeypatch, {"/api/workflows": body})
        env = _run(["list", "--where", "cloud", "--offset", "40"], capsys)["data"]
        assert "total" not in env


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

    def test_response_over_cap_refuses_to_truncate(self, cloud_target, monkeypatch, capsys):
        # Shrink the cap so the mocked body exceeds it. An oversize body used to
        # be truncated, fail to parse, and get swallowed into (status, None) —
        # indistinguishable from an empty body. It must fail loudly instead.
        monkeypatch.setattr(workflow_cmd, "_HTTP_MAX_BYTES", 4)
        _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid/content": _WORKFLOW_CONTENT_RESPONSE})
        env = _run(["get", "wf-uuid", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_too_large"
        assert env["error"]["details"]["limit_bytes"] == 4


class TestHttpRequestCap:
    """``_http_request`` must distinguish an oversize body from an empty one."""

    def test_oversize_body_raises_rather_than_returning_none(self, cloud_target, monkeypatch):
        monkeypatch.setattr(workflow_cmd, "_HTTP_MAX_BYTES", 4)
        _patch_urlopen(monkeypatch, {"/api/workflows": (b'{"data": []}', 200)})
        target = workflow_cmd._resolve_where_target("cloud")
        with pytest.raises(workflow_cmd._ResponseTooLarge):
            workflow_cmd._http_request(target.url("workflows"), target)

    def test_body_exactly_at_cap_still_parses(self, cloud_target, monkeypatch):
        # Boundary: len(raw) == cap is a *complete* body, not a truncated one.
        body = b'{"a": 1}'
        monkeypatch.setattr(workflow_cmd, "_HTTP_MAX_BYTES", len(body))
        _patch_urlopen(monkeypatch, {"/api/workflows": (body, 200)})
        target = workflow_cmd._resolve_where_target("cloud")
        assert workflow_cmd._http_request(target.url("workflows"), target) == (200, {"a": 1})

    def test_empty_body_still_returns_none(self, cloud_target, monkeypatch):
        # The (status, None) contract for a genuinely empty body is unchanged.
        _patch_urlopen(monkeypatch, {"/api/workflows": (b"", 200)})
        target = workflow_cmd._resolve_where_target("cloud")
        assert workflow_cmd._http_request(target.url("workflows"), target) == (200, None)

    def test_non_utf8_body_raises_unparseable(self, cloud_target, monkeypatch):
        # UnicodeDecodeError is a ValueError but not a JSONDecodeError. A non-empty
        # body that won't decode must not masquerade as "no data" (200, None); it is
        # malformed and must raise _ResponseUnparseable so a call site maps it to a
        # loud envelope rather than reporting empty success.
        _patch_urlopen(monkeypatch, {"/api/workflows": (b"\xff\xfe\x00not json", 200)})
        target = workflow_cmd._resolve_where_target("cloud")
        with pytest.raises(workflow_cmd._ResponseUnparseable):
            workflow_cmd._http_request(target.url("workflows"), target)

    def test_non_utf8_body_surfaces_envelope_not_traceback(self, cloud_target, monkeypatch, capsys):
        # End-to-end: the undecodable body must reach the user as an envelope.
        _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid/content": (b"\xff\xfe\x00not json", 200)})
        env = _run(["get", "wf-uuid", "--where", "cloud"], capsys)
        assert env["ok"] is False

    @pytest.mark.parametrize("verb", ["list", "get", "save", "delete"])
    def test_every_call_site_routes_oversize_to_envelope(self, verb, cloud_target, tmp_path, monkeypatch, capsys):
        # Each of the four _http_request call sites must catch _ResponseTooLarge;
        # an uncaught one would surface as a traceback rather than an envelope.
        monkeypatch.setattr(workflow_cmd, "_HTTP_MAX_BYTES", 4)
        # `save` writes through `/api/userdata/`; the other three read `/api/workflows`.
        _patch_urlopen(monkeypatch, {"/api/workflows": (b'{"data": []}', 200), "/api/userdata/": (b'{"a": 1}', 200)})
        args = {
            "list": ["list"],
            "get": ["get", "wf-uuid"],
            "delete": ["delete", "wf-uuid"],
        }.get(verb)
        if args is None:
            wf_path = tmp_path / "wf.json"
            wf_path.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}))
            args = ["save", str(wf_path), "--name", "x"]
        env = _run([*args, "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_too_large"
        assert env["error"]["details"]["operation"] == verb
        # The hint must speak to the operation that actually failed; a single
        # hardcoded "saved workflow is too large" is wrong for list/save/delete.
        assert env["error"]["hint"] == workflow_cmd._TOO_LARGE_HINTS[verb]

    @pytest.mark.parametrize("verb", ["save", "delete"])
    def test_mutating_verbs_do_not_claim_the_write_was_rejected(self, verb):
        # save/delete have already sent their request by the time the response
        # is read, so the server-side change may have landed. The hint must not
        # imply otherwise, or users will wrongly retry a completed mutation.
        assert "may still have been" in workflow_cmd._TOO_LARGE_HINTS[verb]


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


_MIRRORED_RECORD = {
    "id": "33333333-3333-3333-3333-333333333333",
    "name": "my workflow",
    "default_view": "workflow",
    "latest_version": 2,
    "created_by": "user-uuid",
    "created_at": "2026-05-23T00:00:00Z",
    "updated_at": "2026-05-23T00:00:00Z",
}

# Cloud's `POST /api/userdata/{file}?full_info=true` receipt: a FileInfo dict.
_USERDATA_SAVE_RESPONSE = {"path": "workflows/my workflow.json", "size": 64, "modified": 1700000003000}


def _record_listing(*rows: dict) -> dict:
    return {"data": list(rows), "pagination": {"total": len(rows), "limit": 100, "offset": 0}}


def _cloud_wf(tmp_path: Path) -> tuple[Path, dict]:
    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps(_LOCAL_WORKFLOW))
    return wf_path, _LOCAL_WORKFLOW


def _http_409(url: str = "https://cloud.example.com/api/userdata/x"):
    return urllib.error.HTTPError(url, 409, "Conflict", {}, io.BytesIO(b"File already exists"))


class TestSave:
    """Cloud `workflow save` writes the file the Cloud Workflows sidebar lists.

    The sidebar lists `userdata/workflows/*.json`; a record created only through
    `POST /api/workflows` has no file behind it and never shows there. So the save
    is the frontend's own write, `POST /api/userdata/<encoded workflows/name.json>`,
    which Cloud mirrors into its `/api/workflows` record store. The record id is
    then read back by exact name.
    """

    def test_writes_the_sidebar_userdata_file(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, wf_data = _cloud_wf(tmp_path)
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _USERDATA_SAVE_RESPONSE,
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
            },
        )
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        assert env["where"] == "cloud"

        write = calls[0]
        assert write["method"] == "POST"
        # The whole `workflows/<name>.json` path is one percent-encoded segment,
        # exactly as the frontend's `encodeURIComponent(file)` sends it.
        assert write["url"].startswith("https://cloud.example.com/api/userdata/workflows%2Fmy%20workflow.json?")
        assert "overwrite=true" in write["url"]
        assert "full_info=true" in write["url"]
        # The body is the graph itself, not the `/api/workflows` {name, workflow_json} envelope.
        assert json.loads(write["body"]) == wf_data

        # No bare `/api/workflows` record write: Cloud creates the record from the file.
        assert not [c for c in calls if c["method"] == "POST" and "/api/workflows" in c["url"]]

        data = env["data"]
        assert data["filename"] == "my workflow.json"
        assert data["name"] == "my workflow"
        assert data["workflow_id"] == _MIRRORED_RECORD["id"]
        assert data["latest_version"] == 2
        assert data["size"] == 64
        assert data["source"] == str(wf_path)
        assert "warnings" not in data

    def test_record_is_read_back_by_exact_name(self, cloud_target, tmp_path, monkeypatch, capsys):
        # Cloud's `name` filter is a case-insensitive substring match, so the
        # listing can carry near-misses. Only the exact name is the mirror.
        wf_path, _ = _cloud_wf(tmp_path)
        near_miss = {**_MIRRORED_RECORD, "id": "44444444-4444-4444-4444-444444444444", "name": "my workflow (copy)"}
        cased = {**_MIRRORED_RECORD, "id": "55555555-5555-5555-5555-555555555555", "name": "My Workflow"}
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _USERDATA_SAVE_RESPONSE,
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(near_miss, cased, _MIRRORED_RECORD),
            },
        )
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["data"]["workflow_id"] == _MIRRORED_RECORD["id"]
        lookup = calls[1]
        assert lookup["method"] == "GET"
        assert "name=my+workflow" in lookup["url"]

    def test_subdir_name_is_a_single_encoded_segment(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        record = {**_MIRRORED_RECORD, "name": "team/flux"}
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": {"path": "team/flux.json"},
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(record),
            },
        )
        env = _run(["save", str(wf_path), "--name", "team/flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert "/api/userdata/workflows%2Fteam%2Fflux.json?" in calls[0]["url"]
        assert env["data"]["filename"] == "team/flux.json"
        assert env["data"]["workflow_id"] == record["id"]

    def test_json_ext_is_not_doubled(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        record = {**_MIRRORED_RECORD, "name": "flux"}
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": {},
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(record),
            },
        )
        env = _run(["save", str(wf_path), "--name", "flux.json", "--where", "cloud"], capsys)
        assert "workflows%2Fflux.json?" in calls[0]["url"]
        # The record name Cloud derives drops the `.json`.
        assert env["data"]["name"] == "flux"
        assert env["data"]["workflow_id"] == record["id"]

    def test_uppercase_ext_still_gets_json_appended_on_cloud(self, cloud_target, tmp_path, monkeypatch, capsys):
        # Cloud mirrors only names ending in a lowercase `.json` into a record,
        # so `flux.JSON` is a stem, not an extension, on this route.
        wf_path, _ = _cloud_wf(tmp_path)
        calls = _patch_urlopen(monkeypatch, {"/api/userdata/": {}, "/api/workflows?": _record_listing()})
        env = _run(["save", str(wf_path), "--name", "flux.JSON", "--where", "cloud"], capsys)
        assert "workflows%2Fflux.JSON.json?" in calls[0]["url"]
        assert env["data"]["filename"] == "flux.JSON.json"

    def test_no_overwrite_sends_overwrite_false(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _USERDATA_SAVE_RESPONSE,
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
            },
        )
        env = _run(["save", str(wf_path), "--name", "my workflow", "--no-overwrite", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert "overwrite=false" in calls[0]["url"]

    def test_no_overwrite_collision_is_workflow_exists(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        calls = _patch_urlopen(monkeypatch, {"/api/userdata/": _http_409()})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--no-overwrite", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["where"] == "cloud"
        assert env["error"]["code"] == "workflow_exists"
        assert env["error"]["details"]["filename"] == "my workflow.json"
        assert "--no-overwrite" in env["error"]["hint"]
        # Nothing was written, so there is no record to look up.
        assert len(calls) == 1

    def test_unauthorized_write_is_cloud_unauthorized(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        denied = urllib.error.HTTPError(
            "https://cloud.example.com/api/userdata/x", 401, "Unauthorized", {}, io.BytesIO(b"")
        )
        _patch_urlopen(monkeypatch, {"/api/userdata/": denied})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_unauthorized"
        assert env["error"]["details"]["operation"] == "save"

    def test_unreachable_write_is_an_error(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path, _ = _cloud_wf(tmp_path)
        _patch_urlopen(monkeypatch, {"/api/userdata/": urllib.error.URLError("refused")})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"

    @pytest.mark.parametrize(
        "listing",
        [
            pytest.param(_record_listing(), id="no-record"),
            pytest.param(
                _record_listing(_MIRRORED_RECORD, {**_MIRRORED_RECORD, "id": "66666666-6666-6666-6666-666666666666"}),
                id="two-records-one-name",
            ),
            pytest.param(urllib.error.URLError("reset"), id="lookup-unreachable"),
            pytest.param((b"<html>502</html>", 200), id="lookup-unparseable"),
            pytest.param(
                # A full page may be truncated: the exact match it holds may not be the only one.
                _record_listing(
                    _MIRRORED_RECORD,
                    *[{**_MIRRORED_RECORD, "id": f"id-{i}", "name": f"my workflow {i}"} for i in range(99)],
                ),
                id="full-page",
            ),
        ],
    )
    def test_unresolved_record_keeps_the_save_and_warns(self, listing, cloud_target, tmp_path, monkeypatch, capsys):
        # The file write has already landed when the record lookup runs, so a
        # lookup that cannot name exactly one record must not fail the save (an
        # agent would re-issue a save that already succeeded). It reports a null
        # id and says why.
        wf_path, _ = _cloud_wf(tmp_path)
        _patch_urlopen(monkeypatch, {"/api/userdata/": _USERDATA_SAVE_RESPONSE, "/api/workflows?": listing})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["filename"] == "my workflow.json"
        assert env["data"]["workflow_id"] is None
        assert env["data"]["latest_version"] is None
        assert [w["code"] for w in env["data"]["warnings"]] == ["workflow_record_unresolved"]

    @pytest.mark.parametrize(
        "content",
        [
            {"workflow_json": {"nodes": [], "links": []}},
            urllib.error.URLError("mirror content unavailable"),
        ],
    )
    def test_stale_or_unreadable_mirror_does_not_return_record_id(
        self, content, cloud_target, tmp_path, monkeypatch, capsys
    ):
        # The best-effort mirror may fail while an older same-name record remains.
        wf_path, _ = _cloud_wf(tmp_path)
        _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _USERDATA_SAVE_RESPONSE,
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
                "/content": content,
            },
        )
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] is None
        assert env["data"]["warnings"][0]["code"] == "workflow_record_unresolved"

    @pytest.mark.parametrize("name", [".hidden", "team/.hidden/flow", ".json", "team/.json"])
    def test_hidden_names_are_refused_before_cloud_write(self, name, cloud_target, tmp_path, monkeypatch, capsys):
        # Cloud's userdata listing excludes every dot-prefixed component.
        wf_path, _ = _cloud_wf(tmp_path)
        calls = _patch_urlopen(monkeypatch, {"/api/userdata/": {}, "/api/workflows?": _record_listing()})
        env = _run(["save", str(wf_path), "--name", name, "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_description_is_ignored_with_a_warning(self, cloud_target, tmp_path, monkeypatch, capsys):
        # The sidebar file has no description field; say so rather than drop it silently.
        wf_path, wf_data = _cloud_wf(tmp_path)
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _USERDATA_SAVE_RESPONSE,
                "/content": {"workflow_json": _LOCAL_WORKFLOW},
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
            },
        )
        env = _run(
            ["save", str(wf_path), "--name", "my workflow", "--description", "hi", "--where", "cloud"],
            capsys,
        )
        assert env["ok"] is True
        assert [w["code"] for w in env["data"]["warnings"]] == ["description_ignored"]
        assert json.loads(calls[0]["body"]) == wf_data

    @pytest.mark.parametrize("name", ["../escape", "/abs", "a\\b", ".index", "sub/.index.json"])
    def test_unsafe_or_reserved_names_are_refused_before_any_call(
        self, name, cloud_target, tmp_path, monkeypatch, capsys
    ):
        # `.index.json` is the frontend's favorites file in the same folder; a
        # save under that name would replace the user's bookmarks.
        wf_path, _ = _cloud_wf(tmp_path)
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf_path), "--name", name, "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_non_utf8_file_surfaces_read_error(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf = tmp_path / "bin.json"
        wf.write_bytes(b"\xff\xfe\x00")
        _patch_urlopen(monkeypatch, {})
        env = _run(["save", str(wf), "--name", "x", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_read_error"

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


def test_cloud_save_list_get_delete_round_trip(cloud_target, tmp_path, monkeypatch, capsys):
    # Mirror the verified Cloud handler boundaries: file POST mirrors a record,
    # file DELETE mirrors record removal, record DELETE alone removes only a record.
    state = {"file": None, "record": False}

    def file(req):
        if req.get_method() == "POST":
            state["file"] = json.loads(req.data)
            state["record"] = True
            return _USERDATA_SAVE_RESPONSE
        if req.get_method() == "DELETE":
            state["file"] = None
            state["record"] = False
            return (b"", 204)
        return state["file"]

    def record(req):
        if not state["record"]:
            return urllib.error.HTTPError(req.full_url, 404, "Not found", {}, io.BytesIO())
        if req.full_url.endswith("/content"):
            return {"id": "v2", "version": 2, "workflow_json": state["file"]}
        if req.get_method() == "DELETE":
            state["record"] = False
            return (b"", 204)
        return _MIRRORED_RECORD

    def listing(req):
        return _record_listing(_MIRRORED_RECORD) if state["record"] else _record_listing()

    _patch_urlopen(
        monkeypatch,
        {
            "/api/userdata/": file,
            "/api/workflows?": listing,
            "/api/workflows/": record,
            "/api/user": {"id": "user-uuid"},
        },
    )
    src, workflow_json = _cloud_wf(tmp_path)
    saved = _run(["save", str(src), "--name", "my workflow", "--where", "cloud"], capsys)
    workflow_id = saved["data"]["workflow_id"]
    listed = _run(["list", "--where", "cloud"], capsys)
    assert listed["data"]["workflows"][0]["id"] == workflow_id
    out = tmp_path / "downloaded.json"
    fetched = _run(["get", workflow_id, "--out", str(out), "--where", "cloud"], capsys)
    assert fetched["ok"] is True
    assert json.loads(out.read_text()) == workflow_json
    deleted = _run(["delete", workflow_id, "--where", "cloud"], capsys)
    assert deleted["ok"] is True
    assert state == {"file": None, "record": False}
    listed = _run(["list", "--where", "cloud"], capsys)
    assert listed["data"]["count"] == 0


class TestCloudDeleteFile:
    @pytest.mark.parametrize(
        ("content", "creator", "user"),
        [
            pytest.param({"nodes": []}, "user-uuid", {"id": "user-uuid"}, id="unrelated-file"),
            pytest.param(_LOCAL_WORKFLOW, "other-user", {"id": "user-uuid"}, id="other-members-record"),
            pytest.param(_LOCAL_WORKFLOW, None, {"id": "user-uuid"}, id="unknown-record-owner"),
            pytest.param(_LOCAL_WORKFLOW, "user-uuid", {}, id="unknown-current-user"),
        ],
    )
    def test_unverified_file_association_leaves_file_and_record_untouched(
        self, content, creator, user, cloud_target, monkeypatch, capsys
    ):
        # A same-name record can be stale after a best-effort sync failure, or
        # belong to a different workspace member. Neither authorizes file deletion.
        record = {**_MIRRORED_RECORD, "created_by": creator}
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _LOCAL_WORKFLOW,
                "/content": {"workflow_json": content},
                "/api/workflows?": _record_listing(record),
                "/api/workflows/": record,
                "/api/user": user,
            },
        )
        env = _run(["delete", record["id"], "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_filename_ambiguous"
        assert all(call["method"] == "GET" for call in calls)

    @pytest.mark.parametrize("dual_delete", [True, False])
    def test_delete_removes_sidebar_file_and_record(self, dual_delete, cloud_target, monkeypatch, capsys):
        # Cloud's userdata delete mirrors best-effort; record DELETE alone leaves the file behind.
        state = {"file": True, "record": True}

        def record(req):
            if req.full_url.endswith("/content"):
                return {"workflow_json": _LOCAL_WORKFLOW}
            if req.get_method() == "GET":
                return _MIRRORED_RECORD
            if not state["record"]:
                return urllib.error.HTTPError(req.full_url, 404, "Not found", {}, io.BytesIO())
            state["record"] = False
            return (b"", 204)

        def file(req):
            if req.get_method() == "DELETE":
                state["file"] = False
                if dual_delete:
                    state["record"] = False
                return (b"", 204)
            return _LOCAL_WORKFLOW

        _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": file,
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
                "/api/workflows/": record,
                "/api/user": {"id": "user-uuid"},
            },
        )
        env = _run(["delete", _MIRRORED_RECORD["id"], "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        assert state == {"file": False, "record": False}
        assert env["data"]["filename"] == "my workflow.json"

    def test_duplicate_names_leave_file_and_records_untouched(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _LOCAL_WORKFLOW,
                "/api/workflows?": _record_listing(_MIRRORED_RECORD, {**_MIRRORED_RECORD, "id": "other-id"}),
                "/api/workflows/": _MIRRORED_RECORD,
            },
        )
        env = _run(["delete", _MIRRORED_RECORD["id"], "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_filename_ambiguous"
        assert all(c["method"] == "GET" for c in calls)

    @pytest.mark.parametrize("name", [" leading space", ".index", "../outside"])
    def test_unsafe_record_name_cannot_delete_another_file(self, name, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": _LOCAL_WORKFLOW,
                "/api/workflows?": _record_listing({**_MIRRORED_RECORD, "name": name}),
                "/api/workflows/": {**_MIRRORED_RECORD, "name": name},
            },
        )
        env = _run(["delete", _MIRRORED_RECORD["id"], "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert all(c["method"] == "GET" for c in calls)

    def test_partial_failure_reports_file_already_deleted(self, cloud_target, monkeypatch, capsys):
        def record(req):
            if req.full_url.endswith("/content"):
                return {"workflow_json": _LOCAL_WORKFLOW}
            if req.get_method() == "GET":
                return _MIRRORED_RECORD
            return urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, io.BytesIO())

        def file(req):
            return (b"", 204) if req.get_method() == "DELETE" else _LOCAL_WORKFLOW

        _patch_urlopen(
            monkeypatch,
            {
                "/api/userdata/": file,
                "/api/workflows?": _record_listing(_MIRRORED_RECORD),
                "/api/workflows/": record,
                "/api/user": {"id": "user-uuid"},
            },
        )
        env = _run(["delete", _MIRRORED_RECORD["id"], "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_delete_incomplete"
        assert env["error"]["details"]["file_deleted"] is True
        assert env["error"]["details"]["workflow_id"] == _MIRRORED_RECORD["id"]

    def test_record_only_workflow_still_deletes(self, cloud_target, monkeypatch, capsys):
        def record(req):
            return _MIRRORED_RECORD if req.get_method() == "GET" else (b"", 204)

        missing = urllib.error.HTTPError("https://x/userdata/x", 404, "Not Found", {}, io.BytesIO())
        calls = _patch_urlopen(monkeypatch, {"/api/userdata/": missing, "/api/workflows/": record})
        env = _run(["delete", _MIRRORED_RECORD["id"], "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert [c["method"] for c in calls] == ["GET", "GET", "DELETE"]


class TestDelete:
    def test_sends_delete_request(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid": {"id": "wf-uuid"}})
        env = _run(["delete", "wf-uuid", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["deleted"] is True
        assert [c["method"] for c in calls] == ["GET", "DELETE"]

    def test_404_surfaces_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError("https://x/workflows/g", 404, "Not Found", {}, io.BytesIO(b"{}"))
        _patch_urlopen(monkeypatch, {"/api/workflows/ghost": err})
        env = _run(["delete", "ghost", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"


# ---------------------------------------------------------------------------
# unparseable 200 body — must surface a loud error, not empty/success
# ---------------------------------------------------------------------------

# A non-empty 200 body the client can't decode as JSON. Four shapes, all malformed:
#   - a valid-UTF-8 non-JSON page (e.g. an HTML proxy/error page returned 200) -> JSONDecodeError;
#   - a non-UTF-8 byte string that isn't valid UTF-8 at all -> UnicodeDecodeError;
#   - valid JSON encoded as UTF-16, which the *contract* treats as malformed. ``json.loads``
#     auto-detects UTF-16/32 byte input (RFC 4627), so handed raw bytes it would silently
#     accept this; decoding as UTF-8 first surfaces it as UnicodeDecodeError. Locks that in.
#   - a JSON integer past CPython's 4300-digit int/str conversion limit: ``json.loads`` raises
#     a *bare* ``ValueError`` (not ``JSONDecodeError``), which the narrow catch would have let
#     escape as a raw traceback. Locks in the broadened ``ValueError``-base catch.
# ``JSONDecodeError`` and ``UnicodeDecodeError`` both subclass ``ValueError``; catching the base
# maps all four. None may be reported as "no data" (empty list / null id).
_UNPARSEABLE_BODIES = [
    pytest.param(b"<html>502 Bad Gateway</html>", id="non-json"),
    pytest.param(b"\xff\xfe\x00bad", id="non-utf8"),
    pytest.param(json.dumps({"data": [], "id": "x"}).encode("utf-16"), id="utf16-json"),
    pytest.param(b"1" + b"0" * 4400, id="bigint-valueerror"),
]


class TestUnparseableResponse:
    @pytest.mark.parametrize("raw", _UNPARSEABLE_BODIES)
    def test_list_surfaces_error_not_empty(self, cloud_target, monkeypatch, capsys, raw):
        _patch_urlopen(monkeypatch, {"/api/workflows": (raw, 200)})
        env = _run(["list", "--where", "cloud"], capsys)
        # Must NOT masquerade as a successful, genuinely-empty list.
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_unparseable"
        assert env["error"]["details"]["operation"] == "list"

    @pytest.mark.parametrize("raw", _UNPARSEABLE_BODIES)
    def test_save_surfaces_error_not_null_id(self, cloud_target, tmp_path, monkeypatch, capsys, raw):
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}))
        calls = _patch_urlopen(monkeypatch, {"/api/userdata/": (raw, 200)})
        env = _run(["save", str(wf_path), "--name", "x", "--where", "cloud"], capsys)
        # The save's write is the userdata POST; its receipt is what came back malformed.
        assert "/api/userdata/workflows%2Fx.json" in calls[0]["url"]
        # Must NOT claim success off a body it could not read (an HTML 200 can be a
        # proxy page that never reached Cloud).
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_unparseable"
        assert env["error"]["details"]["operation"] == "save"

    @pytest.mark.parametrize("raw", _UNPARSEABLE_BODIES)
    def test_get_surfaces_error_not_missing_content(self, cloud_target, monkeypatch, capsys, raw):
        _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid/content": (raw, 200)})
        env = _run(["get", "wf-uuid", "--where", "cloud"], capsys)
        # Must NOT be reported as missing/invalid content — it's a malformed transport body.
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_unparseable"
        assert env["error"]["details"]["operation"] == "get"

    @pytest.mark.parametrize("raw", _UNPARSEABLE_BODIES)
    def test_delete_surfaces_error_not_success(self, cloud_target, monkeypatch, capsys, raw):
        _patch_urlopen(monkeypatch, {"/api/workflows/wf-uuid": (raw, 200)})
        env = _run(["delete", "wf-uuid", "--where", "cloud"], capsys)
        # Must NOT claim a successful delete off an unparseable body.
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_unparseable"
        assert env["error"]["details"]["operation"] == "delete"


class TestListMalformedShape:
    """A valid-JSON but wrongly-shaped 200 body reaches ``list`` (``get``/``save`` already
    guard with ``isinstance``). It must map to an error, not raise a raw ``AttributeError``
    and not coerce to a masquerading empty list."""

    @pytest.mark.parametrize("raw", [pytest.param(b"[1, 2, 3]", id="array"), pytest.param(b"42", id="scalar")])
    def test_list_non_dict_body_surfaces_error(self, cloud_target, monkeypatch, capsys, raw):
        _patch_urlopen(monkeypatch, {"/api/workflows": (raw, 200)})
        env = _run(["list", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(b'{"data": 42}', id="scalar"),
            pytest.param(b'{"data": "oops"}', id="string"),
            pytest.param(b'{"data": {"a": 1}}', id="object"),
        ],
    )
    def test_list_non_list_data_surfaces_error(self, cloud_target, monkeypatch, capsys, raw):
        # A dict body with a non-list ``data`` must not raise a raw TypeError (scalar)
        # nor iterate silently into a masquerading empty listing (str/dict).
        _patch_urlopen(monkeypatch, {"/api/workflows": (raw, 200)})
        env = _run(["list", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"

    def test_list_missing_data_is_empty(self, cloud_target, monkeypatch, capsys):
        # A dict body with no ``data`` key is a legitimately-empty listing, not an error.
        _patch_urlopen(monkeypatch, {"/api/workflows": (b"{}", 200)})
        env = _run(["list", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 0
