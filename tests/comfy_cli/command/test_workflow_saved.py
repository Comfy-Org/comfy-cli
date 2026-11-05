"""Tests for `comfy workflow list/get/save/delete` — saved workflows.

The cloud path wraps the `/api/workflows` surface documented at
docs.comfy.org/api-reference/cloud/workflow. The local path (`--where local`)
wraps the running ComfyUI's `/userdata` file store under `workflows/`, mirroring
the ComfyUI frontend's layout. All HTTP is mocked; the live round-trip is
verified manually.
"""

from __future__ import annotations

import http.client
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
        # save and delete read /api/features first. The cap fits its flag-off
        # answer exactly, so they take the record path and reach the oversize
        # /api/workflows body, rather than reading the flag as off because the
        # features body itself was too large.
        flag_off = json.dumps(_FLAG_OFF).encode()
        monkeypatch.setattr(workflow_cmd, "_HTTP_MAX_BYTES", len(flag_off))
        oversize = json.dumps({"data": [], "pad": "x" * len(flag_off)}).encode()
        calls = _patch_urlopen(monkeypatch, {"/api/features": (flag_off, 200), "/api/workflows": (oversize, 200)})
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
        if verb in ("save", "delete"):
            assert calls[0]["url"] == "https://cloud.example.com/api/features"
            assert "/api/workflows" in calls[-1]["url"]

    @pytest.mark.parametrize("verb", ["save", "delete"])
    def test_mutating_verbs_do_not_claim_the_write_was_rejected(self, verb):
        # save/delete have already sent their request by the time the response
        # is read, so the server-side change may have landed. The hint must not
        # imply otherwise, or users will wrongly retry a completed mutation.
        assert "may still have been" in workflow_cmd._TOO_LARGE_HINTS[verb]


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
        calls = _patch_urlopen(monkeypatch, {"/api/features": _FLAG_OFF, "/api/workflows": created})
        env = _run(["save", str(wf_path), "--name", "my workflow", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "new-uuid"
        # POSTed JSON carries the workflow_json + name.
        assert calls[-1]["method"] == "POST"
        sent = json.loads(calls[-1]["body"])
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
# save into a Cloud Workflows-sidebar folder (the cli_cloud_save_enabled flag)
# ---------------------------------------------------------------------------

_FLAG_OFF = {"cli_cloud_save_enabled": False}
_FLAG_ON = {"cli_cloud_save_enabled": True}
# An editor-format workflow: the only kind the sidebar can open.
_UI_DATA = {"nodes": [{"id": 1, "type": "KSampler", "widgets_values": [20]}], "links": [], "version": 0.4}
# An API-format workflow: saved the record-only way even with the flag on.
_WF_DATA = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}
_RECORD_CREATED = {"id": "new-uuid", "name": "flux", "latest_version": 1}
_WORKSPACE = {"id": "ws_123", "name": "Personal", "type": "personal", "role": "owner"}
_SAVED_FILE_ID = "2c2fd5e2-c367-48cf-936f-c561f2414c19"
_SAVED_FILE_FLAGS = {**_FLAG_ON, "private_saved_file_links_enabled": True}


def _patch_cloud(monkeypatch: pytest.MonkeyPatch, handler):
    """Route every request through ``handler(method, url, body)``.

    It returns a JSON-able payload, ``(bytes, status)``, or an Exception to
    raise. Records each call as ``_patch_urlopen`` does.
    """
    calls: list[dict] = []

    def _fake(req, timeout=None):
        call = {"url": req.full_url, "method": req.get_method(), "body": req.data, "headers": dict(req.headers)}
        calls.append(call)
        payload = handler(call["method"], call["url"], call["body"])
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, tuple):
            return _fake_resp(*payload)
        return _fake_resp(json.dumps(payload).encode())

    import comfy_cli.http as http_mod

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", _fake)
    return calls


def _sidebar(
    features=_FLAG_ON,
    *,
    taken=(),
    nested=(),
    conflict=(),
    records=(),
    workspace=_WORKSPACE,
    lookup_after_write=None,
    save_reply=None,
):
    """A fake Cloud.

    ``taken`` folders list a file, ``nested`` folders hold only a file in a
    subfolder (listed only with ``recurse=true``), a POST into a ``conflict``
    folder gets 409, and ``records`` are the names of live workflow records.
    A written file is mirrored into a record named after its path, as Cloud
    does. ``lookup_after_write``, when set, answers the record search once a
    file has been written.
    """
    import urllib.parse

    state = {"records": list(records), "written": False}

    def handler(method, url, body):
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/api/features":
            return features
        if parsed.path == "/api/workspaces/current":
            return workspace
        if parsed.path == "/api/userdata" and method == "GET":
            assert query.get("recurse") == ["true"]
            folder = query["dir"][0].removeprefix("workflows/")
            if folder in taken:
                return [{"path": "old.json"}]
            if folder in nested:
                return [{"path": "nested/old.json"}]
            return _http_error(404)
        if parsed.path.startswith("/api/userdata/") and method == "POST":
            key = urllib.parse.unquote(parsed.path.removeprefix("/api/userdata/"))
            assert query == {"overwrite": ["false"]}
            folder = key.split("/")[1]
            if folder in conflict:
                return _http_error(409)
            state["written"] = True
            state["records"].append(key.removeprefix("workflows/").removesuffix(".json"))
            return save_reply if save_reply is not None else {"path": key, "size": len(body), "modified": 1700000000000}
        if parsed.path == "/api/workflows" and method == "GET":
            if state["written"] and lookup_after_write is not None:
                return lookup_after_write
            needle = query["name"][0].lower()
            rows = [{"id": f"rec-{n}", "name": n, "latest_version": 1} for n in state["records"] if needle in n.lower()]
            return {"data": rows, "pagination": {"total": len(rows), "limit": 100, "offset": 0}}
        if parsed.path == "/api/workflows" and method == "POST":
            return _RECORD_CREATED
        raise AssertionError(f"unexpected {method} {url}")

    return handler


def _posts(calls):
    return [c for c in calls if c["method"] == "POST"]


class TestSaveToSidebarFolder:
    @pytest.fixture
    def wf_path(self, tmp_path):
        p = tmp_path / "wf.json"
        p.write_text(json.dumps(_UI_DATA, indent=2))
        return p

    @pytest.mark.parametrize("deployment", ["standard", "dep_9"])
    @pytest.mark.parametrize("file_id", [_SAVED_FILE_ID, _SAVED_FILE_ID.upper()])
    def test_saved_file_link_uses_the_file_id_from_the_save_reply(
        self, cloud_target, wf_path, monkeypatch, capsys, deployment, file_id
    ):
        calls = _patch_cloud(
            monkeypatch,
            _sidebar(features=_SAVED_FILE_FLAGS, save_reply={"id": file_id, "size": 42}),
        )
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud", "--deployment", deployment], capsys)
        data = env["data"]
        assert env["ok"] is True
        assert data["url"] == (
            f"https://cloud.example.com/?workspace=ws_123&savedFile={_SAVED_FILE_ID}&deployment={deployment}"
        )
        assert data["saved_file_id"] == _SAVED_FILE_ID
        assert data["workflow_id"] == "rec-flux/flux"
        assert data["latest_version"] == 1
        assert data["folder"] == "flux"
        assert data["filename"] == "workflows/flux/flux.json"
        assert data["size"] == 42
        assert data["workspace_id"] == "ws_123"
        assert data["deployment"] == deployment
        assert sum(c["url"].endswith("/api/features") for c in calls) == 1
        (post,) = _posts(calls)
        assert post["body"] == wf_path.read_bytes()
        assert post["url"].endswith("?overwrite=false")

    @pytest.mark.parametrize("flag", [False, None, "true", 1, [], {}])
    def test_link_flag_requires_boolean_true(self, cloud_target, wf_path, monkeypatch, capsys, flag):
        features = {**_FLAG_ON, "private_saved_file_links_enabled": flag}
        _patch_cloud(monkeypatch, _sidebar(features=features, save_reply={"id": _SAVED_FILE_ID}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["url"] == "https://cloud.example.com/?workspace=ws_123&workflows=flux&deployment=standard"
        assert env["data"]["saved_file_id"] is None

    @pytest.mark.parametrize(
        "reply",
        [{}, {"id": None}, {"id": 42}, {"id": []}, {"id": "not-a-uuid"}, {"id": "bad&savedFile=x"}, [], (b"oops", 200)],
    )
    def test_missing_or_invalid_file_id_keeps_the_folder_link(self, cloud_target, wf_path, monkeypatch, capsys, reply):
        _patch_cloud(monkeypatch, _sidebar(features=_SAVED_FILE_FLAGS, save_reply=reply))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["url"] == "https://cloud.example.com/?workspace=ws_123&workflows=flux&deployment=standard"
        assert env["data"]["saved_file_id"] is None

    def test_pretty_output_prints_the_json_link(self, cloud_target, wf_path, monkeypatch, capsys):
        _patch_cloud(monkeypatch, _sidebar(features=_SAVED_FILE_FLAGS, save_reply={"id": _SAVED_FILE_ID}))
        args = ["save", str(wf_path), "--name", "flux", "--where", "cloud"]
        env = _run(args, capsys)
        _patch_cloud(monkeypatch, _sidebar(features=_SAVED_FILE_FLAGS, save_reply={"id": _SAVED_FILE_ID}))
        renderer = _force_json_renderer()
        renderer.mode = OutputMode.PRETTY
        messages = []
        monkeypatch.setattr(workflow_cmd, "rprint", messages.append)
        result = CliRunner().invoke(workflow_cmd.app, args, standalone_mode=False)
        assert result.exit_code == 0
        assert f"open it: {env['data']['url']}" in messages

    def test_file_link_without_workspace_names_the_saved_file_in_the_warning(
        self, cloud_target, wf_path, monkeypatch, capsys
    ):
        _patch_cloud(monkeypatch, _sidebar(features=_SAVED_FILE_FLAGS, save_reply={"id": _SAVED_FILE_ID}, workspace={}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["url"] == f"https://cloud.example.com/?savedFile={_SAVED_FILE_ID}"
        assert env["data"]["saved_file_id"] == _SAVED_FILE_ID
        assert env["data"]["warnings"][0]["code"] == "workspace_unknown"
        assert "saved file" in env["data"]["warnings"][0]["message"]

    def test_writes_the_file_into_a_new_folder_and_prints_its_link(self, cloud_target, wf_path, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar())
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        data = env["data"]
        assert data["filename"] == "workflows/flux/flux.json"
        assert data["folder"] == "flux"
        assert data["name"] == "flux/flux"
        assert data["workspace_id"] == "ws_123"
        assert data["deployment"] == "standard"
        assert data["url"] == "https://cloud.example.com/?workspace=ws_123&workflows=flux&deployment=standard"
        assert data["saved_file_id"] is None
        assert "warnings" not in data
        (post,) = _posts(calls)
        assert post["url"] == "https://cloud.example.com/api/userdata/workflows%2Fflux%2Fflux.json?overwrite=false"
        # The file's bytes go up as they are on disk; no record-only POST is made.
        assert post["body"] == wf_path.read_bytes()

    def test_output_keeps_the_record_id_and_version(self, cloud_target, wf_path, monkeypatch, capsys):
        # The record-only save returned workflow_id and latest_version, and
        # `workflow get` / `workflow delete` take that id. The sidebar save
        # still returns them: the id of the record Cloud mirrored the file into.
        _patch_cloud(monkeypatch, _sidebar(records={"flux/flux-old"}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["workflow_id"] == "rec-flux/flux"
        assert env["data"]["latest_version"] == 1
        assert env["data"]["source"] == str(wf_path)

    @pytest.mark.parametrize(
        "lookup",
        [
            pytest.param(_http_error(500), id="http-500"),
            pytest.param(http.client.IncompleteRead(b"{"), id="truncated"),
            pytest.param({"data": []}, id="not-found"),
        ],
    )
    def test_an_unknown_record_still_saves_with_a_warning(self, cloud_target, wf_path, monkeypatch, capsys, lookup):
        _patch_cloud(monkeypatch, _sidebar(lookup_after_write=lookup))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["filename"] == "workflows/flux/flux.json"
        assert env["data"]["workflow_id"] is None
        assert env["data"]["latest_version"] is None
        assert [w["code"] for w in env["data"]["warnings"]] == ["workflow_record_unknown"]

    def test_deployment_option_goes_into_the_link(self, cloud_target, wf_path, monkeypatch, capsys):
        _patch_cloud(monkeypatch, _sidebar())
        env = _run(["save", str(wf_path), "--name", "flux", "--deployment", "dep_9", "--where", "cloud"], capsys)
        assert env["data"]["deployment"] == "dep_9"
        assert env["data"]["url"].endswith("&deployment=dep_9")

    def test_bad_deployment_id_is_refused_before_any_request(self, cloud_target, wf_path, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar())
        env = _run(["save", str(wf_path), "--name", "flux", "--deployment", "a&b=c", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_a_folder_with_files_moves_on_to_a_numbered_one(self, cloud_target, wf_path, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar(taken={"flux", "flux (2)"}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["filename"] == "workflows/flux (3)/flux.json"
        # Spaces are %20 in the link: the editor reads a '+' as a literal plus.
        assert "workflows=flux%20%283%29" in env["data"]["url"]
        assert len(_posts(calls)) == 1

    def test_a_folder_with_only_nested_files_is_taken(self, cloud_target, wf_path, monkeypatch, capsys):
        # Without recurse=true Cloud leaves out workflows/flux/nested/old.json
        # and answers 404, which read as an empty folder.
        _patch_cloud(monkeypatch, _sidebar(nested={"flux"}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"

    def test_an_existing_record_of_the_mirror_name_is_never_added_to(self, cloud_target, wf_path, monkeypatch, capsys):
        # A record named flux/flux with no file behind it: writing
        # workflows/flux/flux.json would add a version to it. A record whose
        # name only contains it does not count.
        calls = _patch_cloud(monkeypatch, _sidebar(records={"flux/flux", "Flux/Flux", "old flux/flux"}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"
        assert env["data"]["workflow_id"] == "rec-flux (2)/flux"
        (post,) = _posts(calls)
        assert "flux%20%282%29%2Fflux.json" in post["url"]

    def test_a_record_search_past_the_page_cap_counts_as_taken(self, cloud_target, wf_path, monkeypatch, capsys):
        import urllib.parse

        base = _sidebar()
        full_page = {"data": [{"id": f"r{i}", "name": f"x flux/flux {i}"} for i in range(100)]}

        def handler(method, url, body):
            parsed = urllib.parse.urlsplit(url)
            name = urllib.parse.parse_qs(parsed.query).get("name", [""])[0]
            if parsed.path == "/api/workflows" and method == "GET" and name == "flux/flux":
                return full_page
            return base(method, url, body)

        _patch_cloud(monkeypatch, handler)
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"

    @staticmethod
    def _paged_search(pages):
        """A fake Cloud whose search for ``flux/flux`` answers ``pages`` by offset; the rest is ``_sidebar()``."""
        import urllib.parse

        base = _sidebar()

        def handler(method, url, body):
            parsed = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/workflows" and method == "GET" and query.get("name") == ["flux/flux"]:
                return pages[int(query["offset"][0])]
            return base(method, url, body)

        return handler

    @staticmethod
    def _search_offsets(calls):
        import urllib.parse

        return [
            urllib.parse.parse_qs(urllib.parse.urlsplit(c["url"]).query)["offset"][0]
            for c in calls
            if c["method"] == "GET" and "/api/workflows?name=flux%2Fflux&" in c["url"]
        ]

    def test_a_short_page_with_more_to_come_keeps_searching(self, cloud_target, wf_path, monkeypatch, capsys):
        # Cloud sends 2 rows where 100 were asked for and says more follow. The
        # record named flux/flux is on the next page, so flux is taken, and the
        # next page starts after the 2 rows sent, not at offset 100.
        pages = {
            0: {
                "data": [{"id": "r1", "name": "a flux/flux"}, {"id": "r2", "name": "b flux/flux"}],
                "pagination": {"total": 3, "limit": 2, "offset": 0, "has_more": True},
            },
            2: {
                "data": [{"id": "r3", "name": "flux/flux"}],
                "pagination": {"total": 3, "limit": 2, "offset": 2, "has_more": False},
            },
        }
        calls = _patch_cloud(monkeypatch, self._paged_search(pages))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"
        assert self._search_offsets(calls) == ["0", "2"]

    def test_a_full_page_with_nothing_more_still_reads_the_next_page(self, cloud_target, wf_path, monkeypatch, capsys):
        # A full page does not end the search even with has_more false; the
        # empty page after it does, and flux is free.
        full_page = {
            "data": [{"id": f"r{i}", "name": f"x flux/flux {i}"} for i in range(100)],
            "pagination": {"total": 100, "limit": 100, "offset": 0, "has_more": False},
        }
        empty = {"data": [], "pagination": {"total": 100, "limit": 100, "offset": 100, "has_more": False}}
        calls = _patch_cloud(monkeypatch, self._paged_search({0: full_page, 100: empty}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux"
        # The search before the write and the one after it each read both pages.
        assert self._search_offsets(calls) == ["0", "100", "0", "100"]

    def test_a_record_added_after_the_count_is_still_found(self, cloud_target, wf_path, monkeypatch, capsys):
        # Cloud counts the rows before it reads the page. A record named
        # flux/flux added in between lands on the next page while the full
        # first page says has_more false. The search must still find it, so
        # flux is taken and nothing is written into it.
        full_page = {
            "data": [{"id": f"r{i}", "name": f"x flux/flux {i}"} for i in range(100)],
            "pagination": {"total": 100, "limit": 100, "offset": 0, "has_more": False},
        }
        late = {
            "data": [{"id": "r-late", "name": "flux/flux"}],
            "pagination": {"total": 101, "limit": 100, "offset": 100, "has_more": False},
        }
        calls = _patch_cloud(monkeypatch, self._paged_search({0: full_page, 100: late}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"
        assert self._search_offsets(calls) == ["0", "100"]
        (post,) = _posts(calls)
        assert "flux%20%282%29%2Fflux.json" in post["url"]

    def test_more_to_come_with_no_rows_counts_as_taken(self, cloud_target, wf_path, monkeypatch, capsys):
        # has_more true on an empty page cannot move the offset; asking again
        # would get the same page, so the search has no answer.
        empty = {"data": [], "pagination": {"total": 5, "limit": 100, "offset": 0, "has_more": True}}
        calls = _patch_cloud(monkeypatch, self._paged_search({0: empty}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"
        assert self._search_offsets(calls) == ["0"]

    def test_a_failed_record_search_saves_nothing(self, cloud_target, wf_path, monkeypatch, capsys):
        base = _sidebar()

        def handler(method, url, body):
            if method == "GET" and "/api/workflows?" in url:
                return _http_error(500)
            return base(method, url, body)

        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert _posts(calls) == []

    def test_a_409_moves_on_and_nothing_is_overwritten(self, cloud_target, wf_path, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar(conflict={"flux"}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["data"]["folder"] == "flux (2)"
        posts = _posts(calls)
        assert len(posts) == 2
        assert all(c["url"].endswith("?overwrite=false") for c in posts)
        assert not any(c["method"] in ("PUT", "DELETE") for c in calls)

    def test_every_folder_taken_saves_nothing(self, cloud_target, wf_path, monkeypatch, capsys):
        taken = {"flux"} | {f"flux ({n})" for n in range(2, workflow_cmd._MAX_FOLDER_TRIES + 1)}
        calls = _patch_cloud(monkeypatch, _sidebar(taken=taken))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_folder_unavailable"
        assert _posts(calls) == []

    @pytest.mark.parametrize(
        ("name", "folder"),
        [
            ("flux.json", "flux"),
            ("team/flux", "team-flux"),
            ("a,b\\c", "a-b-c"),
            ("../..secret", "-.secret"),
            ("  .hidden.  ", "hidden"),
            ("two   spaces", "two spaces"),
            ("...", "workflow"),
        ],
    )
    def test_folder_names_are_one_safe_segment(self, name, folder):
        assert workflow_cmd._cloud_folder_name(name) == folder

    def test_description_is_reported_as_ignored(self, cloud_target, wf_path, monkeypatch, capsys):
        _patch_cloud(monkeypatch, _sidebar())
        env = _run(["save", str(wf_path), "--name", "flux", "--description", "d", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert [w["code"] for w in env["data"]["warnings"]] == ["description_ignored"]

    @pytest.mark.parametrize(
        "workspace",
        [
            pytest.param(_http_error(500), id="http-500"),
            pytest.param(http.client.IncompleteRead(b"{"), id="truncated"),
            pytest.param({"id": "bad id"}, id="bad-id"),
            pytest.param({}, id="no-id"),
        ],
    )
    def test_unknown_workspace_still_saves_and_links_the_folder(
        self, cloud_target, wf_path, monkeypatch, capsys, workspace
    ):
        _patch_cloud(monkeypatch, _sidebar(workspace=workspace))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workspace_id"] is None
        assert env["data"]["url"] == "https://cloud.example.com/?workflows=flux"
        assert [w["code"] for w in env["data"]["warnings"]] == ["workspace_unknown"]

    def test_a_failed_write_is_an_error_not_a_retry(self, cloud_target, wf_path, monkeypatch, capsys):
        base = _sidebar()

        def handler(method, url, body):
            if method == "POST":
                return _http_error(500)
            return base(method, url, body)

        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert len(_posts(calls)) == 1

    def test_api_format_keeps_the_record_save_with_a_warning(self, cloud_target, tmp_path, monkeypatch, capsys):
        # The sidebar opens a file as an editor graph, which builds nodes only
        # from nodes[]: an API-format file there would open empty.
        wf_path = tmp_path / "api.json"
        wf_path.write_text(json.dumps(_WF_DATA))
        calls = _patch_cloud(monkeypatch, _sidebar(features=_SAVED_FILE_FLAGS, save_reply={"id": _SAVED_FILE_ID}))
        env = _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "new-uuid"
        assert "url" not in env["data"]
        assert "saved_file_id" not in env["data"]
        assert [w["code"] for w in env["data"]["warnings"]] == ["api_format_saved_as_record"]
        (post,) = _posts(calls)
        assert post["url"] == "https://cloud.example.com/api/workflows"
        assert json.loads(post["body"]) == {"name": "flux", "workflow_json": _WF_DATA}
        assert not any("/api/userdata" in c["url"] for c in calls)


class TestSaveFlagOffKeepsTheRecordSave:
    """Flag off, missing, or unreadable: the save is the record-only POST, as before the flag."""

    @pytest.mark.parametrize(
        "features",
        [
            pytest.param(_FLAG_OFF, id="off"),
            pytest.param({**_FLAG_OFF, "private_saved_file_links_enabled": True}, id="only-file-links-on"),
            pytest.param({}, id="missing"),
            pytest.param({"cli_cloud_save_enabled": "true"}, id="not-a-bool"),
            pytest.param([], id="not-an-object"),
            pytest.param(_http_error(500), id="http-500"),
            pytest.param(_http_error(401), id="http-401"),
            pytest.param(urllib.error.URLError("down"), id="unreachable"),
            pytest.param(TimeoutError("slow"), id="timeout"),
            pytest.param((b"<html>", 200), id="unparseable"),
            pytest.param(http.client.IncompleteRead(b'{"cli_'), id="truncated"),
            pytest.param(http.client.RemoteDisconnected("closed"), id="disconnected"),
        ],
    )
    def test_record_save_unchanged(self, cloud_target, tmp_path, monkeypatch, capsys, features):
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(_UI_DATA))
        calls = _patch_cloud(monkeypatch, _sidebar(features=features))
        env = _run(["save", str(wf_path), "--name", "flux", "--description", "d", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"] == {
            "workflow_id": "new-uuid",
            "name": "flux",
            "latest_version": 1,
            "source": str(wf_path),
        }
        assert [c["url"] for c in calls] == [
            "https://cloud.example.com/api/features",
            "https://cloud.example.com/api/workflows",
        ]
        assert calls[0]["method"] == "GET"
        assert json.loads(calls[1]["body"]) == {"name": "flux", "workflow_json": _UI_DATA, "description": "d"}

    def test_features_read_sends_the_users_auth(self, cloud_target, tmp_path, monkeypatch, capsys):
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(_UI_DATA))
        calls = _patch_cloud(monkeypatch, _sidebar(features=_FLAG_OFF))
        _run(["save", str(wf_path), "--name", "flux", "--where", "cloud"], capsys)
        # urllib stores header names capitalized: X-API-Key -> X-api-key.
        assert calls[0]["url"].endswith("/api/features")
        assert calls[0]["headers"].get("X-api-key") == "test-key"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_sends_delete_request(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/features": _FLAG_OFF, "/api/workflows/wf-uuid": (b"", 204)})
        env = _run(["delete", "wf-uuid", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["deleted"] is True
        assert calls[-1]["method"] == "DELETE"

    def test_404_surfaces_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError("https://x/workflows/g", 404, "Not Found", {}, io.BytesIO(b"{}"))
        _patch_urlopen(monkeypatch, {"/api/features": _FLAG_OFF, "/api/workflows/ghost": err})
        env = _run(["delete", "ghost", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"


def _sidebar_delete(
    features=_FLAG_ON,
    *,
    record=None,
    file=None,
    record_delete=(b"", 204),
    others=(),
    content=None,
    search=None,
):
    """A fake Cloud for delete: record ``rec-1`` and, unless ``file`` is None, the sidebar file of its name.

    ``others`` are more live records, each ``(id, name)``, that the name
    search lists beside ``rec-1``. ``content`` is ``rec-1``'s latest version
    (``_UI_DATA`` unless given). ``search``, when set, answers the name search
    in place of the records. It refuses any userdata DELETE: the CLI must
    never delete a sidebar file.
    """
    import urllib.parse

    record = {"id": "rec-1", "name": "flux/flux", "latest_version": 1} if record is None else record
    content = {"version": 1, "workflow_json": _UI_DATA} if content is None else content

    def handler(method, url, body):
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path
        if path == "/api/features":
            return features
        if path == "/api/workflows/rec-1" and method == "GET":
            return record
        if path == "/api/workflows/rec-1/content" and method == "GET":
            return content
        if path == "/api/workflows" and method == "GET":
            if search is not None:
                return search
            needle = urllib.parse.parse_qs(parsed.query)["name"][0].lower()
            rows = [{"id": "rec-1", "name": record["name"], "latest_version": 1}]
            rows += [{"id": i, "name": n, "latest_version": 1} for i, n in others]
            rows = [r for r in rows if needle in r["name"].lower()]
            return {"data": rows, "pagination": {"total": len(rows), "limit": 100, "offset": 0}}
        if path.startswith("/api/userdata/") and method == "GET":
            return _http_error(404) if file is None else file
        if path == "/api/workflows/rec-1" and method == "DELETE":
            return record_delete
        raise AssertionError(f"unexpected {method} {url}")

    return handler


class TestDeleteKeepsTheSidebarFile:
    """Flag on: delete removes only the record, as with the flag off, and names a sidebar file that stays."""

    _RECORD_DELETE = {"url": "https://cloud.example.com/api/workflows/rec-1", "method": "DELETE"}

    def _deletes(self, calls):
        return [{"url": c["url"], "method": c["method"]} for c in calls if c["method"] != "GET"]

    def test_a_kept_file_is_named_in_a_warning(self, cloud_target, monkeypatch, capsys):
        # Cloud's DELETE /api/workflows/<id> leaves workflows/flux/flux.json,
        # which stays in the sidebar. The CLI says so instead of deleting it:
        # Cloud's file DELETE takes no condition and could remove a newer edit.
        calls = _patch_cloud(monkeypatch, _sidebar_delete(file=(json.dumps(_UI_DATA).encode(), 200)))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["workflow_id"] == "rec-1"
        assert env["data"]["deleted"] is True
        [warning] = env["data"]["warnings"]
        assert warning["code"] == "sidebar_file_kept"
        assert warning["filename"] == "workflows/flux/flux.json"
        assert "workflows/flux/flux.json" in warning["message"]
        assert self._deletes(calls) == [self._RECORD_DELETE]
        # The record's name, its file, the records of that name and the
        # record's content are all read before the DELETE renames the record.
        assert [(c["method"], c["url"]) for c in calls] == [
            ("GET", "https://cloud.example.com/api/features"),
            ("GET", "https://cloud.example.com/api/workflows/rec-1"),
            ("GET", "https://cloud.example.com/api/userdata/workflows%2Fflux%2Fflux.json"),
            ("GET", "https://cloud.example.com/api/workflows?name=flux%2Fflux&limit=100&offset=0&sort=name&order=asc"),
            ("GET", "https://cloud.example.com/api/workflows/rec-1/content"),
            ("DELETE", "https://cloud.example.com/api/workflows/rec-1"),
        ]

    @pytest.mark.parametrize(
        "others, content",
        [
            # The finding's case: file A at workflows/flux/flux.json mirrors
            # record rec-A, and rec-1 is a record-only workflow B of the same
            # name with other content.
            pytest.param([("rec-A", "flux/flux")], {"workflow_json": _WF_DATA}, id="other-record-other-content"),
            # B was posted with A's exact content: the file still cannot be told apart.
            pytest.param([("rec-A", "flux/flux")], None, id="other-record-same-content"),
            # A's record is gone and B is the only record of the name.
            pytest.param([], {"workflow_json": _WF_DATA}, id="only-record-other-content"),
        ],
    )
    def test_a_same_named_file_of_another_workflow_is_not_named(
        self, cloud_target, monkeypatch, capsys, others, content
    ):
        # Cloud lets several live records share a name, and a record made by
        # POST /api/workflows has no file. So deleting rec-1 must not call
        # workflows/flux/flux.json rec-1's file, or the person would remove
        # another workflow's file. The delete itself is unchanged.
        handler = _sidebar_delete(file=(json.dumps(_UI_DATA).encode(), 200), others=others, content=content)
        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]

    def test_a_record_with_a_longer_name_still_gives_the_warning(self, cloud_target, monkeypatch, capsys):
        # Cloud's name search matches a substring, so it also lists "flux/flux v2";
        # only an exact name counts as a second record of this name.
        handler = _sidebar_delete(file=(json.dumps(_UI_DATA).encode(), 200), others=[("rec-C", "flux/flux v2")])
        _patch_cloud(monkeypatch, handler)
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        [warning] = env["data"]["warnings"]
        assert warning["code"] == "sidebar_file_kept"
        assert warning["filename"] == "workflows/flux/flux.json"

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"search": _http_error(500)}, id="name-search-fails"),
            pytest.param({"search": {"data": "x"}}, id="name-search-odd"),
            pytest.param({"content": _http_error(500)}, id="content-read-fails"),
            pytest.param({"content": {"version": 1}}, id="content-without-workflow-json"),
            pytest.param({"file": (b"<html>", 200)}, id="file-not-json"),
        ],
    )
    def test_an_unproven_file_gives_the_plain_delete(self, cloud_target, monkeypatch, capsys, kwargs):
        handler = _sidebar_delete(**{"file": (json.dumps(_UI_DATA).encode(), 200), **kwargs})
        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]

    def test_a_name_search_past_the_page_cap_gives_the_plain_delete(self, cloud_target, monkeypatch, capsys):
        monkeypatch.setattr(workflow_cmd, "_RECORD_PAGE_SIZE", 1)
        monkeypatch.setattr(workflow_cmd, "_MAX_RECORD_PAGES", 1)
        handler = _sidebar_delete(file=(json.dumps(_UI_DATA).encode(), 200))
        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]

    def test_no_file_gives_the_plain_delete(self, cloud_target, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar_delete(file=None))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]

    @pytest.mark.parametrize(
        "file",
        [
            pytest.param(_http_error(500), id="500"),
            pytest.param(http.client.IncompleteRead(b"{"), id="truncated"),
            pytest.param(urllib.error.URLError("unreachable"), id="unreachable"),
        ],
    )
    def test_a_failed_file_read_gives_the_plain_delete(self, cloud_target, monkeypatch, capsys, file):
        calls = _patch_cloud(monkeypatch, _sidebar_delete(file=file))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]

    def test_a_failed_record_read_gives_the_plain_delete(self, cloud_target, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar_delete(record=_http_error(500), file=(b"{}", 200)))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert self._deletes(calls) == [self._RECORD_DELETE]
        assert not any("/api/userdata/" in c["url"] for c in calls)

    @pytest.mark.parametrize("name", ["../x", "a//b", "a\\b", "", None])
    def test_a_name_that_is_no_file_path_is_not_looked_up(self, cloud_target, monkeypatch, capsys, name):
        record = {"id": "rec-1", "name": name}
        calls = _patch_cloud(monkeypatch, _sidebar_delete(record=record, file=(b"{}", 200)))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert not any("/api/userdata/" in c["url"] for c in calls)

    def test_flag_off_never_looks_for_a_file(self, cloud_target, monkeypatch, capsys):
        calls = _patch_cloud(monkeypatch, _sidebar_delete(features=_FLAG_OFF, file=(b"{}", 200)))
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["data"] == {"workflow_id": "rec-1", "deleted": True}
        assert [c["url"] for c in calls] == [
            "https://cloud.example.com/api/features",
            "https://cloud.example.com/api/workflows/rec-1",
        ]

    def test_a_failed_record_delete_is_the_same_error_as_before(self, cloud_target, monkeypatch, capsys):
        handler = _sidebar_delete(file=(b"{}", 200), record_delete=_http_error(404))
        calls = _patch_cloud(monkeypatch, handler)
        env = _run(["delete", "rec-1", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"
        assert self._deletes(calls) == [self._RECORD_DELETE]


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
        _patch_urlopen(monkeypatch, {"/api/features": _FLAG_OFF, "/api/workflows": (raw, 200)})
        env = _run(["save", str(wf_path), "--name", "x", "--where", "cloud"], capsys)
        # Must NOT claim success with a null workflow_id.
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
        _patch_urlopen(monkeypatch, {"/api/features": _FLAG_OFF, "/api/workflows/wf-uuid": (raw, 200)})
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
