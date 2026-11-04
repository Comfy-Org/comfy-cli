"""Loopback contract POC for saving sidebar-visible workflow files.

This deliberately exercises the root CLI and the real urllib client.  The
server is only a faithful test double for the pinned userdata handler contract;
it does not claim to prove a deployed Cloud database or browser integration.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import cmdline
from comfy_cli.output.renderer import reset_renderer_for_testing
from comfy_cli.target import Target

_FAKE_KEY_A = "fixture-workspace-a-key"
_FAKE_KEY_B = "fixture-workspace-b-key"
_FRONTEND_WORKFLOW = {
    "last_node_id": 2,
    "last_link_id": 1,
    "nodes": [
        {
            "id": 1,
            "type": "KSampler",
            "pos": [20, 30],
            "size": [315, 262],
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}],
            "widgets_values": [123, "fixed", 20, 8.0],
        },
        {
            "id": 2,
            "type": "VAEDecode",
            "pos": [400, 30],
            "size": [210, 46],
            "inputs": [{"name": "samples", "type": "LATENT", "link": 1}],
            "outputs": [],
        },
    ],
    "links": [[1, 1, 0, 2, 0, "LATENT"]],
    "groups": [],
    "config": {},
    "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
    "version": 0.4,
}


class _UserdataContract:
    """State and sanitized receipts for one loopback endpoint."""

    def __init__(self, *, cloud: bool):
        self.cloud = cloud
        self.files: dict[str, dict[str, bytes]] = {"workspace-a": {}, "workspace-b": {}, "local": {}}
        self.receipts: list[dict[str, Any]] = []
        self.mirror_fails = False

    def scope(self, handler: BaseHTTPRequestHandler) -> str | None:
        if not self.cloud:
            return "local"
        key = handler.headers.get("X-API-Key")
        if key is None:
            handler.send_error(401)
            return None
        scope = {_FAKE_KEY_A: "workspace-a", _FAKE_KEY_B: "workspace-b"}.get(key)
        if scope is None:
            handler.send_error(403)
            return None
        return scope

    def record(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        parsed = urllib.parse.urlsplit(handler.path)
        self.receipts.append(
            {
                "method": handler.command,
                "path": parsed.path,
                "query": urllib.parse.parse_qs(parsed.query),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "body_bytes": len(body),
                "auth": "X-API-Key:<redacted>" if handler.headers.get("X-API-Key") else None,
            }
        )


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", "0")))

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> tuple[str, str] | None:
        prefix = "/api/userdata/" if self.server.contract.cloud else "/userdata/"
        path = urllib.parse.urlsplit(self.path).path
        if not path.startswith(prefix):
            return None
        relative = urllib.parse.unquote(path[len(prefix) :])
        if not relative.startswith("workflows/"):
            return None
        return relative, relative.removeprefix("workflows/")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self._body()
        self.server.contract.record(self, body)
        scope = self.server.contract.scope(self)
        if scope is None:
            return
        route = self._route()
        if route is None:
            self.send_error(404)
            return
        relative, key = route
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        overwrite = query.get("overwrite") == ["true"]
        if key in self.server.contract.files[scope] and not overwrite:
            self.send_error(409)
            return
        self.server.contract.files[scope][key] = body
        # Matches the pinned handler: the durable file write is authoritative;
        # SyncFromUserdata is best effort and does not turn its failure into a
        # failed or repeated upload. The response intentionally has no ID.
        info = {"path": relative, "modified": 1700000002000, "size": len(body)}
        self._send(200, json.dumps(info).encode())

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.server.contract.record(self, b"")
        scope = self.server.contract.scope(self)
        if scope is None:
            return
        parsed = urllib.parse.urlsplit(self.path)
        collection = "/api/userdata" if self.server.contract.cloud else "/userdata"
        if parsed.path == collection:
            query = urllib.parse.parse_qs(parsed.query)
            if query != {
                "dir": ["workflows"],
                "recurse": ["true"],
                "split": ["false"],
                "full_info": ["true"],
            }:
                self.send_error(400)
                return
            listing = [
                {"path": key, "modified": 1700000002000, "size": len(body)}
                for key, body in sorted(self.server.contract.files[scope].items())
            ]
            self._send(200, json.dumps(listing).encode())
            return
        route = self._route()
        if route is None:
            self.send_error(404)
            return
        _relative, key = route
        body = self.server.contract.files[scope].get(key)
        if body is None:
            self.send_error(404)
            return
        self._send(200, body)

    def do_DELETE(self) -> None:  # noqa: N802 - negative control only
        body = self._body()
        self.server.contract.record(self, body)
        self.send_error(405)


class _Server(ThreadingHTTPServer):
    def __init__(self, contract: _UserdataContract):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.contract = contract


@contextmanager
def _serve(*, cloud: bool):
    contract = _UserdataContract(cloud=cloud)
    server = _Server(contract)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield contract, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _target(kind: str, base_url: str, *, key: str | None = None) -> Target:
    return Target(
        kind=kind,
        base_url=base_url,
        path_prefix="/api" if kind == "cloud" else "",
        history_path="history_v2" if kind == "cloud" else "history",
        jobs_path="jobs" if kind == "cloud" else None,
        api_key=key,
        host="127.0.0.1" if kind == "local" else None,
    )


def _invoke(args: list[str]) -> tuple[dict[str, Any], int]:
    reset_renderer_for_testing()
    result = CliRunner().invoke(cmdline.app, ["--json", "workflow", *args])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"no JSON output; rc={result.exit_code}, exception={result.exception!r}"
    return json.loads(lines[-1]), result.exit_code


def _sidebar_get(base_url: str, key: str, *, api_key: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(f"workflows/{key}", safe="")
    request = urllib.request.Request(f"{base_url}/api/userdata/{encoded}", headers={"X-API-Key": api_key}, method="GET")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - loopback fixture URL
        return json.loads(response.read())


def _sidebar_list(base_url: str, *, api_key: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"dir": "workflows", "recurse": "true", "split": "false", "full_info": "true"})
    request = urllib.request.Request(f"{base_url}/api/userdata?{query}", headers={"X-API-Key": api_key}, method="GET")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - loopback fixture URL
        return json.loads(response.read())


@pytest.fixture(autouse=True)
def _reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def test_cloud_save_and_sidebar_userdata_readback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    with _serve(cloud=True) as (cloud, cloud_url):
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )

        envelope, rc = _invoke(["save", str(workflow_file), "--name", "team/demo", "--where", "cloud"])

        assert rc == 0 and envelope["ok"] is True
        assert envelope["where"] == "cloud"
        assert envelope["data"]["workflow_id"] is None
        assert envelope["data"]["filename"] == "team/demo.json"
        listing = _sidebar_list(cloud_url, api_key=_FAKE_KEY_A)
        assert [row["path"] for row in listing] == ["team/demo.json"]
        assert _sidebar_get(cloud_url, "team/demo.json", api_key=_FAKE_KEY_A) == _FRONTEND_WORKFLOW
        write = cloud.receipts[0]
        assert write == {
            "method": "POST",
            "path": "/api/userdata/workflows%2Fteam%2Fdemo.json",
            "query": {"overwrite": ["true"], "full_info": ["true"]},
            "body_sha256": hashlib.sha256(workflow_file.read_bytes()).hexdigest(),
            "body_bytes": len(workflow_file.read_bytes()),
            "auth": "X-API-Key:<redacted>",
        }
        assert not any(r["method"] == "DELETE" or r["path"].startswith("/api/workflows") for r in cloud.receipts)


def test_local_route_is_unchanged_and_cloud_receives_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    out = tmp_path / "readback.json"
    with _serve(cloud=True) as (cloud, cloud_url), _serve(cloud=False) as (local, local_url):

        def resolve_target(**kwargs):
            if kwargs.get("where") == "local":
                return _target("local", local_url)
            return _target("cloud", cloud_url, key=_FAKE_KEY_A)

        monkeypatch.setattr("comfy_cli.target.resolve_target", resolve_target)
        saved, save_rc = _invoke(["save", str(workflow_file), "--name", "local-demo", "--where", "local"])
        got, get_rc = _invoke(["get", "local-demo.json", "--out", str(out), "--where", "local"])

        assert save_rc == get_rc == 0
        assert saved["where"] == got["where"] == "local"
        assert json.loads(out.read_text()) == _FRONTEND_WORKFLOW
        assert [r["method"] for r in local.receipts] == ["POST", "GET"]
        assert all(r["auth"] is None for r in local.receipts)
        assert cloud.receipts == []


def test_save_overwrites_same_name_but_contract_refuses_no_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    replacement = {**_FRONTEND_WORKFLOW, "last_node_id": 99}
    with _serve(cloud=True) as (cloud, cloud_url):
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )
        first, first_rc = _invoke(["save", str(workflow_file), "--name", "same", "--where", "cloud"])
        workflow_file.write_text(json.dumps(replacement))
        second, second_rc = _invoke(["save", str(workflow_file), "--name", "same", "--where", "cloud"])

        assert first_rc == second_rc == 0 and first["ok"] and second["ok"]
        assert _sidebar_get(cloud_url, "same.json", api_key=_FAKE_KEY_A) == replacement

        encoded = urllib.parse.quote("workflows/same.json", safe="")
        request = urllib.request.Request(
            f"{cloud_url}/api/userdata/{encoded}?overwrite=false&full_info=true",
            data=json.dumps(_FRONTEND_WORKFLOW).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": _FAKE_KEY_A},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as collision:
            urllib.request.urlopen(request)  # noqa: S310 - loopback fixture URL
        assert collision.value.code == 409
        assert _sidebar_get(cloud_url, "same.json", api_key=_FAKE_KEY_A) == replacement
        assert [r["query"]["overwrite"] for r in cloud.receipts if r["method"] == "POST"] == [
            ["true"],
            ["true"],
            ["false"],
        ]


def test_missing_id_and_best_effort_mirror_failure_do_not_repeat_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    with _serve(cloud=True) as (cloud, cloud_url):
        cloud.mirror_fails = True
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )
        envelope, rc = _invoke(["save", str(workflow_file), "--name", "no-id", "--where", "cloud"])

        assert rc == 0 and envelope["ok"] is True
        assert envelope["data"]["workflow_id"] is None
        assert [w["code"] for w in envelope["data"]["warnings"]] == ["workflow_record_id_unavailable"]
        assert [(r["method"], r["path"]) for r in cloud.receipts] == [("POST", "/api/userdata/workflows%2Fno-id.json")]


@pytest.mark.parametrize("key, expected_status", [(None, 401), ("fixture-invalid-key", 403)])
def test_auth_failure_never_falls_back_or_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str | None, expected_status: int
):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    with _serve(cloud=True) as (cloud, cloud_url), _serve(cloud=False) as (local, local_url):
        monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=key))
        envelope, rc = _invoke(["save", str(workflow_file), "--name", "denied", "--where", "cloud"])

        assert rc == 1 and envelope["ok"] is False
        assert envelope["where"] == "cloud"
        assert envelope["error"]["details"]["status"] == expected_status
        assert not any(cloud.files[scope] for scope in cloud.files)
        assert local.receipts == []
        assert local_url not in json.dumps(cloud.receipts)


def test_workspace_scopes_do_not_fall_back_or_cross_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    with _serve(cloud=True) as (cloud, cloud_url):
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )
        envelope, rc = _invoke(["save", str(workflow_file), "--name", "scoped", "--where", "cloud"])
        assert rc == 0 and envelope["ok"] is True
        assert "scoped.json" in cloud.files["workspace-a"]
        assert cloud.files["workspace-b"] == {}
        with pytest.raises(urllib.error.HTTPError) as missing:
            _sidebar_get(cloud_url, "scoped.json", api_key=_FAKE_KEY_B)
        assert missing.value.code == 404


@pytest.mark.parametrize("name", ["../escape", ".hidden", "team/.hidden/demo", ".index.json"])
def test_unsafe_sidebar_names_refuse_before_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str):
    workflow_file = tmp_path / "frontend.json"
    workflow_file.write_text(json.dumps(_FRONTEND_WORKFLOW))
    with _serve(cloud=True) as (cloud, cloud_url):
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )
        envelope, rc = _invoke(["save", str(workflow_file), "--name", name, "--where", "cloud"])
        assert rc == 1 and envelope["error"]["code"] == "invalid_argument"
        assert cloud.receipts == []


def test_malformed_json_refuses_before_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    workflow_file = tmp_path / "broken.json"
    workflow_file.write_text('{"nodes": [}')
    with _serve(cloud=True) as (cloud, cloud_url):
        monkeypatch.setattr(
            "comfy_cli.target.resolve_target", lambda **kwargs: _target("cloud", cloud_url, key=_FAKE_KEY_A)
        )
        envelope, rc = _invoke(["save", str(workflow_file), "--name", "broken", "--where", "cloud"])
        assert rc == 1 and envelope["error"]["code"] == "workflow_invalid_json"
        assert cloud.receipts == []
