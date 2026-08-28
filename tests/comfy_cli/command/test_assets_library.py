"""``comfy assets library`` error envelopes.

Pins the 404 mapping for ``assets library ensure``. Found in prod (Langfuse
2026-08-25, ``use_asset_as_input``): an agent passed a FILE NAME
(``comfyorg_logo.png``) where the content hash belongs, the API answered 404,
and the CLI reported ``workflow_not_found`` / "workflow not found (ensure)"
with a hint to list *workflows* — the shared cloud-HTTP helper's 404 branch
was written for the saved-workflow commands and hardcoded their vocabulary.
The agent had to guess its way past a message about the wrong resource.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import error_codes
from comfy_cli.caller import Caller
from comfy_cli.command import assets_library
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch):
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


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(assets_library.app, args, standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    return json.loads(captured.strip().splitlines()[-1])


def _http_error(code: int, body: bytes = b""):
    return urllib.error.HTTPError("https://cloud.example.com/api/assets/from-hash", code, "err", {}, io.BytesIO(body))


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcome):
    calls: list[dict] = []

    class _Resp:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return json.dumps(outcome).encode()

    def _fake(req, timeout=None):
        calls.append({"url": req.full_url, "method": req.get_method(), "body": req.data})
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


class TestEnsure:
    def test_404_is_asset_not_found_not_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(404))
        env = _run(["ensure", "--hash", "comfyorg_logo.png", "--where", "cloud"], capsys)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "asset_not_found"
        assert error_codes.is_registered(err["code"])
        assert "workflow" not in err["message"].lower()
        assert "comfyorg_logo.png" in err["message"]
        # The remediation names the asset commands, not `workflow list`.
        assert "assets library ls" in err["hint"]
        assert "workflow list" not in err["hint"]
        assert err["details"]["hash"] == "comfyorg_logo.png"
        assert err["details"]["operation"] == "ensure"

    def test_401_is_still_cloud_unauthorized(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(401))
        env = _run(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert env["error"]["code"] == "cloud_unauthorized"

    def test_success_reports_id_hash_and_created_new(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1", "hash": "a" * 64})
        env = _run(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"] == {"id": "asset-1", "hash": "a" * 64, "created_new": True}
        assert calls[0]["url"].endswith("/api/assets/from-hash") and calls[0]["method"] == "POST"
