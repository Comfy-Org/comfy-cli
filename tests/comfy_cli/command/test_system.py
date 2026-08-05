"""Tests for `comfy system-stats` and `comfy free` — the ComfyUI
`/system_stats` and `/free` passthrough.

All HTTP is mocked at the `comfy_client._OPENER` seam (the opener the
`Client` class actually calls through), matching the mocking style used for
`Client`-based commands elsewhere (e.g. `comfy download`).
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest
import typer

from comfy_cli import comfy_client
from comfy_cli.caller import Caller
from comfy_cli.command import system as system_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer
from comfy_cli.target import Target


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


LOCAL_TARGET = Target(kind="local", base_url="http://127.0.0.1:8188", path_prefix="", host="127.0.0.1", port=8188)
CLOUD_TARGET = Target(
    kind="cloud",
    base_url="https://cloud.example.com",
    path_prefix="/api",
    history_path="history_v2",
    jobs_path="jobs",
    api_key="test-api-key",
)


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n: int | None = None) -> bytes:
        # Mirror http.client.HTTPResponse.read(amt) — bodies are read with a
        # byte cap, so a no-arg-only fake would not match the real API.
        return self._body if n is None else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _force_renderer() -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _envelope(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr().out
    assert captured.strip(), "no envelope on stdout"
    return json.loads(captured.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# system-stats
# ---------------------------------------------------------------------------


_RAW_STATS = {
    "system": {"os": "posix", "ram_total": 34359738368, "ram_free": 12884901888, "comfyui_version": "0.3.45"},
    "devices": [
        {
            "name": "cuda:0 NVIDIA GeForce RTX 4090",
            "type": "cuda",
            "index": 0,
            "vram_total": 25769803776,
            "vram_free": 21474836480,
            "torch_vram_total": 4294967296,
            "torch_vram_free": 2147483648,
        }
    ],
}


class TestSystemStats:
    def test_emits_passthrough_envelope(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.system.resolve_target", return_value=LOCAL_TARGET),
            patch.object(
                comfy_client._OPENER,
                "open",
                side_effect=lambda req, timeout=None: _FakeResp(json.dumps(_RAW_STATS).encode()),
            ),
        ):
            system_cmd.system_stats_execute(renderer, where="local")
        env = _envelope(capsys)
        assert env["ok"] is True
        assert env["command"] == "system-stats"
        # Passed through unmodified — same dict, not a re-shaped one.
        assert env["data"] == _RAW_STATS
        assert isinstance(env["data"]["devices"][0]["vram_free"], int)

    def test_local_unreachable_server_not_running(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.system.resolve_target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=urllib.error.URLError("Connection refused")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                system_cmd.system_stats_execute(renderer, where="local")
        assert exc_info.value.exit_code == 1
        env = _envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"

    def test_cloud_unreachable_is_cloud_http_error(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.system.resolve_target", return_value=CLOUD_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=urllib.error.URLError("Connection refused")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                system_cmd.system_stats_execute(renderer, where="cloud")
        assert exc_info.value.exit_code == 1
        env = _envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"

    def test_http_error_status_surfaced(self, capsys):
        renderer = _force_renderer()

        def _raise(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, None)

        with (
            patch("comfy_cli.command.system.resolve_target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=_raise),
        ):
            with pytest.raises(typer.Exit):
                system_cmd.system_stats_execute(renderer, where="local")
        env = _envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["details"]["status"] == 500


# ---------------------------------------------------------------------------
# free
# ---------------------------------------------------------------------------


class TestFree:
    def _run_free(self, capsys, *, unload_models=True, free_memory=False, target=LOCAL_TARGET):
        renderer = _force_renderer()
        calls: list[dict] = []

        def _fake(req, timeout=None):
            calls.append({"url": req.full_url, "method": req.get_method(), "body": req.data})
            return _FakeResp(b"")

        with (
            patch("comfy_cli.command.system.resolve_target", return_value=target),
            patch.object(comfy_client._OPENER, "open", side_effect=_fake),
        ):
            system_cmd.free_execute(
                renderer,
                where="local" if not target.is_cloud else "cloud",
                unload_models=unload_models,
                free_memory=free_memory,
            )
        return calls, _envelope(capsys)

    def test_default_body(self, capsys):
        calls, env = self._run_free(capsys)
        assert calls[0]["method"] == "POST"
        assert calls[0]["url"].endswith("/free")
        assert json.loads(calls[0]["body"]) == {"unload_models": True, "free_memory": False}
        assert env["ok"] is True
        assert env["command"] == "free"
        assert env["data"]["requested"] == {"unload_models": True, "free_memory": False}
        assert "does not interrupt" in env["data"]["note"]

    @pytest.mark.parametrize(
        "unload_models,free_memory",
        [
            (False, False),
            (True, True),
            (False, True),
        ],
    )
    def test_flag_combinations(self, capsys, unload_models, free_memory):
        calls, env = self._run_free(capsys, unload_models=unload_models, free_memory=free_memory)
        assert json.loads(calls[0]["body"]) == {"unload_models": unload_models, "free_memory": free_memory}
        assert env["data"]["requested"] == {"unload_models": unload_models, "free_memory": free_memory}

    def test_local_unreachable_server_not_running(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.system.resolve_target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=urllib.error.URLError("Connection refused")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                system_cmd.free_execute(renderer, where="local")
        assert exc_info.value.exit_code == 1
        env = _envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"

    def test_cloud_unreachable_is_cloud_http_error(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.system.resolve_target", return_value=CLOUD_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=urllib.error.URLError("Connection refused")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                system_cmd.free_execute(renderer, where="cloud")
        assert exc_info.value.exit_code == 1
        env = _envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
