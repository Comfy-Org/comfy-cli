"""A typed download with no `--relative-path` lands where the running ComfyUI scans.

Comfy Desktop points ComfyUI at a shared library through
`--extra-model-paths-config`, so `<workspace>/models/<kind>` is a folder the server never
lists. The server's own `/internal/folder_paths` puts the default folder first.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer.testing

from comfy_cli.command.models import models
from comfy_cli.command.models.models import _default_destination, app

runner = typer.testing.CliRunner()


def _server(body: bytes, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/internal/folder_paths"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def local_url(monkeypatch):
    servers = []

    def start(body: bytes, status: int = 200) -> None:
        server = _server(body, status)
        servers.append(server)
        monkeypatch.setenv("COMFY_LOCAL_URL", f"http://127.0.0.1:{server.server_port}")

    monkeypatch.delenv("COMFY_WHERE", raising=False)
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_uses_the_servers_first_folder_for_the_kind(local_url, tmp_path):
    library = str(tmp_path / "ComfyUI-Shared" / "models" / "loras")
    local_url(json.dumps({"loras": [library, "/install/models/loras"], "checkpoints": ["/x"]}).encode())
    assert _default_destination("loras", "SDXL") == os.path.join(library, "SDXL")


@pytest.mark.parametrize(
    "body,status",
    [
        (b'{"loras": ["relative/loras"]}', 200),  # never a guess at the server's cwd
        (b'{"loras": []}', 200),
        (b'{"checkpoints": ["/x"]}', 200),  # kind the server does not know
        (b'{"loras": "/x"}', 200),
        (b"[]", 200),
        (b"not json", 200),
        (b"{}", 404),  # an older server has no such route
    ],
)
def test_keeps_the_workspace_folder_when_the_server_gives_no_answer(local_url, body, status):
    local_url(body, status)
    assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")


def test_keeps_the_workspace_folder_when_no_server_is_running(monkeypatch):
    monkeypatch.delenv("COMFY_WHERE", raising=False)
    monkeypatch.setenv("COMFY_LOCAL_URL", "http://127.0.0.1:9")
    assert _default_destination("loras", "") == os.path.join("models", "loras", "")


def test_never_asks_a_server_on_another_machine(monkeypatch):
    monkeypatch.delenv("COMFY_WHERE", raising=False)
    monkeypatch.setenv("COMFY_LOCAL_URL", "http://192.0.2.1:8188")
    with patch("comfy_cli.http.request_json") as request:
        assert _default_destination("loras", "") == os.path.join("models", "loras", "")
    request.assert_not_called()


def test_a_model_kind_nobody_entered_asks_nothing(local_url):
    local_url(b'{"": ["/x"]}')
    assert _default_destination("", "") == os.path.join("models", "", "")


def test_a_typed_download_lands_in_the_library_and_says_so(local_url, tmp_path):
    library = tmp_path / "ComfyUI-Shared" / "models" / "loras"
    local_url(json.dumps({"loras": [str(library)]}).encode())
    fake_download = Mock()
    cfg = Mock()
    cfg.get_or_override.return_value = None
    cfg.get.return_value = None
    with (
        patch.object(models, "config_manager", cfg),
        patch.object(models, "get_workspace", return_value=tmp_path / "install"),
        patch.object(models, "check_civitai_url", return_value=(True, False, 1, 2)),
        patch.object(models, "check_huggingface_url", return_value=(False, None, None, None, None)),
        patch.object(
            models, "_resolve_civitai_source", return_value=("m.safetensors", "https://civitai.com/x", "lora", "SDXL")
        ),
        patch.object(models, "download_file", fake_download),
    ):
        result = runner.invoke(
            app, ["download", "--url", "https://civitai.com/models/1", "--filename", "m.safetensors"]
        )
    assert result.exit_code == 0, result.output
    dest = Path(fake_download.call_args.args[1])
    assert dest == library / "SDXL" / "m.safetensors"
    assert "ComfyUI-Shared" in result.output.replace("\n", "")
