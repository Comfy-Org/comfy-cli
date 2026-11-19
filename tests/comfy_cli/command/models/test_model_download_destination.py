"""A typed download with no `--relative-path` lands where the running ComfyUI scans.

Comfy Desktop points ComfyUI at a shared library through
`--extra-model-paths-config`, so `<workspace>/models/<kind>` is a folder the server never
lists. The server's own `/internal/folder_paths` puts the default folder first.

The folder it names is checked before it is used: a loopback answer is not proof of a
shared filesystem (Docker, `ssh -L`, WSL), and the responder need not be ComfyUI at all.
Every check falls back to the workspace path rather than failing the download.
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
from comfy_cli.workspace_manager import WorkspaceType

runner = typer.testing.CliRunner()


def _server(body: bytes, status: int = 200):
    """A loopback HTTP server answering `/internal/folder_paths` with `body`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Answer the one route under test, asserting nothing else is asked for."""
            assert self.path == "/internal/folder_paths"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            """Keep the server's request log out of the test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def local_url(monkeypatch):
    """Point the CLI at a throwaway loopback server started by the returned callable."""
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


@pytest.fixture
def library(tmp_path):
    """A shared-library folder that exists on disk, as a real ComfyUI's would."""
    path = tmp_path / "ComfyUI-Shared" / "models" / "loras"
    path.mkdir(parents=True)
    return path


def test_uses_the_servers_first_folder_for_the_kind(local_url, library):
    local_url(json.dumps({"loras": [str(library), "/install/models/loras"], "checkpoints": ["/x"]}).encode())
    assert _default_destination("loras", "SDXL") == os.path.join(str(library), "SDXL")


@pytest.mark.parametrize(
    "body,status",
    [
        (b'{"loras": ["relative/loras"]}', 200),  # never a guess at the server's cwd
        (b'{"loras": []}', 200),
        (b'{"checkpoints": ["/x"]}', 200),  # kind the server does not know
        (b'{"loras": "/x"}', 200),
        (b'{"loras": [""]}', 200),
        (b'{"loras": [null]}', 200),
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


class TestTheFolderTheServerNamesIsChecked:
    """A loopback reply is server-chosen text, not a fact about this filesystem."""

    def test_a_folder_that_does_not_exist_here_keeps_the_workspace_path(self, local_url, tmp_path):
        """ComfyUI in Docker, WSL, or behind `ssh -L` answers on 127.0.0.1 with its own
        paths. Taking one would fail on `mkdir`, or as root write where nothing scans."""
        local_url(json.dumps({"loras": [str(tmp_path / "root" / "ComfyUI" / "models" / "loras")]}).encode())
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")

    def test_a_file_where_a_folder_should_be_keeps_the_workspace_path(self, local_url, tmp_path):
        not_a_dir = tmp_path / "loras"
        not_a_dir.write_text("")
        local_url(json.dumps({"loras": [str(not_a_dir)]}).encode())
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")

    @pytest.mark.parametrize("folder", ["\\\\attacker\\share\\loras", "//attacker/share/loras", "\\\\?\\C:\\loras"])
    def test_a_unc_or_device_path_is_refused_without_touching_the_filesystem(self, local_url, folder, monkeypatch):
        """Stat-ing a UNC path on Windows is itself the attack: it opens an outbound
        SMB connection that can leak the user's NTLM credentials."""
        touched = []
        real_isdir = os.path.isdir
        monkeypatch.setattr(os.path, "isdir", lambda p: touched.append(p) or real_isdir(p))
        local_url(json.dumps({"loras": [folder]}).encode())
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")
        assert touched == []

    def test_a_folder_is_normalized_before_use(self, local_url, library):
        local_url(json.dumps({"loras": [str(library / ".." / "loras")]}).encode())
        assert _default_destination("loras", "SDXL") == os.path.join(str(library), "SDXL")


class TestTheLookupIsLocalByConstruction:
    """It is a filesystem question, so `where` resolution must not reroute it."""

    def test_a_configured_cloud_target_does_not_stop_the_local_lookup(self, local_url, library, monkeypatch):
        """`resolve_target(where=None)` would pick cloud here, never ask the running
        server, and refresh a cloud session for a Target it then discards."""
        monkeypatch.setenv("COMFY_WHERE", "cloud")
        local_url(json.dumps({"loras": [str(library)]}).encode())
        with patch("comfy_cli.credentials.resolve_cloud_credential") as credential:
            assert _default_destination("loras", "SDXL") == os.path.join(str(library), "SDXL")
        credential.assert_not_called()

    def test_an_explicit_workspace_is_never_overridden_by_a_running_server(self, local_url, library, monkeypatch):
        """Nothing ties the probed server to the selected workspace, so `--workspace`
        has to win: otherwise another install's server silently redirects the file."""
        local_url(json.dumps({"loras": [str(library)]}).encode())
        monkeypatch.setattr(models.workspace_manager, "workspace_type", WorkspaceType.SPECIFIED)
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")


def test_a_stalling_server_cannot_hold_up_the_download(local_url, library, monkeypatch):
    """The socket timeout is per-operation, so a drip-feeding listener could stall an
    optional lookup indefinitely. The wall-clock deadline is what bounds it."""
    monkeypatch.setattr(models, "_FOLDER_PATHS_DEADLINE_S", 0.05)
    started = threading.Event()
    release = threading.Event()

    def never_answers(*_args, **_kwargs):
        started.set()
        release.wait(30)
        return 200, {"loras": [str(library)]}

    monkeypatch.setattr("comfy_cli.http.request_json", never_answers)
    local_url(b"{}")
    try:
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")
        assert started.wait(5), "the probe never ran"
    finally:
        release.set()


def test_a_hung_filesystem_cannot_hold_up_the_download_either(local_url, library, monkeypatch):
    """The folder check is a filesystem call on a server-chosen path, so it has to run
    inside the deadline too — `os.path.isdir` on an unresponsive hard mount blocks
    uninterruptibly, and on the caller thread it would outlive the bound entirely."""
    monkeypatch.setattr(models, "_FOLDER_PATHS_DEADLINE_S", 0.05)
    entered = threading.Event()
    release = threading.Event()
    real_isdir = os.path.isdir

    def hangs(path):
        if str(path) == str(library):
            entered.set()
            release.wait(30)
        return real_isdir(path)

    monkeypatch.setattr(os.path, "isdir", hangs)
    local_url(json.dumps({"loras": [str(library)]}).encode())
    try:
        assert _default_destination("loras", "SDXL") == os.path.join("models", "loras", "SDXL")
        assert entered.wait(5), "the folder check never ran"
    finally:
        release.set()


def test_a_typed_download_lands_in_the_library_and_says_so(local_url, library, tmp_path):
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


def test_a_server_supplied_folder_cannot_repaint_the_terminal(local_url, tmp_path):
    """The destination is printed through rich markup, so server text needs the
    control-character strip as well as the markup escape."""
    hostile = tmp_path / "\x1b[2Jloras[red]"
    hostile.mkdir()
    local_url(json.dumps({"loras": [str(hostile)]}).encode())
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
        patch.object(models, "download_file", Mock()),
    ):
        result = runner.invoke(
            app, ["download", "--url", "https://civitai.com/models/1", "--filename", "m.safetensors"]
        )
    assert result.exit_code == 0, result.output
    assert "\x1b[2J" not in result.output
