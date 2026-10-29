"""Read-only provenance: network-reported paths are not local filesystem proof."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from comfy_cli import local_server_identity as identity
from comfy_cli.command.stop_port import Listener


@pytest.fixture
def process(tmp_path):
    script = tmp_path / "main.py"
    script.write_text("")
    return Listener(
        123, [sys.executable, str(script), "--listen", "127.0.0.1"], str(tmp_path), "python", "127.0.0.1", 10.0
    )


def _stats(process):
    return {"system": {"argv": process.cmdline[1:], "comfyui_version": "test", "python": sys.executable}}


@pytest.mark.parametrize("change", ["missing", "pid", "start", "argv", "cwd", "no-start"])
def test_process_identity_must_survive_the_request(process, monkeypatch, change):
    after = {
        "missing": None,
        "pid": replace(process, pid=456),
        "start": replace(process, create_time=11.0),
        "argv": replace(process, cmdline=["ssh", "-L", "8188:remote:8188"]),
        "cwd": replace(process, cwd="/elsewhere"),
        "no-start": replace(process, create_time=None),
    }[change]
    monkeypatch.setattr(identity, "_same_filesystem_namespace", lambda pid: True)
    assert not identity._verified(_stats(process), process, after)


@pytest.mark.parametrize(
    "change", ["missing-argv", "wrong-argv", "wrong-script", "missing-version", "forwarder", "unknown-cwd", "container"]
)
def test_reported_paths_do_not_replace_os_evidence(process, monkeypatch, change):
    stats = _stats(process)
    if change == "missing-argv":
        del stats["system"]["argv"]
    elif change == "wrong-argv":
        stats["system"]["argv"] = [str(Path(process.cwd) / "main.py"), "--base-directory", "/private"]
    elif change == "wrong-script":
        stats["system"]["argv"][0] = "/remote/main.py"
    elif change == "missing-version":
        del stats["system"]["comfyui_version"]
    elif change == "forwarder":
        process = replace(process, cmdline=["ssh", "-L", "8188:remote:8188"])
    elif change == "unknown-cwd":
        process = replace(process, cmdline=[sys.executable, "main.py"], cwd=None)
        stats["system"]["argv"] = ["main.py"]
    monkeypatch.setattr(identity, "_same_filesystem_namespace", lambda pid: change != "container")
    assert not identity._verified(stats, process, process)


def test_relative_script_uses_verified_process_cwd(process, monkeypatch):
    process = replace(process, cmdline=[sys.executable, "main.py", *process.cmdline[2:]])
    monkeypatch.setattr(identity, "_same_filesystem_namespace", lambda pid: True)
    assert identity._verified(_stats(process), process, process)


@pytest.mark.parametrize("failure", [None, "mount", "root", "unreadable"])
def test_linux_requires_same_mount_namespace_and_process_root(monkeypatch, failure):
    monkeypatch.setattr(identity, "sys", SimpleNamespace(platform="linux"))
    seen = []

    def samefile(path, other):
        seen.append((str(path), str(other)))
        if failure == "unreadable":
            raise PermissionError("restricted /proc")
        return not (
            (failure == "mount" and "ns/mnt" in str(path)) or (failure == "root" and str(path).endswith("root"))
        )

    monkeypatch.setattr(Path, "samefile", samefile)
    assert identity._same_filesystem_namespace(123) == (failure is None)
    if failure is None:
        assert len(seen) == 2


def _conn(ip):
    return SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip=ip, port=8188))


@pytest.mark.parametrize(
    "bound,accepted", [("127.0.0.2", False), ("127.0.0.1", True), ("0.0.0.0", True), ("::", False)]
)
def test_listener_must_bind_the_actual_address(process, monkeypatch, bound, accepted):
    fake = SimpleNamespace(
        pid=123,
        info={"name": "python", "cmdline": process.cmdline},
        net_connections=lambda **kw: [_conn(bound)],
        cwd=lambda: process.cwd,
        create_time=lambda: 10.0,
    )
    fake.cmdline = lambda: process.cmdline
    fake.name = lambda: "python"
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [fake])
    monkeypatch.setattr(psutil, "Process", lambda pid: fake)
    monkeypatch.setattr(identity, "_is_local_address", lambda ip: True)
    assert (identity._listener("127.0.0.1", 8188) is not None) == accepted


def test_remote_address_never_borrows_a_local_wildcard_listener(monkeypatch):
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {})
    scan = Mock(side_effect=AssertionError("remote endpoint must not scan listeners"))
    monkeypatch.setattr(psutil, "process_iter", scan)
    assert identity._listener("192.0.2.77", 8188) is None
    scan.assert_not_called()


def test_reused_port_with_multiple_owners_is_ambiguous(process, monkeypatch):
    def fake(pid):
        return SimpleNamespace(
            pid=pid,
            info={"name": "python", "cmdline": process.cmdline},
            net_connections=lambda **kw: [_conn("127.0.0.1")],
            cwd=lambda: process.cwd,
            create_time=lambda: 10.0,
        )

    owners = {pid: fake(pid) for pid in (123, 456)}
    for owner in owners.values():
        owner.cmdline = lambda: process.cmdline
        owner.name = lambda: "python"
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: list(owners.values()))
    monkeypatch.setattr(psutil, "Process", lambda pid: owners[pid])
    assert identity._listener("127.0.0.1", 8188) is None


@pytest.fixture
def live_server(tmp_path, request):
    """Own a real Python main.py listener; terminate only this fixture's child."""
    script = tmp_path / "ComfyUI" / getattr(request, "param", "main.py")
    script.parent.mkdir()
    port_file = tmp_path / "port"
    script.write_text("""import json,sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path == "/redirect":
   self.send_response(302); self.send_header("Location", "/system_stats"); self.end_headers(); return
  body=json.dumps({"system":{"argv":sys.argv,"comfyui_version":"test","python":sys.executable},"host":self.headers.get("Host")}).encode()
  self.send_response(200); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
 def log_message(self,*args): pass
server=HTTPServer(("127.0.0.1",0),Handler)
open(sys.argv[1],"w").write(str(server.server_port))
server.serve_forever()
""")
    child = subprocess.Popen(
        [sys.executable, str(script), str(port_file)], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 10
        while not port_file.exists():
            assert child.poll() is None, child.stderr.read().decode()
            assert time.monotonic() < deadline, "fixture listener did not start"
            time.sleep(0.01)
        yield int(port_file.read_text()), script, child
    finally:
        child.terminate()
        child.communicate(timeout=5)


def test_actual_local_process_is_proven_and_host_header_preserved(live_server, monkeypatch):
    port, _, child = live_server
    # An HTTP proxy must not reroute the proof-bearing fetch.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    document = identity.fetch_json(f"http://localhost:{port}/system_stats", verify_filesystem=True)
    assert document.filesystem_verified
    assert document.data["host"] == f"localhost:{port}"
    assert child.poll() is None


def test_actual_redirect_does_not_supply_identity(live_server):
    port, _, _ = live_server
    with pytest.raises(urllib.error.HTTPError, match="redirect refused"):
        identity.fetch_json(f"http://127.0.0.1:{port}/redirect", verify_filesystem=True)


def test_ipv6_then_ipv4_connection_fallback_remains_supported(live_server, monkeypatch):
    port, _, _ = live_server
    original = socket.getaddrinfo

    def resolve(host, *args, **kwargs):
        if host == "localhost":
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
            ]
        return original(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    document = identity.fetch_json(f"http://localhost:{port}/system_stats", verify_filesystem=True)
    assert document.filesystem_verified
    assert document.data["host"] == f"localhost:{port}"


@pytest.mark.parametrize("live_server", ["forwarder.py"], indirect=True)
def test_real_python_forwarder_cannot_supply_filesystem_proof(live_server):
    port, _, child = live_server
    document = identity.fetch_json(f"http://127.0.0.1:{port}/system_stats", verify_filesystem=True)
    assert document.data["system"]["comfyui_version"] == "test"
    assert not document.filesystem_verified
    assert child.poll() is None


@pytest.mark.parametrize("prefix", [["worker.py"], ["-c", "pass"], ["-m", "http.server"]])
def test_later_main_py_token_is_not_the_executed_script(process, monkeypatch, prefix):
    decoy = process.cmdline[1:]
    process = replace(process, cmdline=[sys.executable, *prefix, *decoy])
    stats = {"system": {"argv": decoy, "comfyui_version": "test"}}
    monkeypatch.setattr(identity, "_same_filesystem_namespace", lambda pid: True)
    assert not identity._verified(stats, process, process)


@pytest.mark.parametrize("flags", [["-u"], ["-s", "-B"], ["-X", "dev"], ["-Wignore"], ["--"]])
def test_python_interpreter_flags_before_actual_main_py_are_supported(process, monkeypatch, flags):
    stats = _stats(process)
    process = replace(process, cmdline=[sys.executable, *flags, *process.cmdline[1:]])
    monkeypatch.setattr(identity, "_same_filesystem_namespace", lambda pid: True)
    assert identity._verified(stats, process, process)


def test_unreadable_process_table_preserves_server_metadata(live_server, monkeypatch):
    port, _, _ = live_server

    def denied(*args, **kwargs):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "process_iter", denied)
    document = identity.fetch_json(f"http://127.0.0.1:{port}/system_stats", verify_filesystem=True)
    assert document.data["system"]["comfyui_version"] == "test"
    assert not document.filesystem_verified


def test_real_listener_proof_selects_desktop_inventory(live_server, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from comfy_cli import desktop_install
    from comfy_cli.cmdline import app

    port, script, child = live_server
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "installations.json").write_text(json.dumps([{"id": "live", "installPath": str(script.parent.parent)}]))
    models = script.parent / "models" / "loras"
    models.mkdir(parents=True)
    (models / "private_v1.safetensors").write_bytes(b"actual-local")
    (script.parent / "input").mkdir()
    (script.parent / "output").mkdir()
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "type": "LoraLoaderModelOnly",
                        "mode": 0,
                        "widgets_values": ["private_v1.safetensors", 1.0],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(desktop_install, "app_data_dirs", lambda: (registry,))
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)
    result = CliRunner().invoke(
        app,
        ["build", "inventory", str(workflow), "--desktop", "--server", f"http://localhost:{port}"],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout.splitlines()[-1])["data"]
    assert data["desktop"]["install_id"] == "live"
    assert data["desktop"]["python"] == sys.executable
    assert data["models"][0]["local"][0]["bytes"] == 12
    assert data["input_roots"] == [str(script.parent / "input")]
    assert data["desktop"]["output_dir"] == str(script.parent / "output")
    assert data["local_server"]["server"] == f"http://localhost:{port}"
    assert child.poll() is None


def test_body_read_failure_does_not_fall_back_to_another_server(monkeypatch):
    class BrokenBody:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise TimeoutError("body stalled after HTTP response")

    class OtherServer(BrokenBody):
        def read(self, size):
            return b"{}"

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8188)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.2", 8188)),
        ],
    )
    opened = Mock(side_effect=[BrokenBody(), OtherServer()])
    monkeypatch.setattr(identity, "build_http_only_opener", lambda *args: SimpleNamespace(open=opened))
    with pytest.raises(TimeoutError, match="body stalled"):
        identity.fetch_json("http://localhost:8188/system_stats")
    opened.assert_called_once()


def test_listener_reads_fresh_create_time_instead_of_iterator_cache(process, monkeypatch):
    cached = SimpleNamespace(pid=123, create_time=lambda: 10.0)
    fresh = SimpleNamespace(
        pid=123,
        cmdline=lambda: process.cmdline,
        name=lambda: "python",
        net_connections=lambda **kw: [_conn("127.0.0.1")],
        cwd=lambda: process.cwd,
        create_time=lambda: 20.0,
    )
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [cached])
    constructor = Mock(return_value=fresh)
    monkeypatch.setattr(psutil, "Process", constructor)
    listener = identity._listener("127.0.0.1", 8188)
    assert listener.create_time == 20.0
    constructor.assert_called_once_with(123)
