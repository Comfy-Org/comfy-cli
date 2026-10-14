"""``comfy agent permissions`` against a temp data dir; approving lives in the agent panel, not here."""

from __future__ import annotations

import http.server
import json
import socket
import threading
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.agent import StateError, read_state
from comfy_cli.agent.command import NOTE, app
from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

SCHEMA = json.loads((Path(__file__).parents[3] / "comfy_cli" / "schemas" / "agent.json").read_text())


def _pin_renderer(mode: OutputMode = OutputMode.JSON):
    """A renderer emits one envelope per process; pin a fresh one per invoke."""
    r = Renderer.resolve(
        is_stdout_tty=mode is OutputMode.PRETTY,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=mode is OutputMode.JSON,
    )
    r.mode = mode
    set_renderer(r)


def _pin_json_renderer():
    _pin_renderer(OutputMode.JSON)


@pytest.fixture(autouse=True)
def _renderer_lifecycle():
    _pin_json_renderer()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """A fake home, so the default data dir (``~/.comfy-agent``) is a temp dir."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    return h


def _envelope(result) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def _validate(data: dict) -> None:
    jsonschema.Draft202012Validator(SCHEMA).validate(data)


@pytest.mark.parametrize("bad", ['{"paths": 1}', '{"paths": [1, 2]}', "[]", '"x"'])
def test_state_that_the_agent_would_not_load_is_reported_not_replaced(tmp_path: Path, bad: str):
    root = tmp_path / "data"
    root.mkdir()
    (root / "permissions.json").write_text(bad)
    with pytest.raises(StateError):
        read_state(root)
    _pin_json_renderer()
    res = CliRunner().invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "agent_state_unreadable"
    assert (root / "permissions.json").read_text() == bad, "a file the agent cannot load is never overwritten"


def test_pretty_output_escapes_rich_markup_from_disk(tmp_path: Path):
    """A reason of ``notes [/] here`` came from a file; printed raw it raises MarkupError."""
    root = tmp_path / "data"
    root.mkdir()
    folder = tmp_path / "[bold]draft"
    folder.mkdir()
    (root / "permissions.json").write_text(
        json.dumps({"paths": [{"path": str(folder), "reason": "notes [/] here", "approved_at": "now"}]})
    )
    (root / "egress-allow.json").write_text(
        json.dumps({"hosts": [{"host": "[link=http://evil]x[/link]", "reason": "[red]r", "approved_at": "now"}]})
    )
    _pin_renderer(OutputMode.PRETTY)
    res = CliRunner().invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.output
    assert res.exception is None
    assert "notes [/] here" in res.output
    assert "[bold]draft" in res.output


def test_read_state_ignores_a_boolean_or_out_of_range_port(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    for bad in (True, 0, 70000, "8190"):
        (root / "agent.json").write_text(json.dumps({"port": bad}))
        assert read_state(root).port is None
    (root / "agent.json").write_text(json.dumps({"port": 8190}))
    assert read_state(root).port == 8190


class _Resp:
    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_sandbox_status_treats_non_object_health_as_unavailable(monkeypatch):
    from comfy_cli.agent import command

    monkeypatch.setattr(command, "_open_health", lambda url: _Resp(b"[1, 2]"))
    assert command._sandbox_status(8190) is None
    monkeypatch.setattr(command, "_open_health", lambda url: _Resp(b'{"sandbox": {"mode": "seatbelt"}}'))
    assert command._sandbox_status(8190) == {"sandbox": {"mode": "seatbelt"}}


def test_sandbox_status_normalizes_malformed_nested_health(monkeypatch):
    from comfy_cli.agent import command

    monkeypatch.setattr(command, "_open_health", lambda url: _Resp(b'{"sandbox": [], "cli": {"workspace": true}}'))
    assert command._sandbox_status(8086) == {"sandbox": None}
    monkeypatch.setattr(
        command,
        "_open_health",
        lambda url: _Resp(b'{"sandbox": {"shell": "enabled"}, "cli": {"workspace": "/opt/ComfyUI"}}'),
    )
    assert command._sandbox_status(8086) == {"sandbox": {"shell": "enabled"}, "comfy_path": "/opt/ComfyUI"}


def test_sandbox_status_survives_a_stale_port_that_is_not_http():
    """An SSH banner on the port raises http.client.BadStatusLine, which is not
    an OSError; it must read as "no agent", not a traceback."""
    from comfy_cli.agent import command

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        try:
            conn.recv(1024)
            conn.sendall(b"SSH-2.0-OpenSSH_9.0\r\n")
        finally:
            conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        assert command._sandbox_status(port) is None
    finally:
        t.join(timeout=5)
        srv.close()


def test_sandbox_status_ignores_http_proxy_for_loopback(monkeypatch):
    """A shell behind the agent's egress proxy exports HTTP_PROXY; the global
    urlopen would route the loopback probe through it and show a live agent
    as absent."""
    from comfy_cli.agent import command

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib name
            body = json.dumps({"sandbox": {"mode": "test"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Health)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        assert command._sandbox_status(port) == {"sandbox": {"mode": "test"}}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data_dir": "/x", "note": "n"},
        {"data_dir": "/x", "path": {"path": "/p"}, "note": "n"},
        {"data_dir": "/x", "agent": {"running": False}, "sandbox": None, "comfy_path": None, "paths": [], "hosts": []},
        {
            "data_dir": "/x",
            "agent": {"running": False, "port": None},
            "sandbox": None,
            "comfy_path": None,
            "pending": [],
            "paths": [],
            "hosts": [],
        },
    ],
)
def test_agent_schema_rejects_incomplete_payloads(payload):
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload)


# --- pending requests: the agent records, the user answers in the agent panel ---


def _write_pending(root: Path, *requests: dict, paths: list | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "permissions.json").write_text(json.dumps({"paths": paths or [], "pending": list(requests)}))


def _req(id_: str, kind: str, target: str, reason: str = "the agent asked") -> dict:
    return {"id": id_, "kind": kind, "target": target, "reason": reason, "requested_at": "2026-09-14T10:00:00Z"}


def test_read_state_returns_pending_and_skips_malformed_entries(tmp_path: Path):
    root = tmp_path / "data"
    good = _req("0123456789abcdef", "host", "models.example.com")
    _write_pending(
        root,
        good,
        {"id": "no-kind", "target": "/x"},
        {"id": 7, "kind": "path", "target": "/x"},
        {"id": "fedcba9876543210", "kind": "shell", "target": "x"},
        "not an object",
    )
    state = read_state(root)
    assert state.pending == [good]
    assert state.paths == []
    (root / "permissions.json").write_text(json.dumps({"paths": []}))
    assert read_state(root).pending == []
    (root / "permissions.json").write_text(json.dumps({"paths": [], "pending": "x"}))
    assert read_state(root).pending == []


def test_cli_permissions_lists_pending_and_points_at_the_agent_panel(tmp_path: Path):
    root = tmp_path / "data"
    req = _req("0123456789abcdef", "host", "models.example.com", "a VAE [/]")
    _write_pending(root, req, paths=[{"path": "/refs", "reason": "photos", "approved_at": "now"}])
    (root / "egress-allow.json").write_text(json.dumps({"hosts": [{"host": "cdn.example.com", "reason": "r"}]}))
    runner = CliRunner()
    res = runner.invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.stdout
    data = _envelope(res)["data"]
    assert data["pending"] == [req]
    assert [e["path"] for e in data["paths"]] == ["/refs"]
    assert [e["host"] for e in data["hosts"]] == ["cdn.example.com"]
    assert data["note"] == NOTE
    assert "agent panel" in data["note"]
    # No command that approves or grants is handed out: the model can run this
    # CLI, so a command here would let it answer its own request.
    assert "grant" not in data
    assert "agent allow" not in res.stdout and "agent deny" not in res.stdout
    _validate(data)

    _pin_renderer(OutputMode.PRETTY)
    res = runner.invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.output
    assert res.exception is None
    out = res.output
    # The request's id, target and reason each appear on the pending line.
    # (A ``find`` rather than ``in``: CodeQL reads ``"<host>" in s`` as URL
    # sanitization and flags it, and this is an assertion, not a check.)
    for needle in (req["id"], req["target"], req["reason"]):
        assert out.find(needle) >= 0, f"{needle!r} missing from:\n{out}"
    assert out.index(req["id"]) < out.index("hosts the user approved")
    assert "ago" in out
    flat = " ".join(out.split())
    assert NOTE in flat
    assert "agent allow" not in flat and "agent deny" not in flat


@pytest.mark.parametrize("from_env", [False, True])
def test_cli_permissions_reads_the_agents_default_data_dir(home: Path, tmp_path: Path, monkeypatch, from_env: bool):
    # A bare `comfy agent permissions` is what the skill tells users to run: it
    # must read the dir the agent uses (AGENT_DATA_DIR, else ~/.comfy-agent),
    # or it shows nothing waiting while a request is.
    root = tmp_path / "agent-data" if from_env else home / ".comfy-agent"
    if from_env:
        monkeypatch.setenv("AGENT_DATA_DIR", str(root))
    else:
        monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    req = _req("0123456789abcdef", "path", "/refs")
    _write_pending(root, req)
    res = CliRunner().invoke(app, ["permissions"])
    assert res.exit_code == 0, res.stdout
    data = _envelope(res)["data"]
    assert data["data_dir"] == str(root)
    assert data["pending"] == [req]


def test_cli_permissions_prints_no_panel_note_with_nothing_pending(tmp_path: Path):
    root = tmp_path / "data"
    _write_pending(root)
    _pin_renderer(OutputMode.PRETTY)
    res = CliRunner().invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.output
    assert NOTE not in " ".join(res.output.split())


@pytest.mark.parametrize(
    "argv",
    [
        ["allow", "--approve", "0123456789abcdef"],
        ["allow", "--path", "/tmp", "--reason", "x"],
        ["allow", "--host", "models.example.com"],
        ["deny", "0123456789abcdef"],
    ],
)
def test_allow_and_deny_are_not_commands(tmp_path: Path, argv: list[str]):
    """A script the agent's model runs could otherwise approve its own request."""
    root = tmp_path / "data"
    _write_pending(root, _req("0123456789abcdef", "host", "models.example.com"))
    before = sorted((p.name, p.read_text()) for p in root.iterdir())
    res = CliRunner().invoke(app, [*argv, "--data-dir", str(root)])
    assert res.exit_code != 0
    assert "No such command" in res.output
    assert sorted((p.name, p.read_text()) for p in root.iterdir()) == before
