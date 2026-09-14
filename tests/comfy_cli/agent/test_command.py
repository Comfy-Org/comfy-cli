"""``comfy agent permissions`` / ``comfy agent allow`` against a temp data dir."""

from __future__ import annotations

import http.server
import json
import os
import socket
import stat
import sys
import threading
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.agent import StateError, allow_host, allow_path, read_state, vet_host, vet_path
from comfy_cli.agent.command import app
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
    """A fake home with no credential store in it, and the data dir outside it."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    return h


def _envelope(result) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def _validate(data: dict) -> None:
    jsonschema.Draft202012Validator(SCHEMA).validate(data)


def test_allow_path_records_and_deduplicates(tmp_path: Path):
    root = tmp_path / "data"
    folder = tmp_path / "Pictures"
    folder.mkdir()
    p, added = allow_path(root, str(folder), "reference photos")
    assert added and p == folder
    _, again = allow_path(root, str(folder), "again")
    assert not again
    state = read_state(root)
    assert [e["path"] for e in state.paths] == [str(folder)]
    assert state.paths[0]["reason"] == "reference photos"
    assert state.paths[0]["approved_at"].endswith("Z")
    assert not state.running


def test_vet_path_refuses_what_the_agent_refuses(home: Path, tmp_path: Path):
    (home / ".ssh").mkdir()
    (home / "Documents").mkdir()
    with pytest.raises(ValueError):
        vet_path("relative/folder")
    with pytest.raises(ValueError):
        vet_path(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(home / ".ssh"))
    with pytest.raises(ValueError, match="contains"):
        vet_path(str(home))
    assert vet_path(str(home / "Documents")) == home / "Documents"


def test_vet_path_refuses_the_home_folder_before_any_store_exists(home: Path):
    """A later ssh-keygen or aws configure would put keys inside a folder the
    agent can already reach; the home folder is refused whether or not a
    store is there yet, as the agent's own store does."""
    assert not any(home.iterdir())
    with pytest.raises(ValueError, match="contains"):
        vet_path(str(home))
    with pytest.raises(ValueError, match="contains"):
        vet_path(str(home.parent))


@pytest.mark.parametrize(
    "rel",
    [
        ".config/gh",
        ".git-credentials",
        ".cache/huggingface/token",
        "Library/Application Support/Google/Chrome",
        ".mozilla/firefox",
        ".local/share/keyrings",
        "AppData/Local/Google/Chrome/User Data",
        "AppData/Roaming/Microsoft/Credentials",
    ],
)
def test_vet_path_denies_the_agents_whole_credential_list(home: Path, rel: str):
    """The list mirrors the agent's CredentialDenyDirs: tokens for services on
    the egress allow list and browser profiles on every platform, not only
    the dotfile stores. The store itself and a parent of it are both refused."""
    store = home / rel
    store.mkdir(parents=True)
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(store))
    with pytest.raises(ValueError, match="contains"):
        vet_path(str(store.parent))


def test_vet_path_folds_case_where_the_filesystem_does(home: Path, monkeypatch):
    """The default macOS volume is case-insensitive and resolve() does not
    correct case, so ~/.SSH opens ~/.ssh there. Comparing folded paths on
    darwin closes it; the test forces the platform so it runs everywhere."""
    (home / ".ssh").mkdir()
    (home / ".SSH").mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(home / ".SSH"))
    (home / "library" / "keychains").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(home / "library" / "keychains"))


def test_vet_path_refuses_a_dot_env_name_anywhere(home: Path, tmp_path: Path):
    """The agent denies .env / .env.* / .envrc by name wherever they live."""
    for name in (".env", ".ENV", ".env.local", ".envrc"):
        d = tmp_path / "proj" / name
        d.mkdir(parents=True, exist_ok=True)  # .env and .ENV are one folder on a case-folding volume
        with pytest.raises(ValueError, match=r"\.env"):
            vet_path(str(d))
    (tmp_path / "proj" / "src").mkdir()
    assert vet_path(str(tmp_path / "proj" / "src")) == (tmp_path / "proj" / "src").resolve()


def test_vet_host_strips_scheme_port_and_case():
    assert vet_host("HTTPS://Models.Example.com:443/x") == "models.example.com"
    assert vet_host("cdn.example.com.") == "cdn.example.com"
    assert vet_host("192.168.1.20:8188") == "192.168.1.20"
    assert vet_host("[2001:db8::1]:443") == "2001:db8::1"
    assert vet_host("hf.co") == "hf.co"
    assert vet_host("bücher.example") == "xn--bcher-kva.example"
    with pytest.raises(ValueError):
        vet_host("")


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ("*", "wildcard"),
        ("*.example.com", "wildcard"),
        ("com", "no domain"),
        ("localhost", "no domain"),
        ("a.com,b.com", "more than one host"),
        ("user:pw@evil.com", "credentials"),
        ("https://user:pw@evil.com/", "credentials"),
        ("127.0.0.1", "loopback"),
        ("::1", "loopback"),
        ("169.254.169.254", "link-local"),
        ("0.0.0.0", "loopback, link-local or reserved"),
        ("t.comfy.org", "telemetry"),
        ("api.mixpanel.com", "telemetry"),
        ("storage.googleapis.com", "object storage"),
        ("r2.cloudflarestorage.com", "object storage"),
        ("bad_host.example.com", "not a host name"),
        ("-x.example.com", "not a host name"),
        ("a b.example.com", "not a host name"),
    ],
)
def test_vet_host_refuses_what_the_proxy_could_not_use_or_would_never_open(raw: str, why: str):
    """The proxy matches an approved host exactly, refuses the known-refused
    hosts whatever the file says, and exempts loopback from its port rule, so
    each of these would record something useless or dangerous."""
    with pytest.raises(ValueError, match=why):
        vet_host(raw)


def test_allow_host_records(tmp_path: Path):
    root = tmp_path / "data"
    h, added = allow_host(root, "https://models.example.com/a", "a VAE")
    assert (h, added) == ("models.example.com", True)
    assert json.loads((root / "egress-allow.json").read_text())["hosts"][0]["host"] == "models.example.com"


def test_cli_allow_and_permissions_round_trip(tmp_path: Path):
    root = tmp_path / "data"
    folder = tmp_path / "refs"
    folder.mkdir()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["allow", "--path", str(folder), "--host", "models.example.com", "--reason", "test", "--data-dir", str(root)],
    )
    assert res.exit_code == 0, res.stdout
    env = _envelope(res)
    assert env["ok"] is True
    assert env["data"]["path"]["added"] is True
    assert env["data"]["host"]["host"] == "models.example.com"
    assert "no agent has run from this data dir" in env["data"]["warning"]
    _validate(env["data"])

    _pin_json_renderer()
    res = runner.invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.stdout
    env = _envelope(res)
    assert env["data"]["agent"]["running"] is False
    assert [e["path"] for e in env["data"]["paths"]] == [str(folder)]
    assert [e["host"] for e in env["data"]["hosts"]] == ["models.example.com"]
    _validate(env["data"])


def test_cli_allow_does_not_warn_when_an_agent_published_itself(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "agent.json").write_text(json.dumps({"port": 8190}))
    res = CliRunner().invoke(app, ["allow", "--host", "models.example.com", "--data-dir", str(root)])
    assert res.exit_code == 0, res.stdout
    data = _envelope(res)["data"]
    assert "warning" not in data
    _validate(data)


def test_cli_allow_refuses_a_missing_folder(tmp_path: Path):
    runner = CliRunner()
    res = runner.invoke(app, ["allow", "--path", str(tmp_path / "nope"), "--data-dir", str(tmp_path / "data")])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "agent_refused"


def test_cli_allow_needs_something(tmp_path: Path):
    res = CliRunner().invoke(app, ["allow", "--data-dir", str(tmp_path)])
    assert res.exit_code == 2
    assert _envelope(res)["error"]["code"] == "agent_bad_args"


def test_cli_allow_refuses_a_bad_host_without_persisting_the_path(tmp_path: Path):
    root = tmp_path / "data"
    folder = tmp_path / "refs"
    folder.mkdir()
    res = CliRunner().invoke(app, ["allow", "--path", str(folder), "--host", "   ", "--data-dir", str(root)])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "agent_refused"
    assert not (root / "permissions.json").exists()
    assert read_state(root).paths == []


def test_cli_allow_reads_every_state_file_before_the_first_write(tmp_path: Path):
    """A broken egress-allow.json must not leave --path approved behind a
    failure exit, and the failure is the file, not a refusal."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "egress-allow.json").write_text("{not json")
    folder = tmp_path / "refs"
    folder.mkdir()
    res = CliRunner().invoke(app, ["allow", "--path", str(folder), "--host", "ok.example.com", "--data-dir", str(root)])
    assert res.exit_code == 1
    err = _envelope(res)["error"]
    assert err["code"] == "agent_state_unreadable"
    assert "egress-allow.json" in err["message"]
    assert not (root / "permissions.json").exists()


@pytest.mark.parametrize("bad", ['{"paths": 1}', '{"paths": [1, 2]}', "[]", '"x"'])
def test_state_that_the_agent_would_not_load_is_reported_not_replaced(tmp_path: Path, bad: str):
    root = tmp_path / "data"
    root.mkdir()
    (root / "permissions.json").write_text(bad)
    with pytest.raises(StateError):
        read_state(root)
    folder = tmp_path / "refs"
    folder.mkdir()
    with pytest.raises(StateError):
        allow_path(root, str(folder), "x")
    assert (root / "permissions.json").read_text() == bad, "a file the agent cannot load is never overwritten"
    _pin_json_renderer()
    res = CliRunner().invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "agent_state_unreadable"


def test_allow_keeps_other_top_level_keys(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "permissions.json").write_text(json.dumps({"version": 2, "paths": [], "denied": ["x"]}))
    (root / "egress-allow.json").write_text(json.dumps({"version": 2, "hosts": []}))
    folder = tmp_path / "refs"
    folder.mkdir()
    allow_path(root, str(folder), "r")
    allow_host(root, "models.example.com", "r")
    perms = json.loads((root / "permissions.json").read_text())
    assert perms["version"] == 2 and perms["denied"] == ["x"] and len(perms["paths"]) == 1
    assert json.loads((root / "egress-allow.json").read_text())["version"] == 2


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a folder the user cannot write")
def test_cli_allow_reports_an_unwritable_data_dir_as_an_envelope(tmp_path: Path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        res = CliRunner().invoke(app, ["allow", "--host", "models.example.com", "--data-dir", str(ro / "data")])
        assert res.exit_code == 1
        assert res.exception is None or isinstance(res.exception, SystemExit)
        err = _envelope(res)["error"]
        assert err["code"] == "agent_state_unwritable"
    finally:
        ro.chmod(stat.S_IRWXU)


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
    _pin_renderer(OutputMode.PRETTY)
    res = CliRunner().invoke(app, ["allow", "--path", str(folder), "--reason", "[/]", "--data-dir", str(root)])
    assert res.exit_code == 0, res.output
    assert res.exception is None


def test_vet_path_follows_symlinks_into_credential_stores(home: Path, tmp_path: Path):
    (home / ".ssh").mkdir()
    link = tmp_path / "photos"
    os.symlink(home / ".ssh", link, target_is_directory=True)
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(link))
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    os.symlink(real, alias, target_is_directory=True)
    assert vet_path(str(alias)) == real.resolve()


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
    ],
)
def test_agent_schema_rejects_incomplete_payloads(payload):
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload)


def test_vet_path_refuses_the_agent_data_dir_its_parents_and_its_children(tmp_path: Path):
    """`--data-dir /tmp/x --path /tmp` would approve the agent's own state
    (and everything beside it). The data dir, anything under it, and any
    folder that contains it are refused; a sibling is fine."""
    root = tmp_path / "state" / "agent"
    root.mkdir(parents=True)
    child = root / "project"
    child.mkdir()
    sibling = tmp_path / "state" / "photos"
    sibling.mkdir()
    with pytest.raises(ValueError, match="agent"):
        vet_path(str(root), root=root)
    with pytest.raises(ValueError, match="agent"):
        vet_path(str(child), root=root)
    with pytest.raises(ValueError, match="agent"):
        vet_path(str(tmp_path / "state"), root=root)
    with pytest.raises(ValueError, match="agent"):
        vet_path(str(tmp_path), root=root)
    assert vet_path(str(sibling), root=root) == sibling.resolve()
    with pytest.raises(ValueError):
        allow_path(root, str(tmp_path), "x")
    assert not (root / "permissions.json").exists()


def test_cli_allow_refuses_the_data_dirs_parent(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    res = CliRunner().invoke(app, ["allow", "--path", str(tmp_path), "--data-dir", str(root)])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "agent_refused"
    assert not (root / "permissions.json").exists()
