"""``comfy stop --port <p>`` — finding, verifying, and stopping an untracked ComfyUI.

The whole point of this command is that it kills a process comfy-cli never
started, so these tests are weighted toward the *refusals*: an unidentifiable
listener, a listener that answers HTTP but disagrees with the process table,
and the dry run that must touch nothing.

psutil is faked at the two seams the module actually uses — ``process_iter``
for discovery and ``Process``/``wait_procs`` for the teardown — so the real
exception classes (``AccessDenied`` &c.) stay real.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jsonschema
import psutil
import pytest
import typer

from comfy_cli.command import stop_port
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas"


@pytest.fixture(autouse=True)
def reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _json_renderer() -> Renderer:
    r = Renderer(mode=OutputMode.JSON, command="stop")
    set_renderer(r)
    return r


def _envelope(captured_out: str) -> dict:
    lines = [ln for ln in captured_out.splitlines() if ln.strip()]
    return json.loads(lines[-1])


# --------------------------------------------------------------------------- #
# Fake psutil process table
# --------------------------------------------------------------------------- #


def _conn(port: int, status=psutil.CONN_LISTEN):
    return SimpleNamespace(status=status, laddr=SimpleNamespace(ip="127.0.0.1", port=port))


class FakeProc:
    def __init__(self, pid, cmdline, conns, *, denied=False, cwd="/workspace", name="python"):
        self.pid = pid
        self._cmdline = list(cmdline)
        self._conns = list(conns)
        self._denied = denied
        self._cwd = cwd
        self._name = name

    @property
    def info(self):
        return {"pid": self.pid, "name": self._name, "cmdline": list(self._cmdline)}

    def net_connections(self, kind="inet"):
        if self._denied:
            raise psutil.AccessDenied(self.pid)
        return list(self._conns)

    def cwd(self):
        if self._cwd is None:
            raise psutil.AccessDenied(self.pid)
        return self._cwd


COMFY_CMDLINE = ["/usr/bin/python3", "main.py", "--port", "8188"]


def _table(*procs):
    return patch.object(stop_port.psutil, "process_iter", return_value=list(procs))


class TestFindListener:
    def test_no_listener(self):
        with _table(FakeProc(10, ["nginx"], [_conn(80)])):
            assert stop_port.find_listener(8188) is None

    def test_finds_the_listener(self):
        with _table(FakeProc(10, ["nginx"], [_conn(80)]), FakeProc(42, COMFY_CMDLINE, [_conn(8188)])):
            found = stop_port.find_listener(8188)
        assert found is not None
        assert found.pid == 42
        assert found.cmdline == COMFY_CMDLINE
        assert found.cwd == "/workspace"

    def test_access_denied_process_does_not_hide_the_real_one(self):
        # On macOS every process owned by another user raises AccessDenied from
        # net_connections(); one of those must not abort the scan.
        with _table(FakeProc(9, ["root-thing"], [], denied=True), FakeProc(42, COMFY_CMDLINE, [_conn(8188)])):
            found = stop_port.find_listener(8188)
        assert found is not None and found.pid == 42

    def test_access_denied_only_reads_as_no_listener(self):
        with _table(FakeProc(9, ["root-thing"], [], denied=True)):
            assert stop_port.find_listener(8188) is None

    def test_established_connection_on_the_port_is_not_a_listener(self):
        # A *client* connected to 8188 has a conn with that remote port; only
        # LISTEN counts.
        with _table(FakeProc(11, COMFY_CMDLINE, [_conn(8188, status=psutil.CONN_ESTABLISHED)])):
            assert stop_port.find_listener(8188) is None

    def test_unreadable_cmdline_still_yields_the_pid(self):
        with _table(FakeProc(42, [], [_conn(8188)], cwd=None)):
            found = stop_port.find_listener(8188)
        assert found is not None
        assert found.pid == 42
        assert found.cmdline == []
        assert found.cwd is None


# --------------------------------------------------------------------------- #
# cmdline identity
# --------------------------------------------------------------------------- #


class TestLooksLikeComfyui:
    @pytest.mark.parametrize(
        "cmdline",
        [
            ["/usr/bin/python3", "main.py"],
            ["python", "main.py", "--port", "8188"],
            ["/opt/py/bin/python3.12", "/srv/ComfyUI/main.py"],
            # Windows interpreter. Spelled with forward slashes so the case is
            # portable: the module uses `os.path.basename`, which is
            # `ntpath.basename` on Windows and splits the backslash form there.
            ["C:/Python311/python.exe", "main.py"],
            # ComfyUI Desktop: bundled interpreter in the app's managed venv,
            # absolute path to the checkout's main.py.
            [
                "/Users/x/Library/Application Support/ComfyUI/.venv/bin/python",
                "/Users/x/Documents/ComfyUI/main.py",
                "--listen",
                "127.0.0.1",
            ],
        ],
    )
    def test_accepts_python_running_main_py(self, cmdline):
        assert stop_port.looks_like_comfyui(cmdline) is True

    @pytest.mark.parametrize(
        "cmdline",
        [
            [],
            ["/usr/bin/python3"],
            ["/usr/bin/node", "main.py"],
            ["nginx: master process /usr/sbin/nginx"],
            ["/usr/bin/python3", "-m", "http.server", "8188"],
            # `main.py` must be a *token* basename, not a substring.
            ["/usr/bin/python3", "notmain.py"],
            # argv[0] is not python, even though a main.py is on the line.
            ["/usr/bin/uv", "run", "main.py"],
        ],
    )
    def test_rejects_everything_else(self, cmdline):
        assert stop_port.looks_like_comfyui(cmdline) is False


# --------------------------------------------------------------------------- #
# HTTP probe + verdict
# --------------------------------------------------------------------------- #


def _stats(argv=None, version="0.3.99"):
    system = {"comfyui_version": version}
    if argv is not None:
        system["argv"] = argv
    return stop_port.Probe(answered=True, stats={"system": system})


class TestProbeSystemStats:
    def _urlopen(self, body: bytes):
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_parses_json(self):
        payload = {"system": {"comfyui_version": "0.3.1"}}
        with patch.object(
            stop_port.urllib.request, "urlopen", return_value=self._urlopen(json.dumps(payload).encode())
        ):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats == payload

    def test_connection_refused_is_not_answered(self):
        with patch.object(stop_port.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is False

    def test_timeout_is_not_answered(self):
        with patch.object(stop_port.urllib.request, "urlopen", side_effect=TimeoutError()):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is False

    def test_http_error_counts_as_answered(self):
        # A 404 means something IS serving this port — it is not wedged, it is
        # simply not ComfyUI.
        err = urllib.error.HTTPError("http://127.0.0.1:8188/system_stats", 404, "Not Found", {}, None)
        with patch.object(stop_port.urllib.request, "urlopen", side_effect=err):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats is None

    def test_non_json_body_counts_as_answered_without_stats(self):
        with patch.object(stop_port.urllib.request, "urlopen", return_value=self._urlopen(b"<html>hi</html>")):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats is None


class TestVerifyListener:
    def _listener(self, cmdline=None):
        return stop_port.Listener(pid=42, cmdline=list(cmdline or COMFY_CMDLINE))

    def test_cmdline_is_a_precondition_http_cannot_substitute(self):
        # A perfect /system_stats answer must NOT verify a listener whose
        # command line isn't ComfyUI: never kill on HTTP evidence alone.
        verdict = stop_port.verify_listener(
            self._listener(["/usr/sbin/nginx"]),
            8188,
            probe=_stats(argv=["main.py", "--port", "8188"]),
        )
        assert verdict.verified is False
        assert verdict.http == "not_probed"

    def test_agreeing_server_verifies(self):
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=_stats(argv=["main.py", "--port", "8188"]))
        assert verdict.verified is True
        assert verdict.http == "agreed"

    def test_agreeing_server_with_absolute_argv0_verifies(self):
        listener = self._listener(["/usr/bin/python3", "/srv/ComfyUI/main.py", "--listen"])
        verdict = stop_port.verify_listener(listener, 8188, probe=_stats(argv=["main.py", "--listen"]))
        assert verdict.verified is True

    def test_disagreeing_argv_refuses(self):
        # The reverse-proxy false positive: the port answers like ComfyUI, but
        # the process holding it was started with different arguments.
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=_stats(argv=["main.py", "--port", "9999"]))
        assert verdict.verified is False
        assert verdict.http == "disagreed"

    def test_missing_comfyui_version_refuses(self):
        probe = stop_port.Probe(answered=True, stats={"system": {"os": "posix"}})
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=probe)
        assert verdict.verified is False
        assert verdict.http == "disagreed"

    def test_answered_without_a_json_payload_refuses(self):
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=stop_port.Probe(answered=True))
        assert verdict.verified is False
        assert verdict.http == "disagreed"

    def test_unreachable_server_still_verifies_on_cmdline_evidence(self):
        # The wedged-server case this command exists for.
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=stop_port.Probe(answered=False))
        assert verdict.verified is True
        assert verdict.http == "unreachable"

    def test_server_without_argv_verifies_on_version_plus_cmdline(self):
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=_stats(argv=None))
        assert verdict.verified is True
        assert verdict.http == "agreed_no_argv"


# --------------------------------------------------------------------------- #
# Teardown
# --------------------------------------------------------------------------- #


class FakeKillProc:
    """A process that dies at a configurable point of the escalation."""

    def __init__(self, pid, *, children=(), survives_terminate=False, survives_kill=False, log=None):
        self.pid = pid
        self._children = list(children)
        self._survives_terminate = survives_terminate
        self._survives_kill = survives_kill
        self.log = log if log is not None else []
        self.terminated = False
        self.killed = False

    def children(self, recursive=False):
        return list(self._children)

    def terminate(self):
        self.terminated = True
        self.log.append(("terminate", self.pid))

    def kill(self):
        self.killed = True
        self.log.append(("kill", self.pid))

    @property
    def gone(self):
        if self.killed:
            return not self._survives_kill
        if self.terminated:
            return not self._survives_terminate
        return False


def _fake_wait_procs(procs, timeout=None):
    gone = [p for p in procs if p.gone]
    alive = [p for p in procs if not p.gone]
    return gone, alive


class TestKillProcessTree:
    def _run(self, parent):
        with (
            patch.object(stop_port.psutil, "Process", return_value=parent),
            patch.object(stop_port.psutil, "wait_procs", side_effect=_fake_wait_procs),
        ):
            return stop_port.kill_process_tree(parent.pid)

    def test_terminates_the_listener_itself_not_only_its_children(self):
        # The `utils.kill_all` trap: that helper kills only children, which
        # would leave a directly-identified listener running.
        log = []
        parent = FakeKillProc(42, log=log)
        ok, survivors = self._run(parent)
        assert ok is True and survivors == []
        assert parent.terminated is True
        assert log == [("terminate", 42)]

    def test_reaps_recursive_children_before_the_parent(self):
        log = []
        kids = [FakeKillProc(43, log=log), FakeKillProc(44, log=log)]
        parent = FakeKillProc(42, children=kids, log=log)
        ok, survivors = self._run(parent)
        assert ok is True and survivors == []
        assert [entry[1] for entry in log] == [43, 44, 42]
        assert all(k.terminated for k in kids)

    def test_escalates_to_kill_when_terminate_is_ignored(self):
        parent = FakeKillProc(42, survives_terminate=True)
        ok, survivors = self._run(parent)
        assert ok is True and survivors == []
        assert parent.terminated is True and parent.killed is True

    def test_reports_survivors_when_kill_does_not_land(self):
        parent = FakeKillProc(42, survives_terminate=True, survives_kill=True)
        ok, survivors = self._run(parent)
        assert ok is False and survivors == [42]

    def test_already_gone_process_is_a_success(self):
        with patch.object(stop_port.psutil, "Process", side_effect=psutil.NoSuchProcess(42)):
            ok, survivors = stop_port.kill_process_tree(42)
        assert ok is True and survivors == []


# --------------------------------------------------------------------------- #
# The command body: envelopes + the dry run
# --------------------------------------------------------------------------- #


class TestStopPortExecute:
    def _config(self, bg_info=None):
        cfg = MagicMock()
        cfg.background = bg_info
        return cfg

    def test_no_listener_emits_port_not_listening(self, capsys):
        renderer = _json_renderer()
        with patch.object(stop_port, "find_listener", return_value=None):
            with pytest.raises(typer.Exit) as exc:
                stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert exc.value.exit_code == 1

        env = _envelope(capsys.readouterr().out)
        assert env["ok"] is False
        assert env["error"]["code"] == "port_not_listening"
        assert env["error"]["details"]["port"] == 8188

    def test_unverified_listener_is_refused_and_reported(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=99, cmdline=["/usr/sbin/nginx", "-g", "daemon off;"], cwd="/etc")
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "kill_process_tree") as killer,
        ):
            with pytest.raises(typer.Exit) as exc:
                stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert exc.value.exit_code == 1
        killer.assert_not_called()

        env = _envelope(capsys.readouterr().out)
        assert env["error"]["code"] == "unverified_process"
        details = env["error"]["details"]
        assert details["pid"] == 99
        assert details["cmdline"] == ["/usr/sbin/nginx", "-g", "daemon off;"]
        assert details["reason"]

    def test_dry_run_reports_the_victim_and_kills_nothing(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd="/srv/ComfyUI")
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=_stats(argv=["main.py", "--port", "8188"])),
            patch.object(stop_port, "kill_process_tree") as killer,
            patch.object(stop_port, "ConfigManager") as cfg,
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=True)

        killer.assert_not_called()
        cfg.assert_not_called()

        env = _envelope(capsys.readouterr().out)
        assert env["ok"] is True
        assert env["changed"] is False
        assert env["data"] == {
            "stopped": False,
            "dry_run": True,
            "verified": True,
            "untracked": True,
            "pid": 42,
            "port": 8188,
            "cmdline": COMFY_CMDLINE,
            "cwd": "/srv/ComfyUI",
        }

    def test_dry_run_omits_cwd_when_unreadable(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd=None)
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree"),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=True)
        assert "cwd" not in _envelope(capsys.readouterr().out)["data"]

    def test_success_envelope(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=None)
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree", return_value=(True, [])),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)

        env = _envelope(capsys.readouterr().out)
        assert env["ok"] is True
        assert env["changed"] is True
        assert env["data"] == {"stopped": True, "pid": 42, "port": 8188, "untracked": True}
        cfg.remove_background.assert_not_called()

    def test_failed_kill_emits_stop_failed(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree", return_value=(False, [42])),
            patch.object(stop_port, "ConfigManager") as cfg,
        ):
            with pytest.raises(typer.Exit) as exc:
                stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert exc.value.exit_code == 1
        # A failed stop must not drop the background record.
        cfg.assert_not_called()

        env = _envelope(capsys.readouterr().out)
        assert env["error"]["code"] == "stop_failed"
        assert env["error"]["details"]["survivors"] == [42]

    def test_matching_background_pid_is_forgotten(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 42))
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree", return_value=(True, [])),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        cfg.remove_background.assert_called_once()
        capsys.readouterr()

    def test_different_background_pid_on_the_same_port_is_kept(self, capsys):
        # Port equality is not identity: dropping the record here would orphan
        # a still-running server this CLI does own.
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 777))
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree", return_value=(True, [])),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        cfg.remove_background.assert_not_called()
        capsys.readouterr()

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_payloads_validate_against_the_shipped_schema(self, capsys, dry_run):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd="/srv/ComfyUI")
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree", return_value=(True, [])),
            patch.object(stop_port, "ConfigManager", return_value=self._config()),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=dry_run)

        schema = json.loads((SCHEMAS_DIR / "stop.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(_envelope(capsys.readouterr().out)["data"])

    def test_pretty_mode_dry_run_escapes_foreign_markup(self, capsys):
        # The cmdline is another process's argv; Rich would otherwise render
        # markup out of it (or crash on an unbalanced tag).
        set_renderer(Renderer(mode=OutputMode.PRETTY, command="stop"))
        renderer = Renderer(mode=OutputMode.PRETTY, command="stop")
        listener = stop_port.Listener(pid=42, cmdline=["/usr/bin/python3", "main.py", "--x", "[/]"])
        with (
            patch.object(stop_port, "find_listener", return_value=listener),
            patch.object(stop_port, "probe_system_stats", return_value=stop_port.Probe(answered=False)),
            patch.object(stop_port, "kill_process_tree") as killer,
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=True)
        out = capsys.readouterr().out
        killer.assert_not_called()
        assert "Would stop ComfyUI on port 8188" in out
        # Pretty mode emits no envelope.
        assert "envelope" not in out
