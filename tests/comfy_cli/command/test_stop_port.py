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

import http.client
import io
import json
import urllib.error
import urllib.request
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


@pytest.fixture
def comfy_tree(tmp_path) -> Path:
    """A directory shaped like a real ComfyUI checkout."""
    root = tmp_path / "ComfyUI"
    root.mkdir()
    for name in ("main.py", "nodes.py", "execution.py", "folder_paths.py", "server.py"):
        (root / name).write_text("")
    (root / "comfy").mkdir()
    (root / "comfy_extras").mkdir()
    return root


# --------------------------------------------------------------------------- #
# Fake psutil process table
# --------------------------------------------------------------------------- #


def _conn(port: int, status=psutil.CONN_LISTEN, ip="127.0.0.1"):
    return SimpleNamespace(status=status, laddr=SimpleNamespace(ip=ip, port=port))


class FakeProc:
    def __init__(self, pid, cmdline, conns, *, denied=False, cwd="/workspace", name="python", create_time=1000.0):
        self.pid = pid
        self._cmdline = list(cmdline)
        self._conns = list(conns)
        self._denied = denied
        self._cwd = cwd
        self._name = name
        self._create_time = create_time

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

    def create_time(self):
        return self._create_time


class LegacyFakeProc:
    """psutil < 6, where the accessor is still spelled ``connections()``.

    Deliberately *not* a FakeProc subclass: the whole point is that
    ``net_connections`` does not exist on it.
    """

    def __init__(self, pid, cmdline, conns, *, cwd="/workspace", name="python", create_time=1000.0):
        self.pid = pid
        self._cmdline = list(cmdline)
        self._conns = list(conns)
        self._cwd = cwd
        self._name = name
        self._create_time = create_time

    @property
    def info(self):
        return {"pid": self.pid, "name": self._name, "cmdline": list(self._cmdline)}

    def connections(self, kind="inet"):
        return list(self._conns)

    def cwd(self):
        return self._cwd

    def create_time(self):
        return self._create_time


COMFY_CMDLINE = ["/usr/bin/python3", "main.py", "--port", "8188"]


def _table(*procs):
    return patch.object(stop_port.psutil, "process_iter", return_value=list(procs))


# A `/system_stats` answer that omits `argv`: it corroborates any python
# `main.py` command line without having to match its arguments, so the command
# tests below can vary the cmdline freely.
_AGREEING = stop_port.Probe(answered=True, stats={"system": {"comfyui_version": "0.3.99"}})

_TORN_DOWN = stop_port.Teardown(ok=True, survivors=[])


def _finds(listener):
    """``find_listener``: the target, then nothing — the post-kill port recheck."""
    return patch.object(stop_port, "find_listener", side_effect=[listener, None])


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

    def test_carries_the_bound_address_and_create_time(self):
        # Both are load-bearing downstream: the probe must reach the address the
        # server actually bound, and create_time pins identity across the probe.
        with _table(FakeProc(42, COMFY_CMDLINE, [_conn(8188, ip="::1")], create_time=1234.5)):
            found = stop_port.find_listener(8188)
        assert found is not None
        assert found.laddr_ip == "::1"
        assert found.create_time == 1234.5

    def test_psutil_5_connections_accessor_is_still_understood(self):
        # `net_connections` is the psutil >= 6 spelling. On 5.x the missing
        # attribute would otherwise escape as a raw AttributeError traceback
        # instead of a structured envelope.
        with _table(LegacyFakeProc(42, COMFY_CMDLINE, [_conn(8188)])):
            found = stop_port.find_listener(8188)
        assert found is not None and found.pid == 42


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


class FakeResponse:
    """Enough of an HTTPResponse for the capped reader: chunked, closeable."""

    def __init__(self, body: bytes, *, chunk: int | None = None):
        self._buf = io.BytesIO(body)
        self._chunk = chunk

    def read(self, size=-1):
        if self._chunk is not None and (size is None or size < 0 or size > self._chunk):
            size = self._chunk
        return self._buf.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestProbeSystemStats:
    def test_parses_json(self):
        payload = {"system": {"comfyui_version": "0.3.1"}}
        with patch.object(stop_port, "_open_url", return_value=FakeResponse(json.dumps(payload).encode())):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats == payload

    def test_probes_the_address_the_listener_bound(self):
        seen = []

        def _fake_open(url, timeout):
            seen.append(url)
            return FakeResponse(b"{}")

        with patch.object(stop_port, "_open_url", side_effect=_fake_open):
            stop_port.probe_system_stats(8188, host="[::1]")
        assert seen == ["http://[::1]:8188/system_stats"]

    def test_connection_refused_is_not_answered(self):
        with patch.object(stop_port, "_open_url", side_effect=urllib.error.URLError("refused")):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is False

    def test_timeout_is_not_answered(self):
        with patch.object(stop_port, "_open_url", side_effect=TimeoutError()):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is False

    @pytest.mark.parametrize(
        "exc",
        [
            http.client.BadStatusLine("\x16\x03\x01"),
            http.client.IncompleteRead(b"{"),
            http.client.LineTooLong("header line"),
        ],
    )
    def test_protocol_garbage_is_not_answered_rather_than_a_traceback(self, exc):
        # `http.client.HTTPException` is neither URLError nor OSError: urllib
        # wraps OSError from the request, not from reading the response. A port
        # speaking TLS (a --tls-keyfile ComfyUI) or a server dying mid-body
        # would otherwise escape as an uncaught traceback with no envelope.
        with patch.object(stop_port, "_open_url", side_effect=exc):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is False

    def test_http_error_counts_as_answered(self):
        # A 404 means something IS serving this port — it is not wedged, it is
        # simply not ComfyUI.
        err = urllib.error.HTTPError("http://127.0.0.1:8188/system_stats", 404, "Not Found", {}, None)
        with patch.object(stop_port, "_open_url", side_effect=err):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats is None

    def test_non_json_body_counts_as_answered_without_stats(self):
        with patch.object(stop_port, "_open_url", return_value=FakeResponse(b"<html>hi</html>")):
            probe = stop_port.probe_system_stats(8188)
        assert probe.answered is True
        assert probe.stats is None

    def test_oversized_body_is_capped_not_buffered(self):
        body = b"x" * (stop_port.MAX_PROBE_BYTES + 4096)
        with patch.object(stop_port, "_open_url", return_value=FakeResponse(body)):
            probe = stop_port.probe_system_stats(8188)
        # Answered (something is serving) but never treated as /system_stats.
        assert probe.answered is True
        assert probe.stats is None

    def test_trickling_body_gives_up_at_the_deadline(self):
        # `timeout` bounds each socket operation, not the whole transfer, so a
        # peer feeding one byte at a time could hold the command open forever.
        never_ending = FakeResponse(b"x" * (stop_port.MAX_PROBE_BYTES + 1), chunk=1)
        with patch.object(stop_port, "_open_url", return_value=never_ending):
            probe = stop_port.probe_system_stats(8188, timeout=0.0)
        assert probe.answered is False

    def test_opener_disables_proxies_and_redirects(self):
        # The default opener honors http_proxy/ALL_PROXY (urllib's proxy_bypass
        # has no implicit localhost exemption) and follows up to 10 redirects,
        # so the probe could leave the machine entirely.
        with patch.object(stop_port.urllib.request, "build_opener") as build:
            stop_port._open_url("http://127.0.0.1:8188/system_stats", 1.0)
        handlers = build.call_args.args
        proxy = next(h for h in handlers if isinstance(h, urllib.request.ProxyHandler))
        assert proxy.proxies == {}
        assert stop_port._NoRedirect in handlers
        # Returning None leaves the 3xx unhandled, so urllib raises HTTPError
        # and the response reads as "answered, but not ComfyUI".
        assert stop_port._NoRedirect().redirect_request(None, None, 302, "Found", {}, "http://evil.example/") is None


class TestProbeHost:
    @pytest.mark.parametrize(
        ("laddr_ip", "expected"),
        [
            ("127.0.0.1", "127.0.0.1"),
            # Wildcard binds: probe the matching loopback.
            ("0.0.0.0", "127.0.0.1"),
            ("::", "[::1]"),
            (None, "127.0.0.1"),
            ("", "127.0.0.1"),
            # `--listen <lan-ip>` and v6-only binds were previously probed at
            # 127.0.0.1, read as unreachable, and fell through uncorroborated.
            ("192.168.1.50", "192.168.1.50"),
            ("::1", "[::1]"),
            ("fd00::5", "[fd00::5]"),
            # v4-mapped v6 sockets report this shape on some platforms.
            ("::ffff:0.0.0.0", "127.0.0.1"),
            ("::ffff:192.168.1.50", "192.168.1.50"),
            # Scoped/garbage addresses can't be probed; loopback is the safe
            # fallback and a non-answer is handled conservatively anyway.
            ("fe80::1%en0", "127.0.0.1"),
            ("not-an-ip", "127.0.0.1"),
        ],
    )
    def test_maps_bound_address_to_probe_host(self, laddr_ip, expected):
        assert stop_port.probe_host(laddr_ip) == expected


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

    def test_unreachable_server_verifies_on_cmdline_plus_the_checkout(self, comfy_tree):
        # The wedged-server case this command exists for: the server can't
        # corroborate itself, so the checkout on disk is the second signal.
        listener = stop_port.Listener(
            pid=42, cmdline=["/usr/bin/python3", str(comfy_tree / "main.py"), "--port", "8188"]
        )
        verdict = stop_port.verify_listener(listener, 8188, probe=stop_port.Probe(answered=False))
        assert verdict.verified is True
        assert verdict.http == "unreachable"

    def test_unreachable_server_relative_script_resolves_against_cwd(self, comfy_tree):
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd=str(comfy_tree))
        verdict = stop_port.verify_listener(listener, 8188, probe=stop_port.Probe(answered=False))
        assert verdict.verified is True

    def test_unreachable_non_comfyui_main_py_is_refused(self, tmp_path):
        # The hole the cmdline check alone leaves open: an unrelated
        # `python main.py` service that isn't speaking HTTP on the port at all.
        (tmp_path / "main.py").write_text("")
        listener = stop_port.Listener(pid=42, cmdline=["/usr/bin/python3", str(tmp_path / "main.py")])
        verdict = stop_port.verify_listener(listener, 8188, probe=stop_port.Probe(answered=False))
        assert verdict.verified is False
        assert verdict.http == "unreachable"
        assert "ComfyUI checkout" in verdict.reason

    def test_unreachable_unresolvable_script_is_refused(self):
        # Relative `main.py` with an unreadable cwd: nothing to corroborate with.
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd=None)
        verdict = stop_port.verify_listener(listener, 8188, probe=stop_port.Probe(answered=False))
        assert verdict.verified is False

    def test_answering_server_does_not_need_the_checkout_on_disk(self):
        # `/system_stats` corroboration is the stronger signal; a checkout we
        # cannot see (container, remote mount) must not veto it.
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd="/nonexistent")
        verdict = stop_port.verify_listener(listener, 8188, probe=_stats(argv=["main.py", "--port", "8188"]))
        assert verdict.verified is True

    def test_probes_the_bound_address(self):
        # Regression: hardcoding 127.0.0.1 made every `--listen <lan-ip>` or
        # v6-only server read as unreachable, skipping the cross-check.
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, laddr_ip="::1")
        with patch.object(
            stop_port, "probe_system_stats", return_value=_stats(argv=["main.py", "--port", "8188"])
        ) as probe:
            stop_port.verify_listener(listener, 8188)
        assert probe.call_args.kwargs["host"] == "[::1]"

    def test_server_without_argv_verifies_on_version_plus_cmdline(self):
        verdict = stop_port.verify_listener(self._listener(), 8188, probe=_stats(argv=None))
        assert verdict.verified is True
        assert verdict.http == "agreed_no_argv"


class TestLooksLikeComfyuiTree:
    def test_accepts_a_real_checkout(self, comfy_tree):
        listener = stop_port.Listener(pid=1, cmdline=["python", str(comfy_tree / "main.py")])
        assert stop_port.looks_like_comfyui_tree(listener) is True

    def test_rejects_a_bare_main_py(self, tmp_path):
        (tmp_path / "main.py").write_text("")
        listener = stop_port.Listener(pid=1, cmdline=["python", str(tmp_path / "main.py")])
        assert stop_port.looks_like_comfyui_tree(listener) is False

    def test_rejects_a_partial_match(self, tmp_path):
        # Two markers is under the threshold — one incidental `nodes.py` next to
        # a `main.py` shouldn't authorize a kill.
        (tmp_path / "main.py").write_text("")
        (tmp_path / "nodes.py").write_text("")
        listener = stop_port.Listener(pid=1, cmdline=["python", str(tmp_path / "main.py")])
        assert stop_port.looks_like_comfyui_tree(listener) is False


# --------------------------------------------------------------------------- #
# Teardown
# --------------------------------------------------------------------------- #


class FakeKillProc:
    """A process that dies at a configurable point of the escalation."""

    def __init__(
        self,
        pid,
        *,
        children=(),
        survives_terminate=False,
        survives_kill=False,
        log=None,
        create_time=1000.0,
        children_denied=False,
    ):
        self.pid = pid
        self._children = list(children)
        self._survives_terminate = survives_terminate
        self._survives_kill = survives_kill
        self.log = log if log is not None else []
        self.terminated = False
        self.killed = False
        self._create_time = create_time
        self._children_denied = children_denied

    def create_time(self):
        return self._create_time

    def children(self, recursive=False):
        if self._children_denied:
            raise psutil.AccessDenied(self.pid)
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
    def _run(self, parent, **kwargs):
        with (
            patch.object(stop_port.psutil, "Process", return_value=parent),
            patch.object(stop_port.psutil, "wait_procs", side_effect=_fake_wait_procs),
        ):
            return stop_port.kill_process_tree(parent.pid, **kwargs)

    def test_terminates_the_listener_itself_not_only_its_children(self):
        # The `utils.kill_all` trap: that helper kills only children, which
        # would leave a directly-identified listener running.
        log = []
        parent = FakeKillProc(42, log=log)
        result = self._run(parent)
        assert result.ok is True and result.survivors == []
        assert parent.terminated is True
        assert log == [("terminate", 42)]

    def test_reaps_recursive_children_before_the_parent(self):
        log = []
        kids = [FakeKillProc(43, log=log), FakeKillProc(44, log=log)]
        parent = FakeKillProc(42, children=kids, log=log)
        result = self._run(parent)
        assert result.ok is True and result.survivors == []
        assert [entry[1] for entry in log] == [43, 44, 42]
        assert all(k.terminated for k in kids)

    def test_escalates_to_kill_when_terminate_is_ignored(self):
        parent = FakeKillProc(42, survives_terminate=True)
        result = self._run(parent)
        assert result.ok is True and result.survivors == []
        assert parent.terminated is True and parent.killed is True

    def test_reports_survivors_when_kill_does_not_land(self):
        parent = FakeKillProc(42, survives_terminate=True, survives_kill=True)
        result = self._run(parent)
        assert result.ok is False and result.survivors == [42]

    def test_already_gone_process_is_a_success(self):
        with patch.object(stop_port.psutil, "Process", side_effect=psutil.NoSuchProcess(42)):
            result = stop_port.kill_process_tree(42)
        assert result.ok is True and result.survivors == []

    def test_recycled_pid_is_not_signalled(self):
        # Discovery and teardown are separated by an HTTP probe that can block
        # for the full timeout. If the listener exits in that window and the OS
        # hands its pid to a stranger, the stranger's whole tree must not die.
        stranger = FakeKillProc(42, create_time=9999.0)
        result = self._run(stranger, create_time=1000.0)
        assert result.ok is True and result.survivors == []
        assert stranger.terminated is False and stranger.killed is False

    def test_matching_create_time_still_signals(self):
        parent = FakeKillProc(42, create_time=1000.0)
        result = self._run(parent, create_time=1000.0)
        assert result.ok is True
        assert parent.terminated is True

    def test_unreadable_children_are_reported_not_silently_dropped(self):
        # `ok` can only speak for the parent when the child list can't be read;
        # claiming a clean stop over live workers holding GPU memory is worse
        # than admitting the gap.
        parent = FakeKillProc(42, children_denied=True)
        result = self._run(parent)
        assert result.ok is True
        assert result.children_enumerated is False


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
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree"),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=True)
        assert "cwd" not in _envelope(capsys.readouterr().out)["data"]

    def test_success_envelope(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=None)
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN),
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
        cfg = self._config(bg_info=("127.0.0.1", 8188, 42))
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=stop_port.Teardown(ok=False, survivors=[42])),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            with pytest.raises(typer.Exit) as exc:
                stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert exc.value.exit_code == 1
        # A failed stop must not drop the background record.
        cfg.remove_background.assert_not_called()

        env = _envelope(capsys.readouterr().out)
        assert env["error"]["code"] == "stop_failed"
        assert env["error"]["details"]["survivors"] == [42]

    def test_port_still_held_after_a_clean_kill_is_not_reported_as_success(self, capsys):
        # SO_REUSEPORT, split v4/v6 sockets, or children we could not enumerate:
        # the identified process died, but the port never came free.
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        sibling = stop_port.Listener(pid=43, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=None)
        with (
            patch.object(stop_port, "find_listener", side_effect=[listener, sibling]),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            with pytest.raises(typer.Exit) as exc:
                stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert exc.value.exit_code == 1
        cfg.remove_background.assert_not_called()

        env = _envelope(capsys.readouterr().out)
        assert env["error"]["code"] == "stop_failed"
        assert env["error"]["details"]["survivors"] == [43]

    def test_unenumerable_children_are_flagged_in_the_envelope(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        teardown = stop_port.Teardown(ok=True, survivors=[], children_enumerated=False)
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=teardown),
            patch.object(stop_port, "ConfigManager", return_value=self._config()),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        assert _envelope(capsys.readouterr().out)["data"]["children_unknown"] is True

    def test_matching_background_pid_is_forgotten(self, capsys):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 42))
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        cfg.remove_background.assert_called_once()
        capsys.readouterr()

    def test_recorded_launch_wrapper_is_torn_down_with_its_listener(self, capsys):
        # `comfy launch --background` records the *wrapper's* pid, not the
        # `python main.py` it spawns, so the exact-pid match below could never
        # fire for a server this CLI started: the record outlived every
        # `comfy stop --port`, and the wrapper (whose redirector threads never
        # exit on their own) leaked.
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, create_time=1000.0)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 7))
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "_ancestor_pids", return_value=[7, 1]),
            patch.object(stop_port, "_is_launch_wrapper", return_value=True),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN) as killer,
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        killer.assert_called_once_with(7)
        cfg.remove_background.assert_called_once()
        capsys.readouterr()

    def test_recycled_ancestor_pid_is_not_mistaken_for_the_wrapper(self, capsys):
        # The recorded pid being an ancestor is not enough: it could have been
        # recycled onto an unrelated ancestor (a login shell). Signalling that
        # tree, or forgetting a background server we still own, would be worse
        # than leaving a stale record.
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, create_time=1000.0)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 7))
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "_ancestor_pids", return_value=[7, 1]),
            patch.object(stop_port, "_is_launch_wrapper", return_value=False),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN) as killer,
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        killer.assert_called_once_with(42, create_time=1000.0)
        cfg.remove_background.assert_not_called()
        capsys.readouterr()

    def test_different_background_pid_on_the_same_port_is_kept(self, capsys):
        # Port equality is not identity: dropping the record here would orphan
        # a still-running server this CLI does own.
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE)
        cfg = self._config(bg_info=("127.0.0.1", 8188, 777))
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "_ancestor_pids", return_value=[]),
            patch.object(stop_port, "kill_process_tree", return_value=_TORN_DOWN),
            patch.object(stop_port, "ConfigManager", return_value=cfg),
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=False)
        cfg.remove_background.assert_not_called()
        capsys.readouterr()

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_payloads_validate_against_the_shipped_schema(self, capsys, dry_run):
        renderer = _json_renderer()
        listener = stop_port.Listener(pid=42, cmdline=COMFY_CMDLINE, cwd="/srv/ComfyUI")
        teardown = stop_port.Teardown(ok=True, survivors=[], children_enumerated=False)
        with (
            _finds(listener),
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree", return_value=teardown),
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
            patch.object(stop_port, "probe_system_stats", return_value=_AGREEING),
            patch.object(stop_port, "kill_process_tree") as killer,
        ):
            stop_port.stop_port_execute(renderer, port=8188, dry_run=True)
        out = capsys.readouterr().out
        killer.assert_not_called()
        assert "Would stop ComfyUI on port 8188" in out
        # Pretty mode emits no envelope.
        assert "envelope" not in out
