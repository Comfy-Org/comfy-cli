"""Unit tests for `comfy run --json` (NDJSON streaming through the renderer).

One dialect: the legacy `JsonEmitter` (`{"event": …, "error": {"kind": …}}`)
is gone. `comfy run --json` now emits the renderer's `event/1` stream —
`{"schema": "event/1", "type": …}` lines — terminated by a single
`{"schema": "envelope/1", "type": "envelope", ok, data, error}` line.
See `docs/json-output.md` for the contract these tests pin in place.

The tests cover:
  - every event type emitted at the right time and shape
  - every error.code for each documented failure path (all registered
    in `comfy_cli.error_codes`)
  - the final envelope on both success and failure, with exit codes
    (1 for errors, 130 for cancellation)
  - stream archetypes from the spec table
  - the duck-typed output filter rule
  - the cached/executed overlap semantics
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
import typer
from websocket import WebSocketException, WebSocketTimeoutException

from comfy_cli.command.run import (
    WorkflowExecution,
    _classify_api_workflow,
    execute,
)
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing


@pytest.fixture(autouse=True)
def ndjson_renderer():
    """Install a fresh NDJSON renderer per test (the state `comfy run --json`
    runs in after `force_stream()`), and reset the singleton afterwards."""
    renderer = Renderer(mode=OutputMode.NDJSON, command="run")
    set_renderer(renderer)
    yield renderer
    reset_renderer_for_testing()


@pytest.fixture
def simple_workflow():
    return {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1},
            "_meta": {"title": "Latent"},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "x", "images": ["1", 0]},
            "_meta": {"title": "Save"},
        },
    }


@pytest.fixture
def workflow_file(simple_workflow):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(simple_workflow, f)
        f.flush()
        path = f.name
    yield path
    os.unlink(path)


def _parse_lines(out: str) -> list[dict]:
    lines = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(json.loads(line))
    return lines


def _events(lines: list[dict]) -> list[dict]:
    return [ln for ln in lines if ln.get("type") != "envelope"]


def _envelope(lines: list[dict]) -> dict:
    assert lines, "expected at least one NDJSON line"
    last = lines[-1]
    assert last.get("type") == "envelope", f"last line is not the envelope: {last}"
    return last


def _run_execute_capture(workflow_path, capsys, **overrides):
    """Run execute() and return (parsed stdout lines, exit_code)."""
    kwargs = dict(
        host="127.0.0.1",
        port=8188,
        wait=True,
        verbose=False,
        timeout=30,
    )
    kwargs.update(overrides)
    exit_code = 0
    try:
        execute(workflow_path, **kwargs)
    except typer.Exit as e:
        exit_code = e.exit_code or 0
    out, _err = capsys.readouterr()
    return _parse_lines(out), exit_code


def _make_http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8188/prompt",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def _make_workflow_execution(workflow, *, with_progress: bool = False):
    """Build a `WorkflowExecution`. `with_progress=True` attaches a MagicMock
    progress object — needed by tests that exercise `update_overall_progress`."""
    progress = None
    if with_progress:
        progress = MagicMock()
        progress.add_task.return_value = 0
    return WorkflowExecution(
        workflow=workflow,
        host="127.0.0.1",
        port=8188,
        verbose=False,
        local_paths=False,
        progress=progress,
        timeout=30,
    )


class TestStreamShape:
    """Dialect-level invariants: schema fields, pretty-mode no-op, UTF-8."""

    def test_events_are_noop_in_pretty_mode(self, simple_workflow, capsys):
        set_renderer(Renderer(mode=OutputMode.PRETTY))
        ex = _make_workflow_execution(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2", "output": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}})
        ex.on_cached({"nodes": ["1"]})
        out, _ = capsys.readouterr()
        # No NDJSON lines in pretty mode (event() is a no-op there).
        for line in out.splitlines():
            assert not line.strip().startswith("{"), f"unexpected JSON in pretty mode: {line!r}"

    def test_every_line_carries_schema_and_type(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        assert exit_code == 0
        for ln in _events(lines):
            assert ln["schema"] == "event/1", ln
            assert isinstance(ln["type"], str) and ln["type"], ln
        env = _envelope(lines)
        assert env["schema"] == "envelope/1"
        assert env["ok"] is True

    def test_non_ascii_round_trips_as_utf8(self, capsys, ndjson_renderer):
        # The renderer writes UTF-8 (ensure_ascii=False); consumers get the
        # original characters back after json.loads.
        ndjson_renderer.error(code="workflow_not_found", message="found: 猫_00001_.png")
        out, _ = capsys.readouterr()
        env = json.loads(out.strip())
        assert env["error"]["message"] == "found: 猫_00001_.png"
        assert "猫" in out

    def test_error_envelope_is_emitted_once(self, capsys, ndjson_renderer):
        ndjson_renderer.error(code="workflow_not_found", message="first")
        ndjson_renderer.error(code="workflow_empty", message="second")
        out, _ = capsys.readouterr()
        lines = _parse_lines(out)
        assert len(lines) == 1
        assert lines[0]["error"]["code"] == "workflow_not_found"


class TestClassifyApiWorkflow:
    def test_well_formed(self):
        assert _classify_api_workflow({"1": {"class_type": "X", "inputs": {}}})[0] == "ok"

    def test_empty_dict(self):
        assert _classify_api_workflow({})[0] == "empty"

    def test_invalid_first_node(self):
        assert _classify_api_workflow({"foo": "bar"})[0] == "invalid"

    def test_invalid_not_a_dict(self):
        assert _classify_api_workflow([])[0] == "invalid"


class TestPreFlightFailures:
    """Single error envelope, no events, exit 1."""

    def test_workflow_not_found(self, capsys):
        lines, exit_code = _run_execute_capture("/nonexistent.json", capsys)
        assert exit_code == 1
        assert len(lines) == 1
        env = _envelope(lines)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_workflow_invalid_json(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ this is not json")
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert exit_code == 1
            assert _envelope(lines)["error"]["code"] == "workflow_invalid_json"
        finally:
            os.unlink(path)

    def test_workflow_read_error_unicode(self, capsys):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
            f.write(b"\xff\xfe\xfa\x00")  # invalid UTF-8
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert exit_code == 1
            assert _envelope(lines)["error"]["code"] == "workflow_read_error"
        finally:
            os.unlink(path)

    def test_workflow_empty_api(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert _envelope(lines)["error"]["code"] == "workflow_empty"
        finally:
            os.unlink(path)

    def test_workflow_not_api_format(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"foo": "bar"}, f)
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert _envelope(lines)["error"]["code"] == "workflow_not_api_format"
        finally:
            os.unlink(path)

    def test_server_not_running(self, workflow_file, capsys):
        with patch("comfy_cli.command.run.check_comfy_server_running", return_value=False):
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        assert exit_code == 1
        env = _envelope(lines)
        assert env["error"]["code"] == "server_not_running"
        assert env["error"]["details"]["port"] == 8188


class TestSuccessfulRun:
    def test_no_wait_emits_prompt_preview_then_queued(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run._spawn_watcher", return_value=True),
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p123"}).encode()
            lines, exit_code = _run_execute_capture(workflow_file, capsys, wait=False)
        assert exit_code == 0
        # prompt_preview is always emitted before queued so agents have a
        # full audit trail of the submitted workflow graph.
        assert [e["type"] for e in _events(lines)] == ["prompt_preview", "queued"]
        assert _events(lines)[0]["prompt"]
        queued = _events(lines)[1]
        assert queued["prompt_id"] == "p123"
        assert queued["validation_warnings"] == []
        env = _envelope(lines)
        assert env["ok"] is True
        assert env["data"]["status"] == "queued"
        assert env["data"]["prompt_id"] == "p123"

    def test_envelope_after_success(self, workflow_file, capsys):
        """Mocked WS flow → queued + executing/executed/output events + ok envelope."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance

            def msg(t, **d):
                return json.dumps({"type": t, "data": {"prompt_id": "p", **d}})

            ws_instance.recv.side_effect = [
                msg("executing", node="1"),
                msg(
                    "executed", node="1", output={"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}
                ),
                msg("executing", node=None),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys, wait=True)

        assert exit_code == 0
        types = [e["type"] for e in _events(lines)]
        assert types == ["prompt_preview", "queued", "executing", "executed", "output"]
        executed = next(e for e in _events(lines) if e["type"] == "executed")
        assert executed["outputs"][0]["filename"] == "x.png"
        assert executed["outputs"][0]["category"] == "images"
        env = _envelope(lines)
        assert env["ok"] is True
        assert env["data"]["status"] == "completed"
        assert env["data"]["prompt_id"] == "p"
        assert len(env["data"]["outputs"]) == 1
        assert env["data"]["executed_node_ids"] == ["1"]


class TestQueueHttpErrors:
    """Verify the HTTP error mapping for /prompt failures."""

    def _setup_and_run(self, workflow_file, http_response, capsys, status=None, body=b""):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            if status is None:
                # Success path mock
                mock_open.return_value.read.return_value = http_response
            else:
                mock_open.side_effect = _make_http_error(status, body)
            return _run_execute_capture(workflow_file, capsys)

    def test_400_with_node_errors_routes_to_prompt_rejected(self, workflow_file, capsys):
        body = json.dumps(
            {
                "error": {"type": "x", "message": "y"},
                "node_errors": {"1": {"errors": [{"type": "z", "message": "bad"}], "class_type": "X"}},
            }
        ).encode()
        lines, exit_code = self._setup_and_run(workflow_file, None, capsys, status=400, body=body)
        assert exit_code == 1
        env = _envelope(lines)
        assert env["error"]["code"] == "prompt_rejected"
        node_errors = env["error"]["details"]["node_errors"]
        assert isinstance(node_errors, list)
        assert any(rec["node_id"] == "1" for rec in node_errors)

    @pytest.mark.parametrize(
        "status,body,code",
        [
            (400, b"plain bad request", "client_error"),
            (401, b"unauthorized", "client_error"),
            (403, b"forbidden", "client_error"),
            (429, b"too many", "client_error"),
            (500, b"oops", "server_error"),
            (503, b"down", "server_error"),
        ],
    )
    def test_http_status_routes_to_code(self, workflow_file, capsys, status, body, code):
        lines, exit_code = self._setup_and_run(workflow_file, None, capsys, status=status, body=body)
        assert exit_code == 1
        env = _envelope(lines)
        assert env["error"]["code"] == code
        assert env["error"]["details"]["status"] == status
        assert env["error"]["details"]["body"] == body.decode()

    def test_200_with_non_json_body_routes_to_invalid_response(self, workflow_file, capsys):
        lines, exit_code = self._setup_and_run(workflow_file, b"<html>garbage</html>", capsys)
        env = _envelope(lines)
        assert env["error"]["code"] == "invalid_response"
        assert env["error"]["details"]["status"] == 200

    def test_200_without_prompt_id_routes_to_invalid_response(self, workflow_file, capsys):
        lines, exit_code = self._setup_and_run(workflow_file, json.dumps({"other": "x"}).encode(), capsys)
        assert _envelope(lines)["error"]["code"] == "invalid_response"

    def test_200_with_utf16_bom_body_routes_to_invalid_response(self, workflow_file, capsys):
        # `json.loads(bytes)` sniffs encoding before parsing — a UTF-16 BOM
        # makes it raise `UnicodeDecodeError`, not `JSONDecodeError`.
        lines, exit_code = self._setup_and_run(workflow_file, b"\x00\x01\xff\xfeNOT JSON \x80\x81", capsys)
        env = _envelope(lines)
        assert env["error"]["code"] == "invalid_response"
        assert env["error"]["details"]["status"] == 200

    def test_url_error_routes_to_connection_error(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            mock_open.side_effect = urllib.error.URLError("refused")
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        assert _envelope(lines)["error"]["code"] == "connection_error"

    def test_validation_warnings_on_200_with_partial_node_errors(self, workflow_file, capsys):
        """200 + non-empty node_errors → `queued` with validation_warnings populated."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            body = json.dumps(
                {
                    "prompt_id": "p",
                    "node_errors": {"3": {"errors": [{"type": "x", "message": "skipped"}], "class_type": "X"}},
                }
            ).encode()
            mock_open.return_value.read.return_value = body
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        queued = next(e for e in _events(lines) if e["type"] == "queued")
        warnings = queued["validation_warnings"]
        assert isinstance(warnings, list)
        rec = next(rec for rec in warnings if rec["node_id"] == "3")
        assert rec["class_type"] == "X"
        assert rec["errors"][0]["message"] == "skipped"


class TestQueuedEventShape:
    def test_queued_nodes_manifest_from_workflow(self, workflow_file, capsys, simple_workflow):
        """`nodes` lists one entry per workflow node with node_id, class_type, title."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run._spawn_watcher", return_value=True),
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            lines, exit_code = _run_execute_capture(workflow_file, capsys, wait=False)
        queued = next(e for e in _events(lines) if e["type"] == "queued")
        assert queued["client_id"]
        nodes = queued["nodes"]
        assert len(nodes) == 2
        by_id = {n["node_id"]: n for n in nodes}
        assert by_id["1"]["class_type"] == "EmptyLatentImage"
        assert by_id["1"]["title"] == "Latent"  # _meta.title wins
        assert by_id["2"]["class_type"] == "SaveImage"
        assert by_id["2"]["title"] == "Save"


class TestWebSocketEvents:
    def _run_with_ws_messages(self, workflow_file, recv_side_effect, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = recv_side_effect
            return _run_execute_capture(workflow_file, capsys)

    def test_websocket_timeout(self, workflow_file, capsys):
        lines, exit_code = self._run_with_ws_messages(
            workflow_file,
            WebSocketTimeoutException("timed out"),
            capsys,
        )
        assert exit_code == 1
        env = _envelope(lines)
        assert env["error"]["code"] == "ws_timeout"
        assert env["error"]["details"]["timeout"] == 30

    def test_connection_lost_websocket(self, workflow_file, capsys):
        lines, exit_code = self._run_with_ws_messages(
            workflow_file,
            WebSocketException("dropped"),
            capsys,
        )
        assert exit_code == 1
        assert _envelope(lines)["error"]["code"] == "ws_disconnected"

    def test_keyboard_interrupt_maps_to_cancelled_exit_130(self, workflow_file, capsys):
        lines, exit_code = self._run_with_ws_messages(
            workflow_file,
            KeyboardInterrupt(),
            capsys,
        )
        assert exit_code == 130
        env = _envelope(lines)
        assert env["ok"] is False
        assert env["error"]["code"] == "cancelled"

    def test_malformed_frame_is_skipped_run_completes(self, workflow_file, capsys):
        """Malformed JSON frames are silently skipped mid-stream. A valid
        executing(node=None) frame following the bad one still terminates
        the run normally with an ok envelope."""
        lines, exit_code = self._run_with_ws_messages(
            workflow_file,
            ["{not json", json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}})],
            capsys,
        )
        assert exit_code == 0
        env = _envelope(lines)
        assert env["ok"] is True
        assert env["data"]["status"] == "completed"

    def test_execution_error(self, workflow_file, capsys):
        messages = [
            json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
            json.dumps(
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": "p",
                        "node_id": "1",
                        "node_type": "EmptyLatentImage",
                        "exception_type": "RuntimeError",
                        "exception_message": "boom",
                        "traceback": ['  File "x.py"\n', "    raise RuntimeError\n"],
                    },
                }
            ),
        ]
        lines, exit_code = self._run_with_ws_messages(workflow_file, messages, capsys)
        assert exit_code == 1
        # An `execution_error` event precedes the error envelope.
        assert any(e["type"] == "execution_error" for e in _events(lines))
        env = _envelope(lines)
        assert env["error"]["code"] == "execution_error"
        assert env["error"]["message"] == "EmptyLatentImage (node 1): boom"
        details = env["error"]["details"]
        assert details["node_id"] == "1"
        assert details["class_type"] == "EmptyLatentImage"
        assert details["exception_type"] == "RuntimeError"
        assert details["title"] == "Latent"  # from _meta.title
        # The envelope carries only the traceback tail; the full traceback
        # stays on the execution_error event.
        assert isinstance(details["traceback_tail"], list)
        assert any("raise RuntimeError" in frame for frame in details["traceback_tail"])
        assert "traceback" not in details

    def test_execution_error_node_id_coerced_to_str(self, workflow_file, capsys):
        # If ComfyUI ever sends node_id as an int, the contract still
        # requires a string in details.node_id.
        messages = [
            json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
            json.dumps(
                {
                    "type": "execution_error",
                    "data": {
                        "prompt_id": "p",
                        "node_id": 7,
                        "node_type": "EmptyLatentImage",
                        "exception_type": "RuntimeError",
                        "exception_message": "boom",
                        "traceback": [],
                    },
                }
            ),
        ]
        lines, exit_code = self._run_with_ws_messages(workflow_file, messages, capsys)
        env = _envelope(lines)
        assert env["error"]["code"] == "execution_error"
        assert env["error"]["details"]["node_id"] == "7"
        assert isinstance(env["error"]["details"]["node_id"], str)

    def test_server_side_interrupt_maps_to_cancelled_exit_130(self, workflow_file, capsys):
        messages = [
            json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
            json.dumps({"type": "execution_interrupted", "data": {"prompt_id": "p"}}),
        ]
        lines, exit_code = self._run_with_ws_messages(workflow_file, messages, capsys)
        assert exit_code == 130
        assert _envelope(lines)["error"]["code"] == "cancelled"


class TestOutputObject:
    def _exec(self, simple_workflow):
        return _make_workflow_execution(simple_workflow, with_progress=True)

    def _executed_event(self, capsys):
        events = _parse_lines(capsys.readouterr().out)
        return next(e for e in events if e["type"] == "executed")

    def test_duck_typed_filter_skips_strings(self, simple_workflow, capsys):
        """ComfyUI's `text` output key emits a list of strings; the filter must skip non-file shapes."""
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": "2",
                "output": {
                    "text": ["hello"],
                    "images": [{"filename": "x.png", "subfolder": "", "type": "output"}],
                },
            }
        )
        executed = self._executed_event(capsys)
        assert len(executed["outputs"]) == 1
        assert executed["outputs"][0]["category"] == "images"

    def test_duck_typed_filter_skips_booleans(self, simple_workflow, capsys):
        """`animated` key emits list of bool — must be skipped."""
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": "2",
                "output": {
                    "animated": [True],
                    "images": [{"filename": "x.png", "subfolder": "", "type": "output"}],
                },
            }
        )
        executed = self._executed_event(capsys)
        assert len(executed["outputs"]) == 1

    def test_audio_category_recognized(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": "2",
                "output": {
                    "audio": [{"filename": "a.wav", "subfolder": "sf", "type": "output"}],
                },
            }
        )
        executed = self._executed_event(capsys)
        assert executed["outputs"][0]["category"] == "audio"
        assert executed["outputs"][0]["filename"] == "a.wav"
        assert executed["outputs"][0]["subfolder"] == "sf"

    def test_output_url_has_correct_format(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": "2",
                "output": {
                    "images": [{"filename": "x.png", "subfolder": "", "type": "output"}],
                },
            }
        )
        executed = self._executed_event(capsys)
        url = executed["outputs"][0]["url"]
        assert url.startswith("http://127.0.0.1:8188/view?")
        assert "filename=x.png" in url
        assert "type=output" in url

    def test_missing_subfolder_defaults_to_empty_string(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": "2",
                "output": {
                    "images": [{"filename": "x.png", "type": "output"}],
                },
            }
        )
        executed = self._executed_event(capsys)
        assert executed["outputs"][0]["subfolder"] == ""


class TestNodeBookkeeping:
    """`cached_node_ids` / `executed_node_ids` aggregation surfaced in the
    success envelope (ported from the emitter's `completed` event)."""

    def test_cached_and_executed_can_overlap(self, simple_workflow):
        """Cached output-bearing nodes appear in both lists."""
        ex = _make_workflow_execution(simple_workflow)
        ex.prompt_id = "p"
        ex.on_cached({"nodes": ["2"]})
        ex.on_executed({"node": "2"})
        assert "2" in ex.cached_node_ids
        assert "2" in ex.executed_node_ids

    def test_node_ids_coerced_to_str(self, simple_workflow, capsys):
        ex = _make_workflow_execution(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executing({"node": 2})
        ex.on_cached({"nodes": [1]})
        ex.on_executed({"node": 2})
        assert ex.executed_node_ids == ["2"]
        assert ex.cached_node_ids == ["1"]
        events = _parse_lines(capsys.readouterr().out)
        for ev in events:
            assert isinstance(ev["node"], str), f"{ev['type']} node is {type(ev['node']).__name__}"

    def test_title_falls_back_to_class_type(self):
        ex = _make_workflow_execution({"1": {"class_type": "EmptyLatentImage", "inputs": {}}})
        assert ex.get_node_title("1") == "EmptyLatentImage"

    def test_title_falls_back_to_node_id_for_unknown(self):
        ex = _make_workflow_execution({})
        assert ex.get_node_title("unknown") == "unknown"


UI_WORKFLOW = {
    "nodes": [
        {
            "id": 1,
            "type": "EmptyLatentImage",
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10]}],
            "widgets_values": [64, 64, 1],
            "mode": 0,
        },
        {
            "id": 2,
            "type": "PreviewImage",
            "inputs": [{"name": "images", "link": 10}],
            "outputs": [],
            "mode": 0,
        },
    ],
    "links": [[10, 1, 0, 2, 0, "IMAGE"]],
}

OBJECT_INFO = {
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        },
        "input_order": {"required": ["width", "height", "batch_size"]},
        "output_node": False,
        "display_name": "Empty Latent Image",
    },
    "PreviewImage": {
        "input": {"required": {"images": ["IMAGE"]}},
        "input_order": {"required": ["images"]},
        "output_node": True,
        "display_name": "Preview Image",
    },
}


@pytest.fixture
def ui_workflow_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(UI_WORKFLOW, f)
        f.flush()
        path = f.name
    yield path
    os.unlink(path)


class TestWorkflowPathExpansion:
    """Regression: `~/wf.json` must be expanded before the existence check.
    Otherwise scripted callers passing literal `~/...` see a misleading
    workflow_not_found."""

    def test_tilde_path_is_expanded_before_existence_check(self, capsys, monkeypatch, tmp_path):
        workflow_path = tmp_path / "wf.json"
        workflow_path.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}))
        monkeypatch.setenv("HOME", str(tmp_path))
        lines, exit_code = _run_execute_capture("~/wf.json", capsys, print_prompt=True)
        assert _events(lines)[0]["type"] == "prompt_preview", lines

    def test_tilde_path_to_missing_file_reports_expanded_path(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        lines, exit_code = _run_execute_capture("~/missing.json", capsys, print_prompt=True)
        env = _envelope(lines)
        assert env["error"]["code"] == "workflow_not_found"
        # The error message should name the resolved path so the user can
        # see exactly where we looked.
        assert str(tmp_path) in env["error"]["message"]


class TestCliRunnerIntegration:
    """End-to-end: the typer entry callback chain (consent prompt, decorators,
    config init) must not leak any prose to stdout in stream mode. Direct
    `execute()` tests bypass this seam; agents on a fresh machine with
    no recorded consent are exactly where the original prompt-corrupts-stream
    bug would have hidden."""

    def _make_workflow_file(self, tmp_path):
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}))
        return str(wf_path)

    def test_cli_json_print_prompt_emits_clean_ndjson(self, tmp_path):
        # Smoke: default config state, --json --print-prompt → every stdout
        # line is valid JSON with `schema` and `type`, ending in an envelope.
        from typer.testing import CliRunner

        from comfy_cli.cmdline import app

        runner = CliRunner()  # non-TTY by default
        result = runner.invoke(
            app,
            ["run", "--workflow", self._make_workflow_file(tmp_path), "--json", "--print-prompt"],
            env={"COMFY_WHERE": "local"},
        )
        assert result.exit_code == 0, f"stdout={result.stdout!r}\nexc={result.exception!r}"
        lines = _parse_lines(result.stdout)
        assert lines, "expected at least one NDJSON line"
        for ln in lines:
            assert "schema" in ln
            assert "type" in ln
        env = _envelope(lines)
        assert env["ok"] is True
        # Consent prompt text must not appear.
        assert "Do you agree" not in result.stdout
        assert "improve the application" not in result.stdout

    def test_cli_json_with_fresh_consent_state_stays_clean(self, tmp_path):
        # The exact regression scenario: a fresh machine where consent has
        # never been recorded. The entry callback enables session-only
        # tracking via the non-TTY branch (PROVIDERS swapped out so no
        # network), and the resulting stdout must still be clean NDJSON.
        from typer.testing import CliRunner

        from comfy_cli.cmdline import app
        from comfy_cli.config_manager import ConfigManager

        _Cls = ConfigManager.__closure__[0].cell_contents
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        with (
            patch.object(_Cls, "get_config_path", return_value=str(cfg_dir)),
            patch("comfy_cli.tracking.PROVIDERS", []),
        ):
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["run", "--workflow", self._make_workflow_file(tmp_path), "--json", "--print-prompt"],
                env={"COMFY_WHERE": "local"},
            )
        assert result.exit_code == 0, f"stdout={result.stdout!r}\nexc={result.exception!r}"
        for ln in _parse_lines(result.stdout):
            assert "type" in ln
        assert "Do you agree" not in result.stdout
        assert "tracking" not in result.stdout.lower()


class TestPromptPreviewAlwaysEmitted:
    """In stream mode the converted workflow graph is always emitted as a
    `prompt_preview` event before `queued`. Agents debugging conversions
    or building an audit trail get full visibility without re-running
    with a flag."""

    def test_api_input_emits_prompt_preview_before_queued(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        types = [e["type"] for e in _events(lines)]
        assert types[0] == "prompt_preview"
        assert "queued" in types
        assert types.index("prompt_preview") < types.index("queued")
        assert _events(lines)[0]["prompt"]["1"]["class_type"] == "EmptyLatentImage"

    def test_ui_input_emits_converted_then_prompt_preview_then_queued(self, ui_workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch("comfy_cli.command.run.request.urlopen") as mock_post,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_post.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)
        types = [e["type"] for e in _events(lines)]
        # Ordering: converted, prompt_preview, queued (then per-node events).
        c = types.index("converted")
        p = types.index("prompt_preview")
        q = types.index("queued")
        assert c < p < q

    def test_prompt_preview_excludes_client_id_and_extra_data(self, workflow_file, capsys):
        # The audit trail must carry only the workflow graph, never the
        # POST envelope's runtime fields (client_id, extra_data with api_key).
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys, api_key="sk-secret")
        preview = next(e for e in _events(lines) if e["type"] == "prompt_preview")
        prompt = preview["prompt"]
        assert "client_id" not in prompt
        assert "extra_data" not in prompt
        assert "sk-secret" not in json.dumps(prompt)


class TestPrintPrompt:
    """`--print-prompt` returns the would-be `/prompt` body and exits 0
    without POSTing. UI input still needs `/object_info`; API input
    doesn't touch the server at all."""

    def test_api_input_emits_prompt_preview_and_envelope_only(self, workflow_file, capsys):
        # No server probe, no /object_info fetch — API input is printed as-is.
        with (
            patch("comfy_cli.command.run.check_comfy_server_running") as mock_check,
            patch("comfy_cli.command.run.fetch_object_info") as mock_fetch,
            patch("comfy_cli.command.run.request.urlopen") as mock_post,
        ):
            lines, exit_code = _run_execute_capture(workflow_file, capsys, print_prompt=True)
        assert mock_check.call_count == 0
        assert mock_fetch.call_count == 0
        assert mock_post.call_count == 0
        assert exit_code == 0
        assert [e["type"] for e in _events(lines)] == ["prompt_preview"]
        preview = _events(lines)[0]
        assert preview["schema"] == "event/1"
        assert preview["prompt"]["1"]["class_type"] == "EmptyLatentImage"
        env = _envelope(lines)
        assert env["ok"] is True
        assert env["data"]["status"] == "preview"

    def test_ui_input_emits_converted_then_prompt_preview(self, ui_workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch("comfy_cli.command.run.request.urlopen") as mock_post,
        ):
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys, print_prompt=True)
        assert mock_post.call_count == 0
        assert [e["type"] for e in _events(lines)] == ["converted", "prompt_preview"]
        prompt = _events(lines)[1]["prompt"]
        assert isinstance(prompt, dict)
        # The converted prompt should have entries for the UI nodes.
        assert len(prompt) >= 1
        for entry in prompt.values():
            assert "class_type" in entry

    def test_ui_input_with_unreachable_object_info_routes_to_connection_error(self, ui_workflow_file, capsys):
        # --print-prompt skips the pre-flight server probe, but UI conversion
        # still needs /object_info, so an unreachable host surfaces here.
        with (
            patch("comfy_cli.command.run.request.urlopen", side_effect=urllib.error.URLError("Connection refused")),
        ):
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys, print_prompt=True)
        assert exit_code == 1
        assert _envelope(lines)["error"]["code"] == "connection_error"

    def test_api_input_works_with_offline_server(self, workflow_file, capsys):
        # Hard-fail the server probe — the API path must not call it under --print-prompt.
        with patch(
            "comfy_cli.command.run.check_comfy_server_running", side_effect=AssertionError("must not be called")
        ):
            lines, exit_code = _run_execute_capture(workflow_file, capsys, print_prompt=True)
        assert _events(lines)[0]["type"] == "prompt_preview"

    def test_print_prompt_does_not_include_api_key_or_client_id(self, workflow_file, capsys):
        # The prompt_preview body should only carry the workflow graph,
        # not the runtime POST envelope (which would otherwise leak the api_key).
        lines, exit_code = _run_execute_capture(workflow_file, capsys, print_prompt=True, api_key="sk-secret")
        prompt = _events(lines)[0]["prompt"]
        assert "extra_data" not in prompt
        assert "client_id" not in prompt
        assert "sk-secret" not in json.dumps(prompt)

    def test_print_prompt_text_mode_pretty_prints_json(self, workflow_file, capsys):
        set_renderer(Renderer(mode=OutputMode.PRETTY))
        try:
            execute(workflow_file, host="127.0.0.1", port=8188, print_prompt=True)
        except typer.Exit:
            pass
        out, _err = capsys.readouterr()
        parsed = json.loads(out)
        assert "1" in parsed
        assert parsed["1"]["class_type"] == "EmptyLatentImage"

    def test_print_prompt_does_not_post_when_workflow_invalid(self, capsys):
        # Pre-flight failures (workflow_not_found, workflow_not_api_format)
        # still produce an error envelope and exit 1 under --print-prompt.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a workflow"}, f)
            path = f.name
        try:
            lines, exit_code = _run_execute_capture(path, capsys, print_prompt=True)
            assert exit_code == 1
            assert _envelope(lines)["error"]["code"] == "workflow_not_api_format"
        finally:
            os.unlink(path)


class TestConvertedAndConversionErrors:
    """UI-input event path and the conversion_error / conversion_crash codes."""

    def test_converted_event_for_ui_input(self, ui_workflow_file, capsys):
        """`converted` is the first event when input is UI format."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)

        converted = _events(lines)[0]
        assert converted["type"] == "converted"
        assert converted["schema"] == "event/1"
        assert converted["node_count"] == 2  # the UI workflow has 2 nodes

    def test_conversion_error_code(self, ui_workflow_file, capsys):
        """WorkflowConversionError → code=conversion_error."""
        from comfy_cli.workflow_to_api import WorkflowConversionError

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=WorkflowConversionError("broken graph"),
            ),
        ):
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)

        assert exit_code == 1
        assert _envelope(lines)["error"]["code"] == "conversion_error"

    def test_conversion_crash_code_with_exception_type(self, ui_workflow_file, capsys):
        """Unexpected converter crash → code=conversion_crash with details.exception_type."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=KeyError("missing field"),
            ),
        ):
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)

        env = _envelope(lines)
        assert env["error"]["code"] == "conversion_crash"
        assert env["error"]["details"]["exception_type"] == "KeyError"

    def test_workflow_empty_after_conversion(self, capsys):
        """UI conversion producing {} → workflow_empty."""
        empty_ui = {"nodes": [], "links": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_ui, f)
            f.flush()
            path = f.name
        try:
            with (
                patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
                patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
                patch("comfy_cli.command.run.convert_ui_to_api", return_value={}),
            ):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert _envelope(lines)["error"]["code"] == "workflow_empty"
        finally:
            os.unlink(path)


class TestObjectInfoFailures:
    """HTTP and network errors on /object_info."""

    def test_object_info_unavailable_on_http_error(self, ui_workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            # _make_http_error builds a /prompt URL by default — build the
            # /object_info HTTPError inline so the test exercises that path.
            mock_open.side_effect = urllib.error.HTTPError(
                url="http://127.0.0.1:8188/object_info",
                code=503,
                msg="HTTP 503",
                hdrs=None,
                fp=io.BytesIO(b"service unavailable"),
            )
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)

        env = _envelope(lines)
        assert env["error"]["code"] == "object_info_unavailable"
        assert env["error"]["details"]["status"] == 503
        assert "service unavailable" in env["error"]["details"]["body"]

    def test_object_info_connection_error_on_urlerror(self, ui_workflow_file, capsys):
        """URLError on /object_info → connection_error (NOT object_info_unavailable)."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            mock_open.side_effect = urllib.error.URLError("connection refused")
            lines, exit_code = _run_execute_capture(ui_workflow_file, capsys)

        assert _envelope(lines)["error"]["code"] == "connection_error"


class TestNodeCachedIntegration:
    """`execution_cached` WS message → execution_cached events with class_type / title."""

    def test_execution_cached_event_shape(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "execution_cached", "data": {"prompt_id": "p", "nodes": ["1", "2"]}}),
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys)

        cached_events = [e for e in _events(lines) if e["type"] == "execution_cached"]
        assert len(cached_events) == 2
        # Node 1 has _meta.title="Latent"; class_type=EmptyLatentImage
        n1 = next(e for e in cached_events if e["node"] == "1")
        assert n1["class_type"] == "EmptyLatentImage"
        assert n1["title"] == "Latent"
        # Node 2 has _meta.title="Save"; class_type=SaveImage
        n2 = next(e for e in cached_events if e["node"] == "2")
        assert n2["class_type"] == "SaveImage"
        assert n2["title"] == "Save"

        # All cached nodes also appear in the envelope's cached_node_ids.
        env = _envelope(lines)
        assert env["ok"] is True
        assert set(env["data"]["cached_node_ids"]) == {"1", "2"}


class TestNodeExecutedFiresEvenWithoutOutputs:
    """`executed` must fire whenever the server emits `executed` for our
    prompt, even when there's no `output` dict or it's empty (outputs=[])."""

    def _exec(self, simple_workflow):
        return _make_workflow_execution(simple_workflow, with_progress=True)

    def test_output_node_id_coerced_to_str(self, simple_workflow, capsys):
        # If the server ever sends `node` as an int, every emit site coerces —
        # outputs[i].node_id must too, since the contract says node_id is str.
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": 2,
                "output": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]},
            }
        )
        events = _parse_lines(capsys.readouterr().out)
        executed = next(e for e in events if e["type"] == "executed")
        assert isinstance(executed["node"], str)
        assert executed["outputs"]
        for out in executed["outputs"]:
            assert isinstance(out["node_id"], str), (
                f"outputs[i].node_id leaked non-str: {type(out['node_id']).__name__}"
            )
            assert out["node_id"] == "2"

    def test_executed_with_missing_output(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2"})  # no `output` key at all
        events = _parse_lines(capsys.readouterr().out)
        executed = [e for e in events if e["type"] == "executed"]
        assert len(executed) == 1
        assert executed[0]["outputs"] == []
        assert executed[0]["node"] == "2"

    def test_executed_with_non_dict_output(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2", "output": []})  # list instead of dict
        events = _parse_lines(capsys.readouterr().out)
        executed = [e for e in events if e["type"] == "executed"]
        assert len(executed) == 1
        assert executed[0]["outputs"] == []

    def test_executed_with_empty_dict_output(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2", "output": {}})
        events = _parse_lines(capsys.readouterr().out)
        executed = [e for e in events if e["type"] == "executed"]
        assert len(executed) == 1
        assert executed[0]["outputs"] == []


class TestFormatImagePathDefensive:
    """`format_image_path` must be defensive against missing `type` / `subfolder`
    keys — the duck-type filter only requires `filename`."""

    def _exec(self, simple_workflow):
        return _make_workflow_execution(simple_workflow, with_progress=True)

    def test_no_keyerror_on_missing_type(self, simple_workflow):
        ex = self._exec(simple_workflow)
        # Should not raise — `type` missing, should default to "output"
        url = ex.format_image_path({"filename": "x.png", "subfolder": ""})
        assert "filename=x.png" in url
        assert "type=output" in url

    def test_no_keyerror_on_missing_subfolder(self, simple_workflow):
        ex = self._exec(simple_workflow)
        url = ex.format_image_path({"filename": "x.png", "type": "output"})
        assert "filename=x.png" in url


class TestVerboseNoOpInJsonMode:
    """`--verbose` has no effect in stream mode. Regression against a bug
    where `log_node()` printed Rich-formatted lines to stdout when
    verbose=True, corrupting the NDJSON stream."""

    def test_verbose_does_not_corrupt_json_stream(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "execution_cached", "data": {"prompt_id": "p", "nodes": ["1"]}}),
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "2"}}),
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            try:
                execute(
                    workflow_file,
                    host="127.0.0.1",
                    port=8188,
                    wait=True,
                    verbose=True,
                    timeout=30,
                )
            except typer.Exit:
                pass
            out, _err = capsys.readouterr()
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Any Rich-formatted leak would make json.loads raise on a bare
            # "Cached : ..." line.
            json.loads(line)


class TestErrorPathCoverage:
    """Less-trodden paths: /object_info timeout/non-JSON, queue()
    TimeoutError/OSError, on_executed/on_progress None guards, on_cached
    None entries, two consecutive executing pattern."""

    def _make_workflow(self):
        return {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 64, "height": 64, "batch_size": 1},
                "_meta": {"title": "Latent"},
            },
            "2": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "x", "images": ["1", 0]},
            },
        }

    def _make_exec(self, workflow):
        return _make_workflow_execution(workflow)

    def test_object_info_timeout_routes_to_connection_error(self, capsys):
        """fetch_object_info(timeout → connection_error)."""
        ui_wf = {
            "nodes": [
                {
                    "id": 1,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [64, 64, 1],
                    "mode": 0,
                }
            ],
            "links": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ui_wf, f)
            path = f.name
        try:
            with (
                patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
                patch("comfy_cli.command.run.request.urlopen", side_effect=TimeoutError("timed out")),
            ):
                lines, exit_code = _run_execute_capture(path, capsys)
            assert _envelope(lines)["error"]["code"] == "connection_error"
        finally:
            os.unlink(path)

    def test_object_info_non_json_body_routes_to_object_info_unavailable(self, capsys):
        """fetch_object_info(200 + non-JSON body → object_info_unavailable status=200)."""
        ui_wf = {
            "nodes": [
                {
                    "id": 1,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": [64, 64, 1],
                    "mode": 0,
                }
            ],
            "links": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ui_wf, f)
            path = f.name
        try:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"<html>not json</html>"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            with (
                patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
                patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp),
            ):
                lines, exit_code = _run_execute_capture(path, capsys)
            env = _envelope(lines)
            assert env["error"]["code"] == "object_info_unavailable"
            assert env["error"]["details"]["status"] == 200
        finally:
            os.unlink(path)

    def test_queue_timeout_error_routes_to_connection_error(self, workflow_file, capsys):
        """queue()'s urlopen TimeoutError → connection_error."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", side_effect=TimeoutError("post timed out")),
            patch("comfy_cli.command.run.WebSocket"),
        ):
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        assert _envelope(lines)["error"]["code"] == "connection_error"

    def test_queue_oserror_routes_to_connection_error(self, workflow_file, capsys):
        """queue()'s urlopen OSError → connection_error."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", side_effect=OSError("network unreachable")),
            patch("comfy_cli.command.run.WebSocket"),
        ):
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        assert _envelope(lines)["error"]["code"] == "connection_error"

    def test_on_executed_none_node_id_does_not_emit(self, capsys):
        """If server emits `executed` without `node`, skip rather than emit
        a malformed event with node=null."""
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        # Missing "node" key entirely
        ex.on_executed({"output": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}})
        # Explicit None
        ex.on_executed({"node": None})
        out, _ = capsys.readouterr()
        # No events emitted because we skipped pathological frames
        assert out.strip() == "", f"unexpected output for None node: {out!r}"

    def test_on_progress_none_node_id_does_not_emit(self, capsys):
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        ex.on_progress({"value": 1, "max": 10})  # missing node
        ex.on_progress({"node": None, "value": 2, "max": 10})
        out, _ = capsys.readouterr()
        assert out.strip() == ""

    def test_on_progress_emits_progress_event(self, capsys):
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        ex.on_progress({"node": "1", "value": 5, "max": 10})
        events = _parse_lines(capsys.readouterr().out)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "progress"
        assert ev["node"] == "1"
        assert ev["completed"] == 5
        assert ev["total"] == 10
        assert ev["prompt_id"] == "p"

    @pytest.mark.parametrize("malformed", [None, 42, "string", [1, 2, 3], True])
    def test_on_message_skips_non_dict_payloads(self, capsys, malformed):
        # A bad JSON frame (scalar, array, etc.) must not raise out of the
        # recv loop — that would tear down the run without a terminal
        # envelope and break the stream contract.
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        assert ex.on_message(malformed) is True
        out, _err = capsys.readouterr()
        assert out == ""

    def test_on_message_skips_when_data_is_not_dict(self, capsys):
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        # message is a dict but `data` is the wrong shape.
        assert ex.on_message({"type": "executing", "data": "not a dict"}) is True
        assert ex.on_message({"type": "executing", "data": [1, 2, 3]}) is True
        assert ex.on_message({"type": "executing", "data": None}) is True
        out, _err = capsys.readouterr()
        assert out == ""

    def test_on_executing_skips_when_node_key_missing(self, capsys):
        # Missing `node` key is a protocol violation; we skip rather than
        # treating it as None (which means "execution done").
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        assert ex.on_executing({"prompt_id": "p"}) is True

    def test_on_cached_skips_none_entries(self, capsys):
        wf = self._make_workflow()
        ex = self._make_exec(wf)
        ex.prompt_id = "p"
        ex.on_cached({"nodes": ["1", None, "2"]})
        events = _parse_lines(capsys.readouterr().out)
        assert len(events) == 2
        assert {ev["node"] for ev in events} == {"1", "2"}

    def test_two_consecutive_executing_includes_intermediate(self, workflow_file, capsys):
        """`executed_node_ids` is the union of nodes that emitted `executing`
        OR `executed` — intermediate compute nodes that only fire `executing`
        are still included so consumers see the complete 'what ran' picture."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
                # node 2 starts without an executed for 1 — intermediate compute node
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "2"}}),
                json.dumps(
                    {
                        "type": "executed",
                        "data": {
                            "prompt_id": "p",
                            "node": "2",
                            "output": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]},
                        },
                    }
                ),
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            lines, exit_code = _run_execute_capture(workflow_file, capsys)
        env = _envelope(lines)
        assert env["ok"] is True
        # Both nodes ran; both should appear in executed_node_ids
        # (1 via executing only, 2 via both events with dedup)
        assert set(env["data"]["executed_node_ids"]) == {"1", "2"}
        # And node 2 should only appear once (dedup verified)
        assert env["data"]["executed_node_ids"].count("2") == 1


class TestTimeoutAppliesToConnectAndPost:
    """`--timeout` must bound every blocking network call (ws.connect, /prompt
    POST, ws.recv) so the terminal-envelope guarantee holds under server hangs."""

    def test_queue_passes_timeout_to_urlopen(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            # Single executing(node=None) → on_executing returns False → loop exits
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            try:
                execute(
                    workflow_file,
                    host="127.0.0.1",
                    port=8188,
                    wait=True,
                    verbose=False,
                    timeout=42,
                )
            except typer.Exit:
                pass
            _ = capsys.readouterr()
        # Verify urlopen was called with timeout=42
        assert mock_open.called
        call = mock_open.call_args
        timeout_arg = call.kwargs.get("timeout")
        if timeout_arg is None and len(call.args) >= 2:
            timeout_arg = call.args[1]
        assert timeout_arg == 42, f"urlopen not called with timeout=42, got {timeout_arg!r}"

    def test_preflight_probe_passes_timeout(self, workflow_file, capsys):
        # Pre-flight probe gets the same --timeout as everything else,
        # otherwise a slow-to-respond ComfyUI would be falsely reported
        # "not running" by the probe's default 5s.
        with patch("comfy_cli.command.run.check_comfy_server_running", return_value=False) as mock_probe:
            try:
                execute(
                    workflow_file,
                    host="127.0.0.1",
                    port=8188,
                    timeout=55,
                )
            except typer.Exit:
                pass
            _ = capsys.readouterr()
        assert mock_probe.called
        call = mock_probe.call_args
        timeout_arg = call.kwargs.get("timeout")
        if timeout_arg is None and len(call.args) >= 3:
            timeout_arg = call.args[2]
        assert timeout_arg == 55, f"check_comfy_server_running not called with timeout=55, got {timeout_arg!r}"

    def test_connect_passes_timeout_to_ws_connect(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket") as MockWs,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ws_instance = MagicMock()
            MockWs.return_value = ws_instance
            ws_instance.recv.side_effect = [
                json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}}),
            ]
            try:
                execute(
                    workflow_file,
                    host="127.0.0.1",
                    port=8188,
                    wait=True,
                    verbose=False,
                    timeout=37,
                )
            except typer.Exit:
                pass
            _ = capsys.readouterr()
        # Verify ws.connect was called with timeout=37
        assert ws_instance.connect.called
        connect_call = ws_instance.connect.call_args
        timeout_arg = connect_call.kwargs.get("timeout")
        if timeout_arg is None and len(connect_call.args) >= 2:
            timeout_arg = connect_call.args[1]
        assert timeout_arg == 37, f"ws.connect not called with timeout=37, got {timeout_arg!r}"


class TestNoWaitQueueErrorRegression:
    """--no-wait + queue HTTPError must not crash on the progress-stop path
    (progress is None in --no-wait mode)."""

    def test_no_wait_with_400_emits_prompt_rejected(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            body = json.dumps(
                {
                    "error": {"type": "x", "message": "y"},
                    "node_errors": {"1": {"errors": [{"type": "z", "message": "bad"}], "class_type": "X"}},
                }
            ).encode()
            mock_open.side_effect = _make_http_error(400, body)
            lines, exit_code = _run_execute_capture(workflow_file, capsys, wait=False)
        assert _envelope(lines)["error"]["code"] == "prompt_rejected"
        # The big invariant: it didn't crash with AttributeError on `progress.stop()`


def test_on_executed_emits_output_event():
    from comfy_cli.command.run.execution import WorkflowExecution

    ex = WorkflowExecution(
        workflow={"9": {}},
        host="127.0.0.1",
        port=8188,
        verbose=False,
        progress=None,
        local_paths=None,
        timeout=5,
    )
    events = []
    ex.renderer = MagicMock()
    ex.renderer.event.side_effect = lambda typ, **kw: events.append((typ, kw))
    ex.prompt_id = "p1"

    ex.on_executed({"node": "9", "output": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}})

    output_events = [e for e in events if e[0] == "output"]
    assert len(output_events) == 1, events
    assert "a.png" in output_events[0][1]["url"]
    # And outputs are still recorded exactly once for the state file.
    assert sum(1 for u in ex.outputs if "a.png" in u) == 1


def test_on_executed_records_node_keyed_output_entries():
    """Local parity with the cloud history record: the execution keeps a
    node-keyed `output_entries` list alongside the flat `outputs` URLs so
    `run --wait` can group local outputs by node / foreach item."""
    from comfy_cli.command.run.execution import WorkflowExecution

    ex = WorkflowExecution(
        workflow={"9": {}, "12": {}},
        host="127.0.0.1",
        port=8188,
        verbose=False,
        progress=None,
        local_paths=None,
        timeout=5,
    )
    ex.renderer = MagicMock()
    ex.prompt_id = "p1"

    ex.on_executed({"node": "9", "output": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}})
    ex.on_executed({"node": "12", "output": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]}})
    # Duplicate frame: outputs and entries both stay deduped.
    ex.on_executed({"node": "9", "output": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}})

    assert [e["node_id"] for e in ex.output_entries] == ["9", "12"]
    assert ex.output_entries[0]["filename"] == "a.png"
    assert ex.output_entries[0]["type"] == "output"
    # Entry URLs are the same values recorded in the flat list (1:1, in order).
    assert [e["url"] for e in ex.output_entries] == ex.outputs


def test_cloud_route_load_failure_emits_envelope(tmp_path, capsys):
    """execute_cloud must emit a workflow_not_found envelope, not crash on e.code."""
    import json

    import pytest
    import typer

    from comfy_cli.command.run import execute_cloud

    with pytest.raises(typer.Exit) as exc:
        execute_cloud(str(tmp_path / "missing.json"), wait=True, timeout=5)
    assert exc.value.exit_code == 1
    out, _ = capsys.readouterr()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    env = json.loads(lines[-1])
    assert env["ok"] is False and env["error"]["code"] == "workflow_not_found"
