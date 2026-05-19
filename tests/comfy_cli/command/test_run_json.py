"""Unit tests for `comfy run --json` (NDJSON output mode).

See `docs/json-output.md` for the contract these tests pin in place.
The tests cover:
  - every event type emitted at the right time and shape
  - every error.kind for each documented failure path
  - schema_version: 1 on every line
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
    JsonEmitter,
    WorkflowExecution,
    _classify_api_workflow,
    execute,
)


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


def _run_execute_capture(workflow_path, capsys, **overrides):
    """Run execute() and return the parsed JSON events from stdout."""
    kwargs = dict(
        host="127.0.0.1",
        port=8188,
        wait=True,
        verbose=False,
        timeout=30,
        json_mode=True,
    )
    kwargs.update(overrides)
    try:
        execute(workflow_path, **kwargs)
    except typer.Exit:
        pass
    out, _err = capsys.readouterr()
    events = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _make_http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8188/prompt",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def _make_workflow_execution(workflow, *, with_progress: bool = False, json_mode: bool = True):
    """Build a `WorkflowExecution` with a `JsonEmitter` pre-wired to the
    workflow. `with_progress=True` attaches a MagicMock progress object —
    needed by tests that exercise `update_overall_progress`."""
    e = JsonEmitter(json_mode=json_mode)
    e.set_workflow(workflow)
    progress = None
    if with_progress:
        progress = MagicMock()
        progress.add_task.return_value = 0
    return WorkflowExecution(
        workflow=workflow,
        host="127.0.0.1",
        port=8188,
        verbose=False,
        progress=progress,
        timeout=30,
        emitter=e,
    )


class TestJsonEmitter:
    """Direct emitter tests — verify event shape, schema_version, no-op in non-JSON mode."""

    def test_noop_in_human_mode(self, capsys):
        e = JsonEmitter(json_mode=False)
        e.set_client_id("cid")
        e.emit_queued("pid", None)
        e.emit_completed()
        e.emit_failed("workflow_not_found", "x")
        out, _ = capsys.readouterr()
        assert out == ""

    def test_every_event_has_schema_version_1(self, capsys, simple_workflow):
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("cid")
        e.emit_converted(2)
        e.emit_queued("pid", None)
        e.emit_node_cached("1")
        e.emit_node_executing("2")
        e.emit_node_progress("2", 5, 10)
        e.emit_node_executed("2", [])
        e.emit_completed()
        e.emit_failed("execution_error", "x", node_id="2")
        out, _ = capsys.readouterr()
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 8
        for line in lines:
            event = json.loads(line)
            assert event.get("schema_version") == 1, f"Missing schema_version on: {line}"

    def test_queued_includes_validation_warnings_empty(self, capsys):
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        e.emit_queued("p", [])
        out, _ = capsys.readouterr()
        event = json.loads(out.strip())
        assert event["event"] == "queued"
        assert event["validation_warnings"] == []
        assert event["prompt_id"] == "p"
        assert event["client_id"] == "c"
        # nodes manifest is always present (empty when no workflow set)
        assert event["nodes"] == []

    def test_queued_includes_validation_warnings_list(self, capsys):
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        warnings = [{"node_id": "5", "errors": [{"type": "x", "message": "y"}]}]
        e.emit_queued("p", warnings)
        out, _ = capsys.readouterr()
        event = json.loads(out.strip())
        assert event["validation_warnings"] == warnings

    def test_queued_nodes_manifest_from_workflow(self, capsys, simple_workflow):
        """`nodes` should list one entry per workflow node with node_id, class_type, title."""
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("c")
        e.emit_queued("p", None)
        event = json.loads(capsys.readouterr().out.strip())
        nodes = event["nodes"]
        assert len(nodes) == 2
        by_id = {n["node_id"]: n for n in nodes}
        assert by_id["1"]["class_type"] == "EmptyLatentImage"
        assert by_id["1"]["title"] == "Latent"  # _meta.title wins
        assert by_id["2"]["class_type"] == "SaveImage"
        assert by_id["2"]["title"] == "Save"

    def test_node_progress_includes_class_type_and_title(self, capsys, simple_workflow):
        """node_progress carries class_type+title so stateless consumers can
        render the running node without buffering a prior node_executing event."""
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("c")
        e.emit_node_progress("1", 5, 10)
        event = json.loads(capsys.readouterr().out.strip())
        assert event["event"] == "node_progress"
        assert event["class_type"] == "EmptyLatentImage"
        assert event["title"] == "Latent"
        assert event["value"] == 5
        assert event["max"] == 10

    def test_emit_node_handlers_coerce_node_id_to_str(self, capsys, simple_workflow):
        """If the server ever sends an int node_id, emit_* must coerce to str."""
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("c")
        e.emit_node_executing(2)
        e.emit_node_progress(2, 1, 10)
        e.emit_node_cached(2)
        e.emit_node_executed(2, [])
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        for ev in events:
            assert isinstance(ev["node_id"], str), f"{ev['event']} node_id is {type(ev['node_id']).__name__}"
            assert ev["node_id"] == "2"
        e.emit_completed()
        completed = json.loads(capsys.readouterr().out.strip())
        assert all(isinstance(nid, str) for nid in completed["cached_node_ids"])
        assert all(isinstance(nid, str) for nid in completed["executed_node_ids"])

    def test_completed_aggregates_outputs_and_node_ids(self, capsys, simple_workflow):
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("c")
        e.emit_node_cached("1")
        out1 = {
            "category": "images",
            "node_id": "2",
            "class_type": "SaveImage",
            "title": "Save",
            "filename": "x.png",
            "subfolder": "",
            "type": "output",
            "url": "http://x",
        }
        e.emit_node_executed("2", [out1])
        e.emit_completed()
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        completed = events[-1]
        assert completed["event"] == "completed"
        assert completed["cached_node_ids"] == ["1"]
        assert completed["executed_node_ids"] == ["2"]
        assert completed["outputs"] == [out1]
        assert isinstance(completed["elapsed_seconds"], float)
        assert completed["elapsed_seconds"] >= 0

    def test_cached_and_executed_can_overlap(self, capsys, simple_workflow):
        """Cached output-bearing nodes emit both execution_cached and executed."""
        e = JsonEmitter(json_mode=True)
        e.set_workflow(simple_workflow)
        e.set_client_id("c")
        e.emit_node_cached("2")
        e.emit_node_executed("2", [])
        e.emit_completed()
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        completed = events[-1]
        assert "2" in completed["cached_node_ids"]
        assert "2" in completed["executed_node_ids"]

    def test_failed_event_carries_universal_and_extras(self, capsys):
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        e.emit_failed("client_error", "Bad request", status_code=401, body="unauthorized")
        event = json.loads(capsys.readouterr().out.strip())
        assert event["event"] == "failed"
        assert event["error"]["kind"] == "client_error"
        assert event["error"]["message"] == "Bad request"
        assert event["error"]["status_code"] == 401
        assert event["error"]["body"] == "unauthorized"
        assert event["client_id"] == "c"
        assert event["prompt_id"] is None  # never set
        assert isinstance(event["elapsed_seconds"], float)

    def test_fail_helper_emits_event_and_returns_exit(self, capsys):
        # JSON mode: the helper emits a `failed` event, returns a typer.Exit
        # (not raised — caller raises so `from e` chaining is clean), and
        # does NOT print prose (stdout stays NDJSON-only).
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        result = e.fail("client_error", "Bad request", status_code=403, body="forbidden")
        assert isinstance(result, typer.Exit)
        assert result.exit_code == 1
        out = capsys.readouterr().out
        event = json.loads(out.strip())
        assert event["event"] == "failed"
        assert event["error"]["kind"] == "client_error"
        assert event["error"]["message"] == "Bad request"
        assert event["error"]["status_code"] == 403

    def test_fail_helper_wraps_text_mode_message_in_bold_red(self, capsys):
        # Non-JSON mode: the helper auto-wraps `message` in
        # [bold red]...[/bold red] and returns typer.Exit (no event on stdout).
        e = JsonEmitter(json_mode=False)
        result = e.fail("client_error", "Bad request")
        assert isinstance(result, typer.Exit)
        out = capsys.readouterr().out
        assert "Bad request" in out
        # Rich strips markup tags but still applies the formatting; the
        # message content must reach stdout. No NDJSON in text mode.
        assert "failed" not in out  # no event was emitted

    def test_fail_helper_rich_message_overrides_text_only(self, capsys):
        # `rich_message` replaces the auto-wrapped text; JSON event still
        # carries the original `message`.
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        e.fail("client_error", "machine-readable", rich_message="human-friendly")
        event = json.loads(capsys.readouterr().out.strip())
        assert event["error"]["message"] == "machine-readable"
        # In text mode it'd flip — verify separately.
        e2 = JsonEmitter(json_mode=False)
        e2.fail("client_error", "machine-readable", rich_message="human-friendly")
        out = capsys.readouterr().out
        assert "human-friendly" in out
        assert "machine-readable" not in out

    def test_title_falls_back_to_class_type(self, simple_workflow):
        e = JsonEmitter(json_mode=True)
        # Drop _meta.title from node 1
        wf = {"1": {"class_type": "EmptyLatentImage", "inputs": {}}}
        e.set_workflow(wf)
        assert e.get_title("1") == "EmptyLatentImage"

    def test_title_falls_back_to_node_id_for_unknown(self):
        e = JsonEmitter(json_mode=True)
        e.set_workflow({})
        assert e.get_title("unknown") == "unknown"

    def test_ascii_safe_emission(self, capsys):
        """ensure_ascii=True: non-ASCII becomes \\u escapes."""
        e = JsonEmitter(json_mode=True)
        e.set_client_id("c")
        e.emit_failed("workflow_not_found", "found: 猫_00001_.png")
        out, _ = capsys.readouterr()
        # The wire must contain \u escapes, not raw UTF-8 bytes.
        assert "\\u732b" in out
        assert "猫" not in out


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
    """Single `failed` event, prompt_id=null, client_id=null."""

    def test_workflow_not_found(self, capsys):
        events = _run_execute_capture("/nonexistent.json", capsys)
        assert len(events) == 1
        assert events[0]["event"] == "failed"
        assert events[0]["error"]["kind"] == "workflow_not_found"
        assert events[0]["prompt_id"] is None
        assert events[0]["client_id"] is None
        assert events[0]["schema_version"] == 1

    def test_workflow_invalid_json(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ this is not json")
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                events = _run_execute_capture(path, capsys)
            assert len(events) == 1
            assert events[0]["error"]["kind"] == "workflow_invalid_json"
        finally:
            os.unlink(path)

    def test_workflow_read_error_unicode(self, capsys):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as f:
            f.write(b"\xff\xfe\xfa\x00")  # invalid UTF-8
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                events = _run_execute_capture(path, capsys)
            assert len(events) == 1
            assert events[0]["error"]["kind"] == "workflow_read_error"
        finally:
            os.unlink(path)

    def test_workflow_empty_api(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                events = _run_execute_capture(path, capsys)
            assert len(events) == 1
            assert events[0]["error"]["kind"] == "workflow_empty"
        finally:
            os.unlink(path)

    def test_workflow_format_invalid(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"foo": "bar"}, f)
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                events = _run_execute_capture(path, capsys)
            assert len(events) == 1
            assert events[0]["error"]["kind"] == "workflow_format_invalid"
        finally:
            os.unlink(path)

    def test_connection_error_server_down(self, workflow_file, capsys):
        with patch("comfy_cli.command.run.check_comfy_server_running", return_value=False):
            events = _run_execute_capture(workflow_file, capsys)
        assert len(events) == 1
        assert events[0]["error"]["kind"] == "connection_error"


class TestSuccessfulRun:
    def test_no_wait_emits_prompt_preview_then_queued(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "p123"}).encode()
            events = _run_execute_capture(workflow_file, capsys, wait=False)
        # prompt_preview is always emitted in --json before queued so agents
        # have a full audit trail of the submitted workflow graph.
        assert [e["event"] for e in events] == ["prompt_preview", "queued"]
        assert events[0]["prompt"]
        assert events[1]["prompt_id"] == "p123"
        assert events[1]["validation_warnings"] == []

    def test_completed_event_after_success(self, workflow_file, capsys):
        """Mocked WS flow → expect queued + node_* + completed."""
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
            events = _run_execute_capture(workflow_file, capsys, wait=True)

        terminal = events[-1]
        assert terminal["event"] == "completed"
        assert terminal["prompt_id"] == "p"
        assert len(terminal["outputs"]) == 1
        assert terminal["outputs"][0]["filename"] == "x.png"
        assert terminal["outputs"][0]["category"] == "images"
        assert terminal["executed_node_ids"] == ["1"]


class TestQueueHttpErrors:
    """Verify the 5-way HTTP error mapping for /prompt failures."""

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

    def test_400_with_node_errors_routes_to_validation_error(self, workflow_file, capsys):
        body = json.dumps(
            {
                "error": {"type": "x", "message": "y"},
                "node_errors": {"1": {"errors": [{"type": "z", "message": "bad"}], "class_type": "X"}},
            }
        ).encode()
        events = self._setup_and_run(workflow_file, None, capsys, status=400, body=body)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "validation_error"
        node_errors = terminal["error"]["node_errors"]
        assert isinstance(node_errors, list)
        assert any(rec["node_id"] == "1" for rec in node_errors)

    @pytest.mark.parametrize(
        "status,body,kind",
        [
            (401, b"unauthorized", "client_error"),
            (403, b"forbidden", "client_error"),
            (429, b"too many", "client_error"),
            (500, b"oops", "server_error"),
            (503, b"down", "server_error"),
        ],
    )
    def test_http_status_routes_to_kind(self, workflow_file, capsys, status, body, kind):
        events = self._setup_and_run(workflow_file, None, capsys, status=status, body=body)
        terminal = events[-1]
        assert terminal["error"]["kind"] == kind
        assert terminal["error"]["status_code"] == status
        assert terminal["error"]["body"] == body.decode()

    def test_200_with_non_json_body_routes_to_invalid_response(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            mock_open.return_value.read.return_value = b"<html>garbage</html>"
            events = _run_execute_capture(workflow_file, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "invalid_response"
        assert terminal["error"]["status_code"] == 200

    def test_200_without_prompt_id_routes_to_invalid_response(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            mock_open.return_value.read.return_value = json.dumps({"other": "x"}).encode()
            events = _run_execute_capture(workflow_file, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "invalid_response"

    def test_200_with_utf16_bom_body_routes_to_invalid_response(self, workflow_file, capsys):
        # `json.loads(bytes)` sniffs encoding before parsing — a UTF-16 BOM
        # makes it raise `UnicodeDecodeError`, not `JSONDecodeError`.
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            mock_open.return_value.read.return_value = b"\x00\x01\xff\xfeNOT JSON \x80\x81"
            events = _run_execute_capture(workflow_file, capsys)
        terminal = events[-1]
        assert terminal["event"] == "failed"
        assert terminal["error"]["kind"] == "invalid_response"
        assert terminal["error"]["status_code"] == 200

    def test_url_error_routes_to_connection_error(self, workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
            patch("comfy_cli.command.run.WebSocket"),
        ):
            mock_open.side_effect = urllib.error.URLError("refused")
            events = _run_execute_capture(workflow_file, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "connection_error"

    def test_validation_warnings_on_200_with_partial_node_errors(self, workflow_file, capsys):
        """200 + non-empty node_errors → emit `queued` with validation_warnings populated."""
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
            events = _run_execute_capture(workflow_file, capsys)
        queued = next(e for e in events if e["event"] == "queued")
        warnings = queued["validation_warnings"]
        assert isinstance(warnings, list)
        assert any(rec["node_id"] == "3" for rec in warnings)
        rec = next(rec for rec in warnings if rec["node_id"] == "3")
        assert rec["class_type"] == "X"
        assert rec["errors"][0]["message"] == "skipped"


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
        events = self._run_with_ws_messages(
            workflow_file,
            WebSocketTimeoutException("timed out"),
            capsys,
        )
        terminal = events[-1]
        assert terminal["error"]["kind"] == "timeout"
        assert isinstance(terminal["error"]["timeout_seconds"], float)

    def test_connection_lost_websocket(self, workflow_file, capsys):
        events = self._run_with_ws_messages(
            workflow_file,
            WebSocketException("dropped"),
            capsys,
        )
        terminal = events[-1]
        assert terminal["error"]["kind"] == "connection_lost"

    def test_keyboard_interrupt_emits_execution_interrupted(self, workflow_file, capsys):
        events = self._run_with_ws_messages(
            workflow_file,
            KeyboardInterrupt(),
            capsys,
        )
        terminal = events[-1]
        assert terminal["event"] == "failed"
        assert terminal["error"]["kind"] == "execution_interrupted"

    def test_malformed_frame_is_skipped_run_completes(self, workflow_file, capsys):
        """We silently skip malformed JSON frames mid-stream. A valid
        executing(node=None) frame following the bad one should still
        terminate the run normally with `completed`."""
        events = self._run_with_ws_messages(
            workflow_file,
            ["{not json", json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": None}})],
            capsys,
        )
        # No crash, normal completion path reached.
        terminal = events[-1]
        assert terminal["event"] == "completed"

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
        events = self._run_with_ws_messages(workflow_file, messages, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "execution_error"
        assert terminal["error"]["node_id"] == "1"
        assert terminal["error"]["class_type"] == "EmptyLatentImage"
        assert terminal["error"]["exception_type"] == "RuntimeError"
        assert terminal["error"]["title"] == "Latent"  # from _meta.title
        assert isinstance(terminal["error"]["traceback"], str)
        assert "raise RuntimeError" in terminal["error"]["traceback"]

    def test_execution_error_node_id_coerced_to_str(self, workflow_file, capsys):
        # If ComfyUI ever sends node_id as an int in execution_error (other
        # node_id-bearing events all string-coerce defensively), the
        # contract still requires a string.
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
        events = self._run_with_ws_messages(workflow_file, messages, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "execution_error"
        assert terminal["error"]["node_id"] == "7"
        assert isinstance(terminal["error"]["node_id"], str)

    def test_execution_interrupted(self, workflow_file, capsys):
        messages = [
            json.dumps({"type": "executing", "data": {"prompt_id": "p", "node": "1"}}),
            json.dumps({"type": "execution_interrupted", "data": {"prompt_id": "p"}}),
        ]
        events = self._run_with_ws_messages(workflow_file, messages, capsys)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "execution_interrupted"


class TestOutputObject:
    def _exec(self, simple_workflow):
        return _make_workflow_execution(simple_workflow, with_progress=True)

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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = next(e for e in events if e["event"] == "node_executed")
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = next(e for e in events if e["event"] == "node_executed")
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = next(e for e in events if e["event"] == "node_executed")
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        url = events[-1]["outputs"][0]["url"]
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert events[-1]["outputs"][0]["subfolder"] == ""


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
        events = _run_execute_capture("~/wf.json", capsys, print_prompt=True)
        assert events[0]["event"] == "prompt_preview", events

    def test_tilde_path_to_missing_file_reports_expanded_path(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        events = _run_execute_capture("~/missing.json", capsys, print_prompt=True)
        assert events[0]["event"] == "failed"
        assert events[0]["error"]["kind"] == "workflow_not_found"
        # The error message should name the resolved path so the user can
        # see exactly where we looked.
        assert str(tmp_path) in events[0]["error"]["message"]


class TestCliRunnerIntegration:
    """End-to-end: the typer entry callback chain (consent prompt, decorators,
    config init) must not leak any prose to stdout in JSON mode. Direct
    `execute()` tests bypass this seam; agents on a fresh machine with
    no recorded consent are exactly where the original prompt-corrupts-stream
    bug would have hidden."""

    def _make_workflow_file(self, tmp_path):
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps({"1": {"class_type": "X", "inputs": {}}}))
        return str(wf_path)

    def test_cli_json_print_prompt_emits_clean_ndjson(self, tmp_path):
        # Smoke: default config state, --json --print-prompt → every stdout
        # line is valid JSON with `event` and `schema_version`.
        from typer.testing import CliRunner

        from comfy_cli.cmdline import app

        runner = CliRunner()  # non-TTY by default
        result = runner.invoke(
            app, ["run", "--workflow", self._make_workflow_file(tmp_path), "--json", "--print-prompt"]
        )
        assert result.exit_code == 0, f"stdout={result.stdout!r}\nexc={result.exception!r}"
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines, "expected at least one NDJSON line"
        for line in lines:
            event = json.loads(line)
            assert "event" in event
            assert "schema_version" in event
        # Consent prompt text must not appear.
        assert "Do you agree" not in result.stdout
        assert "improve the application" not in result.stdout

    def test_cli_json_with_fresh_consent_state_stays_clean(self, tmp_path):
        # The exact regression scenario: a fresh machine where consent has
        # never been recorded. The entry callback enables session-only
        # tracking via the non-TTY branch (mocked Mixpanel client so no
        # network), and the resulting stdout must still be clean NDJSON.
        from typer.testing import CliRunner

        from comfy_cli.cmdline import app
        from comfy_cli.config_manager import ConfigManager

        _Cls = ConfigManager.__closure__[0].cell_contents
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        with (
            patch.object(_Cls, "get_config_path", return_value=str(cfg_dir)),
            patch("comfy_cli.tracking.mp") as mock_mp,
        ):
            mock_mp.track.return_value = None
            runner = CliRunner()
            result = runner.invoke(
                app, ["run", "--workflow", self._make_workflow_file(tmp_path), "--json", "--print-prompt"]
            )
        assert result.exit_code == 0, f"stdout={result.stdout!r}\nexc={result.exception!r}"
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            assert "event" in event
        assert "Do you agree" not in result.stdout
        assert "tracking" not in result.stdout.lower()


class TestPromptPreviewAlwaysEmitted:
    """In JSON mode the converted workflow graph is always emitted as a
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
            events = _run_execute_capture(workflow_file, capsys)
        kinds = [e["event"] for e in events]
        assert kinds[0] == "prompt_preview"
        assert "queued" in kinds
        assert kinds.index("prompt_preview") < kinds.index("queued")
        assert events[0]["prompt"]["1"]["class_type"] == "EmptyLatentImage"

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
            events = _run_execute_capture(ui_workflow_file, capsys)
        kinds = [e["event"] for e in events]
        # Ordering: converted, prompt_preview, queued (then node_* / completed).
        c = kinds.index("converted")
        p = kinds.index("prompt_preview")
        q = kinds.index("queued")
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
            events = _run_execute_capture(workflow_file, capsys, api_key="sk-secret")
        preview = next(e for e in events if e["event"] == "prompt_preview")
        prompt = preview["prompt"]
        assert "client_id" not in prompt
        assert "extra_data" not in prompt
        assert "sk-secret" not in json.dumps(prompt)


class TestPrintPrompt:
    """`--print-prompt` returns the would-be `/prompt` body and exits 0
    without POSTing. UI input still needs `/object_info`; API input
    doesn't touch the server at all."""

    def test_api_input_emits_prompt_preview_and_no_other_events(self, workflow_file, capsys):
        # No server probe, no /object_info fetch — API input is printed as-is.
        with (
            patch("comfy_cli.command.run.check_comfy_server_running") as mock_check,
            patch("comfy_cli.command.run.fetch_object_info") as mock_fetch,
            patch("comfy_cli.command.run.request.urlopen") as mock_post,
        ):
            events = _run_execute_capture(workflow_file, capsys, print_prompt=True)
        assert mock_check.call_count == 0
        assert mock_fetch.call_count == 0
        assert mock_post.call_count == 0
        assert len(events) == 1
        assert events[0]["event"] == "prompt_preview"
        assert events[0]["schema_version"] == 1
        assert isinstance(events[0]["prompt"], dict)
        assert "1" in events[0]["prompt"]
        assert events[0]["prompt"]["1"]["class_type"] == "EmptyLatentImage"

    def test_ui_input_emits_converted_then_prompt_preview(self, ui_workflow_file, capsys):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch("comfy_cli.command.run.request.urlopen") as mock_post,
        ):
            events = _run_execute_capture(ui_workflow_file, capsys, print_prompt=True)
        assert mock_post.call_count == 0
        assert [e["event"] for e in events] == ["converted", "prompt_preview"]
        prompt = events[1]["prompt"]
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
            events = _run_execute_capture(ui_workflow_file, capsys, print_prompt=True)
        assert events[-1]["event"] == "failed"
        assert events[-1]["error"]["kind"] == "connection_error"

    def test_api_input_works_with_offline_server(self, workflow_file, capsys):
        # Hard-fail the server probe — the API path must not call it under --print-prompt.
        with patch(
            "comfy_cli.command.run.check_comfy_server_running", side_effect=AssertionError("must not be called")
        ):
            events = _run_execute_capture(workflow_file, capsys, print_prompt=True)
        assert len(events) == 1
        assert events[0]["event"] == "prompt_preview"

    def test_print_prompt_does_not_include_api_key_or_client_id(self, workflow_file, capsys):
        # The prompt_preview body should only carry the workflow graph,
        # not the runtime POST envelope (which would otherwise leak the api_key).
        events = _run_execute_capture(workflow_file, capsys, print_prompt=True, api_key="sk-secret")
        prompt = events[0]["prompt"]
        assert "extra_data" not in prompt
        assert "client_id" not in prompt
        assert "sk-secret" not in json.dumps(prompt)

    def test_print_prompt_text_mode_pretty_prints_json(self, workflow_file, capsys):
        try:
            execute(workflow_file, host="127.0.0.1", port=8188, print_prompt=True, json_mode=False)
        except typer.Exit:
            pass
        out, _err = capsys.readouterr()
        parsed = json.loads(out)
        assert "1" in parsed
        assert parsed["1"]["class_type"] == "EmptyLatentImage"

    def test_print_prompt_does_not_post_when_workflow_invalid(self, capsys):
        # Pre-flight failures (workflow_not_found, workflow_format_invalid)
        # still trigger `failed` and exit 1 under --print-prompt.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a workflow"}, f)
            path = f.name
        try:
            events = _run_execute_capture(path, capsys, print_prompt=True)
            assert events[-1]["event"] == "failed"
            assert events[-1]["error"]["kind"] == "workflow_format_invalid"
        finally:
            os.unlink(path)


class TestConvertedAndConversionErrors:
    """UI-input event path and the conversion_error / conversion_crash kinds."""

    def test_converted_event_for_ui_input(self, ui_workflow_file, capsys):
        """Spec lines 84-98: `converted` is the first event when input is UI format."""
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
            events = _run_execute_capture(ui_workflow_file, capsys)

        assert events[0]["event"] == "converted"
        assert events[0]["schema_version"] == 1
        assert events[0]["node_count"] == 2  # the UI workflow has 2 nodes

    def test_conversion_error_kind(self, ui_workflow_file, capsys):
        """WorkflowConversionError → kind=conversion_error, no extras."""
        from comfy_cli.workflow_to_api import WorkflowConversionError

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=WorkflowConversionError("broken graph"),
            ),
        ):
            events = _run_execute_capture(ui_workflow_file, capsys)

        terminal = events[-1]
        assert terminal["error"]["kind"] == "conversion_error"
        assert terminal["client_id"] is None  # before WorkflowExecution
        assert terminal["prompt_id"] is None

    def test_conversion_crash_kind_with_exception_type(self, ui_workflow_file, capsys):
        """Unexpected converter crash → kind=conversion_crash with exception_type extra."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=OBJECT_INFO),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=KeyError("missing field"),
            ),
        ):
            events = _run_execute_capture(ui_workflow_file, capsys)

        terminal = events[-1]
        assert terminal["error"]["kind"] == "conversion_crash"
        assert terminal["error"]["exception_type"] == "KeyError"
        assert terminal["client_id"] is None
        assert terminal["prompt_id"] is None

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
                events = _run_execute_capture(path, capsys)
            assert events[-1]["error"]["kind"] == "workflow_empty"
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
            events = _run_execute_capture(ui_workflow_file, capsys)

        terminal = events[-1]
        assert terminal["error"]["kind"] == "object_info_unavailable"
        assert terminal["error"]["status_code"] == 503
        assert "service unavailable" in terminal["error"]["body"]
        assert terminal["client_id"] is None  # pre-WorkflowExecution

    def test_object_info_connection_error_on_urlerror(self, ui_workflow_file, capsys):
        """URLError on /object_info → connection_error (NOT object_info_unavailable)."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            mock_open.side_effect = urllib.error.URLError("connection refused")
            events = _run_execute_capture(ui_workflow_file, capsys)

        terminal = events[-1]
        assert terminal["error"]["kind"] == "connection_error"


class TestNodeCachedIntegration:
    """`execution_cached` WS message → node_cached events with class_type / title."""

    def test_node_cached_event_shape(self, workflow_file, capsys):
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
            events = _run_execute_capture(workflow_file, capsys)

        cached_events = [e for e in events if e["event"] == "node_cached"]
        assert len(cached_events) == 2
        # Node 1 has _meta.title="Latent"; class_type=EmptyLatentImage
        n1 = next(e for e in cached_events if e["node_id"] == "1")
        assert n1["class_type"] == "EmptyLatentImage"
        assert n1["title"] == "Latent"
        # Node 2 has _meta.title="Save"; class_type=SaveImage
        n2 = next(e for e in cached_events if e["node_id"] == "2")
        assert n2["class_type"] == "SaveImage"
        assert n2["title"] == "Save"

        # All cached nodes also appear in completed.cached_node_ids
        completed = events[-1]
        assert completed["event"] == "completed"
        assert set(completed["cached_node_ids"]) == {"1", "2"}


class TestNodeExecutedFiresEvenWithoutOutputs:
    """`node_executed` must fire whenever the server emits `executed` for our
    prompt, even when there's no `output` dict or it's empty (outputs=[])."""

    def _exec(self, simple_workflow):
        return _make_workflow_execution(simple_workflow, with_progress=True)

    def test_output_node_id_coerced_to_str(self, simple_workflow, capsys):
        # If the server ever sends `node` as an int, every other emit site
        # coerces — outputs[i].node_id must too, since the contract says
        # node_id is always str.
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed(
            {
                "node": 2,
                "output": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]},
            }
        )
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = next(e for e in events if e["event"] == "node_executed")
        assert isinstance(executed["node_id"], str)
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = [e for e in events if e["event"] == "node_executed"]
        assert len(executed) == 1
        assert executed[0]["outputs"] == []
        assert executed[0]["node_id"] == "2"

    def test_executed_with_non_dict_output(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2", "output": []})  # list instead of dict
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = [e for e in events if e["event"] == "node_executed"]
        assert len(executed) == 1
        assert executed[0]["outputs"] == []

    def test_executed_with_empty_dict_output(self, simple_workflow, capsys):
        ex = self._exec(simple_workflow)
        ex.prompt_id = "p"
        ex.on_executed({"node": "2", "output": {}})
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        executed = [e for e in events if e["event"] == "node_executed"]
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
    """Spec lines 24-25: `--verbose` has no effect in JSON mode. Regression
    against a bug where `log_node()` printed Rich-formatted lines to stdout
    when verbose=True, corrupting the NDJSON stream."""

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
                    json_mode=True,
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
    None entries, two consecutive node_executing pattern."""

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
        """fetch_object_info(timeout → connection_error). Previously untested."""
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
                events = _run_execute_capture(path, capsys)
            assert events[-1]["error"]["kind"] == "connection_error"
        finally:
            os.unlink(path)

    def test_object_info_non_json_body_routes_to_object_info_unavailable(self, capsys):
        """fetch_object_info(200 + non-JSON body → object_info_unavailable status_code=200)."""
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
                events = _run_execute_capture(path, capsys)
            terminal = events[-1]
            assert terminal["error"]["kind"] == "object_info_unavailable"
            assert terminal["error"]["status_code"] == 200
        finally:
            os.unlink(path)

    def test_queue_timeout_error_routes_to_connection_error(self, workflow_file, capsys):
        """queue()'s urlopen TimeoutError → connection_error."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", side_effect=TimeoutError("post timed out")),
            patch("comfy_cli.command.run.WebSocket"),
        ):
            events = _run_execute_capture(workflow_file, capsys)
        assert events[-1]["error"]["kind"] == "connection_error"

    def test_queue_oserror_routes_to_connection_error(self, workflow_file, capsys):
        """queue()'s urlopen OSError → connection_error."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", side_effect=OSError("network unreachable")),
            patch("comfy_cli.command.run.WebSocket"),
        ):
            events = _run_execute_capture(workflow_file, capsys)
        assert events[-1]["error"]["kind"] == "connection_error"

    def test_on_executed_none_node_id_does_not_emit(self, capsys):
        """If server emits `executed` without `node`, skip rather than emit
        a malformed event with node_id=null."""
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

    @pytest.mark.parametrize("malformed", [None, 42, "string", [1, 2, 3], True])
    def test_on_message_skips_non_dict_payloads(self, capsys, malformed):
        # A bad JSON frame (scalar, array, etc.) must not raise out of the
        # recv loop — that would tear down the run without a terminal
        # `failed` event and break the stream contract.
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
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(events) == 2
        assert {ev["node_id"] for ev in events} == {"1", "2"}

    def test_two_consecutive_node_executing_includes_intermediate(self, workflow_file, capsys):
        """`executed_node_ids` is the union of nodes that emitted `node_executing`
        OR `node_executed` — intermediate compute nodes that only fire `executing`
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
                # node 2 starts without a node_executed for 1 — intermediate compute node
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
            events = _run_execute_capture(workflow_file, capsys)
        completed = events[-1]
        assert completed["event"] == "completed"
        # Both nodes ran; both should appear in executed_node_ids
        # (1 via node_executing only, 2 via both events with dedup)
        assert set(completed["executed_node_ids"]) == {"1", "2"}
        # And node 2 should only appear once (dedup verified)
        assert completed["executed_node_ids"].count("2") == 1


class TestTimeoutAppliesToConnectAndPost:
    """`--timeout` must bound every blocking network call (ws.connect, /prompt
    POST, ws.recv) so the terminal-event guarantee holds under server hangs."""

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
                    json_mode=True,
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
                    json_mode=True,
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
                    json_mode=True,
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

    def test_no_wait_with_400_emits_validation_error(self, workflow_file, capsys):
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
            events = _run_execute_capture(workflow_file, capsys, wait=False)
        terminal = events[-1]
        assert terminal["error"]["kind"] == "validation_error"
        # The big invariant: it didn't crash with AttributeError on `progress.stop()`
