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
    execute,
    fetch_object_info,
    is_ui_workflow,
)


@pytest.fixture
def workflow():
    return {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1},
            "_meta": {"title": "Empty Latent"},
        },
        "2": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["1", 0]},
            "_meta": {"title": "Preview"},
        },
    }


@pytest.fixture
def workflow_file(workflow):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(workflow, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_execution(workflow):
    progress = MagicMock()
    progress.add_task.return_value = 0
    return WorkflowExecution(
        workflow=workflow,
        host="127.0.0.1",
        port=8188,
        verbose=False,
        progress=progress,
        local_paths=False,
        timeout=30,
    )


def _make_msg(msg_type, prompt_id, **data_fields):
    return json.dumps({"type": msg_type, "data": {"prompt_id": prompt_id, **data_fields}})


class TestIsUiWorkflow:
    def test_detects_ui_workflow(self):
        assert is_ui_workflow({"nodes": [{"id": 1}], "links": []})

    def test_rejects_api_workflow(self):
        assert not is_ui_workflow({"1": {"class_type": "X", "inputs": {}}})

    def test_rejects_non_dict(self):
        assert not is_ui_workflow(["nodes", "links"])
        assert not is_ui_workflow(None)

    def test_requires_both_keys(self):
        assert not is_ui_workflow({"nodes": []})
        assert not is_ui_workflow({"links": []})

    def test_rejects_api_workflow_with_nodes_and_links_as_keys(self):
        # A pathological API workflow where node IDs happen to be the strings
        # "nodes" and "links" should not be mistaken for UI format.
        api = {
            "nodes": {"class_type": "Foo", "inputs": {}},
            "links": {"class_type": "Bar", "inputs": {}},
        }
        assert not is_ui_workflow(api)

    def test_rejects_when_values_are_not_lists(self):
        assert not is_ui_workflow({"nodes": "string", "links": "string"})
        assert not is_ui_workflow({"nodes": 1, "links": 2})


def _make_http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8188/object_info",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def _ok_response(body: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestFetchObjectInfo:
    def test_returns_parsed_json_on_success(self):
        payload = {"KSampler": {"input": {}, "output_node": False}}
        with patch(
            "comfy_cli.command.run.request.urlopen",
            return_value=_ok_response(json.dumps(payload).encode()),
        ) as mock_open:
            result = fetch_object_info("127.0.0.1", 8188, timeout=30)
        assert result == payload
        assert mock_open.call_args[0][0] == "http://127.0.0.1:8188/object_info"

    def test_http_error_exits_cleanly(self):
        with patch(
            "comfy_cli.command.run.request.urlopen",
            side_effect=_make_http_error(500, b"server exploded"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                fetch_object_info("127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_network_error_exits_cleanly(self):
        with patch(
            "comfy_cli.command.run.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                fetch_object_info("127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_timeout_exits_cleanly(self):
        with patch("comfy_cli.command.run.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(typer.Exit) as exc_info:
                fetch_object_info("127.0.0.1", 8188, timeout=5)
            assert exc_info.value.exit_code == 1

    def test_invalid_json_exits_cleanly(self):
        with patch(
            "comfy_cli.command.run.request.urlopen",
            return_value=_ok_response(b"<html>not json</html>"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                fetch_object_info("127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1


class TestWorkflowExecutionAuth:
    """X-API-Key is the credential the ComfyUI server forwards to Partner Nodes."""

    def _make_exec(self, workflow, api_key=None):
        progress = MagicMock()
        progress.add_task.return_value = 0
        return WorkflowExecution(
            workflow=workflow,
            host="127.0.0.1",
            port=8188,
            verbose=False,
            progress=progress,
            local_paths=False,
            timeout=30,
            api_key=api_key,
        )

    def test_queue_embeds_api_key_in_extra_data(self, workflow):
        ex = self._make_exec(workflow, api_key="sk-secret")
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        assert body["extra_data"] == {"api_key_comfy_org": "sk-secret"}

    def test_queue_does_not_send_x_api_key_header(self, workflow):
        ex = self._make_exec(workflow, api_key="sk-secret")
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        assert req.get_header("X-api-key") is None

    def test_queue_omits_extra_data_when_no_api_key(self, workflow):
        ex = self._make_exec(workflow)
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        assert "extra_data" not in body
        assert body == {"prompt": workflow, "client_id": ex.client_id}


class TestWatchExecution:
    def test_successful_execution(self, mock_execution):
        prompt_id = "test-prompt"
        mock_execution.prompt_id = prompt_id

        messages = [
            _make_msg("executing", prompt_id, node="1"),
            _make_msg("executed", prompt_id, node="1"),
            _make_msg("executing", prompt_id, node="2"),
            _make_msg("executed", prompt_id, node="2"),
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        mock_execution.ws = mock_ws

        mock_execution.watch_execution()
        assert len(mock_execution.remaining_nodes) == 0

    def test_skips_other_prompt_messages(self, mock_execution):
        prompt_id = "my-prompt"
        mock_execution.prompt_id = prompt_id

        messages = [
            _make_msg("executing", "other-prompt", node="1"),
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        mock_execution.ws = mock_ws

        mock_execution.watch_execution()
        assert "1" in mock_execution.remaining_nodes

    def test_unknown_node_ids_do_not_crash(self, mock_execution):
        prompt_id = "test-prompt"
        mock_execution.prompt_id = prompt_id

        messages = [
            _make_msg("executing", prompt_id, node="1"),
            _make_msg("executing", prompt_id, node="406.0.0.428"),
            json.dumps(
                {"type": "progress", "data": {"prompt_id": prompt_id, "node": "406.0.0.428", "value": 5, "max": 10}}
            ),
            _make_msg("executed", prompt_id, node="406.0.0.428"),
            json.dumps({"type": "execution_cached", "data": {"prompt_id": prompt_id, "nodes": ["999"]}}),
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        mock_execution.ws = mock_ws

        mock_execution.watch_execution()

    def test_unknown_node_ids_verbose(self, workflow):
        prompt_id = "test-prompt"
        progress = MagicMock()
        progress.add_task.return_value = 0
        execution = WorkflowExecution(
            workflow=workflow,
            host="127.0.0.1",
            port=8188,
            verbose=True,
            progress=progress,
            local_paths=False,
            timeout=30,
        )
        execution.prompt_id = prompt_id

        messages = [
            _make_msg("executing", prompt_id, node="406.0.0.428"),
            json.dumps({"type": "execution_cached", "data": {"prompt_id": prompt_id, "nodes": ["999"]}}),
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        execution.ws = mock_ws

        execution.watch_execution()

    def test_collects_image_outputs(self, mock_execution):
        prompt_id = "test-prompt"
        mock_execution.prompt_id = prompt_id

        executed_msg = json.dumps(
            {
                "type": "executed",
                "data": {
                    "prompt_id": prompt_id,
                    "node": "2",
                    "output": {
                        "images": [{"filename": "result.png", "subfolder": "", "type": "output"}],
                    },
                },
            }
        )
        messages = [
            _make_msg("executing", prompt_id, node="2"),
            executed_msg,
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        mock_execution.ws = mock_ws

        mock_execution.watch_execution()
        assert len(mock_execution.outputs) == 1
        assert "result.png" in mock_execution.outputs[0]


class TestExecuteErrorHandling:
    def _run_execute_expect_exit(self, workflow_file, **overrides):
        kwargs = dict(host="127.0.0.1", port=8188, wait=True, verbose=False, local_paths=False, timeout=30)
        kwargs.update(overrides)
        with pytest.raises(typer.Exit) as exc_info:
            execute(workflow_file, **kwargs)
        return exc_info.value.exit_code

    def test_timeout_exits_with_code_1(self, workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.watch_execution.side_effect = WebSocketTimeoutException("timed out")

            code = self._run_execute_expect_exit(workflow_file)
            assert code == 1

    def test_connection_error_exits_with_code_1(self, workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.connect.side_effect = ConnectionError("Connection refused")

            code = self._run_execute_expect_exit(workflow_file)
            assert code == 1

    def test_websocket_exception_exits_with_code_1(self, workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.watch_execution.side_effect = WebSocketException("Connection lost")

            code = self._run_execute_expect_exit(workflow_file)
            assert code == 1

    def test_successful_execution(self, workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress") as MockProgress,
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_progress = MagicMock()
            MockProgress.return_value = mock_progress
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []

            execute(workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            mock_exec.connect.assert_called_once()
            mock_exec.queue.assert_called_once()
            mock_exec.watch_execution.assert_called_once()

    def test_file_not_found_exits(self):
        with pytest.raises(typer.Exit) as exc_info:
            execute("/nonexistent/workflow.json", host="127.0.0.1", port=8188)
        assert exc_info.value.exit_code == 1

    def test_rejects_invalid_workflow_format(self):
        bad = {"1": {"no_class_type_here": "X"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad, f)
            f.flush()
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                with pytest.raises(typer.Exit) as exc_info:
                    execute(path, host="127.0.0.1", port=8188)
                assert exc_info.value.exit_code == 1
        finally:
            os.unlink(path)

    def test_rejects_malformed_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ this is not valid json")
            f.flush()
            path = f.name
        try:
            with patch("comfy_cli.command.run.check_comfy_server_running", return_value=True):
                with pytest.raises(typer.Exit) as exc_info:
                    execute(path, host="127.0.0.1", port=8188)
                assert exc_info.value.exit_code == 1
        finally:
            os.unlink(path)

    def test_rejects_unreadable_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            path = f.name
        try:
            real_open = open

            def fake_open(file, *args, **kwargs):
                if file == path:
                    raise PermissionError(13, "Permission denied", path)
                return real_open(file, *args, **kwargs)

            with (
                patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
                patch("builtins.open", side_effect=fake_open),
            ):
                with pytest.raises(typer.Exit) as exc_info:
                    execute(path, host="127.0.0.1", port=8188)
                assert exc_info.value.exit_code == 1
        finally:
            os.unlink(path)

    def test_progress_stopped_on_error(self, workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress") as MockProgress,
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_progress = MagicMock()
            MockProgress.return_value = mock_progress
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.watch_execution.side_effect = WebSocketTimeoutException("timed out")

            with pytest.raises(typer.Exit):
                execute(workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            mock_progress.stop.assert_called()


class TestExecuteUiWorkflow:
    UI = {
        "nodes": [
            {
                "id": 1,
                "type": "EmptyLatentImage",
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [10]}],
                "widgets_values": [512, 512, 1],
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
    def ui_workflow_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.UI, f)
            f.flush()
            path = f.name
        yield path
        os.unlink(path)

    def test_ui_workflow_is_converted_then_executed(self, ui_workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=self.OBJECT_INFO) as mock_fetch,
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []

            execute(ui_workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)

            mock_fetch.assert_called_once_with("127.0.0.1", 8188, 30)
            api_workflow = MockExec.call_args.args[0]
            assert set(api_workflow) == {"1", "2"}
            assert api_workflow["1"]["class_type"] == "EmptyLatentImage"
            assert api_workflow["2"]["inputs"]["images"] == ["1", 0]
            mock_exec.queue.assert_called_once()

    def test_ui_workflow_exits_when_server_not_running(self, ui_workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=False),
            patch("comfy_cli.command.run.fetch_object_info") as mock_fetch,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute(ui_workflow_file, host="127.0.0.1", port=8188)
            assert exc_info.value.exit_code == 1
            mock_fetch.assert_not_called()

    def test_ui_workflow_exits_cleanly_on_unexpected_converter_crash(self, ui_workflow_file):
        # If the experimental converter crashes with an unexpected error, the
        # CLI should still exit with code 1 and a friendly message — not let a
        # Python traceback escape to the user.
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=self.OBJECT_INFO),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=RuntimeError("simulated converter bug"),
            ),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute(ui_workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            assert exc_info.value.exit_code == 1
            MockExec.assert_not_called()

    def test_ui_workflow_plumbs_api_key_through_to_execution(self, ui_workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.fetch_object_info", return_value=self.OBJECT_INFO) as mock_fetch,
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []

            execute(ui_workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30, api_key="sk-test")

            mock_fetch.assert_called_once_with("127.0.0.1", 8188, 30)
            assert MockExec.call_args.kwargs["api_key"] == "sk-test"

    def test_ui_workflow_exits_when_conversion_yields_nothing(self):
        # All nodes are UI-only (Note/PrimitiveNode/Reroute/GetNode/SetNode) and
        # therefore stripped by the converter → execute() should bail before
        # ever instantiating WorkflowExecution.
        empty_ui = {
            "nodes": [
                {"id": 1, "type": "Note", "inputs": [], "outputs": [], "widgets_values": ["x"]},
                {"id": 2, "type": "Reroute", "inputs": [{"link": None}], "outputs": [{"links": []}]},
            ],
            "links": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(empty_ui, f)
            f.flush()
            path = f.name
        try:
            with (
                patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
                patch("comfy_cli.command.run.fetch_object_info", return_value=self.OBJECT_INFO),
                patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
            ):
                with pytest.raises(typer.Exit) as exc_info:
                    execute(path, host="127.0.0.1", port=8188, wait=True, timeout=30)
                assert exc_info.value.exit_code == 1
                MockExec.assert_not_called()
        finally:
            os.unlink(path)
