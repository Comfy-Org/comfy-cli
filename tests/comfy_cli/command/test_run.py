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
    WorkflowConverterUnavailable,
    WorkflowExecution,
    convert_ui_workflow_via_server,
    execute,
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
        url="http://127.0.0.1:8188/workflow/convert",
        code=code,
        msg=f"HTTP {code}",
        hdrs=None,
        fp=io.BytesIO(body),
    )


class TestConvertUiWorkflowViaServer:
    UI = {"nodes": [{"id": 1, "type": "X"}], "links": []}
    CONVERTED = {"1": {"class_type": "X", "inputs": {}}}

    def test_returns_api_format_on_success(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self.CONVERTED).encode()
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp) as mock_open:
            result = convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)

        assert result == self.CONVERTED
        sent_req = mock_open.call_args[0][0]
        assert sent_req.full_url == "http://127.0.0.1:8188/workflow/convert"
        assert json.loads(sent_req.data) == self.UI

    @pytest.mark.parametrize("code", [404, 405])
    def test_raises_unavailable_on_missing_endpoint(self, code):
        with patch("comfy_cli.command.run.request.urlopen", side_effect=_make_http_error(code)):
            with pytest.raises(WorkflowConverterUnavailable):
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)

    def test_raises_typer_exit_on_server_error(self):
        err = _make_http_error(500, b"conversion blew up")
        with patch("comfy_cli.command.run.request.urlopen", side_effect=err):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_on_network_error(self):
        with patch(
            "comfy_cli.command.run.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>not json</html>"
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_on_non_object_response(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'["not", "an", "object"]'
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_on_empty_object_response(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_when_first_entry_is_not_a_node(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"1": "not-a-node-dict"}'
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1

    def test_raises_typer_exit_when_first_entry_missing_class_type(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"1": {"inputs": {}}}'
        with patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp):
            with pytest.raises(typer.Exit) as exc_info:
                convert_ui_workflow_via_server(self.UI, "127.0.0.1", 8188, timeout=30)
            assert exc_info.value.exit_code == 1


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
    UI = {"nodes": [{"id": 1, "type": "X"}], "links": []}
    CONVERTED = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}}}

    @pytest.fixture
    def ui_workflow_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.UI, f)
            f.flush()
            path = f.name
        yield path
        os.unlink(path)

    def test_ui_workflow_is_converted_then_executed(self, ui_workflow_file):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(self.CONVERTED).encode()

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", return_value=mock_resp) as mock_open,
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []

            execute(ui_workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)

            sent_req = mock_open.call_args[0][0]
            assert sent_req.full_url == "http://127.0.0.1:8188/workflow/convert"
            assert MockExec.call_args.args[0] == self.CONVERTED
            mock_exec.queue.assert_called_once()

    @pytest.mark.parametrize("code", [404, 405])
    def test_ui_workflow_exits_when_endpoint_missing(self, ui_workflow_file, code):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.request.urlopen", side_effect=_make_http_error(code)),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute(ui_workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            assert exc_info.value.exit_code == 1
            MockExec.assert_not_called()

    def test_ui_workflow_exits_when_server_not_running(self, ui_workflow_file):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=False),
            patch("comfy_cli.command.run.request.urlopen") as mock_open,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute(ui_workflow_file, host="127.0.0.1", port=8188)
            assert exc_info.value.exit_code == 1
            mock_open.assert_not_called()
