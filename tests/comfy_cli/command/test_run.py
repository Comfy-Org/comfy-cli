import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import typer
from websocket import WebSocketException, WebSocketTimeoutException

from comfy_cli.command.run import WorkflowExecution, execute, load_api_workflow


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


class TestLoadApiWorkflow:
    def test_valid_api_workflow(self, workflow_file):
        result = load_api_workflow(workflow_file)
        assert result is not None
        assert "1" in result
        assert result["1"]["class_type"] == "EmptyLatentImage"

    def test_rejects_ui_workflow(self):
        ui_workflow = {"nodes": [], "links": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ui_workflow, f)
            f.flush()
            result = load_api_workflow(f.name)
        os.unlink(f.name)
        assert result is None

    def test_rejects_invalid_node(self):
        bad_workflow = {"1": {"not_class_type": "Foo"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_workflow, f)
            f.flush()
            result = load_api_workflow(f.name)
        os.unlink(f.name)
        assert result is None


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
