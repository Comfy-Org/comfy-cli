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
    _count_output_nodes,
    _detect_partner_nodes,
    _resolve_partner_credential,
    _returned_output_node_count,
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
        local_paths=False,
        progress=progress,
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
            local_paths=False,
            progress=progress,
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
        assert body["extra_data"] == {"comfy_usage_source": "comfy-cli", "api_key_comfy_org": "sk-secret"}

    def test_queue_does_not_send_x_api_key_header(self, workflow):
        ex = self._make_exec(workflow, api_key="sk-secret")
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        assert req.get_header("X-api-key") is None

    def test_queue_omits_api_key_when_not_set(self, workflow):
        ex = self._make_exec(workflow)
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        body = json.loads(req.data)
        assert body == {
            "prompt": workflow,
            "client_id": ex.client_id,
            "extra_data": {"comfy_usage_source": "comfy-cli"},
        }

    def test_queue_sends_usage_source_header(self, workflow):
        ex = self._make_exec(workflow)
        with patch("comfy_cli.command.run.request.urlopen") as mock_open:
            mock_open.return_value.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
            ex.queue()
        req = mock_open.call_args[0][0]
        assert req.get_header("Comfy-usage-source") == "comfy-cli"


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
            local_paths=False,
            progress=progress,
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

    def test_no_progress_bar_survives_cached_and_executing(self, workflow):
        """In --json mode the renderer passes progress=None; cached + executing events must not NPE."""
        prompt_id = "test-prompt"
        execution = WorkflowExecution(
            workflow=workflow,
            host="127.0.0.1",
            port=8188,
            verbose=False,
            progress=None,
            local_paths=False,
            timeout=30,
        )
        execution.prompt_id = prompt_id

        messages = [
            json.dumps({"type": "execution_cached", "data": {"prompt_id": prompt_id, "nodes": ["1"]}}),
            _make_msg("executing", prompt_id, node="2"),
            _make_msg("executed", prompt_id, node="2"),
            _make_msg("executing", prompt_id, node=None),
        ]
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = messages
        execution.ws = mock_ws

        execution.watch_execution()
        assert len(execution.remaining_nodes) == 0

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
        kwargs = dict(host="127.0.0.1", port=8188, wait=True, verbose=False, timeout=30)
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
            # The run WebSocket must be closed on the success path (BE-3404) —
            # the finally-block _safe_close, not left open until teardown.
            mock_exec.ws.close.assert_called_once()

    def test_websocket_closed_on_watch_failure(self, workflow_file):
        # BE-3404: the finally-block close also fires when watch_execution
        # raises, so a mid-run error doesn't linger the server-side session.
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.watch_execution.side_effect = WebSocketTimeoutException("timed out")

            with pytest.raises(typer.Exit):
                execute(workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            mock_exec.ws.close.assert_called_once()

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


class TestDetectPartnerNodes:
    """Partner-API nodes (category `partner/...` or the authoritative
    `api_node: true` flag) must be detected before a local submit so we can
    refuse early instead of failing opaquely at execute time with
    `Unauthorized: Please login first`."""

    def _info(self, **categories):
        # Build a minimal /object_info-shape dict from class_type → category.
        return {ct: {"category": cat} for ct, cat in categories.items()}

    def test_finds_partner_nodes_in_workflow(self):
        wf = {
            "1": {"class_type": "Veo3VideoGenerationNode", "inputs": {}},
            "2": {"class_type": "SaveVideo", "inputs": {}},
            "3": {"class_type": "KlingImage2VideoNode", "inputs": {}},
        }
        info = self._info(
            Veo3VideoGenerationNode="partner/video/Veo",
            SaveVideo="video",
            KlingImage2VideoNode="partner/video/Kling",
        )
        assert _detect_partner_nodes(wf, info) == [
            "KlingImage2VideoNode",
            "Veo3VideoGenerationNode",
        ]

    def test_returns_empty_when_no_partner_nodes(self):
        wf = {
            "1": {"class_type": "EmptyLatentImage", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        info = self._info(EmptyLatentImage="latent", KSampler="sampling")
        assert _detect_partner_nodes(wf, info) == []

    def test_ignores_unknown_class_types(self):
        """A workflow with a class_type the server doesn't advertise (custom
        node, typo) is not treated as a partner node — we only flag when
        the server explicitly categorizes it under `partner/*`."""
        wf = {"1": {"class_type": "SomeUnknownThing", "inputs": {}}}
        info = self._info(KSampler="sampling")
        assert _detect_partner_nodes(wf, info) == []

    def test_handles_malformed_workflow_entries(self):
        wf = {
            "1": "not-a-dict",
            "2": {"class_type": None, "inputs": {}},
            "3": {"inputs": {}},  # no class_type
            "4": {"class_type": "Veo3VideoGenerationNode", "inputs": {}},
        }
        info = self._info(Veo3VideoGenerationNode="partner/video/Veo")
        assert _detect_partner_nodes(wf, info) == ["Veo3VideoGenerationNode"]

    def test_finds_partner_nodes_with_partner_prefix(self):
        """Current cloud/ComfyUI categorizes partner nodes under `partner/...`
        (e.g. `partner/video/ByteDance`)."""
        wf = {
            "1": {"class_type": "ByteDanceTextToVideoNode", "inputs": {}},
            "2": {"class_type": "SaveVideo", "inputs": {}},
        }
        info = self._info(
            ByteDanceTextToVideoNode="partner/video/ByteDance",
            SaveVideo="video",
        )
        assert _detect_partner_nodes(wf, info) == ["ByteDanceTextToVideoNode"]

    def test_finds_partner_nodes_via_api_node_flag(self):
        """The authoritative signal is `api_node: true`, even if the category
        doesn't match either prefix."""
        wf = {"1": {"class_type": "SomePartnerNode", "inputs": {}}}
        info = {"SomePartnerNode": {"category": "weird/category", "api_node": True}}
        assert _detect_partner_nodes(wf, info) == ["SomePartnerNode"]

    def test_legacy_api_node_prefix_no_longer_detected(self):
        """One name per concept: the legacy `api node/...` category alias is
        dropped — current servers publish `partner/*` and the authoritative
        `api_node: true` flag, and the CLI speaks only those."""
        wf = {"1": {"class_type": "OldServerNode", "inputs": {}}}
        info = self._info(OldServerNode="api node/video/Veo")
        assert _detect_partner_nodes(wf, info) == []


class TestPartialExecutionDiff:
    """The cloud prunes branches that fail server-side validation and still
    reports `completed`. We diff submitted output nodes against returned ones
    so a vanished branch surfaces instead of passing as a clean success."""

    def _info(self):
        return {
            "SaveVideo": {"category": "video", "output_node": True},
            "SaveImage": {"category": "image", "output_node": True},
            "KSampler": {"category": "sampling", "output_node": False},
        }

    def test_counts_output_nodes(self):
        wf = {
            "1": {"class_type": "KSampler", "inputs": {}},
            "2": {"class_type": "SaveVideo", "inputs": {}},
            "3": {"class_type": "SaveImage", "inputs": {}},
        }
        assert _count_output_nodes(wf, self._info()) == 2

    def test_returns_none_when_object_info_empty(self):
        wf = {"2": {"class_type": "SaveVideo", "inputs": {}}}
        assert _count_output_nodes(wf, {}) is None

    def test_returned_output_node_count(self):
        record = {
            "outputs": {
                "2": {"videos": [{"filename": "a.mp4"}]},
                "3": {},  # produced nothing
            }
        }
        assert _returned_output_node_count(record) == 1

    def test_returned_count_handles_missing_outputs(self):
        assert _returned_output_node_count({}) == 0
        assert _returned_output_node_count({"outputs": None}) == 0

    def test_diff_detects_pruned_branch(self):
        wf = {
            "1": {"class_type": "SaveVideo", "inputs": {}},
            "2": {"class_type": "SaveVideo", "inputs": {}},
        }
        record = {"outputs": {"1": {"videos": [{"filename": "a.mp4"}]}}}
        submitted = _count_output_nodes(wf, self._info())
        returned = _returned_output_node_count(record)
        assert submitted == 2
        assert returned == 1
        assert returned < submitted  # the partial-execution warning trigger


class TestResolvePartnerCredential:
    """The credential the local submit can inject into ``extra_data`` so a
    partner-API node finds it. Precedence session > env > stored key.

    The OAuth session is refreshed when possible (``refresh=True``) but never
    cleared from this best-effort path (``allow_clear=False``): access tokens
    are short-lived, so a signed-in user's token routinely lapses between
    commands — refreshing keeps local runs working, without ever logging the
    user off the shared session. The refresh happens inside
    ``oauth.ensure_fresh_session`` (mocked here); its allow_clear semantics are
    exercised end-to-end in ``tests/comfy_cli/test_credentials.py``.
    """

    def _no_session(self, monkeypatch: pytest.MonkeyPatch):
        from comfy_cli.cloud import oauth

        monkeypatch.setattr(oauth, "ensure_fresh_session", lambda **kw: None)

    def test_uses_env_var_when_no_session(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COMFY_CLOUD_API_KEY", "env-key-123")
        from comfy_cli.auth import store as auth_store

        monkeypatch.setattr(auth_store, "get", lambda _: None)
        self._no_session(monkeypatch)
        assert _resolve_partner_credential() == ("api_key_comfy_org", "env-key-123")

    def test_falls_back_to_stored_provider_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store
        from comfy_cli.target import CLOUD_API_KEY_PROVIDER

        record = MagicMock()
        record.key = "stored-key-456"
        monkeypatch.setattr(
            auth_store,
            "get",
            lambda name: record if name == CLOUD_API_KEY_PROVIDER else None,
        )
        self._no_session(monkeypatch)
        assert _resolve_partner_credential() == ("api_key_comfy_org", "stored-key-456")

    def test_refreshes_and_uses_oauth_token(self, monkeypatch: pytest.MonkeyPatch):
        """A signed-in user whose access token lapsed gets a REFRESHED token
        here — the whole point of BE-3361 — rather than being skipped and
        hitting ``partner_node_requires_credential``."""
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store
        from comfy_cli.cloud import oauth

        # ensure_fresh_session refreshes the lapsed token and returns a fresh,
        # non-expired session carrying the NEW access token.
        refreshed = MagicMock()
        refreshed.is_expired.return_value = False
        refreshed.access_token = "refreshed-bearer-789"
        refreshed.base_url = "https://cloud.comfy.org"
        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(oauth, "ensure_fresh_session", lambda **kw: refreshed)
        assert _resolve_partner_credential() == ("auth_token_comfy_org", "refreshed-bearer-789")

    def test_passes_allow_clear_false_to_refresh(self, monkeypatch: pytest.MonkeyPatch):
        """This best-effort injector must NEVER clear the shared session on a
        fatal refresh: it refreshes with ``allow_clear=False``."""
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store
        from comfy_cli.cloud import oauth

        seen: dict = {}

        def _refresh(**kw):
            seen.update(kw)
            return None

        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(oauth, "ensure_fresh_session", _refresh)
        _resolve_partner_credential()
        assert seen.get("allow_clear") is False

    def test_returns_none_when_nothing_configured(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store

        monkeypatch.setattr(auth_store, "get", lambda _: None)
        self._no_session(monkeypatch)
        assert _resolve_partner_credential() is None

    def test_stale_session_from_transient_failure_falls_through(self, monkeypatch: pytest.MonkeyPatch):
        """A transient refresh failure returns the STALE (expired) session; it
        fails its own expiry check and the resolver falls through — unchanged
        from the pre-BE-3361 behavior on a network flake."""
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store
        from comfy_cli.cloud import oauth

        stale = MagicMock()
        stale.is_expired.return_value = True
        stale.access_token = "stale"
        stale.base_url = "https://cloud.comfy.org"
        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(oauth, "ensure_fresh_session", lambda **kw: stale)
        assert _resolve_partner_credential() is None

    def test_refresh_path_error_falls_through_to_env_key(self, monkeypatch: pytest.MonkeyPatch):
        """The refresh leg does network + file-locked persist; ``ensure_fresh_session``
        only swallows transient/timeout cases, so an unexpected ``OSError`` (lock
        acquire / token persist) would otherwise abort the run. This best-effort
        injector must catch it and still return the env/stored key network-free."""
        monkeypatch.setenv("COMFY_CLOUD_API_KEY", "env-key-fallback")
        from comfy_cli.auth import store as auth_store
        from comfy_cli.cloud import oauth

        def _boom(**kw):
            raise OSError("cannot acquire refresh lock")

        monkeypatch.setattr(auth_store, "get", lambda _: None)
        # refresh=True raises; the network-free fallback reads the store as-is.
        monkeypatch.setattr(oauth, "ensure_fresh_session", _boom)
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)
        assert _resolve_partner_credential() == ("api_key_comfy_org", "env-key-fallback")


class TestExecutePartnerNodePreflight:
    """Submitting a partner-API workflow to a local server with no
    credentials must fail with the structured envelope error
    ``partner_node_requires_credential`` before /prompt is hit — not at
    execute time with an opaque "Unauthorized" string buried in
    /history."""

    PARTNER_WF = {
        "1": {"class_type": "Veo3VideoGenerationNode", "inputs": {"prompt": "x"}},
        "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
    }
    OBJECT_INFO = {
        "Veo3VideoGenerationNode": {
            "category": "partner/video/Veo",
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
        },
        "SaveVideo": {
            "category": "video",
            "output": [],
            "output_name": [],
            "output_node": True,
        },
    }

    def _wf_file(self, tmp_path):
        path = tmp_path / "partner.json"
        path.write_text(json.dumps(self.PARTNER_WF))
        return str(path)

    def test_refuses_when_no_credential(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        wf_file = self._wf_file(tmp_path)
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)

        from comfy_cli.auth import store as auth_store

        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)

        renderer_errors = []
        from comfy_cli.output.renderer import Renderer

        original_error = Renderer.error

        def capture_error(self, *, code, message, hint=None, details=None, exit_code=1):
            renderer_errors.append({"code": code, "message": message, "hint": hint, "details": details})
            return original_error(self, code=code, message=message, hint=hint, details=details, exit_code=exit_code)

        monkeypatch.setattr(Renderer, "error", capture_error)

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value=self.OBJECT_INFO),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute(wf_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            assert exc_info.value.exit_code == 1
            # /prompt must NOT be hit — refuse pre-submit.
            MockExec.assert_not_called()

        codes = [e["code"] for e in renderer_errors]
        assert "partner_node_requires_credential" in codes, f"got error codes: {codes}"
        err = next(e for e in renderer_errors if e["code"] == "partner_node_requires_credential")
        assert "Veo3VideoGenerationNode" in (err["details"] or {}).get("partner_nodes", [])

    def test_proceeds_and_injects_credential_when_available(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """With creds available, the local submit injects them into
        ``extra_data`` so the partner-API node can call out — same as the
        cloud route does. Closes the silent-failure loop."""
        wf_file = self._wf_file(tmp_path)
        monkeypatch.setenv("COMFY_CLOUD_API_KEY", "test-key-abc")
        from comfy_cli.auth import store as auth_store

        monkeypatch.setattr(auth_store, "get", lambda _: None)

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value=self.OBJECT_INFO),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []
            execute(wf_file, host="127.0.0.1", port=8188, wait=True, timeout=30)

            # WorkflowExecution receives the credential via the
            # ``extra_data`` constructor kwarg.
            kwargs = MockExec.call_args.kwargs
            extra = kwargs.get("extra_data") or {}
            assert extra.get("api_key_comfy_org") == "test-key-abc"

    def test_non_partner_workflow_skips_preflight(self, workflow_file, monkeypatch):
        """The preflight must not gate ordinary workflows. ``_fetch_object_info``
        is allowed to be skipped when no partner nodes are present (or
        called but the workflow has no api-node class types)."""
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch(
                "comfy_cli.command.run._fetch_object_info",
                return_value={
                    "EmptyLatentImage": {"category": "latent", "output": ["LATENT"], "output_name": ["LATENT"]},
                    "PreviewAny": {"category": "image", "output": [], "output_name": [], "output_node": True},
                },
            ),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []
            execute(workflow_file, host="127.0.0.1", port=8188, wait=True, timeout=30)
            MockExec.assert_called_once()


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
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "output_node": False,
            "display_name": "Empty Latent Image",
        },
        "PreviewImage": {
            "input": {"required": {"images": ["IMAGE"]}},
            "input_order": {"required": ["images"]},
            "output": [],
            "output_name": [],
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

            mock_fetch.assert_called_once()
            assert mock_fetch.call_args.args == ("127.0.0.1", 8188, 30)
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

            mock_fetch.assert_called_once()
            assert mock_fetch.call_args.args == ("127.0.0.1", 8188, 30)
            assert MockExec.call_args.kwargs["extra_data"]["api_key_comfy_org"] == "sk-test"

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


class TestLanHostUsesPlainWs:
    """LAN hosts (non-loopback, plain HTTP) must connect over ws://, not wss://."""

    def test_lan_host_uses_plain_ws(self, monkeypatch):
        captured = {}
        fake_ws = MagicMock()
        fake_ws.connect.side_effect = lambda url, **kw: captured.setdefault("url", url)
        import comfy_cli.command.run as run_pkg

        monkeypatch.setattr(run_pkg, "WebSocket", lambda: fake_ws)

        ex = WorkflowExecution(
            workflow={},
            host="192.168.1.50",
            port=8188,
            verbose=False,
            progress=None,
            local_paths=None,
            timeout=5,
        )
        ex.connect()

        assert captured["url"].startswith("ws://192.168.1.50:8188/ws"), captured["url"]


class TestWildcardHostSubstitution:
    """0.0.0.0 is a wildcard bind that macOS/Windows clients can't connect to;
    execute() substitutes it with the canonical loopback so downstream uses
    (server probe, /prompt POST, emitted URLs) are portable."""

    def test_zero_zero_zero_zero_substituted_at_entry(self, workflow_file):
        captured = {}

        def fake_check(port, host, *args, **kwargs):
            captured["check_host"] = host
            return False  # short-circuits execute() with a clean exit

        with patch("comfy_cli.command.run.check_comfy_server_running", side_effect=fake_check):
            with pytest.raises(typer.Exit):
                execute(workflow_file, host="0.0.0.0", port=8188)
        assert captured["check_host"] == "127.0.0.1"

    def test_other_local_hosts_not_substituted(self, workflow_file):
        captured = {}

        def fake_check(port, host, *args, **kwargs):
            captured["check_host"] = host
            return False

        with patch("comfy_cli.command.run.check_comfy_server_running", side_effect=fake_check):
            with pytest.raises(typer.Exit):
                execute(workflow_file, host="localhost", port=8188)
        assert captured["check_host"] == "localhost"


# ---------------------------------------------------------------------------
# execute_cloud auto-convert
# ---------------------------------------------------------------------------


class TestExecuteCloudAutoConvert:
    """The cloud path used to bail with `cloud_ui_workflow_unsupported` on any
    frontend-format workflow. It now converts via convert_ui_to_api against the
    cached cloud object_info, mirroring the local path's behavior.
    """

    UI_WORKFLOW = {
        "nodes": [{"id": 1, "type": "KSampler", "inputs": [], "outputs": [], "widgets_values": []}],
        "links": [],
    }
    CONVERTED = {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}}

    @pytest.fixture
    def ui_workflow_file(self, tmp_path):
        path = tmp_path / "ui.json"
        path.write_text(json.dumps(self.UI_WORKFLOW))
        return str(path)

    @pytest.fixture
    def fake_target(self):
        from comfy_cli.target import Target

        return Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            api_key="test-api-key",
        )

    def test_ui_workflow_converts_and_submits(self, ui_workflow_file, fake_target):
        from comfy_cli.comfy_client import SubmitResult
        from comfy_cli.command.run import execute_cloud

        # Wire the conversion path: object_info loader returns a non-empty dict,
        # convert_ui_to_api returns our pre-cooked API workflow, Client submits
        # successfully. The watcher subprocess is stubbed so the test doesn't
        # actually fork.
        mock_client = MagicMock()
        mock_client.submit_prompt.return_value = SubmitResult(prompt_id="prompt-abc", number=1, node_errors={})

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.command.run.convert_ui_to_api", return_value=self.CONVERTED) as mock_convert,
            patch(
                "comfy_cli.cql.engine._load_from_target",
                # An output-node KSampler so preflight's no-outputs check passes
                # (the converter is mocked and ignores object_info anyway).
                return_value={"KSampler": {"output_node": True}},
            ),
            patch("comfy_cli.comfy_client.Client", return_value=mock_client),
            patch("comfy_cli.command.run._spawn_watcher"),
        ):
            execute_cloud(ui_workflow_file, wait=False)

        # Convert was called against our UI workflow + the cloud object_info.
        assert mock_convert.called
        # The CONVERTED workflow was passed to the submit call — not the raw UI form.
        submitted_args, _ = mock_client.submit_prompt.call_args
        assert submitted_args[0] == self.CONVERTED

    def test_ui_workflow_conversion_failure_surfaces_conversion_error(self, ui_workflow_file, fake_target):
        from comfy_cli.command.run import execute_cloud
        from comfy_cli.workflow_to_api import WorkflowConversionError

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch(
                "comfy_cli.cql.engine._load_from_target",
                return_value={"KSampler": {}},
            ),
            patch(
                "comfy_cli.command.run.convert_ui_to_api",
                side_effect=WorkflowConversionError("missing required field"),
            ),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute_cloud(ui_workflow_file, wait=False)
            assert exc_info.value.exit_code == 1

    def test_ui_workflow_no_object_info_surfaces_cql_no_graph(self, ui_workflow_file, fake_target):
        from comfy_cli.command.run import execute_cloud

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch(
                "comfy_cli.cql.engine._load_from_target",
                side_effect=RuntimeError("no cache and no live server"),
            ),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                execute_cloud(ui_workflow_file, wait=False)
            assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# execute_cloud --wait terminal handling
# ---------------------------------------------------------------------------


class TestExecuteCloudWait:
    """The --wait success path: the final cloud history record is stashed on
    the state file (state.record) so downstream consumers (grouped outputs,
    item-named downloads) don't need a second API call."""

    API_WORKFLOW = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    RECORD = {
        "status": {"completed": True, "status_str": "success"},
        "outputs": {
            "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "12": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
        },
    }

    @pytest.fixture
    def api_workflow_file(self, tmp_path):
        path = tmp_path / "api.json"
        path.write_text(json.dumps(self.API_WORKFLOW))
        return str(path)

    @pytest.fixture
    def fake_target(self):
        from comfy_cli.target import Target

        return Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            api_key="test-api-key",
        )

    def _client(self, fake_target, record=None):
        """A real Client (real extract_* helpers) with the network calls mocked."""
        from comfy_cli import comfy_client
        from comfy_cli.comfy_client import SubmitResult

        client = comfy_client.Client(fake_target)
        client.submit_prompt = MagicMock(return_value=SubmitResult(prompt_id="prompt-wait", number=1, node_errors={}))
        client.wait_for_completion = MagicMock(return_value=record or self.RECORD)
        client.get_job_status = MagicMock(return_value=None)
        return client

    def _run_wait(self, workflow_file, fake_target, client):
        from comfy_cli.command.run import execute_cloud

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.cql.engine._load_from_target", return_value={}),
            patch("comfy_cli.comfy_client.Client", return_value=client),
        ):
            execute_cloud(workflow_file, wait=True, timeout=5)

    def test_wait_success_stashes_record_in_state_file(self, api_workflow_file, fake_target):
        from comfy_cli import jobs_state

        client = self._client(fake_target)
        self._run_wait(api_workflow_file, fake_target, client)

        state = jobs_state.read("prompt-wait")
        assert state is not None
        assert state.status == "completed"
        assert state.record == self.RECORD

    def _wait_envelope(self, workflow_file, fake_target, client, capsys):
        """Run --wait with an NDJSON renderer and return the final envelope."""
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        set_renderer(Renderer(mode=OutputMode.NDJSON, command="run"))
        self._run_wait(workflow_file, fake_target, client)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["type"] == "envelope"
        return envelope

    def test_wait_envelope_has_outputs_by_node(self, api_workflow_file, fake_target, capsys):
        client = self._client(fake_target)
        env = self._wait_envelope(api_workflow_file, fake_target, client, capsys)

        data = env["data"]
        url_a = "https://cloud.example.com/api/view?filename=a.png&subfolder=&type=output"
        url_v = "https://cloud.example.com/api/view?filename=v.mp4&subfolder=&type=output"
        assert data["outputs"] == [url_a, url_v]  # flat list untouched
        assert data["outputs_by_node"] == {"9": [url_a], "12": [url_v]}
        # No item_map on the state → key present, empty dict.
        assert data["outputs_by_item"] == {}

    def test_wait_envelope_groups_outputs_by_item_when_item_map_present(self, tmp_path, fake_target, capsys):
        # The real mechanism: compose embeds `_meta.items` in the compiled
        # workflow; run pops it at submit and stashes it on the state file.
        wf = dict(self.API_WORKFLOW)
        wf["_meta"] = {
            "schema": "compose/1",
            "blueprint": "/abs/story.yaml",
            "items": {
                "s1": {"nodes": ["1", "9"], "save_node": "9", "prefix": "outputs/s1"},
                "s2": {"nodes": ["12"], "save_node": "12", "prefix": "outputs/s2"},
            },
        }
        composed_file = tmp_path / "composed.json"
        composed_file.write_text(json.dumps(wf))

        client = self._client(fake_target)
        env = self._wait_envelope(str(composed_file), fake_target, client, capsys)

        url_a = "https://cloud.example.com/api/view?filename=a.png&subfolder=&type=output"
        url_v = "https://cloud.example.com/api/view?filename=v.mp4&subfolder=&type=output"
        assert env["data"]["outputs_by_item"] == {"s1": [url_a], "s2": [url_v]}

    def test_wait_permanent_500_fails_with_cloud_http_error_after_budget(self, api_workflow_file, fake_target, capsys):
        """Transient 429/5xx mid-poll back off and retry (fennec friction #2);
        a PERMANENT 500 exhausts the poll budget and lands on the existing
        cloud_http_error path — never a traceback."""
        import typer

        from comfy_cli import comfy_client
        from comfy_cli.comfy_client import SubmitResult
        from comfy_cli.command.run import execute_cloud
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        client = comfy_client.Client(fake_target)
        client.submit_prompt = MagicMock(return_value=SubmitResult(prompt_id="prompt-500", number=1, node_errors={}))
        client.get_job_status = MagicMock(return_value=None)
        client.get_history = MagicMock(side_effect=comfy_client.HTTPError(500, "Internal Server Error"))

        set_renderer(Renderer(mode=OutputMode.NDJSON, command="run"))
        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.cql.engine._load_from_target", return_value={}),
            patch("comfy_cli.comfy_client.Client", return_value=client),
            patch("comfy_cli.comfy_client.time.sleep"),
        ):
            with pytest.raises(typer.Exit) as excinfo:
                execute_cloud(api_workflow_file, wait=True, timeout=5)

        assert excinfo.value.exit_code == 1
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "cloud_http_error"
        # The poll budget was actually spent before giving up.
        assert client.get_history.call_count == comfy_client._MAX_POLL_FAILURES

    def test_wait_envelope_data_validates_against_run_schema(self, api_workflow_file, fake_target, capsys):
        from pathlib import Path

        import jsonschema

        client = self._client(fake_target)
        env = self._wait_envelope(api_workflow_file, fake_target, client, capsys)

        schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "run.json"
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(env["data"])
        # The new keys are part of the documented contract, not just tolerated.
        assert "outputs_by_node" in schema["properties"]
        assert "outputs_by_item" in schema["properties"]


# ---------------------------------------------------------------------------
# pop_compose_meta — strip compose provenance before preflight/submit
# ---------------------------------------------------------------------------


class TestPopComposeMeta:
    """`compose` embeds `_meta` (compose/1 provenance) in the compiled
    workflow; `run` must pop it before validation and POST. A node that is
    legitimately keyed "_meta" (has a class_type) must be left alone."""

    ITEMS = {"s1": {"nodes": ["1", "9"], "save_node": "9", "prefix": "o/s1"}}

    def test_pops_meta_dict_without_class_type(self):
        from comfy_cli.command.run.loader import pop_compose_meta

        meta = {"schema": "compose/1", "blueprint": "/abs/b.yaml", "items": self.ITEMS}
        wf = {"_meta": dict(meta), "1": {"class_type": "KSampler", "inputs": {}}}
        popped = pop_compose_meta(wf)
        assert popped == meta
        assert "_meta" not in wf
        assert "1" in wf  # nodes untouched

    def test_returns_none_when_absent(self):
        from comfy_cli.command.run.loader import pop_compose_meta

        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        assert pop_compose_meta(wf) is None
        assert wf == {"1": {"class_type": "KSampler", "inputs": {}}}

    def test_node_keyed_meta_with_class_type_is_preserved(self):
        from comfy_cli.command.run.loader import pop_compose_meta

        node = {"class_type": "Note", "inputs": {"text": "hi"}}
        wf = {"_meta": node, "1": {"class_type": "KSampler", "inputs": {}}}
        assert pop_compose_meta(wf) is None
        assert wf["_meta"] is node

    def test_non_dict_meta_left_alone(self):
        from comfy_cli.command.run.loader import pop_compose_meta

        wf = {"_meta": "garbage", "1": {"class_type": "KSampler", "inputs": {}}}
        assert pop_compose_meta(wf) is None
        assert wf["_meta"] == "garbage"

    def test_node_level_meta_titles_not_confused_with_top_level(self):
        from comfy_cli.command.run.loader import pop_compose_meta

        # Per-node `_meta: {title}` blocks live INSIDE nodes — not stripped.
        wf = {"1": {"class_type": "KSampler", "inputs": {}, "_meta": {"title": "K"}}}
        assert pop_compose_meta(wf) is None
        assert wf["1"]["_meta"] == {"title": "K"}

    def test_meta_is_clean_in_preflight_and_stripped_for_submit(self):
        """`_meta` (compose provenance) is recognized by validation — it never
        warns, stripped or not — and `pop_compose_meta` still removes it before
        the submit POST (so the server never sees the provenance block)."""
        from comfy_cli.command.run.loader import pop_compose_meta
        from comfy_cli.cql.engine import Graph

        object_info = {"KSampler": {"output": ["LATENT"], "output_name": ["LATENT"], "input": {"required": {}}}}
        graph = Graph.from_object_info(object_info)

        def meta_warnings(validation):
            return [w for w in validation.get("warnings", []) if "_meta" in str(w)]

        wf = {"_meta": {"schema": "compose/1"}, "1": {"class_type": "KSampler", "inputs": {}}}
        # Recognized provenance: preflight is clean even WITH `_meta` present
        # (the composer adds it; warning on it would be self-inflicted noise).
        assert not meta_warnings(graph.validate_workflow(dict(wf)))

        # And it's still stripped for the submit path, leaving preflight clean.
        pop_compose_meta(wf)
        assert "_meta" not in wf
        assert not meta_warnings(graph.validate_workflow(wf))


class TestRunStripsComposeMeta:
    """Both submit paths strip `_meta` before the POST and stash its `items`
    map on the job state file so downstream consumers (grouped outputs,
    item-named downloads) can join outputs back to foreach items."""

    ITEMS = {"s1": {"nodes": ["1", "9"], "save_node": "9", "prefix": "o/s1"}}
    META = {"schema": "compose/1", "blueprint": "/abs/b.yaml", "items": ITEMS}
    API_WORKFLOW = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    RECORD = {
        "status": {"completed": True, "status_str": "success"},
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
    }

    @pytest.fixture
    def composed_workflow_file(self, tmp_path):
        wf = dict(self.API_WORKFLOW)
        wf["_meta"] = dict(self.META)
        path = tmp_path / "composed.json"
        path.write_text(json.dumps(wf))
        return str(path)

    @pytest.fixture
    def fake_target(self):
        from comfy_cli.target import Target

        return Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            api_key="test-api-key",
        )

    def test_cloud_nowait_strips_meta_and_stashes_item_map(self, composed_workflow_file, fake_target):
        from comfy_cli import jobs_state
        from comfy_cli.comfy_client import SubmitResult
        from comfy_cli.command.run import execute_cloud

        mock_client = MagicMock()
        mock_client.submit_prompt.return_value = SubmitResult(prompt_id="prompt-meta-nw", number=1, node_errors={})

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.cql.engine._load_from_target", return_value={}),
            patch("comfy_cli.comfy_client.Client", return_value=mock_client),
            patch("comfy_cli.command.run._spawn_watcher"),
        ):
            execute_cloud(composed_workflow_file, wait=False)

        submitted = mock_client.submit_prompt.call_args.args[0]
        assert "_meta" not in submitted
        assert set(submitted) == set(self.API_WORKFLOW)

        state = jobs_state.read("prompt-meta-nw")
        assert state is not None
        assert state.item_map == self.ITEMS

    def test_cloud_wait_strips_meta_and_stashes_item_map(self, composed_workflow_file, fake_target):
        from comfy_cli import comfy_client, jobs_state
        from comfy_cli.comfy_client import SubmitResult
        from comfy_cli.command.run import execute_cloud

        client = comfy_client.Client(fake_target)
        client.submit_prompt = MagicMock(return_value=SubmitResult(prompt_id="prompt-meta-w", number=1, node_errors={}))
        client.wait_for_completion = MagicMock(return_value=self.RECORD)
        client.get_job_status = MagicMock(return_value=None)

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.cql.engine._load_from_target", return_value={}),
            patch("comfy_cli.comfy_client.Client", return_value=client),
        ):
            execute_cloud(composed_workflow_file, wait=True, timeout=5)

        submitted = client.submit_prompt.call_args.args[0]
        assert "_meta" not in submitted

        state = jobs_state.read("prompt-meta-w")
        assert state is not None
        assert state.status == "completed"
        assert state.item_map == self.ITEMS

    def test_local_execute_strips_meta_before_submit(self, tmp_path):
        wf = {
            "1": {"class_type": "EmptyLatentImage", "inputs": {}, "_meta": {"title": "latent"}},
            "_meta": dict(self.META),
        }
        path = tmp_path / "composed_local.json"
        path.write_text(json.dumps(wf))

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value={}),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []
            execute(str(path), host="127.0.0.1", port=8188, wait=True, timeout=30)

        submitted = MockExec.call_args.args[0]
        assert "_meta" not in submitted
        # Node-interior `_meta` titles survive untouched.
        assert submitted["1"]["_meta"] == {"title": "latent"}


class TestLocalExecuteItemMapAndGroupedOutputs:
    """Local-path parity with execute_cloud: `execute` stashes the compose
    item_map on the job state file (both --wait and async paths) and the
    --wait envelope carries outputs_by_node / outputs_by_item grouped from
    the execution's node-keyed output entries."""

    ITEMS = {
        "s1": {"nodes": ["1", "9"], "save_node": "9", "prefix": "o/s1"},
        "s2": {"nodes": ["12"], "save_node": "12", "prefix": "o/s2"},
    }
    META = {"schema": "compose/1", "blueprint": "/abs/b.yaml", "items": ITEMS}
    API_WORKFLOW = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    URL_A = "http://127.0.0.1:8188/view?filename=a.png&subfolder=&type=output"
    URL_V = "http://127.0.0.1:8188/view?filename=v.mp4&subfolder=&type=output"

    @pytest.fixture
    def composed_workflow_file(self, tmp_path):
        wf = dict(self.API_WORKFLOW)
        wf["_meta"] = dict(self.META)
        path = tmp_path / "composed_local.json"
        path.write_text(json.dumps(wf))
        return str(path)

    def _mock_exec(self, prompt_id):
        mock_exec = MagicMock()
        mock_exec.prompt_id = prompt_id
        mock_exec.client_id = "cid-local"
        mock_exec.outputs = []
        mock_exec.output_entries = []
        mock_exec.cached_node_ids = []
        mock_exec.executed_node_ids = []
        return mock_exec

    def _run(self, workflow_file, mock_exec, *, wait, extra_patches=()):
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value={}),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution", return_value=mock_exec),
            patch("comfy_cli.command.run._spawn_watcher", return_value=True),
            patch("comfy_cli.command.run._tail_state_file"),
        ):
            execute(workflow_file, host="127.0.0.1", port=8188, wait=wait, timeout=30)

    def test_local_wait_stashes_item_map(self, composed_workflow_file):
        from comfy_cli import jobs_state

        mock_exec = self._mock_exec("local-meta-wait")
        self._run(composed_workflow_file, mock_exec, wait=True)

        state = jobs_state.read("local-meta-wait")
        assert state is not None
        assert state.status == "completed"
        assert state.item_map == self.ITEMS

    def test_local_async_stashes_item_map(self, composed_workflow_file):
        from comfy_cli import jobs_state

        mock_exec = self._mock_exec("local-meta-async")
        self._run(composed_workflow_file, mock_exec, wait=False)

        state = jobs_state.read("local-meta-async")
        assert state is not None
        assert state.item_map == self.ITEMS

    def _wait_envelope(self, workflow_file, mock_exec, capsys):
        from comfy_cli.output import Renderer, set_renderer
        from comfy_cli.output.renderer import OutputMode

        set_renderer(Renderer(mode=OutputMode.NDJSON, command="run"))
        self._run(workflow_file, mock_exec, wait=True)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["type"] == "envelope"
        return envelope

    def test_local_wait_envelope_groups_by_node_and_item(self, composed_workflow_file, capsys):
        mock_exec = self._mock_exec("local-grouped")
        mock_exec.outputs = [self.URL_A, self.URL_V]
        mock_exec.output_entries = [
            {"node_id": "9", "url": self.URL_A, "filename": "a.png", "type": "output"},
            {"node_id": "12", "url": self.URL_V, "filename": "v.mp4", "type": "output"},
        ]

        env = self._wait_envelope(composed_workflow_file, mock_exec, capsys)

        data = env["data"]
        assert data["outputs"] == [self.URL_A, self.URL_V]  # flat list untouched
        assert data["outputs_by_node"] == {"9": [self.URL_A], "12": [self.URL_V]}
        assert data["outputs_by_item"] == {"s1": [self.URL_A], "s2": [self.URL_V]}

    def test_local_wait_envelope_without_item_map_has_empty_by_item(self, tmp_path, capsys):
        # Plain workflow (no _meta) → outputs_by_node still grouped from the
        # node-keyed entries, outputs_by_item stays {}.
        path = tmp_path / "plain.json"
        path.write_text(json.dumps(self.API_WORKFLOW))
        mock_exec = self._mock_exec("local-plain")
        mock_exec.outputs = [self.URL_A]
        mock_exec.output_entries = [{"node_id": "9", "url": self.URL_A, "filename": "a.png", "type": "output"}]

        env = self._wait_envelope(str(path), mock_exec, capsys)

        assert env["data"]["outputs_by_node"] == {"9": [self.URL_A]}
        assert env["data"]["outputs_by_item"] == {}


class TestRunJournal:
    """Successful submits journal one runs.jsonl line into the governing
    project (cwd-anchored); failures of the journal itself never fail the run."""

    API_WORKFLOW = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    @pytest.fixture
    def proj_dir(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "comfy.yaml").write_text("schema: project/1\ndefaults:\n  where: cloud\n")
        monkeypatch.chdir(proj)
        return proj

    @pytest.fixture
    def fake_target(self):
        from comfy_cli.target import Target

        return Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            api_key="test-api-key",
        )

    def _workflow_file(self, proj_dir):
        path = proj_dir / "wf.json"
        path.write_text(json.dumps(self.API_WORKFLOW))
        return str(path)

    def _journal_events(self, proj_dir):
        path = proj_dir / ".comfy" / "runs.jsonl"
        if not path.exists():
            return []
        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def _execute_cloud_nowait(self, workflow_file, fake_target):
        from comfy_cli.comfy_client import SubmitResult
        from comfy_cli.command.run import execute_cloud

        mock_client = MagicMock()
        mock_client.submit_prompt.return_value = SubmitResult(prompt_id="prompt-journal", number=1, node_errors={})
        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.cql.engine._load_from_target", return_value={}),
            patch("comfy_cli.comfy_client.Client", return_value=mock_client),
            patch("comfy_cli.command.run._spawn_watcher"),
        ):
            execute_cloud(workflow_file, wait=False)

    def test_cloud_submit_journals_inside_project(self, proj_dir, fake_target):
        workflow_file = self._workflow_file(proj_dir)
        self._execute_cloud_nowait(workflow_file, fake_target)

        events = self._journal_events(proj_dir)
        assert len(events) == 1
        ev = events[0]
        assert ev["cmd"] == "run"
        assert ev["workflow"] == workflow_file
        assert ev["prompt_id"] == "prompt-journal"
        assert ev["where"] == "cloud"
        assert "ts" in ev

    def test_local_submit_journals_inside_project(self, proj_dir):
        workflow_file = self._workflow_file(proj_dir)
        mock_exec = MagicMock()
        mock_exec.prompt_id = "prompt-local-journal"
        mock_exec.client_id = "cid"
        mock_exec.outputs = []
        mock_exec.output_entries = []
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value={}),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution", return_value=mock_exec),
            patch("comfy_cli.command.run._spawn_watcher", return_value=True),
            patch("comfy_cli.command.run._tail_state_file"),
        ):
            execute(workflow_file, host="127.0.0.1", port=8188, wait=False, timeout=30)

        events = self._journal_events(proj_dir)
        assert len(events) == 1
        ev = events[0]
        assert ev["cmd"] == "run"
        assert ev["workflow"] == workflow_file
        assert ev["prompt_id"] == "prompt-local-journal"
        assert ev["where"] == "local"

    def test_run_outside_project_writes_no_journal(self, tmp_path, monkeypatch, fake_target):
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        workflow_file = plain / "wf.json"
        workflow_file.write_text(json.dumps(self.API_WORKFLOW))
        self._execute_cloud_nowait(str(workflow_file), fake_target)
        assert not (plain / ".comfy").exists()

    def test_journal_failure_does_not_fail_run(self, proj_dir, fake_target, monkeypatch):
        import comfy_cli.project as project_mod

        def _boom(*a, **kw):
            raise RuntimeError("journal exploded")

        monkeypatch.setattr(project_mod, "journal", _boom)
        workflow_file = self._workflow_file(proj_dir)
        # Must not raise — the wrapped hook swallows the failure.
        self._execute_cloud_nowait(workflow_file, fake_target)

        from comfy_cli import jobs_state

        state = jobs_state.read("prompt-journal")
        assert state is not None  # submit completed normally
