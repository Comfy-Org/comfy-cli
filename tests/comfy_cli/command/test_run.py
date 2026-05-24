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
    _detect_partner_nodes,
    _resolve_partner_credential,
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
    """Partner-API nodes (category `api node/...`) must be detected before
    a local submit so we can refuse early instead of failing opaquely at
    execute time with `Unauthorized: Please login first`."""

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
            Veo3VideoGenerationNode="api node/video/Veo",
            SaveVideo="video",
            KlingImage2VideoNode="api node/video/Kling",
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
        the server explicitly categorizes it under `api node/*`."""
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
        info = self._info(Veo3VideoGenerationNode="api node/video/Veo")
        assert _detect_partner_nodes(wf, info) == ["Veo3VideoGenerationNode"]


class TestResolvePartnerCredential:
    """The credential the local submit can inject into ``extra_data`` so a
    partner-API node finds it. Three sources, env > stored key > OAuth."""

    def test_uses_env_var_first(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("COMFY_CLOUD_API_KEY", "env-key-123")
        from comfy_cli.auth import store as auth_store
        monkeypatch.setattr(auth_store, "get", lambda _: None)
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
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)
        assert _resolve_partner_credential() == ("api_key_comfy_org", "stored-key-456")

    def test_falls_back_to_oauth_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store

        session = MagicMock()
        session.is_expired.return_value = False
        session.access_token = "oauth-bearer-789"
        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: session)
        assert _resolve_partner_credential() == ("auth_token_comfy_org", "oauth-bearer-789")

    def test_returns_none_when_nothing_configured(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store
        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)
        assert _resolve_partner_credential() is None

    def test_treats_expired_session_as_no_creds(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
        from comfy_cli.auth import store as auth_store

        session = MagicMock()
        session.is_expired.return_value = True
        session.access_token = "stale"
        monkeypatch.setattr(auth_store, "get", lambda _: None)
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: session)
        assert _resolve_partner_credential() is None


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
            "category": "api node/video/Veo",
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

    def test_proceeds_and_injects_credential_when_available(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
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
                execute(workflow_file, host="0.0.0.0", port=8188, json_mode=True)
        assert captured["check_host"] == "127.0.0.1"

    def test_other_local_hosts_not_substituted(self, workflow_file):
        captured = {}

        def fake_check(port, host, *args, **kwargs):
            captured["check_host"] = host
            return False

        with patch("comfy_cli.command.run.check_comfy_server_running", side_effect=fake_check):
            with pytest.raises(typer.Exit):
                execute(workflow_file, host="localhost", port=8188, json_mode=True)
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
        mock_client.submit_prompt.return_value = SubmitResult(
            prompt_id="prompt-abc", number=1, node_errors={}
        )

        with (
            patch("comfy_cli.target.resolve_target", return_value=fake_target),
            patch("comfy_cli.command.run.convert_ui_to_api", return_value=self.CONVERTED) as mock_convert,
            patch(
                "comfy_cli.cql.engine._load_from_target",
                return_value={"KSampler": {}},  # any truthy dict suffices for the converter
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
