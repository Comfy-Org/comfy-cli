"""Integration/smoke tests for `comfy run --prompt`/`--set` (BE-2535).

The injector is unit-tested offline in
``tests/comfy_cli/cql/test_default_workflow.py``. These tests prove the wiring:
the CLI builds the bundled default graph, injects the prompt/overrides, and
hands the SAME graph to run's existing execute/submit path (no new
websocket/HTTP code).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli.cmdline import run as run_command
from comfy_cli.command.run import execute
from comfy_cli.cql.default_workflow import POSITIVE_PROMPT_ID


class TestExecuteSubmitsPreloadedGraph:
    """`preloaded` short-circuits file loading; the in-memory graph is what
    reaches the submit path (WorkflowExecution)."""

    def test_preloaded_graph_is_submitted(self):
        injected = {
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox in snow"}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
        }
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            mock_exec = MagicMock()
            MockExec.return_value = mock_exec
            mock_exec.outputs = []

            execute(
                None,
                host="127.0.0.1",
                port=8188,
                wait=True,
                timeout=30,
                preloaded=(injected, "default_text2img", False),
            )

            # The exact injected graph is what got handed to the submit path.
            submitted = MockExec.call_args.args[0]
            assert submitted["6"]["inputs"]["text"] == "a red fox in snow"
            mock_exec.queue.assert_called_once()

    def test_preloaded_skips_file_loading(self):
        """A non-existent path never triggers workflow_not_found when preloaded."""
        injected = {"9": {"class_type": "SaveImage", "inputs": {}}}
        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution") as MockExec,
        ):
            MockExec.return_value = MagicMock(outputs=[])
            # Positional workflow is a bogus path — must be ignored.
            execute(
                "/no/such/file.json",
                host="127.0.0.1",
                port=8188,
                wait=True,
                timeout=30,
                preloaded=(injected, "default_text2img", False),
            )
            MockExec.assert_called_once()


class TestRunCliWiring:
    """The `run` command builds + injects the bundled default and forwards it."""

    def _call_run(self, **kwargs):
        # tracking is consent-gated (no-op in tests), but patch it to be safe.
        with (
            patch("comfy_cli.cmdline.tracking.track_event"),
            patch("comfy_cli.command.run.execute") as mock_exec,
            patch("comfy_cli.command.run.execute_cloud") as mock_cloud,
        ):
            run_command(where="local", **kwargs)
            return mock_exec, mock_cloud

    def test_prompt_builds_injected_graph_and_forwards(self):
        mock_exec, _ = self._call_run(prompt="a red fox in snow")
        mock_exec.assert_called_once()
        preloaded = mock_exec.call_args.kwargs["preloaded"]
        assert preloaded is not None
        graph, name, is_ui = preloaded
        assert is_ui is False
        assert name == "default_text2img"
        assert graph[POSITIVE_PROMPT_ID]["inputs"]["text"] == "a red fox in snow"

    def test_set_checkpoint_override_forwarded(self):
        mock_exec, _ = self._call_run(prompt="fox", set_overrides=["checkpoint=sd_xl.safetensors"])
        graph = mock_exec.call_args.kwargs["preloaded"][0]
        assert graph["4"]["inputs"]["ckpt_name"] == "sd_xl.safetensors"

    def test_workflow_path_forwards_no_preloaded(self):
        mock_exec, _ = self._call_run(workflow="wf.json")
        assert mock_exec.call_args.kwargs["preloaded"] is None

    def test_cloud_path_forwards_injected_graph(self):
        with (
            patch("comfy_cli.cmdline.tracking.track_event"),
            patch("comfy_cli.cmdline.where_module.cloud_preflight", return_value=None),
            patch("comfy_cli.command.run.execute") as mock_exec,
            patch("comfy_cli.command.run.execute_cloud") as mock_cloud,
        ):
            run_command(where="cloud", prompt="a red fox in snow")
            mock_exec.assert_not_called()
            mock_cloud.assert_called_once()
            graph = mock_cloud.call_args.kwargs["preloaded"][0]
            assert graph[POSITIVE_PROMPT_ID]["inputs"]["text"] == "a red fox in snow"

    def test_prompt_with_workflow_is_rejected(self):
        with pytest.raises(typer.Exit) as e:
            self._call_run(workflow="wf.json", prompt="fox")
        assert e.value.exit_code == 1

    def test_no_workflow_no_prompt_is_rejected(self):
        with pytest.raises(typer.Exit) as e:
            self._call_run()
        assert e.value.exit_code == 1

    def test_bad_set_address_is_rejected(self):
        with pytest.raises(typer.Exit) as e:
            self._call_run(prompt="fox", set_overrides=["bogus=1"])
        assert e.value.exit_code == 1
