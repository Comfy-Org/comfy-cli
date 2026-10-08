"""Integration/smoke tests for `comfy run --prompt`/`--set`.

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
from comfy_cli.cql.default_workflow import (
    CHECKPOINT_LOADER_ID,
    DEFAULT_CHECKPOINT_NAME,
    POSITIVE_PROMPT_ID,
    build_default_workflow,
)


def _object_info_with_checkpoints(names):
    return {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [list(names), {}]}}}}


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
                preloaded=(injected, "default_text2img", False, False),
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
                preloaded=(injected, "default_text2img", False, False),
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
        graph, name, is_ui, checkpoint_user_set = preloaded
        assert is_ui is False
        assert name == "default_text2img"
        assert checkpoint_user_set is False
        assert graph[POSITIVE_PROMPT_ID]["inputs"]["text"] == "a red fox in snow"

    def test_set_checkpoint_override_forwarded(self):
        mock_exec, _ = self._call_run(prompt="fox", set_overrides=["checkpoint=sd_xl.safetensors"])
        preloaded = mock_exec.call_args.kwargs["preloaded"]
        graph = preloaded[0]
        assert graph["4"]["inputs"]["ckpt_name"] == "sd_xl.safetensors"
        # A user-pinned checkpoint flips the flag so runtime resolution is skipped.
        assert preloaded[3] is True

    def test_set_checkpoint_raw_form_flags_user_set(self):
        mock_exec, _ = self._call_run(prompt="fox", set_overrides=["4.ckpt_name=sd_xl.safetensors"])
        assert mock_exec.call_args.kwargs["preloaded"][3] is True

    def test_non_checkpoint_set_leaves_flag_false(self):
        mock_exec, _ = self._call_run(prompt="fox", set_overrides=["seed=42"])
        assert mock_exec.call_args.kwargs["preloaded"][3] is False

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


class TestRuntimeCheckpointResolutionLocal:
    """Wiring: `execute` resolves the bundled default's checkpoint against the
    server's object_info before submit."""

    def _run_local(self, preloaded, object_info, patch_pprint=False):
        stack = [
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value=object_info),
            # Isolate resolution from the unrelated class_type validation the
            # bundled graph would otherwise trip against a stub object_info.
            patch("comfy_cli.command.run._preflight_validate"),
            patch("comfy_cli.command.run.ExecutionProgress"),
            patch("comfy_cli.command.run.WorkflowExecution"),
        ]
        with stack[0], stack[1], stack[2], stack[3], stack[4] as MockExec:
            MockExec.return_value = MagicMock(outputs=[])
            if patch_pprint:
                with patch("comfy_cli.command.run.preflight.pprint") as mock_pprint:
                    execute(None, host="127.0.0.1", port=8188, wait=True, timeout=30, preloaded=preloaded)
                    return MockExec, mock_pprint
            execute(None, host="127.0.0.1", port=8188, wait=True, timeout=30, preloaded=preloaded)
            return MockExec, None

    def _default_preloaded(self, *, checkpoint_user_set=False):
        return (build_default_workflow(prompt="fox"), "default_text2img", False, checkpoint_user_set)

    def test_absent_pinned_is_substituted(self):
        oi = _object_info_with_checkpoints(["dreamshaper.safetensors", "sd_xl.safetensors"])
        MockExec, mock_pprint = self._run_local(self._default_preloaded(), oi, patch_pprint=True)
        submitted = MockExec.call_args.args[0]
        assert submitted[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == "dreamshaper.safetensors"
        # A human-facing substitution notice is printed.
        assert any("dreamshaper.safetensors" in str(c.args[0]) for c in mock_pprint.call_args_list)

    def test_present_pinned_is_unchanged(self):
        oi = _object_info_with_checkpoints(["other.safetensors", DEFAULT_CHECKPOINT_NAME])
        MockExec, _ = self._run_local(self._default_preloaded(), oi)
        submitted = MockExec.call_args.args[0]
        assert submitted[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == DEFAULT_CHECKPOINT_NAME

    def test_empty_enum_errors_no_checkpoint_available(self):
        oi = _object_info_with_checkpoints([])
        with pytest.raises(typer.Exit) as e:
            self._run_local(self._default_preloaded(), oi)
        assert e.value.exit_code == 1

    def test_empty_object_info_fails_open_and_submits(self):
        MockExec, _ = self._run_local(self._default_preloaded(), {})
        # No error; the pinned default is submitted as-is.
        submitted = MockExec.call_args.args[0]
        assert submitted[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == DEFAULT_CHECKPOINT_NAME

    def test_user_pinned_checkpoint_is_never_substituted(self):
        graph = build_default_workflow(prompt="fox", overrides=["checkpoint=userpick.safetensors"])
        preloaded = (graph, "default_text2img", False, True)  # checkpoint_user_set=True
        oi = _object_info_with_checkpoints(["dreamshaper.safetensors"])
        MockExec, _ = self._run_local(preloaded, oi)
        submitted = MockExec.call_args.args[0]
        assert submitted[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == "userpick.safetensors"


class TestCheckpointResolutionEmptyEnumByTarget:
    """`_resolve_default_checkpoint_or_exit` hard-errors on an empty enum only
    for the local target; Comfy Cloud provisions models per-job so it fails
    open."""

    def test_local_empty_enum_hard_errors(self):
        from comfy_cli.command.run.preflight import _resolve_default_checkpoint_or_exit
        from comfy_cli.output import get_renderer

        wf = build_default_workflow(prompt="fox")
        oi = _object_info_with_checkpoints([])
        with pytest.raises(typer.Exit) as e:
            _resolve_default_checkpoint_or_exit(get_renderer(), wf, oi, where="local")
        assert e.value.exit_code == 1

    def test_cloud_empty_enum_fails_open(self):
        from comfy_cli.command.run.preflight import _resolve_default_checkpoint_or_exit
        from comfy_cli.output import get_renderer

        wf = build_default_workflow(prompt="fox")
        oi = _object_info_with_checkpoints([])
        # No raise: the submit is allowed to proceed with the pinned default.
        _resolve_default_checkpoint_or_exit(get_renderer(), wf, oi, where="cloud")
        assert wf[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == DEFAULT_CHECKPOINT_NAME
