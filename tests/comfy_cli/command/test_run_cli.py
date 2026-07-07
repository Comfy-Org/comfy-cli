"""Tests for the `comfy run-cli` guided walkthrough."""

import json
from unittest.mock import patch

import pytest

from comfy_cli.command import run_cli


def _ok_envelope(**data) -> str:
    return json.dumps({"ok": True, "data": data, "error": None})


class TestStepBuilders:
    def test_classify_treats_json_flag_as_agent(self):
        inv = run_cli.Invocation(argv=["comfy", "--json", "env"], label="agent")
        assert run_cli._classify(inv) == "agent"

    def test_classify_treats_json_stream_as_agent(self):
        inv = run_cli.Invocation(argv=["comfy", "--json-stream", "jobs", "watch", "x"], label="agent")
        assert run_cli._classify(inv) == "agent"

    def test_classify_default_is_human(self):
        inv = run_cli.Invocation(argv=["comfy", "env"], label="human")
        assert run_cli._classify(inv) == "human"

    def test_build_steps_contains_full_surface(self):
        state = run_cli._DemoState(workflow_path="/tmp/x.json")
        steps = run_cli._build_steps(state)
        titles = " ".join(s.title.lower() for s in steps)
        for needle in [
            "env",
            "which",
            "discover",
            "cql",
            "synchronous",
            "json envelope",
            "async",
            "jobs ls",
            "jobs status",
            "jobs watch",
            "fleet",
            "auth",
        ]:
            assert needle in titles, f"missing coverage for: {needle}"

    def test_build_steps_includes_parallel_fleet(self):
        state = run_cli._DemoState(workflow_path="/tmp/x.json")
        steps = run_cli._build_steps(state)
        fleet = [s for s in steps if "FLEET" in s.title]
        assert len(fleet) == 1
        assert fleet[0].custom is not None  # uses custom runner, not plain invocations

    def test_fleet_workflows_have_distinct_colors(self):
        # Distinct colors prevent ComfyUI's per-node cache from collapsing the
        # fleet into a single execution.
        colors = {run_cli.fleet_workflow(i)["1"]["inputs"]["color"] for i in range(run_cli.FLEET_SIZE)}
        assert len(colors) == run_cli.FLEET_SIZE

    def test_capture_prompt_id_extracts_from_last_line(self):
        state = run_cli._DemoState(workflow_path="/tmp/x.json")
        handler = run_cli._capture_prompt_id(state)
        handler(_ok_envelope(prompt_id="abc-123"))
        assert state.async_prompt_id == "abc-123"

    def test_capture_prompt_id_ignores_garbage(self):
        state = run_cli._DemoState(workflow_path="/tmp/x.json")
        handler = run_cli._capture_prompt_id(state)
        handler("not json at all")
        assert state.async_prompt_id is None


class TestRunInvocation:
    def test_streaming_invocation_returns_subprocess_rc(self):
        inv = run_cli.Invocation(argv=["true"], label="trivial")
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            rc = run_cli._run_invocation(inv, pause_seconds=0)
        assert rc == 0
        call_env = run.call_args.kwargs["env"]
        assert call_env["COMFY_OUTPUT"] == "pretty"

    def test_capturing_invocation_runs_on_output(self):
        captured: list[str] = []
        inv = run_cli.Invocation(
            argv=["true"],
            label="cap",
            capture=True,
            on_output=captured.append,
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = _ok_envelope(value=42)
            run.return_value.stderr = ""
            rc = run_cli._run_invocation(inv, pause_seconds=0)
        assert rc == 0
        assert captured == [_ok_envelope(value=42)]


class TestRunStep:
    def _state(self) -> run_cli._DemoState:
        return run_cli._DemoState(workflow_path="/tmp/x.json")

    def test_skip_if_short_circuits(self):
        step = run_cli.Step(
            title="skipped",
            invocations=[run_cli.Invocation(argv=["false"], label="never")],
            skip_if=lambda: True,
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            rc = run_cli._run_step(step, self._state(), pause_seconds=0, show_agent=True)
        assert rc == 0
        run.assert_not_called()

    def test_no_show_agent_skips_agent_invocations(self):
        step = run_cli.Step(
            title="step",
            invocations=[
                run_cli.Invocation(argv=["comfy", "env"], label="human"),
                run_cli.Invocation(argv=["comfy", "--json", "env"], label="agent"),
            ],
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run_cli._run_step(step, self._state(), pause_seconds=0, show_agent=False)
        assert run.call_count == 1
        assert "--json" not in run.call_args_list[0].args[0]

    def test_show_agent_runs_both(self):
        step = run_cli.Step(
            title="step",
            invocations=[
                run_cli.Invocation(argv=["comfy", "env"], label="human"),
                run_cli.Invocation(argv=["comfy", "--json", "env"], label="agent"),
            ],
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run_cli._run_step(step, self._state(), pause_seconds=0, show_agent=True)
        assert run.call_count == 2

    def test_optional_failure_does_not_propagate(self):
        step = run_cli.Step(
            title="step",
            invocations=[run_cli.Invocation(argv=["false"], label="opt", optional=True)],
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 17
            rc = run_cli._run_step(step, self._state(), pause_seconds=0, show_agent=True)
        assert rc == 0

    def test_required_failure_propagates(self):
        step = run_cli.Step(
            title="step",
            invocations=[run_cli.Invocation(argv=["false"], label="req")],
        )
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 3
            rc = run_cli._run_step(step, self._state(), pause_seconds=0, show_agent=True)
        assert rc == 3

    def test_custom_step_runner_is_invoked(self):
        called: dict = {}

        def custom(state, pause_seconds, show_agent):
            called["state"] = state
            called["pause"] = pause_seconds
            called["agent"] = show_agent
            return 0

        step = run_cli.Step(title="custom", custom=custom)
        state = self._state()
        rc = run_cli._run_step(step, state, pause_seconds=0, show_agent=False)
        assert rc == 0
        assert called == {"state": state, "pause": 0, "agent": False}


class TestSubmitOneAsync:
    def test_returns_prompt_id_on_success(self):
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = _ok_envelope(prompt_id="pid-7")
            run.return_value.stderr = ""
            idx, pid, elapsed = run_cli._submit_one_async(["comfy"], idx=3)
        assert idx == 3
        assert pid == "pid-7"
        assert elapsed >= 0

    def test_returns_none_on_nonzero_exit(self):
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ""
            run.return_value.stderr = "boom"
            idx, pid, _ = run_cli._submit_one_async(["comfy"], idx=0)
        assert idx == 0
        assert pid is None

    def test_returns_none_on_malformed_json(self):
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "garbage"
            run.return_value.stderr = ""
            idx, pid, _ = run_cli._submit_one_async(["comfy"], idx=2)
        assert pid is None


class TestExecute:
    def test_execute_writes_workflow_and_runs_all_steps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_cli.tempfile, "gettempdir", lambda: str(tmp_path))

        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = _ok_envelope(prompt_id="pid-1", commands={"env": {}, "run": {}})
            run.return_value.stderr = ""
            rc = run_cli.execute(pause_seconds=0, no_cleanup=False, show_agent=True)

        assert rc == 0
        # 11 invocation-driven steps + 1 custom fleet step. The fleet step uses
        # ThreadPoolExecutor but the subprocess.run mock still records calls.
        assert run.call_count >= 11
        # All temp workflow files cleaned up (single-job + fleet).
        leftover = list(tmp_path.glob("comfy_run_cli_*.json"))
        assert leftover == []

    def test_execute_keeps_workflows_with_no_cleanup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_cli.tempfile, "gettempdir", lambda: str(tmp_path))
        with patch("comfy_cli.command.run_cli.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = _ok_envelope(prompt_id="pid-1")
            run.return_value.stderr = ""
            run_cli.execute(pause_seconds=0, no_cleanup=True, show_agent=True)
        leftover = list(tmp_path.glob("comfy_run_cli_*.json"))
        # 1 single-job workflow + FLEET_SIZE fleet workflows.
        assert len(leftover) == 1 + run_cli.FLEET_SIZE


def test_cloud_print_prompt_does_not_submit(monkeypatch, tmp_path):
    """--print-prompt on the cloud route prints the graph and never submits."""
    import typer

    import comfy_cli.comfy_client as cc
    from comfy_cli.command.run import execute_cloud

    wf = tmp_path / "wf.json"
    wf.write_text('{"1": {"class_type": "X", "inputs": {}}}')

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def submit_prompt(self, *a, **k):
            raise AssertionError("submit_prompt must not be called in --print-prompt mode")

    # execute_cloud does `from comfy_cli.comfy_client import Client`; patching the
    # attribute on the module makes that import pick up the fake. But the dry-run
    # should return BEFORE Client is even constructed.
    monkeypatch.setattr(cc, "Client", FakeClient)

    with pytest.raises(typer.Exit) as exc:
        execute_cloud(str(wf), wait=True, print_prompt=True, timeout=5)
    assert exc.value.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
