"""Execution lifecycle tests for ``comfy run``.

The CLI's ``run`` command historically emitted a single ``run`` event via the
``@track_command()`` decorator. It now manually emits
``execution_start`` / ``execution_success`` / ``execution_error`` against the
canonical PRD §5.1 schema, with ``mixpanel_name="run"`` on the start event to
preserve Mixpanel-side continuity for the 219K/week legacy stream.
"""

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tracked_run(monkeypatch):
    """Patch tracking entry points so test invocations don't talk to the
    consent prompt or any real provider, but capture every ``track_event``
    call for assertions."""
    captured: list[tuple[str, dict, dict]] = []

    def _record(event_name, properties=None, *, mixpanel_name=None):
        captured.append((event_name, dict(properties or {}), {"mixpanel_name": mixpanel_name}))

    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", _record)
    # cmdline.py imports tracking as a module; patch the reference there too
    # so the call site sees the recorder, not the original.
    monkeypatch.setattr("comfy_cli.cmdline.tracking.track_event", _record)
    return captured


def _events(captured):
    """Drop the kwargs tuple; return [(name, properties), ...]."""
    return [(name, props) for name, props, _ in captured]


def _event_names(captured):
    return [name for name, _, _ in captured]


class TestRunHappyPath:
    def test_emits_execution_start_then_success(self, runner, tracked_run):
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute") as mock_execute:
            mock_execute.return_value = None
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local", "--where", "local"])

        assert result.exit_code == 0, f"stdout={result.output!r} exc={result.exception!r}"
        assert _event_names(tracked_run) == ["execution_start", "execution_success"]

    def test_execution_start_uses_mixpanel_name_run_alias(self, runner, tracked_run):
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute"):
            runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local"])

        # Only execution_start carries the alias; success/error do not.
        start_kwargs = next(kw for name, _, kw in tracked_run if name == "execution_start")
        success_kwargs = next(kw for name, _, kw in tracked_run if name == "execution_success")
        assert start_kwargs["mixpanel_name"] == "run"
        assert success_kwargs["mixpanel_name"] is None

    def test_properties_carry_workflow_and_other_kwargs(self, runner, tracked_run):
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute"):
            runner.invoke(
                app,
                [
                    "run",
                    "--workflow",
                    "wf.json",
                    "--where",
                    "local",
                    "--timeout",
                    "60",
                    "--host",
                    "1.2.3.4",
                    "--port",
                    "9000",
                ],
            )

        for name, props in _events(tracked_run):
            if name == "execution_start":
                assert props["workflow"] == "wf.json"
                assert props["timeout"] == 60
                assert props["host"] == "1.2.3.4"
                assert props["port"] == 9000
                break
        else:
            pytest.fail("execution_start not emitted")

    def test_api_key_is_redacted_in_lifecycle_properties(self, runner, tracked_run, monkeypatch):
        from comfy_cli.cmdline import app

        # Avoid env var leakage masking the redaction check.
        monkeypatch.delenv("COMFY_API_KEY", raising=False)
        with patch("comfy_cli.cmdline.run_inner.execute"):
            runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local", "--api-key", "sk-supersecret"])

        for name, props in _events(tracked_run):
            assert props.get("api_key") == "<redacted>", f"{name} leaked api_key={props.get('api_key')!r}"
            assert "sk-supersecret" not in str(props)


class TestRunCloudTarget:
    """``--where cloud`` must be analytics-equivalent to the local target.

    The cloud branch used to `return` straight out of the `try` suite, and
    Python skips a try's ``else:`` clause on `return` — so ``execution_success``
    never fired for a cloud run (only `--print-prompt` got it by accident, via
    the ``except typer.Exit`` handler, back when it raised ``Exit(0)``).
    """

    def test_cloud_submit_emits_execution_start_then_success(self, runner, tracked_run, monkeypatch):
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.where.cloud_preflight_or_exit", lambda *a, **kw: None)
        with patch("comfy_cli.cmdline.run_inner.execute_cloud") as mock_cloud:
            mock_cloud.return_value = None
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud"])

        assert result.exit_code == 0, f"stdout={result.output!r} exc={result.exception!r}"
        assert mock_cloud.called
        assert _event_names(tracked_run) == ["execution_start", "execution_success"]

    def test_cloud_print_prompt_emits_execution_success(self, runner, tracked_run, monkeypatch):
        # execute_cloud()'s --print-prompt branch returns rather than raising
        # Exit(0); the success event must still land.
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.where.cloud_preflight_or_exit", lambda *a, **kw: None)
        with patch("comfy_cli.cmdline.run_inner.execute_cloud") as mock_cloud:
            mock_cloud.return_value = None
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud", "--print-prompt"])

        assert result.exit_code == 0, f"stdout={result.output!r} exc={result.exception!r}"
        assert mock_cloud.call_args.kwargs["print_prompt"] is True
        assert _event_names(tracked_run) == ["execution_start", "execution_success"]

    def test_cloud_failure_still_emits_execution_error_only(self, runner, tracked_run, monkeypatch):
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.where.cloud_preflight_or_exit", lambda *a, **kw: None)
        with patch("comfy_cli.cmdline.run_inner.execute_cloud") as mock_cloud:
            mock_cloud.side_effect = typer.Exit(code=1)
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud"])

        assert result.exit_code == 1
        names = _event_names(tracked_run)
        assert "execution_error" in names
        assert "execution_success" not in names

    def test_cloud_target_does_not_resolve_host_port(self, runner, tracked_run, monkeypatch):
        # host/port aren't applicable to the HTTPS+Bearer cloud path; the
        # if/else must keep that resolution off the cloud branch.
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.where.cloud_preflight_or_exit", lambda *a, **kw: None)
        with (
            patch("comfy_cli.cmdline.run_inner.execute_cloud"),
            patch("comfy_cli.host_port.resolve_host_port") as mock_resolve,
        ):
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud"])

        assert result.exit_code == 0, f"stdout={result.output!r} exc={result.exception!r}"
        assert not mock_resolve.called


class TestRunFailurePath:
    def test_typer_exit_1_emits_execution_error_with_exit_code(self, runner, tracked_run):
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute") as mock_execute:
            mock_execute.side_effect = typer.Exit(code=1)
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local"])

        assert result.exit_code == 1
        names = _event_names(tracked_run)
        assert "execution_start" in names
        assert "execution_error" in names
        assert "execution_success" not in names

        err_props = next(props for name, props in _events(tracked_run) if name == "execution_error")
        assert err_props["error_type"] == "Exit"
        assert err_props["exit_code"] == 1

    def test_unexpected_exception_emits_execution_error(self, runner, tracked_run):
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute") as mock_execute:
            mock_execute.side_effect = ValueError("oops")
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local"])

        # The exception propagates; CliRunner surfaces it as a nonzero exit.
        assert result.exit_code != 0
        names = _event_names(tracked_run)
        assert names == ["execution_start", "execution_error"]
        err_props = next(props for name, props in _events(tracked_run) if name == "execution_error")
        assert err_props["error_type"] == "ValueError"

    def test_typer_exit_0_is_treated_as_success(self, runner, tracked_run):
        # A clean early-exit (e.g. --print-prompt currently doesn't do this for
        # `comfy run`, but any future early-success path should be analytics-
        # equivalent to a normal completion).
        from comfy_cli.cmdline import app

        with patch("comfy_cli.cmdline.run_inner.execute") as mock_execute:
            mock_execute.side_effect = typer.Exit(code=0)
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local"])

        assert result.exit_code == 0
        assert _event_names(tracked_run) == ["execution_start", "execution_success"]


class TestRunCloudLifecycle:
    """Cloud submissions must emit the SAME start→success lifecycle as local.

    Regression: the cloud branch `return`ed after `execute_cloud`, skipping the
    try's `else` — so a successful cloud run fired `execution_start` but never
    `execution_success`. Local runs (which fall through) were unaffected.
    """

    def test_cloud_success_emits_start_then_success(self, runner, tracked_run, monkeypatch):
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.cmdline.where_module.cloud_preflight_or_exit", lambda: None)
        with patch("comfy_cli.cmdline.run_inner.execute_cloud") as mock_cloud:
            mock_cloud.return_value = None
            result = runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud"])

        assert result.exit_code == 0, f"stdout={result.output!r} exc={result.exception!r}"
        assert _event_names(tracked_run) == ["execution_start", "execution_success"]
        mock_cloud.assert_called_once()

    def test_resolved_target_rides_on_terminal_events(self, runner, tracked_run, monkeypatch):
        """The resolved routing target (cloud vs local) is recorded so a
        submission is attributable even when --where was defaulted."""
        from comfy_cli.cmdline import app

        monkeypatch.setattr("comfy_cli.cmdline.where_module.cloud_preflight_or_exit", lambda: None)
        with patch("comfy_cli.cmdline.run_inner.execute_cloud"):
            runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "cloud"])
        cloud_success = next(props for name, props in _events(tracked_run) if name == "execution_success")
        assert cloud_success.get("target") == "cloud"

        tracked_run.clear()
        with patch("comfy_cli.cmdline.run_inner.execute"):
            runner.invoke(app, ["run", "--workflow", "wf.json", "--where", "local"])
        local_success = next(props for name, props in _events(tracked_run) if name == "execution_success")
        assert local_success.get("target") == "local"
