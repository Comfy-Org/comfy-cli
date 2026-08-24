"""Execution lifecycle + sub-action event tests for ``comfy generate``."""

import httpx
import pytest
import typer
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import app as gen_app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "comfyui-test")
    return "comfyui-test"


@pytest.fixture
def captured_events(monkeypatch):
    """Drop-in replacement for the autouse track_event mock that records every
    call so tests can make assertions on the emitted lifecycle."""
    events: list[tuple[str, dict]] = []

    def _record(event_name, properties=None, *, mixpanel_name=None):
        events.append((event_name, dict(properties or {})))

    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", _record)
    monkeypatch.setattr("comfy_cli.command.generate.app.tracking.track_event", _record)
    monkeypatch.setattr("comfy_cli.cmdline.tracking.track_event", _record)
    return events


def _names(events):
    return [name for name, _ in events]


def _props(events, target_name):
    return [props for name, props in events if name == target_name]


class TestSubActionEvents:
    def test_list_fires_generate_list(self, runner, captured_events):
        r = runner.invoke(cli_app, ["generate", "list"])
        assert r.exit_code == 0
        assert "generate:list" in _names(captured_events)

    def test_schema_fires_generate_schema_with_model(self, runner, captured_events):
        r = runner.invoke(cli_app, ["generate", "schema", "flux-pro"])
        assert r.exit_code == 0
        schema_events = _props(captured_events, "generate:schema")
        assert len(schema_events) == 1
        assert schema_events[0]["model"] == "flux-pro"

    def test_schema_without_model_records_none_model(self, runner, captured_events):
        # ``comfy generate schema`` prints an error but the sub-action event
        # still fires so we can see invalid usage in analytics.
        r = runner.invoke(cli_app, ["generate", "schema"])
        assert r.exit_code == 1
        schema_events = _props(captured_events, "generate:schema")
        assert len(schema_events) == 1
        assert schema_events[0]["model"] is None

    def test_resume_fires_generate_resume_with_model_and_job(self, runner, captured_events):
        r = runner.invoke(cli_app, ["generate", "resume", "flux-pro", "job-abc"])
        # Exit code 1 because the synthetic poll won't reach a real server, but
        # the sub-action event must have fired before that.
        resume_events = _props(captured_events, "generate:resume")
        assert len(resume_events) == 1
        assert resume_events[0]["model"] == "flux-pro"
        assert resume_events[0]["job_id"] == "job-abc"
        del r  # exit code not relevant here — sub-action event firing is

    def test_resume_flag_like_first_arg_does_not_become_model(self, runner, captured_events):
        # If the first positional looks like a flag (e.g. ``--help``), don't
        # record it as the model name — that'd be noise in analytics.
        runner.invoke(cli_app, ["generate", "resume", "--help"])
        resume_events = _props(captured_events, "generate:resume")
        assert len(resume_events) == 1
        assert resume_events[0]["model"] is None
        assert resume_events[0]["job_id"] is None

    def test_resume_flag_like_job_id_surfaces_usage_error(self, runner, captured_events):
        # Keep the telemetry-side flag-rejection (line 75) consistent with what
        # ``_resume()`` accepts — otherwise ``resume flux-pro --json`` polls a
        # job called ``--json`` instead of showing the usage hint.
        r = runner.invoke(cli_app, ["generate", "resume", "flux-pro", "--json"])
        assert r.exit_code == 1
        assert "Usage: comfy generate resume" in r.output

    @pytest.mark.parametrize(
        ("argv", "usage"),
        [
            (["--no-json", "generate", "upload"], "Usage: comfy generate upload <file-or-url> [--json]"),
            (
                ["--no-json", "generate", "resume"],
                "Usage: comfy generate resume <model> <job_id> [--download PATH] [--json]",
            ),
        ],
    )
    def test_usage_errors_render_bracketed_flags_literally(self, runner, captured_events, argv, usage):
        # `--no-json` is load-bearing: under CliRunner stdout is not a TTY, so the
        # renderer defaults to JSON (renderer.py precedence rule 6) and the assertion
        # would pass on the envelope's `message` field without ever rendering markup.
        # The pretty branch is what an interactive user actually sees, and it feeds
        # the text through Rich — `[--json]` / `[--download PATH]` survive only
        # because `_fail` runs `message` through `sanitize_markup`. A hand-rolled
        # unescaped `pretty=` override would swallow those brackets (or raise
        # MarkupError on a tag-shaped token); this pins the rendered line so that
        # regresses loudly.
        r = runner.invoke(cli_app, argv)
        assert r.exit_code == 1
        assert usage in r.output

    def test_refresh_fires_generate_refresh(self, runner, captured_events, monkeypatch):
        # Mock the httpx call so we don't actually hit the network.
        monkeypatch.setattr(
            "comfy_cli.command.generate.app.httpx.Client",
            lambda *a, **kw: _FakeRefreshClient(),
        )
        runner.invoke(cli_app, ["generate", "refresh"])
        assert "generate:refresh" in _names(captured_events)

    def test_upload_fires_generate_upload(self, runner, captured_events, api_key, monkeypatch):
        # Stub the upload boundary so the event fires without network IO.
        def _raise_apiError(*a, **kw):
            from comfy_cli.command.generate import client as _client

            raise _client.ApiError(0, "", "stubbed")

        monkeypatch.setattr("comfy_cli.command.generate.app.upload.upload_target", _raise_apiError)
        runner.invoke(cli_app, ["generate", "upload", "/tmp/does-not-exist.png"])
        assert "generate:upload" in _names(captured_events)

    def test_no_generate_lifecycle_on_sub_actions(self, runner, captured_events):
        """Sub-actions are not partner-node invocations — they must not fire
        ``generate:start/success/error`` (or it'd inflate the parity-window
        counts) or ``execution_*`` (which is reserved for ``comfy run``)."""
        runner.invoke(cli_app, ["generate", "list"])
        names = _names(captured_events)
        assert "generate:start" not in names
        assert "generate:success" not in names
        assert "generate:error" not in names
        assert "execution_start" not in names
        assert "execution_success" not in names
        assert "execution_error" not in names


class _FakeRefreshClient:
    """Minimal stand-in for httpx.Client used by ``comfy generate refresh``."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, *a, **kw):
        return httpx.Response(200, text="openapi: 3.0.0\npaths: {}\n")


class TestGenerateExecutionHappyPath:
    def test_sync_success_emits_start_then_success(self, runner, captured_events, api_key, monkeypatch):
        resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)

        r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])
        assert r.exit_code == 0, r.stdout

        names = _names(captured_events)
        assert names.count("generate:start") == 1
        assert names.count("generate:success") == 1
        assert "generate:error" not in names
        # Must not bleed into the workflow-lifecycle namespace.
        assert "execution_start" not in names
        assert "execution_success" not in names

    def test_generate_start_props_carry_model_and_partner(self, runner, captured_events, api_key, monkeypatch):
        resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
        runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])

        start_props = _props(captured_events, "generate:start")[0]
        assert start_props["model"] == "dalle"
        # NOTE: start_props captures the snapshot at function entry, which
        # is BEFORE spec lookup populates partner/model_alias on the shared
        # gen_props dict. The success/error events that fire later carry
        # the fully-populated values.
        success_props = _props(captured_events, "generate:success")[0]
        assert success_props["model"] == "dalle"
        assert success_props["partner"] == "openai"
        assert success_props["async"] is False
        assert success_props["has_download"] is False

    def test_download_flag_sets_has_download(self, runner, captured_events, api_key, tmp_path, monkeypatch):
        resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
        monkeypatch.setattr("comfy_cli.command.generate.client.download_bytes", lambda *a, **kw: b"png")
        download = str(tmp_path / "out.png")
        runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--download", download])

        success_props = _props(captured_events, "generate:success")[0]
        assert success_props["has_download"] is True


class TestGenerateExecutionErrorPaths:
    def test_api_error_emits_generate_error_with_kind_api(self, runner, captured_events, api_key, monkeypatch):
        resp = httpx.Response(401, json={"message": "Invalid token"})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)

        r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
        assert r.exit_code == 1

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "api"
        assert "generate:success" not in _names(captured_events)

    def test_network_error_emits_generate_error_with_kind_network(self, runner, captured_events, api_key, monkeypatch):
        def boom(*a, **kw):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(gen_app.client.httpx, "post", boom)
        r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
        assert r.exit_code == 1

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "network"

    def test_non_json_response_emits_generate_error_with_kind_non_json(
        self, runner, captured_events, api_key, monkeypatch
    ):
        resp = httpx.Response(200, text="not really json", headers={"content-type": "text/plain"})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
        r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])
        assert r.exit_code == 1

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "non_json_response"

    def test_schema_error_in_args_emits_generate_error_with_kind_schema(self, runner, captured_events, api_key):
        r = runner.invoke(
            cli_app,
            ["generate", "flux-pro", "--prompt", "x", "--width", "abc", "--height", "1"],
        )
        assert r.exit_code == 1

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "schema"

    def test_unknown_model_emits_generate_error_with_kind_schema(self, runner, captured_events, api_key):
        # Pre-validation consistency: typing a bogus model name fires the full
        # generate:start → generate:error pair (mirrors comfy run firing
        # execution_start → execution_error for a bogus workflow path).
        r = runner.invoke(cli_app, ["generate", "bogus-model-name", "--prompt", "x"])
        assert r.exit_code == 1

        names = _names(captured_events)
        assert names.count("generate:start") == 1
        assert names.count("generate:error") == 1
        err_props = _props(captured_events, "generate:error")[0]
        assert err_props["error_kind"] == "schema"
        assert err_props["model"] == "bogus-model-name"
        # model_alias/partner stay None because the lookup never succeeded.
        assert err_props["model_alias"] is None
        assert err_props["partner"] is None

    def test_upload_failure_emits_generate_error_with_kind_upload(
        self, runner, captured_events, api_key, tmp_path, monkeypatch
    ):
        # _apply_upload_transforms calls ``upload.upload_path`` when a local
        # file path is supplied for a flag with ``upload_mode="url"``. If the
        # upload service errors, ``_generate`` emits ``error_kind=upload``.
        local_file = tmp_path / "webhook.json"
        local_file.write_bytes(b"{}")

        def _boom_upload(path, key):
            from comfy_cli.command.generate import client as _client

            raise _client.ApiError(500, "", "upload service down")

        monkeypatch.setattr("comfy_cli.command.generate.app.upload.upload_path", _boom_upload)

        # flux-pro's ``webhook_url`` flag has ``upload_mode="url"``; passing a
        # local file path triggers the upload transform on the way in.
        r = runner.invoke(
            cli_app,
            [
                "generate",
                "flux-pro",
                "--prompt",
                "x",
                "--width",
                "1",
                "--height",
                "1",
                "--webhook_url",
                str(local_file),
            ],
        )
        assert r.exit_code == 1

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "upload"

    def test_unexpected_exception_emits_generate_error_with_kind_unknown(
        self, runner, captured_events, api_key, monkeypatch
    ):
        # Safety net: anything that isn't a known exception type should still
        # produce a paired generate:error so generate:start isn't orphaned.
        def boom(*a, **kw):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(gen_app.client.httpx, "post", boom)
        r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
        assert r.exit_code != 0

        err_props = _props(captured_events, "generate:error")
        assert len(err_props) == 1
        assert err_props[0]["error_kind"] == "unknown"
        assert err_props[0]["error_type"] == "RuntimeError"


class TestGenerateAsyncSubmission:
    def test_async_emits_generate_submitted_not_success(self, runner, captured_events, api_key, monkeypatch):
        submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
        monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)

        r = runner.invoke(
            cli_app,
            ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1", "--async"],
        )
        assert r.exit_code == 0

        names = _names(captured_events)
        assert "generate:start" in names
        assert "generate:submitted" in names
        # Critical: async submission is not "succeeded" — completion happens on
        # a later ``comfy generate resume`` invocation.
        assert "generate:success" not in names
        assert "generate:error" not in names

        submitted = _props(captured_events, "generate:submitted")[0]
        assert submitted["model"] == "flux-pro"
        assert submitted["job_id"] == "job-xyz"


class TestGenerateHelp:
    def test_per_model_help_does_not_fire_lifecycle(self, runner, captured_events):
        # ``--help`` is a help-display action, not an execution attempt.
        r = runner.invoke(cli_app, ["generate", "flux-pro", "--help"])
        assert r.exit_code == 0
        names = _names(captured_events)
        assert "generate:start" not in names
        assert "generate:success" not in names
        assert "generate:error" not in names


class TestBailHelper:
    """`_bail` owns the fail → track → exit order that ten `_generate` branches
    used to spell out inline. These pin that contract directly, so a future edit
    to the helper can't quietly reorder or drop a step for all ten at once."""

    def test_reports_then_tracks_then_exits(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(gen_app, "_fail", lambda **kw: calls.append(("fail", kw)))
        exc = ValueError("boom")

        with pytest.raises(typer.Exit) as exit_info:
            gen_app._bail(
                lambda kind, e: calls.append(("track", kind, e)),
                exc,
                code="generate_bad_args",
                message="bad",
                kind="schema",
            )

        assert exit_info.value.exit_code == 1
        # The reporting call comes first so the user-facing error is already out
        # when the (best-effort, network-bound) tracking call runs.
        assert [c[0] for c in calls] == ["fail", "track"]
        # `_fail`'s four optional keywords are named parameters on `_bail`, so they
        # reach `_fail` at their defaults even when the call site omits them.
        assert calls[0][1] == {
            "code": "generate_bad_args",
            "message": "bad",
            "hint": None,
            "details": None,
            "legacy_json": False,
            "pretty": None,
        }
        assert calls[1][1:] == ("schema", exc)
        # Chained, so the originating exception stays reachable for debugging.
        assert exit_info.value.__cause__ is exc

    def test_forwards_fail_only_kwargs(self, monkeypatch):
        seen: dict = {}
        monkeypatch.setattr(gen_app, "_fail", lambda **kw: seen.update(kw))

        with pytest.raises(typer.Exit):
            gen_app._bail(
                lambda kind, e: None,
                RuntimeError("x"),
                code="generate_api_error",
                message="API error 401",
                kind="api",
                details={"status": 401},
                pretty="[bold red]API error 401[/bold red]",
                hint="check your key",
                legacy_json=True,
            )

        # `kind`/`exc` are the helper's own; everything else reaches `_fail` as-is.
        assert seen == {
            "code": "generate_api_error",
            "message": "API error 401",
            "details": {"status": 401},
            "pretty": "[bold red]API error 401[/bold red]",
            "hint": "check your key",
            "legacy_json": True,
        }

    def test_unknown_fail_kwarg_is_rejected_at_the_call_site(self, monkeypatch):
        """Named parameters, not `**kwargs`: a misspelled option can't reach `_fail`
        and blow up mid-report, after the envelope decision but before the exit."""
        monkeypatch.setattr(gen_app, "_fail", lambda **kw: None)

        with pytest.raises(TypeError, match="detials"):
            gen_app._bail(
                lambda kind, e: None,
                RuntimeError("x"),
                code="generate_api_error",
                message="boom",
                kind="api",
                detials={"status": 401},
            )

    def test_exits_even_when_tracking_raises(self, monkeypatch):
        """Tracking is best-effort and network-bound; the exit is not. A telemetry
        failure must not escape as a traceback in place of the exit-1 envelope."""
        reported: list[dict] = []
        monkeypatch.setattr(gen_app, "_fail", lambda **kw: reported.append(kw))
        exc = ValueError("boom")

        def _explode(kind, e):
            raise RuntimeError("mixpanel is down")

        with pytest.raises(typer.Exit) as exit_info:
            gen_app._bail(_explode, exc, code="generate_bad_args", message="bad", kind="schema")

        assert exit_info.value.exit_code == 1
        # Still chained to the *original* failure, not to the telemetry error.
        assert exit_info.value.__cause__ is exc
        assert len(reported) == 1
