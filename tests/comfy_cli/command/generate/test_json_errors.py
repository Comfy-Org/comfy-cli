"""`comfy generate` failure paths under the GLOBAL machine-output modes.

Every `comfy generate` error used to reach an envelope-consuming caller as exit
1 with a bare `rich.print` line on stdout and nothing parseable — the failure
mode that made a malformed `comfy --json generate --prompt=x` undiagnosable from
the caller side (the flag token was swallowed by the model-alias positional and
`spec.get_endpoint` raised deep inside).

These tests pin the contract from the caller's side: under global `--json` /
`--json-stream`, a failed `comfy generate` always emits exactly ONE `envelope/1`
error object on stdout with a stable `code`, and exits non-zero. Pretty mode
(forced here with the global `--no-json`, since CliRunner's piped stdout would
otherwise auto-resolve to JSON) keeps its historical rich output.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from comfy_cli import error_codes
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import app as gen_app


@pytest.fixture(autouse=True)
def disable_tracking_prompt(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "comfyui-test")
    return "comfyui-test"


def _sole_envelope(result) -> dict:
    """Parse stdout as exactly one `envelope/1` error line, asserting the shape
    every consumer relies on. The single-line assertion is the point: a second
    JSON object (the legacy flat error) would break naive callers."""
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected a single envelope line, got {lines!r}"
    env = json.loads(lines[0])
    assert env["schema"] == "envelope/1"
    assert env["type"] == "envelope"
    assert env["ok"] is False
    assert env["error"]["message"]
    assert error_codes.is_registered(env["error"]["code"])
    return env


# ─── the BE-4571 case: a flag token where the model alias belongs ────────


def test_json_flag_shaped_target_emits_target_required_envelope(runner):
    """`comfy --json generate --prompt=x` used to exit 1 with NO json on stdout
    and empty stderr. It now names the real problem."""
    r = runner.invoke(cli_app, ["--json", "generate", "--prompt=x"])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_target_required"
    assert "model alias" in env["error"]["message"]
    # Actionable: it points at both the right generate invocation and the local
    # alternative, so the caller isn't left at a dead end.
    assert "run-template" in env["error"]["hint"]


@pytest.mark.parametrize("token", ["--prompt=x", "-p", "--width"])
def test_flag_shaped_targets_all_rejected(runner, token):
    r = runner.invoke(cli_app, ["--json", "generate", token])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_target_required"


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flags_are_not_treated_as_a_flag_shaped_target(runner, flag):
    """The guard must not swallow the help short-circuit."""
    r = runner.invoke(cli_app, ["generate", flag])
    assert r.exit_code == 0
    assert "comfy generate" in r.stdout


def test_json_stream_target_required_envelope_is_the_terminal_line(runner):
    r = runner.invoke(cli_app, ["--json-stream", "generate", "--prompt=x"])
    assert r.exit_code == 1
    last = json.loads(r.stdout.strip().splitlines()[-1])
    assert last["schema"] == "envelope/1"
    assert last["ok"] is False
    assert last["error"]["code"] == "generate_target_required"


def test_pretty_flag_shaped_target_stays_rich(runner):
    r = runner.invoke(cli_app, ["--no-json", "generate", "--prompt=x"])
    assert r.exit_code == 1
    assert "model alias" in r.stdout
    assert "run-template" in r.stdout
    assert "envelope/1" not in r.stdout


# ─── the other JSON-mode error paths ─────────────────────────────────────


def test_json_unknown_model_envelope(runner, api_key):
    r = runner.invoke(cli_app, ["--json", "generate", "not-a-real-model", "--prompt=x"])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_unknown_model"
    assert env["error"]["details"]["model"] == "not-a-real-model"
    assert env["error"]["hint"]  # falls back to the registered hint


def test_json_missing_required_param_envelope(runner, api_key):
    r = runner.invoke(cli_app, ["--json", "generate", "flux-pro", "--prompt", "x"])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_bad_args"
    assert "comfy generate schema" in env["error"]["hint"]


def test_json_bad_timeout_envelope(runner, api_key):
    r = runner.invoke(
        cli_app,
        ["--json", "generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1", "--timeout", "nope"],
    )
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_timeout_invalid"


def test_json_api_error_envelope_carries_status_and_body(runner, api_key, monkeypatch):
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(401, json={"message": "Invalid token"}),
    )
    r = runner.invoke(cli_app, ["--json", "generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_api_error"
    assert env["error"]["details"]["status"] == 401
    assert "Invalid token" in env["error"]["details"]["body"]


def test_json_network_error_envelope(runner, api_key, monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(gen_app.client.httpx, "post", boom)
    r = runner.invoke(cli_app, ["--json", "generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_network_error"


def test_json_schema_subcommand_unknown_model_envelope(runner):
    r = runner.invoke(cli_app, ["--json", "generate", "schema", "not-a-real-model"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_unknown_model"


def test_json_schema_subcommand_usage_envelope(runner):
    r = runner.invoke(cli_app, ["--json", "generate", "schema"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_bad_args"


def test_json_resume_usage_envelope(runner, api_key):
    r = runner.invoke(cli_app, ["--json", "generate", "resume"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_bad_args"


def test_json_upload_usage_envelope(runner, api_key):
    r = runner.invoke(cli_app, ["--json", "generate", "upload"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_bad_args"


def test_json_list_bad_meta_flag_envelope(runner):
    """`--download` with no value used to escape `_list_models` as an unhandled
    SchemaError traceback."""
    r = runner.invoke(cli_app, ["--json", "generate", "list", "--download"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_bad_args"


# ─── pretty-mode regression guards ───────────────────────────────────────


def test_pretty_unknown_model_output_unchanged(runner, api_key):
    r = runner.invoke(cli_app, ["--no-json", "generate", "not-a-real-model", "--prompt=x"])
    assert r.exit_code == 1
    assert "Unknown model" in r.stdout
    assert "comfy generate list" in r.stdout
    assert "envelope/1" not in r.stdout


def test_pretty_missing_required_still_suggests_schema(runner, api_key):
    r = runner.invoke(cli_app, ["--no-json", "generate", "flux-pro", "--prompt", "x"])
    assert r.exit_code == 1
    assert "Missing required" in r.stdout
    assert "comfy generate schema flux-pro" in r.stdout
    assert "envelope/1" not in r.stdout


def _mock_failed_job(monkeypatch):
    """Submit succeeds, the async poll settles on a non-succeeded terminal state."""
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    poll_fail = httpx.Response(200, json={"status": "Content Moderated", "progress": 0.0})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: poll_fail)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)


_GEN_ARGS = ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"]


def test_json_failed_job_envelope(runner, api_key, monkeypatch):
    """A terminal non-succeeded job is a failure, not a result: under global
    `--json` (and no command-local `--json`) it gets an envelope, not the red
    line + raw payload."""
    _mock_failed_job(monkeypatch)
    r = runner.invoke(cli_app, ["--json", *_GEN_ARGS])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_job_failed"
    assert env["error"]["details"]["response"]["status"] == "Content Moderated"


def test_pretty_failed_job_output_unchanged(runner, api_key, monkeypatch):
    _mock_failed_job(monkeypatch)
    r = runner.invoke(cli_app, ["--no-json", *_GEN_ARGS])
    assert r.exit_code == 1
    assert "failed" in r.stdout.lower()
    assert "envelope/1" not in r.stdout
