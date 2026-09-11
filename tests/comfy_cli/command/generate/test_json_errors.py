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


# ─── the flag-token case: a flag token where the model alias belongs ────────


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
    objs = [json.loads(ln) for ln in r.stdout.strip().splitlines() if ln.strip()]
    # Stream mode may legitimately precede the envelope with progress events, so
    # the contract is terminal-line, not sole-line. What must NOT happen is a
    # SECOND error object — the legacy flat `{"error": ...}` alongside the
    # envelope would leave a caller reading whichever it hit first.
    errors = [o for o in objs if "error" in o]
    assert len(errors) == 1, f"expected exactly one error object, got {errors!r}"
    last = objs[-1]
    assert errors[0] is last
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
    r = runner.invoke(cli_app, ["--json", "generate", "flux-pro", "--width", "1", "--height", "1"])
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
    r = runner.invoke(cli_app, ["--no-json", "generate", "flux-pro", "--width", "1", "--height", "1"])
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


# ─── review follow-ups: the remaining traceback / mis-code paths ─────────


def test_json_resume_bad_timeout_envelope(runner, api_key):
    """`generate resume … --timeout nope` parsed the value with a bare `float()`,
    so the `ValueError` escaped `main()` (which traps only KeyboardInterrupt /
    typer.Exit / SystemExit) as a traceback — exit 1, blank stdout, the exact
    shape this change exists to remove. The submit path already guarded it."""
    r = runner.invoke(cli_app, ["--json", "generate", "resume", "flux-pro", "job-1", "--timeout", "nope"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_timeout_invalid"


def test_json_non_json_poll_body_envelope(runner, api_key, monkeypatch):
    """A 200 poll response with an unparseable body (a proxy/CDN HTML error
    page) raised a bare `ValueError` out of `resp.json()` — neither `ApiError`
    nor `httpx.HTTPError`, so every caller's `except` tuple missed it."""
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        lambda *a, **kw: httpx.Response(200, text="<html>502 Bad Gateway</html>"),
    )
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    r = runner.invoke(cli_app, ["--json", *_GEN_ARGS])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    assert env["error"]["code"] == "generate_api_error"
    assert "not JSON" in env["error"]["message"]


def test_json_non_dict_poll_body_envelope(runner, api_key, monkeypatch):
    """Valid JSON that isn't an object would blow up on `.get()` a line later."""
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        lambda *a, **kw: httpx.Response(200, json=["not", "an", "object"]),
    )
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    r = runner.invoke(cli_app, ["--json", *_GEN_ARGS])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_api_error"


def test_http_status_error_is_an_api_error_not_a_network_one(runner, api_key, monkeypatch):
    """`upload_remote_url` calls `raise_for_status`, so a 404 source URL arrives
    as `httpx.HTTPStatusError` — an `httpx.HTTPError` subclass. Reporting that
    as `generate_network_error` tells automation to check connectivity and
    retry, which is the wrong move for a 4xx that reached the server fine."""
    req = httpx.Request("GET", "https://host/missing.png")
    err = httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))
    monkeypatch.setattr(gen_app.upload, "upload_target", lambda *a, **kw: (_ for _ in ()).throw(err))
    r = runner.invoke(cli_app, ["--json", "generate", "upload", "https://host/missing.png"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_api_error"


def test_transport_error_stays_a_network_error(runner, api_key, monkeypatch):
    """The sibling of the above: a genuine transport failure keeps its code."""
    err = httpx.ConnectError("connection refused")
    monkeypatch.setattr(gen_app.upload, "upload_target", lambda *a, **kw: (_ for _ in ()).throw(err))
    r = runner.invoke(cli_app, ["--json", "generate", "upload", "https://host/x.png"])
    assert r.exit_code == 1
    assert _sole_envelope(r)["error"]["code"] == "generate_network_error"


def test_pretty_server_text_with_rich_tags_does_not_raise_markup_error(runner, api_key, monkeypatch):
    """Server-controlled text is interpolated into the pretty error line, and a
    bracketed token Rich reads as a closing tag (`[/path]`) raised MarkupError —
    turning a clean error into a traceback."""
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(400, json={"detail": "bad input at [/path] and [link]"}),
    )
    r = runner.invoke(cli_app, ["--no-json", *_GEN_ARGS])
    assert r.exit_code == 1
    assert r.exception is None or isinstance(r.exception, SystemExit)
    # The literal text survives instead of being eaten as markup.
    assert "[/path]" in r.stdout
    assert "[link]" in r.stdout


def test_pretty_failed_job_reason_with_rich_tags_survives(capsys):
    """Same hazard on the partner's terminal-failure reason, which `_emit_result`
    interpolates into its own red line rather than going through `_fail`."""
    import typer

    from comfy_cli.output.renderer import Renderer, get_renderer, reset_renderer_for_testing, set_renderer

    set_renderer(Renderer.resolve(no_json_flag=True, env={}, is_stdout_tty=True))
    try:
        assert not get_renderer().is_json()
        result = gen_app.poll.PollResult(
            status="failed", raw={"status": "x"}, image_urls=[], error="moderated at [/path]"
        )
        with pytest.raises(typer.Exit):
            gen_app._emit_result(result, request_id="job-1", download=None, as_json=False)
    finally:
        reset_renderer_for_testing()
    assert "[/path]" in capsys.readouterr().out


# ─── #614: markup escaping alone is not enough for remote text ───────────
#
# `rich.markup.escape` neutralizes `[tag]` but passes `\x1b` through untouched, so
# these paths need `sanitize_markup`. Without it a partner API can emit CSI/OSC and
# clear the user's screen, rewrite the window title, or repaint earlier lines to
# spoof CLI output. `generate` prints via a bare `rich.print`, so it gets none of
# `Renderer`'s automatic coverage and must sanitize at each site itself.

# ESC[2J clears the screen; ESC]0;…BEL rewrites the window title; ESC[31m restyles.
_ANSI_ATTACK = "denied \x1b[2J\x1b]0;pwned\x07 really \x1b[31mred"


def _assert_ansi_neutralized(text: str) -> None:
    assert "\x1b[2J" not in text, "screen-clear sequence reached the terminal"
    assert "\x1b]0;" not in text, "window-title sequence reached the terminal"
    assert "\x1b" not in text
    # The human-readable words survive — only the control bytes are dropped.
    assert "denied" in text and "really" in text


def test_pretty_api_error_body_strips_ansi(runner, api_key, monkeypatch):
    """`content=` (not `json=`) on purpose: `e.body` is the response's *raw text*, so
    a compliant server that encodes ESC as `\\u001b` is already inert and would make
    this test vacuous. The hazard is a body carrying live escape bytes, which is
    exactly what a non-JSON error page from a proxy or gateway delivers."""
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(400, content=_ANSI_ATTACK.encode()),
    )
    r = runner.invoke(cli_app, ["--no-json", *_GEN_ARGS])
    assert r.exit_code == 1
    _assert_ansi_neutralized(r.stdout)


def test_pretty_non_json_response_preview_strips_ansi(runner, api_key, monkeypatch):
    """The `resp.text[:500]` preview path — a 200 whose body isn't JSON. Same raw-byte
    exposure as above, reached through a different `_fail` call site."""
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(200, content=_ANSI_ATTACK.encode()),
    )
    r = runner.invoke(cli_app, ["--no-json", *_GEN_ARGS])
    assert r.exit_code == 1
    _assert_ansi_neutralized(r.stdout)


def test_pretty_failed_job_reason_strips_ansi(capsys):
    """The `_emit_result` red line, which builds its own markup instead of `_fail`'s."""
    import typer

    from comfy_cli.output.renderer import Renderer, get_renderer, reset_renderer_for_testing, set_renderer

    set_renderer(Renderer.resolve(no_json_flag=True, env={}, is_stdout_tty=True))
    try:
        assert not get_renderer().is_json()
        result = gen_app.poll.PollResult(status="failed", raw={"status": "x"}, image_urls=[], error=_ANSI_ATTACK)
        with pytest.raises(typer.Exit):
            gen_app._emit_result(result, request_id="job-1", download=None, as_json=False)
    finally:
        reset_renderer_for_testing()
    _assert_ansi_neutralized(capsys.readouterr().out)


def test_pretty_schema_unknown_model_with_markup_does_not_traceback(runner):
    """`generate schema`'s pretty branch interpolated the `SpecError` — which quotes
    the requested alias verbatim — straight into markup, so a name Rich reads as an
    unbalanced closing tag crashed with `MarkupError` and printed NOTHING. Exit 1
    with empty stdout is the undiagnosable failure this module is meant to prevent,
    so it is pinned here even though the line arrived via #621 rather than this PR.
    """
    r = runner.invoke(cli_app, ["--no-json", "generate", "schema", "[/bold]"])
    assert r.exit_code == 1
    assert r.exception is None or isinstance(r.exception, SystemExit), r.exception
    # Something actionable actually reached the user.
    assert r.stdout.strip()
    assert "[/bold]" in r.stdout


def test_json_envelope_preserves_ansi_bytes(runner, api_key, monkeypatch):
    """The machine path must NOT sanitize: `json.dumps` already encodes `\\x1b` as a
    `\\u001b` escape, which is inert on the wire, and stripping it here would
    silently mutate the bytes an agent parses (and diverge from the raw response
    it may be diffing against). Pins the pretty/machine split so a future
    'sanitize everywhere' change can't quietly corrupt envelope data."""
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(400, content=_ANSI_ATTACK.encode()),
    )
    r = runner.invoke(cli_app, ["--json", *_GEN_ARGS])
    assert r.exit_code == 1
    env = _sole_envelope(r)
    # Encoded, not emitted: the envelope line carries no live escape byte, so nothing
    # acts on the terminal even though the data is fully intact.
    assert "\x1b" not in r.stdout
    assert "\\u001b" in r.stdout
    # And the decoded value is the partner's string byte-for-byte — sanitizing here
    # would have silently dropped the ESCs from what the agent parses.
    assert env["error"]["details"]["body"] == _ANSI_ATTACK
