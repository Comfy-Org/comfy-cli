"""End-to-end tests for ``comfy generate`` via Typer's CliRunner.

These cover the dispatch table (list/schema/refresh/resume vs. model alias) and
each major run path with httpx mocked at the boundary.
"""

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import app as gen_app


@pytest.fixture(autouse=True)
def disable_tracking_prompt(monkeypatch):
    """The mixpanel-consent prompt blocks Typer invocations in CI (no TTY).
    Existing CLI tests pass --skip-prompt; we do the same here implicitly."""
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "comfyui-test")
    return "comfyui-test"


# ─── Dispatch / top-level help ────────────────────────────────────────────


def test_no_args_prints_top_help(runner):
    r = runner.invoke(cli_app, ["generate"])
    assert r.exit_code == 0
    assert "comfy generate" in r.stdout
    assert "Examples" in r.stdout


def test_top_help_via_dash_help(runner):
    r = runner.invoke(cli_app, ["generate", "--help"])
    assert r.exit_code == 0
    assert "comfy generate" in r.stdout


# ─── list ────────────────────────────────────────────────────────────────


def test_list_shows_aliases(runner):
    r = runner.invoke(cli_app, ["generate", "list"])
    assert r.exit_code == 0
    assert "flux-pro" in r.stdout


def test_list_partner_filter(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--partner", "bfl"])
    assert r.exit_code == 0
    assert "flux-pro" in r.stdout
    assert "ideogram" not in r.stdout


def test_list_partner_eq_form(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--partner=bfl"])
    assert r.exit_code == 0
    assert "flux-pro" in r.stdout


def test_list_style_filter(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--style", "image-edit"])
    assert r.exit_code == 0
    assert "edit" in r.stdout.lower()


def test_list_query_filter(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--query", "ideogram"])
    assert r.exit_code == 0
    assert "ideogram" in r.stdout


def test_list_no_matches(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--partner", "nonexistent"])
    assert r.exit_code == 0
    assert "No models" in r.stdout


# ─── schema ──────────────────────────────────────────────────────────────


def test_schema_alias(runner):
    r = runner.invoke(cli_app, ["generate", "schema", "flux-pro"])
    assert r.exit_code == 0
    assert "prompt" in r.stdout
    assert "Example" in r.stdout


def test_schema_full_path(runner):
    r = runner.invoke(cli_app, ["generate", "schema", "bfl/flux-pro-1.1/generate"])
    assert r.exit_code == 0
    assert "prompt" in r.stdout


def test_schema_missing_arg(runner):
    r = runner.invoke(cli_app, ["generate", "schema"])
    assert r.exit_code == 1
    assert "Usage" in r.stdout


def test_schema_unknown_model(runner):
    r = runner.invoke(cli_app, ["generate", "schema", "bogus-model"])
    assert r.exit_code == 1
    assert "Unknown model" in r.stdout


# ─── per-model --help passes through to schema view ─────────────────────


def test_per_model_help(runner):
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--help"])
    assert r.exit_code == 0
    assert "Model:" in r.stdout
    assert "prompt" in r.stdout


# ─── generate happy / error paths ───────────────────────────────────────


def test_generate_missing_api_key(runner, monkeypatch):
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"],
    )
    assert r.exit_code == 1
    assert "No API key" in r.stdout


def test_generate_bad_int_suggests_schema(runner, api_key):
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "abc", "--height", "1"],
    )
    assert r.exit_code == 1
    assert "expected integer" in r.stdout
    assert "comfy generate schema" in r.stdout


def test_generate_unknown_model(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "bogus-name", "--prompt", "x"])
    assert r.exit_code == 1
    assert "Unknown model" in r.stdout


def test_generate_missing_required(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x"])
    assert r.exit_code == 1
    assert "Missing required" in r.stdout


def test_generate_bad_timeout(runner, api_key, monkeypatch):
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(200, json={"id": "x", "polling_url": "https://x"}),
    )
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1", "--timeout", "not-a-num"],
    )
    assert r.exit_code == 1
    assert "--timeout" in r.stdout


# ─── generate: async polling path (BFL) ─────────────────────────────────


def test_generate_async_sync_poll_to_ready(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    poll_done = httpx.Response(
        200,
        json={
            "status": "Ready",
            "progress": 1.0,
            "result": {"sample": "https://cdn.example/result.png"},
        },
    )
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: poll_done)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)

    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 0, r.stdout
    assert "https://cdn.example/result.png" in r.stdout


def test_generate_async_returns_job_id(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1", "--async"],
    )
    assert r.exit_code == 0
    assert "Submitted" in r.stdout
    assert "job-xyz" in r.stdout
    assert "comfy generate resume" in r.stdout


def test_generate_async_failure_status(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})
    poll_fail = httpx.Response(200, json={"status": "Content Moderated", "progress": 0.0})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: poll_fail)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 1
    assert "failed" in r.stdout.lower()


# ─── generate: sync JSON response with URL outputs ──────────────────────


def test_generate_sync_prints_url(runner, api_key, monkeypatch):
    resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])
    assert r.exit_code == 0, r.stdout
    assert "https://cdn.example/a.png" in r.stdout


def test_generate_sync_with_download(runner, api_key, tmp_path, monkeypatch):
    resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    monkeypatch.setattr("comfy_cli.command.generate.client.download_bytes", lambda *a, **kw: b"png-bytes")
    download = str(tmp_path / "out.png")
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--download", download])
    assert r.exit_code == 0, r.stdout
    assert Path(download).exists()
    assert Path(download).read_bytes() == b"png-bytes"
    assert "Saved" in r.stdout


def test_generate_json_flag(runner, api_key, monkeypatch):
    resp = httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--json"])
    assert r.exit_code == 0
    # Strip newlines/whitespace from output so we can match across rich's line wrapping
    flat = "".join(r.stdout.split())
    assert '"url":"https://cdn.example/a.png"' in flat


def test_generate_download_no_urls(runner, api_key, monkeypatch):
    resp = httpx.Response(200, json={"data": []})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x", "--download", "/tmp/x.png"])
    assert r.exit_code == 0
    assert "no image urls" in r.stdout.lower()


# ─── generate: sync binary response (Stability returns bytes) ────────────


def test_generate_binary_response_with_download(runner, api_key, tmp_path, monkeypatch):
    resp = httpx.Response(200, content=b"\x89PNGfake", headers={"content-type": "image/png"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    download = str(tmp_path / "ultra.png")
    r = runner.invoke(cli_app, ["generate", "stability-ultra", "--prompt", "x", "--download", download])
    assert r.exit_code == 0, r.stdout
    assert Path(download).exists()


def test_generate_binary_response_no_download(runner, api_key, monkeypatch):
    resp = httpx.Response(200, content=b"\x89PNGfake", headers={"content-type": "image/png"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "stability-ultra", "--prompt", "x"])
    assert r.exit_code == 0
    assert "nothing saved" in r.stdout


# ─── generate: HTTP and network errors ───────────────────────────────────


def test_generate_api_error_surface(runner, api_key, monkeypatch):
    resp = httpx.Response(401, json={"message": "Invalid token"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 1
    assert "API error 401" in r.stdout
    assert "Invalid token" in r.stdout


def test_generate_network_error_surface(runner, api_key, monkeypatch):
    def boom(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(gen_app.client.httpx, "post", boom)
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--width", "1", "--height", "1"])
    assert r.exit_code == 1
    assert "Network error" in r.stdout


def test_generate_non_json_response(runner, api_key, monkeypatch):
    resp = httpx.Response(200, text="not really json", headers={"content-type": "text/plain"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: resp)
    r = runner.invoke(cli_app, ["generate", "dalle", "--prompt", "x"])
    assert r.exit_code == 1
    assert "non-JSON" in r.stdout


# ─── resume ──────────────────────────────────────────────────────────────


def test_resume_missing_args(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "resume"])
    assert r.exit_code == 1
    assert "Usage" in r.stdout


def test_resume_sync_model_rejected(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "resume", "dalle", "abc"])
    assert r.exit_code == 1
    assert "sync" in r.stdout


def test_resume_unknown_model(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "resume", "nope-model", "abc"])
    assert r.exit_code == 1
    assert "Unknown model" in r.stdout


def test_resume_async_succeeds(runner, api_key, monkeypatch):
    poll_done = httpx.Response(
        200,
        json={"status": "Ready", "progress": 1.0, "result": {"sample": "https://cdn.example/done.png"}},
    )
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: poll_done)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    r = runner.invoke(cli_app, ["generate", "resume", "flux-pro", "job-123"])
    assert r.exit_code == 0
    assert "https://cdn.example/done.png" in r.stdout


def test_resume_with_download(runner, api_key, tmp_path, monkeypatch):
    poll_done = httpx.Response(
        200,
        json={"status": "Ready", "progress": 1.0, "result": {"sample": "https://cdn.example/done.png"}},
    )
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: poll_done)
    monkeypatch.setattr("comfy_cli.command.generate.client.download_bytes", lambda *a, **kw: b"bytes")
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    download = str(tmp_path / "resumed.png")
    r = runner.invoke(cli_app, ["generate", "resume", "flux-pro", "job-123", "--download", download])
    assert r.exit_code == 0
    assert Path(download).exists()


# ─── refresh ─────────────────────────────────────────────────────────────


def test_refresh_writes_cache(runner, monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return httpx.Response(
                200,
                text="openapi: 3.0.0\n",
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(gen_app.httpx, "Client", FakeClient)
    monkeypatch.setattr("comfy_cli.command.generate.spec._USER_CACHE", tmp_path / "openapi-cache.yml")

    r = runner.invoke(cli_app, ["generate", "refresh"])
    assert r.exit_code == 0, r.stdout
    assert "Refreshed" in r.stdout
    assert (tmp_path / "openapi-cache.yml").exists()
    assert captured["headers"].get("X-Comfy-Env") == "comfy-cli"


def test_refresh_network_failure(runner, monkeypatch):
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, *a, **kw):
            raise httpx.ConnectError("no net")

    monkeypatch.setattr(gen_app.httpx, "Client", FakeClient)
    r = runner.invoke(cli_app, ["generate", "refresh"])
    assert r.exit_code == 1
    assert "Failed to fetch" in r.stdout


# ─── helpers: _arg_value / _separate_meta_flags ──────────────────────────


def test_arg_value_long_and_eq():
    assert gen_app._arg_value(["--foo", "bar"], "--foo") == "bar"
    assert gen_app._arg_value(["--foo=baz"], "--foo") == "baz"
    assert gen_app._arg_value(["--bar", "v"], "--foo", "-f") is None


def test_arg_value_alternatives():
    assert gen_app._arg_value(["-p", "bfl"], "--partner", "-p") == "bfl"


def test_separate_meta_flags_typical():
    rest, meta = gen_app._separate_meta_flags(["--prompt", "x", "--download", "out.png", "--async", "--timeout", "30"])
    assert rest == ["--prompt", "x"]
    assert meta["download"] == "out.png"
    assert meta["async"] is True
    assert meta["timeout"] == "30"


def test_separate_meta_flags_eq_form():
    _, meta = gen_app._separate_meta_flags(["--download=cat.png", "--json"])
    assert meta == {"download": "cat.png", "json": True}


def test_separate_meta_flags_missing_value_raises():
    from comfy_cli.command.generate.schema import SchemaError

    with pytest.raises(SchemaError):
        gen_app._separate_meta_flags(["--download"])
