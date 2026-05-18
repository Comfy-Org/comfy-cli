"""End-to-end tests for ``comfy generate`` via Typer's CliRunner.

These cover the dispatch table (list/schema/refresh/resume vs. model alias) and
each major run path with httpx mocked at the boundary.
"""

import base64
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
    assert captured["headers"].get("Comfy-Env") == "comfy-cli"


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


# ─── upload subcommand ──────────────────────────────────────────────────


def test_upload_missing_arg(runner, api_key):
    r = runner.invoke(cli_app, ["generate", "upload"])
    assert r.exit_code == 1
    assert "Usage" in r.stdout


def test_upload_local_file(runner, api_key, tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"png-data")
    monkeypatch.setattr(
        "comfy_cli.command.generate.upload.upload_target",
        lambda target, api_key: gen_app.upload.UploadResult(
            url="https://cdn/x.png", expires_at="2099-01-01T00:00:00Z", existing_file=False
        ),
    )
    r = runner.invoke(cli_app, ["generate", "upload", str(img)])
    assert r.exit_code == 0, r.stdout
    assert "Uploaded" in r.stdout
    assert "https://cdn/x.png" in r.stdout


def test_upload_json_output(runner, api_key, tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"png-data")
    monkeypatch.setattr(
        "comfy_cli.command.generate.upload.upload_target",
        lambda target, api_key: gen_app.upload.UploadResult(
            url="https://cdn/x.png", expires_at="2099-01-01T00:00:00Z", existing_file=True
        ),
    )
    r = runner.invoke(cli_app, ["generate", "upload", str(img), "--json"])
    assert r.exit_code == 0
    flat = "".join(r.stdout.split())
    assert '"url":"https://cdn/x.png"' in flat
    assert '"existing_file":true' in flat


def test_upload_does_not_mistake_meta_value_for_target(runner, monkeypatch, tmp_path):
    """`upload --api-key KEY ./img.png` must resolve ./img.png as the target,
    not KEY — regression check for the positional parsing bug."""
    img = tmp_path / "x.png"
    img.write_bytes(b"png-data")
    captured = {}

    def fake_upload(target, api_key):
        captured["target"] = target
        captured["api_key"] = api_key
        return gen_app.upload.UploadResult(url="https://cdn/x.png", expires_at=None, existing_file=False)

    monkeypatch.setattr("comfy_cli.command.generate.upload.upload_target", fake_upload)
    r = runner.invoke(cli_app, ["generate", "upload", "--api-key", "comfyui-test", str(img)])
    assert r.exit_code == 0, r.stdout
    assert captured["target"] == str(img)
    assert captured["api_key"] == "comfyui-test"


def test_upload_propagates_api_error(runner, api_key, tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"png-data")

    def boom(*a, **kw):
        raise gen_app.client.ApiError(500, "fail", "boom")

    monkeypatch.setattr("comfy_cli.command.generate.upload.upload_target", boom)
    r = runner.invoke(cli_app, ["generate", "upload", str(img)])
    assert r.exit_code == 1
    assert "Upload failed" in r.stdout


# ─── auto-upload during generate ────────────────────────────────────────


def test_generate_auto_base64_for_kontext(runner, api_key, tmp_path, monkeypatch):
    """flux-kontext's input_image expects a Base64 string — local files should
    be auto-encoded with no extra steps."""
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNGfake")

    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_):
        captured["body"] = json
        return httpx.Response(200, json={"id": "job-xyz", "polling_url": "https://x/poll"})

    monkeypatch.setattr(gen_app.client.httpx, "post", fake_post)
    r = runner.invoke(
        cli_app,
        ["generate", "flux-kontext", "--prompt", "edit it", "--input_image", str(img), "--async"],
    )
    assert r.exit_code == 0, r.stdout
    assert captured["body"]["input_image"] == base64.b64encode(b"\x89PNGfake").decode("ascii")


def test_generate_auto_upload_leaves_url_alone(runner, api_key, monkeypatch):
    """A pre-existing https:// URL must NOT trigger an upload."""
    upload_called = {"hit": False}

    def fake_upload(*a, **kw):
        upload_called["hit"] = True
        return gen_app.upload.UploadResult(url="x", expires_at=None, existing_file=False)

    monkeypatch.setattr("comfy_cli.command.generate.upload.upload_path", fake_upload)
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_):
        captured["body"] = json
        return httpx.Response(200, json={"id": "x", "polling_url": "https://x"})

    monkeypatch.setattr(gen_app.client.httpx, "post", fake_post)
    r = runner.invoke(
        cli_app,
        [
            "generate",
            "flux-kontext",
            "--prompt",
            "x",
            "--input_image",
            "https://existing/url.png",
            "--async",
        ],
    )
    assert r.exit_code == 0
    assert upload_called["hit"] is False
    assert captured["body"]["input_image"] == "https://existing/url.png"


def test_generate_auto_upload_skipped_for_multipart(runner, api_key, tmp_path, monkeypatch):
    """Multipart endpoints (ideogram-edit) already stream files via httpx —
    they must not be funneled through /customers/storage."""
    img = tmp_path / "x.png"
    img.write_bytes(b"png")

    upload_called = {"hit": False}
    monkeypatch.setattr(
        "comfy_cli.command.generate.upload.upload_path",
        lambda *a, **kw: upload_called.__setitem__("hit", True) or gen_app.upload.UploadResult("x", None, False),
    )
    monkeypatch.setattr(
        gen_app.client.httpx,
        "post",
        lambda *a, **kw: httpx.Response(200, json={"data": [{"url": "https://x/a.png"}]}),
    )
    r = runner.invoke(
        cli_app,
        [
            "generate",
            "ideogram-edit",
            "--prompt",
            "x",
            "--rendering_speed",
            "TURBO",
            "--image",
            str(img),
        ],
    )
    assert r.exit_code == 0
    assert upload_called["hit"] is False


# ─── video models (async polling, generic poller path) ─────────────────


def test_video_kling_async_path(runner, api_key, monkeypatch):
    """End-to-end async path through the generic kling poller."""
    submit = httpx.Response(200, json={"data": {"task_id": "k-xyz"}})
    finished = httpx.Response(
        200,
        json={
            "data": {
                "task_status": "succeed",
                "task_result": {"videos": [{"url": "https://cdn.example/k.mp4"}]},
            }
        },
    )
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: finished)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)

    r = runner.invoke(cli_app, ["generate", "kling", "--prompt", "a cat", "--duration", "5"])
    assert r.exit_code == 0, r.stdout
    assert "https://cdn.example/k.mp4" in r.stdout


def test_video_luma_async_path(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"id": "luma-1", "state": "queued"})
    done = httpx.Response(200, json={"id": "luma-1", "state": "completed", "assets": {"video": "https://cdn/l.mp4"}})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: done)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)

    r = runner.invoke(
        cli_app,
        [
            "generate",
            "luma",
            "--prompt",
            "a cat",
            "--aspect_ratio",
            "16:9",
            "--model",
            "ray-2",
            "--resolution",
            "{}",
            "--duration",
            "{}",
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert "https://cdn/l.mp4" in r.stdout


def test_video_runway_failure_surfaces(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"id": "rw-1"})
    fail = httpx.Response(200, json={"id": "rw-1", "status": "FAILED"})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: fail)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)

    r = runner.invoke(
        cli_app,
        [
            "generate",
            "runway-i2v",
            "--promptImage",
            '"https://x/img.png"',
            "--seed",
            "1",
            "--model",
            "gen4_turbo",
            "--duration",
            "5",
            "--ratio",
            "1280:720",
        ],
    )
    assert r.exit_code == 1
    assert "FAILED" in r.stdout


def test_video_async_submission_shows_resume_alias(runner, api_key, monkeypatch):
    submit = httpx.Response(200, json={"data": {"task_id": "k-async-1"}})
    monkeypatch.setattr(gen_app.client.httpx, "post", lambda *a, **kw: submit)
    r = runner.invoke(cli_app, ["generate", "kling", "--prompt", "x", "--async"])
    assert r.exit_code == 0, r.stdout
    assert "k-async-1" in r.stdout
    assert "comfy generate resume kling k-async-1" in r.stdout


def test_video_resume_kling(runner, api_key, monkeypatch):
    done = httpx.Response(
        200,
        json={
            "data": {
                "task_status": "succeed",
                "task_result": {"videos": [{"url": "https://cdn/resumed.mp4"}]},
            }
        },
    )
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: done)
    monkeypatch.setattr("comfy_cli.command.generate.poll._sleep", lambda *_: None)
    r = runner.invoke(cli_app, ["generate", "resume", "kling", "k-async-1"])
    assert r.exit_code == 0, r.stdout
    assert "https://cdn/resumed.mp4" in r.stdout


def test_list_video_filter(runner):
    r = runner.invoke(cli_app, ["generate", "list", "--style", "text-to-video"])
    assert r.exit_code == 0
    assert "kling" in r.stdout
    assert "luma" in r.stdout
    assert "pika" in r.stdout


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
