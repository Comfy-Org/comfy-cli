"""Tests for ``comfy templates`` — gallery introspection.

Uses a small in-repo fixture index.json (mirroring the real schema) so
the tests don't hit GitHub. Covers filter precedence, the JSON envelope
shape, and the not-found error code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import templates as templates_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

# The genuine spawn helper, captured before the autouse stub replaces it, so the
# spawn-seam tests can exercise the real (Popen-calling) implementation.
_REAL_SPAWN_BACKGROUND_REFRESH = templates_cmd._spawn_background_refresh

FIXTURE = [
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Image",
        "type": "image",
        "templates": [
            {
                "name": "image_flux2",
                "title": "Flux 2 Image",
                "description": "Text-to-image using Flux 2 via the BFL API.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["API", "Text to Image"],
                "models": ["Flux 2"],
                "logos": [{"provider": ["Black Forest Labs"]}],
                "openSource": False,
                "usage": 100,
            },
            {
                "name": "image_z_image",
                "title": "Z Image",
                "description": "Local SDXL-style text-to-image.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["Local", "Text to Image"],
                "models": ["Z Image"],
                "logos": [{"provider": "Z"}],
                "openSource": True,
                "usage": 50,
            },
        ],
    },
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Video",
        "type": "video",
        "templates": [
            {
                "name": "gsc_starter_1",
                "title": "Genesis Starter",
                "description": "Image-to-video starter using Kling.",
                "mediaType": "video",
                "mediaSubtype": "mp4",
                "tags": ["API", "Image to Video"],
                "models": ["Kling 2.5"],
                "logos": [{"provider": ["Kling"]}],
                "openSource": False,
                "usage": 75,
            }
        ],
    },
]


@pytest.fixture
def gallery_file(tmp_path: Path) -> str:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(FIXTURE))
    return str(path)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture(autouse=True)
def no_real_background_refresh(monkeypatch):
    """Never spawn a real detached refresher during tests (stale-while-revalidate).

    ``_spawn_background_refresh`` would otherwise fork ``python -m comfy_cli``
    on every stale-cache serve. Replace it with a counter so tests can assert it
    fired without paying a subprocess/network round-trip.
    """
    calls = {"count": 0}

    def _record():
        calls["count"] += 1
        # The real helper returns True when a refresh is running; mirror that so
        # callers exercise the "refreshing in the background" path.
        return True

    monkeypatch.setattr(templates_cmd, "_spawn_background_refresh", _record)
    return calls


def _force_json_renderer():
    """Pin the renderer to JSON so tests can read envelopes off stdout."""
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _envelope(stdout: str) -> dict:
    """Parse the last JSON line of stdout as the envelope."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope in stdout:\n{stdout}")


def test_ls_default_returns_all_three(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["ok"] is True
    assert env["data"]["total_in_gallery"] == 3
    assert env["data"]["matched"] == 3


def test_ls_type_filter(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--type", "video"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    names = [r["name"] for r in env["data"]["rows"]]
    assert names == ["gsc_starter_1"]


def test_ls_provider_filter_handles_both_scalar_and_array_logos(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    # 'Z' provider was set as a scalar string in the fixture
    result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--provider", "Z"])
    assert result.exit_code == 0
    env = _envelope(result.output)
    names = [r["name"] for r in env["data"]["rows"]]
    assert "image_z_image" in names


def test_ls_limit_applies_after_filter(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(
        templates_cmd.app,
        ["ls", "--gallery", gallery_file, "--tag", "API", "--limit", "1"],
    )
    assert result.exit_code == 0
    env = _envelope(result.output)
    assert env["data"]["matched"] == 2  # API tag in both image_flux2 and gsc_starter_1
    assert env["data"]["shown"] == 1


def test_show_returns_full_template(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["show", "--gallery", gallery_file, "image_flux2"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    tpl = env["data"]["template"]
    assert tpl["name"] == "image_flux2"
    assert tpl["title"] == "Flux 2 Image"
    assert "Black Forest Labs" in tpl["providers"]
    assert tpl["output_type"] == "image"


def test_show_unknown_template_returns_error_code(gallery_file):
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["show", "--gallery", gallery_file, "no_such_template"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "template_not_found"


# ---------------------------------------------------------------------------
# templates fetch
# ---------------------------------------------------------------------------


def _stub_template_workflow_fetch(monkeypatch, body_or_exc):
    """Patch the GitHub workflow-JSON fetch to return a canned body (or raise)."""

    def _impl(name, timeout=15.0):
        if isinstance(body_or_exc, Exception):
            raise body_or_exc
        return body_or_exc

    monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _impl)


def test_fetch_writes_to_stdout_in_pretty_mode(gallery_file, monkeypatch, capsys):
    # Pretty mode (default); workflow JSON streams to stdout.
    reset_renderer_for_testing()
    workflow_body = json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}).encode()
    _stub_template_workflow_fetch(monkeypatch, workflow_body)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2"])
    assert result.exit_code == 0, result.output
    # The workflow JSON itself was written to stdout (the user can pipe it).
    assert '"class_type": "KSampler"' in result.output


def test_fetch_with_out_writes_to_file(gallery_file, tmp_path: Path, monkeypatch, capsys):
    _force_json_renderer()
    workflow_body = json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}).encode()
    _stub_template_workflow_fetch(monkeypatch, workflow_body)

    out_path = tmp_path / "out" / "wf.json"  # nested to verify parent mkdir
    runner = CliRunner()
    result = runner.invoke(
        templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2", "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["ok"] is True
    assert env["data"]["name"] == "image_flux2"
    assert env["data"]["node_count"] == 1
    assert out_path.exists()
    assert out_path.read_bytes() == workflow_body


def test_fetch_unknown_template_surfaces_template_not_found(gallery_file, monkeypatch, capsys):
    _force_json_renderer()
    # The fetch helper should never be called because the gallery check fails first.
    sentinel_called = {"fired": False}

    def _should_not_fire(name, timeout=15.0):
        sentinel_called["fired"] = True
        raise AssertionError("fetch was called for an unknown template")

    monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _should_not_fire)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "no_such_template"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "template_not_found"
    assert sentinel_called["fired"] is False


def test_fetch_upstream_404_surfaces_template_fetch_failed(gallery_file, monkeypatch, capsys):
    import urllib.error

    _force_json_renderer()
    err = urllib.error.HTTPError("https://github/templates/x.json", 404, "Not Found", {}, None)
    _stub_template_workflow_fetch(monkeypatch, err)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "template_fetch_failed"


def test_fetch_non_json_upstream_surfaces_workflow_invalid(gallery_file, monkeypatch, capsys):
    _force_json_renderer()
    _stub_template_workflow_fetch(monkeypatch, b"<html>not json</html>")

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "template_workflow_invalid_json"


# ---------------------------------------------------------------------------
# Gallery cache TTL (BE-3393): fresh-within-TTL serves the cache, expired
# re-fetches, and a fetch failure on an expired cache falls back to stale.
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_file(tmp_path: Path, monkeypatch) -> Path:
    """Point ``_cache_path`` at a tmp file seeded with the fixture index."""
    path = tmp_path / "cache" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(templates_cmd, "_cache_path", lambda: path)
    return path


def _set_mtime(path: Path, seconds_ago: float) -> None:
    """Backdate a file's mtime so the cache reads as ``seconds_ago`` old."""
    when = time.time() - seconds_ago
    os.utime(path, (when, when))


def _fetch_boom(*args, **kwargs):
    raise AssertionError("_fetch_gallery must not be called")


def test_fresh_cache_within_ttl_serves_cache_without_fetching(cache_file, monkeypatch):
    # mtime is "now" (fixture just written) → well within the 24h TTL.
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fetch_boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3


def test_expired_cache_serves_stale_immediately_and_spawns_background_refresh(
    cache_file, monkeypatch, no_real_background_refresh
):
    # Backdate the cache past the TTL so a refresh is due.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    # Stale-while-revalidate (BE-3427): the foreground command must NOT block on
    # a network fetch when a stale cache is present — it serves the stale copy
    # immediately and revalidates in a detached background process.
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fetch_boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output

    env = _envelope(result.output)
    # Served the stale 3-row fixture right away (no inline fetch — _fetch_boom
    # would have raised if it were called).
    assert env["data"]["total_in_gallery"] == 3
    # A background refresh was kicked off for the next invocation.
    assert no_real_background_refresh["count"] == 1
    # The foreground command leaves the cache untouched (the bg process rewrites
    # it), so a failed/slow revalidation can never clobber the last-good copy.
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_refresh_cache_entrypoint_keeps_cache_on_fetch_failure(cache_file, monkeypatch):
    # The detached background refresher (`templates _refresh-cache`) is spawned
    # after the foreground served stale. If its fetch fails (offline), it must
    # exit cleanly and leave the last-known-good cache untouched.
    import urllib.error

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _boom)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["_refresh-cache"])
    assert result.exit_code == 0, result.output
    # The stale cache is untouched (not overwritten by a failed fetch).
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_expired_cache_fetch_failure_with_no_cache_is_fatal(tmp_path, monkeypatch):
    import urllib.error

    # No cache on disk at all → a fetch failure has nothing to fall back to.
    missing = tmp_path / "cache" / "index.json"
    monkeypatch.setattr(templates_cmd, "_cache_path", lambda: missing)

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "gallery_load_failed"


def test_explicit_refresh_fetch_failure_is_fatal_not_stale_fallback(cache_file, monkeypatch):
    import urllib.error

    # Even with a warm cache, `--refresh` is an explicit request: a fetch
    # failure surfaces the error rather than silently serving stale.
    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "gallery_load_failed"


def test_refresh_cache_entrypoint_ignores_garbage_200(cache_file, monkeypatch):
    # A 200 with a non-JSON body (rate-limit HTML / captive portal) must NOT
    # overwrite the last-known-good cache — the background refresher validates
    # before persisting and leaves the stale cache intact on garbage.
    def _garbage(*args, **kwargs):
        return b"<html>rate limited</html>"

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _garbage)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["_refresh-cache"])
    assert result.exit_code == 0, result.output
    # The good cache was left untouched, not clobbered with the HTML garbage.
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_explicit_refresh_garbage_200_is_fatal(cache_file, monkeypatch):
    # A 200-with-garbage body under an explicit `--refresh` surfaces the decode
    # error as gallery_load_failed rather than silently serving stale.
    def _garbage(*args, **kwargs):
        return b"<html>rate limited</html>"

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _garbage)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "gallery_load_failed"
    # Good cache left intact.
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_refresh_cache_entrypoint_swallows_non_200(cache_file, monkeypatch):
    # `_fetch_gallery` raises RuntimeError on a non-200 status; the background
    # refresher must swallow it (never escape as an uncaught traceback) and keep
    # the stale cache intact.
    def _non_200(*args, **kwargs):
        raise RuntimeError("gallery fetch failed: HTTP 429")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _non_200)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["_refresh-cache"])
    assert result.exit_code == 0, result.output
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_non_200_status_under_refresh_is_fatal_not_uncaught(cache_file, monkeypatch):
    # Same RuntimeError under an explicit `--refresh` surfaces cleanly as
    # gallery_load_failed instead of crashing.
    def _non_200(*args, **kwargs):
        raise RuntimeError("gallery fetch failed: HTTP 500")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _non_200)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "gallery_load_failed"


def test_future_mtime_clock_skew_serves_stale_and_revalidates(cache_file, monkeypatch, no_real_background_refresh):
    # A future mtime (clock skew / restored file) yields a negative age; it must
    # read as stale so the cache can't be pinned "fresh" forever. Under SWR that
    # means: served immediately (no inline fetch) and revalidated in the
    # background, rather than the future-dated cache being trusted.
    _set_mtime(cache_file, -2 * 3600)  # mtime 2h in the future

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fetch_boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3
    # A background refresh was kicked off rather than the future-dated cache
    # being trusted forever.
    assert no_real_background_refresh["count"] == 1


def test_refresh_cache_entrypoint_still_persists_when_atomic_rename_used(cache_file, monkeypatch):
    # The background refresher persists a freshly fetched index via the atomic
    # `_persist_cache` path, rewriting the cache for the next invocation.
    refreshed = [
        {
            "moduleName": "default",
            "category": "GENERATION TYPE",
            "title": "Image",
            "type": "image",
            "templates": [{"name": "brand_new_template", "title": "Brand New", "tags": [], "models": [], "logos": []}],
        }
    ]

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: json.dumps(refreshed).encode())

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["_refresh-cache"])
    assert result.exit_code == 0, result.output
    # Cache was rewritten with the refreshed payload for the next `ls`/`show`.
    assert json.loads(cache_file.read_bytes()) == refreshed


def test_readonly_cache_dir_still_serves_fetched_data_on_refresh(cache_file, monkeypatch):
    # `--refresh` fetches synchronously; if persisting the freshly fetched index
    # fails (read-only dir / disk full), the command must still succeed on the
    # in-hand data rather than error out.
    refreshed = [
        {
            "moduleName": "default",
            "category": "GENERATION TYPE",
            "title": "Image",
            "type": "image",
            "templates": [{"name": "brand_new_template", "title": "Brand New", "tags": [], "models": [], "logos": []}],
        }
    ]

    def _boom_mkstemp(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: json.dumps(refreshed).encode())
    # Make the real _persist_cache's write fail (read-only dir / disk full);
    # it must swallow the error and let the command proceed on in-hand data.
    monkeypatch.setattr(templates_cmd.tempfile, "mkstemp", _boom_mkstemp)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    names = [r["name"] for r in env["data"]["rows"]]
    assert names == ["brand_new_template"]


# ---------------------------------------------------------------------------
# Stale-while-revalidate spawn seam (BE-3427): the foreground serves stale and
# fires a *detached* background refresher; here we assert the spawn shape.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch) -> Path:
    """Point ``_cache_path`` at a clean tmp dir so the debounce marker and safe
    cwd never touch (or read a stale marker from) the real user cache."""
    path = tmp_path / "gallery" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(templates_cmd, "_cache_path", lambda: path)
    return path


def test_spawn_background_refresh_is_fully_detached(monkeypatch, isolated_cache):
    # `_spawn_background_refresh` must launch a detached `comfy templates
    # _refresh-cache` — new session (POSIX) / native detach flags (Windows), stdio
    # → /dev/null — so it can outlive the parent without ever blocking it (offline:
    # the parent must not wait on the 15s fetch timeout).
    # Restore the real helper (the autouse fixture stubs it out for other tests).
    monkeypatch.setattr(templates_cmd, "_spawn_background_refresh", _REAL_SPAWN_BACKGROUND_REFRESH)
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(templates_cmd.subprocess, "Popen", _fake_popen)
    assert templates_cmd._spawn_background_refresh() is True

    assert captured["argv"][-2:] == ["templates", "_refresh-cache"]
    assert captured["argv"][0] == sys.executable
    kwargs = captured["kwargs"]
    if sys.platform == "win32":
        flags = kwargs["creationflags"]
        assert flags & templates_cmd.subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & templates_cmd.subprocess.DETACHED_PROCESS
    else:
        assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is templates_cmd.subprocess.DEVNULL
    assert kwargs["stderr"] is templates_cmd.subprocess.DEVNULL
    assert kwargs["stdin"] is templates_cmd.subprocess.DEVNULL
    # The child is anchored in our own cache dir (not the parent's cwd) and opted
    # out of telemetry so it can't race-write config.ini or import a planted
    # comfy_cli.py from an untrusted directory.
    assert kwargs["cwd"] == str(isolated_cache.parent)
    assert kwargs["env"]["COMFY_NO_TELEMETRY"] == "1"
    assert kwargs["env"]["DO_NOT_TRACK"] == "1"


def test_spawn_background_refresh_swallows_spawn_failure(monkeypatch, isolated_cache):
    # If the OS can't spawn the refresher (no fork, exec denied), the foreground
    # command has already served stale — the failure must be swallowed and
    # reported as False (no refresh running), not raised.
    monkeypatch.setattr(templates_cmd, "_spawn_background_refresh", _REAL_SPAWN_BACKGROUND_REFRESH)

    def _boom_popen(*args, **kwargs):
        raise OSError("cannot fork")

    monkeypatch.setattr(templates_cmd.subprocess, "Popen", _boom_popen)
    # Must not raise, and reports failure so the caller doesn't claim a refresh started.
    assert templates_cmd._spawn_background_refresh() is False


# ---------------------------------------------------------------------------
# Wrong-shape / non-UTF-8 payload hardening: a valid-JSON-but-wrong-shape or
# non-UTF-8 200 must never poison the cache or crash the command.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [b'{"error": "rate limited"}', b"null", b"1"])
def test_refresh_cache_entrypoint_ignores_wrong_shape_200(cache_file, monkeypatch, body):
    # A 200 whose body is valid JSON but not the expected array (captive-portal
    # error object, bare null/number) must NOT overwrite the last-known-good cache.
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: body)

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["_refresh-cache"])
    assert result.exit_code == 0, result.output
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_explicit_refresh_wrong_shape_200_is_fatal(cache_file, monkeypatch):
    # Under an explicit `--refresh`, a valid-JSON-but-wrong-shape body surfaces as
    # gallery_load_failed rather than silently poisoning or serving stale.
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: b'{"error": "nope"}')
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "gallery_load_failed"
    assert json.loads(cache_file.read_bytes()) == FIXTURE  # good cache intact


def test_ls_wrong_shape_stale_cache_falls_through_to_fetch(cache_file, monkeypatch, no_real_background_refresh):
    # A stale cache whose *content* is valid JSON but the wrong shape can't be
    # served — SWR must fall through to a synchronous fetch instead of handing a
    # non-list to _flatten_templates (which would raise / silently drop rows).
    cache_file.write_text(json.dumps({"error": "poisoned"}))
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: json.dumps(FIXTURE).encode())
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3
    # It fetched synchronously rather than serving the poisoned cache in the background.
    assert no_real_background_refresh["count"] == 0


def test_explicit_refresh_non_utf8_body_is_fatal_not_uncaught(cache_file, monkeypatch):
    # A non-UTF-8 200 body makes json.loads raise UnicodeDecodeError — a
    # ValueError subclass that is NOT a JSONDecodeError. It must still route
    # through _GALLERY_LOAD_ERRORS as gallery_load_failed, never an uncaught crash.
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: b"\xff\xfe\x00garbage")
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "gallery_load_failed"


# ---------------------------------------------------------------------------
# Exact-name lookups (show/fetch) opt out of stale-while-revalidate so a
# freshly-added template resolves on the same call (BE-3427 review).
# ---------------------------------------------------------------------------


def test_show_stale_cache_fetches_synchronously_for_exact_name(cache_file, monkeypatch, no_real_background_refresh):
    # The stale cache lacks `brand_new_template`; `show` must fetch synchronously
    # (not serve stale + background-refresh) so a template added upstream after
    # the TTL expired resolves immediately instead of reporting not-found.
    refreshed = [
        {
            "moduleName": "default",
            "category": "GENERATION TYPE",
            "title": "Image",
            "type": "image",
            "templates": [{"name": "brand_new_template", "title": "Brand New", "tags": [], "models": [], "logos": []}],
        }
    ]
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)
    monkeypatch.setattr(templates_cmd, "_fetch_gallery", lambda *a, **k: json.dumps(refreshed).encode())
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["show", "brand_new_template"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["template"]["name"] == "brand_new_template"
    # Synchronous fetch — no detached background refresher was spawned.
    assert no_real_background_refresh["count"] == 0


def test_show_stale_cache_falls_back_to_stale_when_offline(cache_file, monkeypatch, no_real_background_refresh):
    # background_ok=False still preserves the offline safety net: when the
    # synchronous fetch fails, `show` falls back to the stale cache rather than
    # erroring, so a known template still resolves offline.
    import urllib.error

    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["show", "image_flux2"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["template"]["name"] == "image_flux2"
    assert no_real_background_refresh["count"] == 0


def test_spawn_background_refresh_debounces_rapid_calls(monkeypatch, isolated_cache):
    # Stale-while-revalidate serves the cache on *every* call past the TTL. Without
    # a debounce, an offline host would spawn a fresh detached refresher each time
    # — unbounded PID fan-out / a local DoS. The second call within the window must
    # NOT spawn again (but still reports True: a refresh is in flight).
    monkeypatch.setattr(templates_cmd, "_spawn_background_refresh", _REAL_SPAWN_BACKGROUND_REFRESH)
    spawns = {"count": 0}

    def _counting_popen(argv, **kwargs):
        spawns["count"] += 1
        return object()

    monkeypatch.setattr(templates_cmd.subprocess, "Popen", _counting_popen)

    assert templates_cmd._spawn_background_refresh() is True
    assert templates_cmd._spawn_background_refresh() is True
    assert spawns["count"] == 1  # second call debounced, no extra process

    # Once the marker ages past the debounce window, a fresh launch is due again.
    marker = templates_cmd._refresh_marker_path()
    _set_mtime(marker, templates_cmd._REFRESH_DEBOUNCE_SECONDS + 5)
    assert templates_cmd._spawn_background_refresh() is True
    assert spawns["count"] == 2
