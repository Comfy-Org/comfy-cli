"""Tests for ``comfy templates`` — gallery introspection.

Uses a small in-repo fixture index.json (mirroring the real schema) so
the tests don't hit GitHub. Covers filter precedence, the JSON envelope
shape, and the not-found error code.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import comfy_cli.http as http_mod
from comfy_cli.caller import Caller
from comfy_cli.command import templates as templates_cmd
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

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
# templates fetch — node_count across both workflow serializations, and the
# envelope ride-along when no file was written.
# ---------------------------------------------------------------------------


UI_WORKFLOW = {
    "id": "abc-123",
    "revision": 0,
    "last_node_id": 6,
    "last_link_id": 5,
    "nodes": [{"id": i, "type": "KSampler"} for i in range(6)],
    "links": [],
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}


def test_fetch_node_count_counts_nodes_not_top_level_keys(gallery_file, tmp_path: Path, monkeypatch):
    """UI-format templates are ``{id, revision, nodes, links, …}`` — ``len(wf)``
    counted those wrapper keys and reported ~10 for every template."""
    _force_json_renderer()
    _stub_template_workflow_fetch(monkeypatch, json.dumps(UI_WORKFLOW).encode())

    out_path = tmp_path / "wf.json"
    runner = CliRunner()
    result = runner.invoke(
        templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2", "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert len(UI_WORKFLOW) == 10  # the number the old `len(wf)` would have reported
    assert env["data"]["node_count"] == 6


def test_fetch_node_count_still_correct_for_api_format(gallery_file, tmp_path: Path, monkeypatch):
    """API format is a bare node map, where the key count *is* the node count."""
    _force_json_renderer()
    api_workflow = {str(i): {"class_type": "KSampler", "inputs": {}} for i in range(3)}
    _stub_template_workflow_fetch(monkeypatch, json.dumps(api_workflow).encode())

    out_path = tmp_path / "wf.json"
    runner = CliRunner()
    result = runner.invoke(
        templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2", "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    assert _envelope(result.output)["data"]["node_count"] == 3


def test_fetch_rides_the_workflow_in_the_envelope_when_no_out(gallery_file, monkeypatch):
    """In JSON mode emit() owns stdout, so without the ride-along a `fetch`
    with no --out returns metadata and loses the workflow entirely."""
    _force_json_renderer()
    _stub_template_workflow_fetch(monkeypatch, json.dumps(UI_WORKFLOW).encode())

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["out"] == "stdout"
    assert env["data"]["workflow"] == UI_WORKFLOW


def test_fetch_with_out_omits_the_workflow_ride_along(gallery_file, tmp_path: Path, monkeypatch):
    """The file is the artifact; duplicating it into every envelope is bloat."""
    _force_json_renderer()
    _stub_template_workflow_fetch(monkeypatch, json.dumps(UI_WORKFLOW).encode())

    out_path = tmp_path / "wf.json"
    runner = CliRunner()
    result = runner.invoke(
        templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2", "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    assert "workflow" not in _envelope(result.output)["data"]


def test_fetch_empty_out_falls_back_to_stdout_not_a_black_hole(gallery_file, monkeypatch):
    """``--out ""`` writes no file; the workflow must still reach the caller.

    The two guards used to disagree — ``if out:`` (falsy → no file) versus
    ``out is None`` (False → no ride-along) — so the workflow landed nowhere.
    """
    _force_json_renderer()
    _stub_template_workflow_fetch(monkeypatch, json.dumps(UI_WORKFLOW).encode())

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "--gallery", gallery_file, "image_flux2", "--out", ""])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["out"] == "stdout"
    assert env["data"]["workflow"] == UI_WORKFLOW


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


def test_expired_cache_refetches_and_rewrites(cache_file, monkeypatch):
    # Backdate the cache past the TTL so a refresh is due.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    # A single-category "refreshed" gallery, distinct from the 3-row fixture.
    refreshed = [
        {
            "moduleName": "default",
            "category": "GENERATION TYPE",
            "title": "Image",
            "type": "image",
            "templates": [
                {
                    "name": "brand_new_template",
                    "title": "Brand New",
                    "description": "Freshly fetched.",
                    "tags": [],
                    "models": [],
                    "logos": [],
                }
            ],
        }
    ]
    fetched = {"count": 0}

    def _fake_fetch(*args, **kwargs):
        fetched["count"] += 1
        return json.dumps(refreshed).encode()

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fake_fetch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    assert fetched["count"] == 1

    env = _envelope(result.output)
    names = [r["name"] for r in env["data"]["rows"]]
    assert names == ["brand_new_template"]
    # Cache was rewritten with the refreshed payload.
    assert json.loads(cache_file.read_bytes()) == refreshed


def test_expired_cache_fetch_failure_falls_back_to_stale(cache_file, monkeypatch, capsys):
    import urllib.error

    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _boom)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    # Offline machine still lists from the stale cache — exit 0, original rows.
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3
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


def test_expired_cache_garbage_200_keeps_stale_and_does_not_clobber(cache_file, monkeypatch):
    # A 200 with a non-JSON body (rate-limit HTML / captive portal) must NOT
    # overwrite the last-known-good cache — we validate before persisting and
    # fall back to the stale cache.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    def _garbage(*args, **kwargs):
        return b"<html>rate limited</html>"

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _garbage)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    # Offline-equivalent: stale cache still serves, exit 0, original rows.
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3
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


def test_non_200_status_falls_back_to_stale_not_uncaught(cache_file, monkeypatch):
    # `_fetch_gallery` raises RuntimeError on a non-200 status; during a TTL
    # auto-refresh that must route through the stale-cache fallback, never
    # escape as an uncaught traceback.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    def _non_200(*args, **kwargs):
        raise RuntimeError("gallery fetch failed: HTTP 429")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _non_200)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["total_in_gallery"] == 3


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


def test_future_mtime_clock_skew_is_treated_as_stale(cache_file, monkeypatch):
    # A future mtime (clock skew / restored file) yields a negative age; it must
    # read as stale so the cache can't be pinned "fresh" forever.
    _set_mtime(cache_file, -2 * 3600)  # mtime 2h in the future

    fetched = {"count": 0}

    def _fake_fetch(*args, **kwargs):
        fetched["count"] += 1
        return json.dumps(FIXTURE).encode()

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fake_fetch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    # A refresh was attempted rather than the future-dated cache being trusted.
    assert fetched["count"] == 1


def test_readonly_cache_dir_still_serves_fetched_data(cache_file, monkeypatch):
    # If persisting the freshly fetched index fails (read-only dir / disk full),
    # the command must still succeed on the in-hand data, not error out.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)

    refreshed = [
        {
            "moduleName": "default",
            "category": "GENERATION TYPE",
            "title": "Image",
            "type": "image",
            "templates": [{"name": "brand_new_template", "title": "Brand New", "tags": [], "models": [], "logos": []}],
        }
    ]

    def _fake_fetch(*args, **kwargs):
        return json.dumps(refreshed).encode()

    def _boom_mkstemp(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(templates_cmd, "_fetch_gallery", _fake_fetch)
    # Make the real _persist_cache's write fail (read-only dir / disk full);
    # it must swallow the error and let the command proceed on in-hand data.
    monkeypatch.setattr(templates_cmd.tempfile, "mkstemp", _boom_mkstemp)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    names = [r["name"] for r in env["data"]["rows"]]
    assert names == ["brand_new_template"]


# ---------------------------------------------------------------------------
# Bounded gallery reads — a remote host must not decide how much memory we use
# ---------------------------------------------------------------------------


class _RecordingResponse:
    """Stands in for a urlopen response, recording how the body was read.

    ``read(None)`` — the unbounded form — is an outright failure here: that is
    the bug being guarded against, and a fake that quietly tolerated it would
    let the regression back in.
    """

    def __init__(self, body: bytes = b"[]", status: int = 200):
        self._body = body
        self.status = status
        self.requested: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n=None):
        if n is None:
            raise AssertionError("unbounded read() — the body must be read with a cap")
        self.requested.append(n)
        return self._body[:n]


def _small_cap(monkeypatch, cap: int = 8) -> None:
    """Run the real ``read_capped`` at a tiny cap so a few bytes trip it.

    Patches the name the module under test resolves, not the primitive itself,
    so the production call path (including which exception it raises and where
    that lands) is exercised end to end without allocating 64 MiB.
    """
    monkeypatch.setattr(
        templates_cmd,
        "read_capped",
        lambda resp, url, max_bytes=cap: http_mod.read_capped(resp, url, max_bytes=max_bytes),
    )


def test_fetch_gallery_reads_with_a_cap_not_unbounded(monkeypatch):
    resp = _RecordingResponse(json.dumps(FIXTURE).encode())
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: resp)

    assert json.loads(templates_cmd._fetch_gallery()) == FIXTURE
    # One byte past the cap, so a body that exactly fills it is still complete.
    assert resp.requested == [http_mod.MAX_RESPONSE_BYTES + 1]


def test_fetch_template_workflow_reads_with_a_cap_not_unbounded(monkeypatch):
    resp = _RecordingResponse(b'{"nodes": []}')
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: resp)

    assert templates_cmd._fetch_template_workflow("image_flux2") == b'{"nodes": []}'
    assert resp.requested == [http_mod.MAX_RESPONSE_BYTES + 1]


def test_oversize_gallery_raises_rather_than_truncating(monkeypatch):
    # Refuse, don't truncate: a short body would be cached and parsed as if it
    # were the whole index.
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: _RecordingResponse(b"x" * 4096))
    _small_cap(monkeypatch)

    with pytest.raises(ResponseTooLarge):
        templates_cmd._fetch_gallery()


def test_oversize_gallery_under_refresh_is_a_clean_envelope(cache_file, monkeypatch):
    # An over-cap body is upstream misbehaviour like any other — it must land in
    # gallery_load_failed, not as an uncaught traceback.
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: _RecordingResponse(b"x" * 4096))
    _small_cap(monkeypatch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls", "--refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "gallery_load_failed"
    # The good cache is left intact — an oversize response must not clobber it.
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_oversize_gallery_during_ttl_refresh_falls_back_to_stale(cache_file, monkeypatch):
    # Same failure on an automatic (TTL-expired) refresh degrades to the stale
    # cache, matching every other fetch failure.
    _set_mtime(cache_file, templates_cmd.GALLERY_TTL_SECONDS + 3600)
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: _RecordingResponse(b"x" * 4096))
    _small_cap(monkeypatch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["ls"])
    assert result.exit_code == 0, result.output
    assert _envelope(result.output)["data"]["total_in_gallery"] == 3


def test_oversize_gallery_under_templates_refresh_is_a_clean_envelope(cache_file, monkeypatch):
    # `templates refresh` calls `_fetch_gallery` directly, so it needs the same
    # error family as `_load_gallery` — this is the call site that used to catch
    # only URLError/OSError.
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: _RecordingResponse(b"x" * 4096))
    _small_cap(monkeypatch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["refresh"])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "gallery_fetch_failed"
    assert json.loads(cache_file.read_bytes()) == FIXTURE


def test_oversize_template_workflow_is_a_clean_envelope(gallery_file, monkeypatch):
    # The workflow fetch happens after the gallery lookup, so point the opener
    # at an oversize body and resolve the name from the local fixture index.
    monkeypatch.setattr(templates_cmd, "plain_urlopen", lambda req, timeout=None: _RecordingResponse(b"x" * 4096))
    _small_cap(monkeypatch)
    _force_json_renderer()

    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, ["fetch", "image_flux2", "--gallery", gallery_file])
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "template_fetch_failed"
