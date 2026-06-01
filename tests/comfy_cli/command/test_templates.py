"""Tests for ``comfy templates`` — gallery introspection.

Uses a small in-repo fixture index.json (mirroring the real schema) so
the tests don't hit GitHub. Covers filter precedence, the JSON envelope
shape, and the not-found error code.
"""

from __future__ import annotations

import json
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
