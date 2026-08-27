"""Tests for ``comfy templates get --where k=v`` (V1-018).

The measured agent loop is ``templates ls`` (to find the name) → ``templates
fetch`` (copying that name back, 100% verbatim). ``get`` fuses the two: the
same ls filter predicates (``_matches``, reused exactly — no new query
language) resolve the template, and when exactly one matches its workflow is
fetched and returned in one envelope.

Contract under test:
  * exactly one match → fetch + return the workflow in one envelope.
  * zero matches → ``template_not_found`` + nearest-candidate suggestions.
  * >1 matches → ``template_ambiguous`` with ≤10 candidates (name + meta).
  * malformed / unknown ``--where`` key → ``template_filter_invalid``.
  * existing ls/show/fetch surfaces untouched (their own tests pin that).
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

# Same schema as the real gallery index (mirrors test_templates.py's fixture,
# with names distinct enough to pin single/ambiguous/none per filter).
GET_FIXTURE = [
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Image",
        "type": "image",
        "templates": [
            {
                "name": "image_flux_dev",
                "title": "Flux Dev Image",
                "description": "Text-to-image with Flux Dev.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["Local", "Text to Image"],
                "models": ["Flux Dev"],
                "logos": [{"provider": ["Black Forest Labs"]}],
                "openSource": True,
                "usage": 90,
            },
            {
                "name": "image_flux_pro_api",
                "title": "Flux Pro (API)",
                "description": "Text-to-image via the BFL API.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["API", "Text to Image"],
                "models": ["Flux Pro"],
                "logos": [{"provider": ["Black Forest Labs"]}],
                "openSource": False,
                "usage": 100,
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
                "name": "video_kling_i2v",
                "title": "Kling Image to Video",
                "description": "Image-to-video via Kling.",
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
    path = tmp_path / "get_index.json"
    path.write_text(json.dumps(GET_FIXTURE))
    return str(path)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
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
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope in stdout:\n{stdout}")


def _stub_workflow_fetch(monkeypatch, body_or_exc):
    def _impl(name, timeout=15.0):
        if isinstance(body_or_exc, Exception):
            raise body_or_exc
        return body_or_exc

    monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _impl)


WORKFLOW_BODY = json.dumps({"9": {"class_type": "KSampler", "inputs": {}}}).encode()


class TestGetSingleMatch:
    def test_unique_filter_fetches_and_returns_workflow(self, gallery_file, monkeypatch):
        _force_json_renderer()
        _stub_workflow_fetch(monkeypatch, WORKFLOW_BODY)
        runner = CliRunner()
        result = runner.invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "type=video"],
        )
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["ok"] is True
        data = env["data"]
        assert data["name"] == "video_kling_i2v"
        assert data["title"] == "Kling Image to Video"
        assert data["output_type"] == "video"
        assert data["node_count"] == 1
        # The whole point: the workflow rides in the same envelope.
        assert data["workflow"] == json.loads(WORKFLOW_BODY)

    def test_filters_and_together_narrow_to_one(self, gallery_file, monkeypatch):
        _force_json_renderer()
        _stub_workflow_fetch(monkeypatch, WORKFLOW_BODY)
        runner = CliRunner()
        result = runner.invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "type=image", "--where", "tag=API"],
        )
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"]["name"] == "image_flux_pro_api"

    def test_name_filter_is_the_ls_substring_predicate(self, gallery_file, monkeypatch):
        _force_json_renderer()
        _stub_workflow_fetch(monkeypatch, WORKFLOW_BODY)
        runner = CliRunner()
        result = runner.invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "name=kling"],
        )
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"]["name"] == "video_kling_i2v"


class TestGetAmbiguous:
    def test_multi_match_errors_with_candidates(self, gallery_file, monkeypatch):
        _force_json_renderer()

        def _should_not_fire(name, timeout=15.0):
            raise AssertionError("workflow fetch must not fire on an ambiguous filter")

        monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _should_not_fire)
        runner = CliRunner()
        result = runner.invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "type=image"],
        )
        assert result.exit_code != 0
        env = _envelope(result.output)
        assert env["ok"] is False
        assert env["error"]["code"] == "template_ambiguous"
        candidates = env["error"]["details"]["candidates"]
        assert len(candidates) == 2
        names = {c["name"] for c in candidates}
        assert names == {"image_flux_dev", "image_flux_pro_api"}
        # One-line meta per candidate so the agent can pick without another ls.
        for c in candidates:
            assert c["title"]
            assert c["output_type"] == "image"

    def test_candidates_are_capped_at_ten(self, tmp_path, monkeypatch):
        _force_json_renderer()
        many = [
            {
                "moduleName": "default",
                "category": "GENERATION TYPE",
                "title": "Image",
                "type": "image",
                "templates": [
                    {
                        "name": f"image_bulk_{i:02d}",
                        "title": f"Bulk {i}",
                        "description": "",
                        "mediaType": "image",
                        "mediaSubtype": "webp",
                        "tags": ["Local"],
                        "models": [],
                        "logos": [],
                    }
                    for i in range(14)
                ],
            }
        ]
        path = tmp_path / "get_index_many.json"
        path.write_text(json.dumps(many))
        runner = CliRunner()
        result = runner.invoke(templates_cmd.app, ["get", "--gallery", str(path), "--where", "type=image"])
        assert result.exit_code != 0
        env = _envelope(result.output)
        assert env["error"]["code"] == "template_ambiguous"
        assert env["error"]["details"]["matched"] == 14
        assert len(env["error"]["details"]["candidates"]) == 10


class TestGetNoMatch:
    def test_zero_matches_errors_with_suggestions(self, gallery_file, monkeypatch):
        _force_json_renderer()

        def _should_not_fire(name, timeout=15.0):
            raise AssertionError("workflow fetch must not fire on zero matches")

        monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _should_not_fire)
        runner = CliRunner()
        # type=video AND tag=Local matches nothing; dropping either filter finds rows.
        result = runner.invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "type=video", "--where", "tag=Local"],
        )
        assert result.exit_code != 0
        env = _envelope(result.output)
        assert env["ok"] is False
        assert env["error"]["code"] == "template_not_found"
        near = env["error"]["details"]["near_misses"]
        # Leave-one-out: dropping `tag` finds the video template, dropping `type`
        # finds the Local image template.
        by_dropped = {n["without"]: n["names"] for n in near}
        assert "video_kling_i2v" in by_dropped["tag"]
        assert "image_flux_dev" in by_dropped["type"]


class TestGetFilterValidation:
    @pytest.mark.parametrize("bad", ["type", "flavor=spicy", "=video"])
    def test_malformed_or_unknown_where_is_rejected(self, gallery_file, bad):
        _force_json_renderer()
        runner = CliRunner()
        result = runner.invoke(templates_cmd.app, ["get", "--gallery", gallery_file, "--where", bad])
        assert result.exit_code != 0
        env = _envelope(result.output)
        assert env["ok"] is False
        assert env["error"]["code"] == "template_filter_invalid"

    def test_no_filters_at_all_is_rejected(self, gallery_file):
        _force_json_renderer()
        runner = CliRunner()
        result = runner.invoke(templates_cmd.app, ["get", "--gallery", gallery_file])
        assert result.exit_code != 0
        env = _envelope(result.output)
        assert env["error"]["code"] == "template_filter_invalid"


class TestGetErrorCodesRegistered:
    def test_new_codes_are_registered(self):
        from comfy_cli import error_codes

        assert error_codes.is_registered("template_ambiguous")
        assert error_codes.is_registered("template_filter_invalid")
