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


# ---------------------------------------------------------------------------
# templates check — workflow-walker unit tests
# ---------------------------------------------------------------------------

# `default`-shape workflow: a top-level CheckpointLoaderSimple carries its model
# in `properties.models`.
_TOP_LEVEL_WF = {
    "nodes": [
        {
            "id": 4,
            "type": "CheckpointLoaderSimple",
            "properties": {
                "models": [
                    {
                        "name": "v1-5-pruned-emaonly.safetensors",
                        "directory": "checkpoints",
                        "url": "https://example.test/ckpt",
                    }
                ]
            },
        },
        {"id": 3, "type": "KSampler", "properties": {}},
    ]
}

# `image_z_image_turbo`-shape workflow: the ONLY model reference lives inside a
# subgraph definition, so a top-level-only walk would find nothing.
_SUBGRAPH_WF = {
    "nodes": [
        {"id": 10, "type": "sg-uuid-1", "properties": {}},
    ],
    "definitions": {
        "subgraphs": [
            {
                "id": "sg-uuid-1",
                "nodes": [
                    {
                        "id": 5,
                        "type": "UNETLoader",
                        "properties": {
                            "models": [
                                {
                                    "name": "z_image_turbo.safetensors",
                                    "directory": "diffusion_models",
                                    "url": "https://example.test/z",
                                }
                            ]
                        },
                    }
                ],
            }
        ]
    },
}


def test_collect_models_top_level():
    reqs = templates_cmd._collect_model_requirements(_TOP_LEVEL_WF)
    assert reqs == [
        {
            "name": "v1-5-pruned-emaonly.safetensors",
            "directory": "checkpoints",
            "url": "https://example.test/ckpt",
        }
    ]


def test_collect_models_subgraph_only():
    # The mandatory subgraph walk is what surfaces this — a top-level walk sees none.
    reqs = templates_cmd._collect_model_requirements(_SUBGRAPH_WF)
    assert reqs == [
        {
            "name": "z_image_turbo.safetensors",
            "directory": "diffusion_models",
            "url": "https://example.test/z",
        }
    ]


def test_collect_models_dedupes_by_directory_and_name():
    wf = {
        "nodes": [
            {"id": 1, "type": "A", "properties": {"models": [{"name": "m.ckpt", "directory": "checkpoints"}]}},
            {"id": 2, "type": "B", "properties": {"models": [{"name": "m.ckpt", "directory": "checkpoints"}]}},
            {"id": 3, "type": "C", "properties": {"models": [{"name": "m.ckpt", "directory": "loras"}]}},
        ]
    }
    reqs = templates_cmd._collect_model_requirements(wf)
    # Same (directory, name) collapses; a different directory stays distinct.
    assert len(reqs) == 2
    assert {(r["directory"], r["name"]) for r in reqs} == {("checkpoints", "m.ckpt"), ("loras", "m.ckpt")}


def test_basename_handles_subfoldered_listings():
    assert templates_cmd._basename("sdxl/model.safetensors") == "model.safetensors"
    assert templates_cmd._basename("model.safetensors") == "model.safetensors"
    assert templates_cmd._basename("a/b/c/model.safetensors") == "model.safetensors"


def test_collect_node_class_types_includes_subgraph_interior():
    types = templates_cmd._collect_node_class_types(_SUBGRAPH_WF)
    # Top-level instance UUID + the interior loader class.
    assert "UNETLoader" in types
    assert "sg-uuid-1" in types


# ---------------------------------------------------------------------------
# templates check — verdict matrix (mocked folder listings + object_info)
# ---------------------------------------------------------------------------


def _stub_folder_listing(monkeypatch, mapping_or_exc):
    """Patch the local folder listing. Pass a {folder: [names] | None} mapping
    (absent folder → None, i.e. 404) or an Exception to raise (server down)."""

    def _impl(target, folder):
        if isinstance(mapping_or_exc, Exception):
            raise mapping_or_exc
        return mapping_or_exc.get(folder)

    monkeypatch.setattr(templates_cmd, "_list_local_folder", _impl)


def _stub_object_info(monkeypatch, graph_or_exc):
    """Patch Graph.load to return a prebuilt graph, or raise (no server)."""
    from comfy_cli.cql import engine

    def _load(cls, *args, **kwargs):
        if isinstance(graph_or_exc, Exception):
            raise graph_or_exc
        return graph_or_exc

    monkeypatch.setattr(engine.Graph, "load", classmethod(_load))


def _no_local_server(monkeypatch):
    from comfy_cli.cql import engine

    _stub_object_info(monkeypatch, engine.LoadError("no local server"))


def _run_check(gallery_file, name, tmp_path, monkeypatch, extra=None):
    # Isolate the on-disk workflow cache so tests never read a stale body.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    runner = CliRunner()
    return runner.invoke(templates_cmd.app, ["check", "--gallery", gallery_file, name, *(extra or [])])


def test_check_all_models_present_is_runnable(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps(_TOP_LEVEL_WF).encode())
    _stub_folder_listing(monkeypatch, {"checkpoints": ["v1-5-pruned-emaonly.safetensors"]})

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "runnable"
    assert env["data"]["models"]["present"] == ["v1-5-pruned-emaonly.safetensors"]
    assert env["data"]["models"]["missing"] == []
    assert env["data"]["models"]["required"] == 1


def test_check_one_missing_is_missing_models(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps(_TOP_LEVEL_WF).encode())
    _stub_folder_listing(monkeypatch, {"checkpoints": ["something-else.safetensors"]})

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "missing-models"
    missing = env["data"]["models"]["missing"]
    assert len(missing) == 1
    # Missing entries carry the download URL so an agent can fetch them.
    assert missing[0]["url"] == "https://example.test/ckpt"
    assert missing[0]["directory"] == "checkpoints"


def test_check_404_folder_marks_missing_with_warning(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps(_SUBGRAPH_WF).encode())
    # diffusion_models absent from the mapping → _list_local_folder returns None (404).
    _stub_folder_listing(monkeypatch, {})

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "missing-models"
    assert env["data"]["models"]["missing"][0]["name"] == "z_image_turbo.safetensors"
    assert any("diffusion_models" in w for w in env["data"]["warnings"])


def test_check_api_required_via_index_beats_present_models(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    # image_flux2 carries the "API" tag in the gallery fixture; a zero-model
    # workflow means we don't even need the server.
    _stub_template_workflow_fetch(monkeypatch, json.dumps({"nodes": []}).encode())

    result = _run_check(gallery_file, "image_flux2", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "api-required"
    assert env["data"]["api"]["dependent"] is True
    assert env["data"]["api"]["source"] == "index"


def test_check_api_required_via_object_info(gallery_file, tmp_path, monkeypatch):
    from comfy_cli.cql import engine

    _force_json_renderer()
    # A non-API-by-index template (image_z_image) whose workflow node IS an api_node
    # per object_info → the object_info tier is what flags it.
    graph = engine.Graph.from_object_info(
        {"SomeApiNode": {"input": {}, "output": [], "output_name": [], "api_node": True}}
    )
    _stub_object_info(monkeypatch, graph)
    _stub_template_workflow_fetch(monkeypatch, json.dumps({"nodes": [{"id": 1, "type": "SomeApiNode"}]}).encode())

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "api-required"
    assert env["data"]["api"]["source"] == "object_info"
    assert env["data"]["api"]["api_nodes"] == ["SomeApiNode"]


def test_check_zero_models_no_loaders_is_runnable(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    wf = {"nodes": [{"id": 1, "type": "KSampler"}, {"id": 2, "type": "SaveImage"}]}
    _stub_template_workflow_fetch(monkeypatch, json.dumps(wf).encode())

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "runnable"
    assert env["data"]["models"]["required"] == 0


def test_check_zero_models_with_loader_is_unknown(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    # A loader-ish node but no declared properties.models → we can't tell.
    wf = {"nodes": [{"id": 1, "type": "CheckpointLoaderSimple"}]}
    _stub_template_workflow_fetch(monkeypatch, json.dumps(wf).encode())

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "unknown"


def test_check_basename_match_against_subfoldered_listing(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps(_TOP_LEVEL_WF).encode())
    # The listing returns a folder-relative path with a subdirectory; basename matching
    # must still count it as present.
    _stub_folder_listing(monkeypatch, {"checkpoints": ["subdir/v1-5-pruned-emaonly.safetensors"]})

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["verdict"] == "runnable"
    assert env["data"]["models"]["present"] == ["v1-5-pruned-emaonly.safetensors"]


def test_check_server_down_surfaces_server_not_running(gallery_file, tmp_path, monkeypatch):
    import urllib.error

    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps(_TOP_LEVEL_WF).encode())
    _stub_folder_listing(monkeypatch, urllib.error.URLError("connection refused"))

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["ok"] is False
    assert env["error"]["code"] == "server_not_running"


def test_check_unknown_template_surfaces_template_not_found(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()

    def _should_not_fire(name, timeout=15.0):
        raise AssertionError("workflow fetch must not run for an unknown template")

    monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", _should_not_fire)
    result = _run_check(gallery_file, "no_such_template", tmp_path, monkeypatch)
    assert result.exit_code != 0
    env = _envelope(result.output)
    assert env["error"]["code"] == "template_not_found"
    assert env["error"]["details"]["close_matches"] == []


def test_check_custom_nodes_surfaced_verbatim(gallery_file, tmp_path, monkeypatch):
    _force_json_renderer()
    _no_local_server(monkeypatch)
    _stub_template_workflow_fetch(monkeypatch, json.dumps({"nodes": []}).encode())

    # Inject a requiresCustomNodes entry into the fixture on disk for this test.
    import json as _json

    cats = _json.loads(Path(gallery_file).read_text())
    cats[0]["templates"][1]["requiresCustomNodes"] = ["ComfyUI-SEEDVR2"]  # image_z_image
    Path(gallery_file).write_text(_json.dumps(cats))

    result = _run_check(gallery_file, "image_z_image", tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    env = _envelope(result.output)
    assert env["data"]["custom_nodes_required"] == ["ComfyUI-SEEDVR2"]
