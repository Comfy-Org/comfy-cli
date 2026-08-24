"""Envelope tests for the ``data.knowledge`` block on the discovery commands.

Contract under test:
  * ``generate schema|list``, ``templates ls|show|get``, ``nodes search|ls`` and
    ``models search`` append ``data.knowledge`` when the loaded bundle has
    something to say about the call, and a ``nudge`` on a thin zero-hit call.
  * ``--select`` projections and error envelopes are never enriched.
  * without a bundle every envelope is exactly what it was before.
  * every enriched ``data`` still validates against its command schema, which
    ``$ref``s ``knowledge_block.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli import knowledge
from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.command import templates as templates_cmd
from comfy_cli.command.models import search as models_search_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "comfy_cli" / "schemas"
FIXTURE_KNOWLEDGE = REPO_ROOT / "tests" / "comfy_cli" / "fixtures" / "knowledge" / "knowledge.json"

GALLERY = [
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
            }
        ],
    },
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Video",
        "type": "video",
        "templates": [
            {
                "name": "video_testvid_i2v",
                "title": "Testvid Image to Video",
                "description": "Image-to-video via Testvid.",
                "mediaType": "video",
                "mediaSubtype": "mp4",
                "tags": ["API", "Image to Video"],
                "models": ["Testvid 2.5"],
                "logos": [{"provider": ["Testvid"]}],
                "openSource": False,
                "usage": 75,
            },
            {
                "name": "video_acme_h3_i2v",
                "title": "Acme Halo 03 Image to Video",
                "description": "Image-to-video via Acme H3.",
                "mediaType": "video",
                "mediaSubtype": "mp4",
                "tags": ["API", "Image to Video", "Lip Sync"],
                "models": ["Acme H3"],
                "logos": [{"provider": ["Acme"]}],
                "openSource": False,
                "usage": 60,
            },
        ],
    },
]


def _object_info() -> dict[str, Any]:
    return {
        "KSampler": {
            "input": {"required": {"model": ["MODEL"], "steps": ["INT", {"default": 20}]}},
            "input_order": {"required": ["model", "steps"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise the latent via the provided model.",
            "output_node": False,
            "python_module": "nodes",
        },
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
            "input_order": {"required": ["samples", "vae"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "VAE Decode",
            "description": "Turn a latent back into pixels.",
            "output_node": False,
            "python_module": "nodes",
        },
        "TestvidImage2VideoNode": {
            "input": {"required": {"start_frame": ["IMAGE"], "prompt": ["STRING", {"multiline": True}]}},
            "input_order": {"required": ["start_frame", "prompt"]},
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
            "category": "api node/video/Testvid",
            "display_name": "Testvid Image to Video",
            "description": "Testvid image-to-video via the partner API.",
            "output_node": False,
            "python_module": "comfy_api_nodes.nodes_testvid",
        },
    }


@pytest.fixture
def gallery_file(tmp_path: Path) -> str:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(GALLERY))
    return str(path)


@pytest.fixture
def object_info_file(tmp_path: Path) -> str:
    path = tmp_path / "object_info.json"
    path.write_text(json.dumps(_object_info()))
    return str(path)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No bundle unless a test opts in with ``bundle``; never any network."""
    reset_renderer_for_testing()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for var in (knowledge.ENV_FILE, knowledge.ENV_URL, knowledge.ENV_TTL):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(knowledge, "_http_get", lambda url: (_ for _ in ()).throw(AssertionError(url)))
    knowledge._reset_for_testing()
    yield
    knowledge._reset_for_testing()
    reset_renderer_for_testing()


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setenv(knowledge.ENV_FILE, str(FIXTURE_KNOWLEDGE))
    knowledge._reset_for_testing()


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


def _validator_for(name: str) -> jsonschema.Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    store = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        sid = s.get("$id")
        if sid:
            store[sid] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def _validate(data: dict, schema_name: str) -> None:
    _validator_for(schema_name).validate(data)
    if "knowledge" in data:
        _validator_for("knowledge_block.json").validate(data["knowledge"])
        assert knowledge._block_bytes(data["knowledge"]) <= knowledge.MAX_BLOCK_BYTES


def _invoke(app, args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    result = CliRunner().invoke(app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_ls_tag_matches_a_capability(self, bundle, gallery_file, capsys):
        env = _invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--tag", "Lip Sync"], capsys)
        assert env["ok"] is True
        data = env["data"]
        _validate(data, "templates.json")
        k = data["knowledge"]
        assert k["picks"][0]["capability"] == "lipsync"
        # Every pick ships, ranked. The ones whose template this gallery does not
        # carry say so; testvid's rank-6 pick names no template, so it stays clean.
        unavailable = {p["model"] for p in k["picks"] if p.get("available_locally") is False}
        assert "lipco-3" in unavailable
        assert "testvid" not in unavailable
        assert [p["rank"] for p in k["picks"]] == sorted(p["rank"] for p in k["picks"])
        # The matching row's template id reverse-resolves to its row.
        assert [m["id"] for m in k["models"]] == ["acme-h3"]
        assert k["models"][0]["matched_on"] == "video_acme_h3_i2v"
        assert k["hit_ids"] == ["acme-h3", "cap:lipsync"]

    def test_enriched_payload_validates_from_discover_schemas_by_id(self, bundle, gallery_file, capsys):
        """A consumer holding only `comfy discover`'s inlined schemas resolves the
        `knowledge_block.json` $ref through `$id`, with no file system behind it."""
        from comfy_cli.discovery import load_all_schemas

        schemas = {entry["name"]: entry["schema"] for entry in load_all_schemas().values()}
        store = {s["$id"]: s for s in schemas.values() if "$id" in s}
        for name in ("templates.json", "nodes.json", "models.json", "generate_list.json", "generate_schema.json"):
            assert schemas[name].get("$id") == f"https://comfy.org/schemas/{name}"
        env = _invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--tag", "Lip Sync"], capsys)
        assert env["data"]["knowledge"]["picks"]
        schema = schemas["templates.json"]
        resolver = jsonschema.RefResolver(base_uri=schema["$id"], referrer=schema, store=store)
        jsonschema.Draft202012Validator(schema, resolver=resolver).validate(env["data"])

    def test_ls_model_filter_hits_the_alias(self, bundle, gallery_file, capsys):
        env = _invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--model", "Testvid"], capsys)
        data = env["data"]
        _validate(data, "templates.json")
        assert data["knowledge"]["models"][0]["id"] == "testvid"
        assert data["knowledge"]["models"][0]["matched_on"] == "testvid"

    def test_ls_zero_hit_gets_a_nudge(self, bundle, gallery_file, capsys):
        env = _invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--name", "faceswap"], capsys)
        data = env["data"]
        _validate(data, "templates.json")
        assert data["matched"] == 0
        assert data["knowledge"]["zero_hit"] is True
        assert "'faceswap'" in data["knowledge"]["nudge"]
        assert "lipsync" in data["knowledge"]["capabilities_available"]

    def test_ls_no_filter_no_match_no_key(self, bundle, gallery_file, capsys):
        # image_flux_dev is not in the trimmed fixture, and no query was given.
        env = _invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--type", "image"], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]

    def test_ls_select_is_never_enriched(self, bundle, gallery_file, capsys):
        env = _invoke(
            templates_cmd.app,
            ["ls", "--gallery", gallery_file, "--tag", "Lip Sync", "--select", "rows.#.name"],
            capsys,
        )
        assert env["ok"] is True
        assert "knowledge" not in json.dumps(env)

    def test_show_reverse_resolves_the_template(self, bundle, gallery_file, capsys):
        env = _invoke(templates_cmd.app, ["show", "--gallery", gallery_file, "video_acme_h3_i2v"], capsys)
        data = env["data"]
        _validate(data, "templates.json")
        assert data["template"]["name"] == "video_acme_h3_i2v"
        assert data["knowledge"]["models"][0]["id"] == "acme-h3"

    def test_show_unknown_template_error_is_not_enriched(self, bundle, gallery_file, capsys):
        env = _invoke(templates_cmd.app, ["show", "--gallery", gallery_file, "video_acme_h3_t2v"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "template_not_found"
        assert "knowledge" not in json.dumps(env)

    def test_get_success_envelope_is_enriched(self, bundle, gallery_file, monkeypatch, capsys):
        body = json.dumps({"9": {"class_type": "KSampler", "inputs": {}}}).encode()
        monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", lambda name, timeout=15.0: body)
        env = _invoke(
            templates_cmd.app,
            ["get", "--gallery", gallery_file, "--where", "name=video_acme_h3_i2v"],
            capsys,
        )
        assert env["ok"] is True
        data = env["data"]
        _validate(data, "templates.json")
        assert data["workflow"] == {"9": {"class_type": "KSampler", "inputs": {}}}
        assert data["knowledge"]["models"][0]["id"] == "acme-h3"

    def test_get_error_paths_are_not_enriched(self, bundle, gallery_file, monkeypatch, capsys):
        monkeypatch.setattr(
            templates_cmd, "_fetch_template_workflow", lambda name, timeout=15.0: (_ for _ in ()).throw(OSError("x"))
        )
        env = _invoke(templates_cmd.app, ["get", "--gallery", gallery_file, "--where", "name=video_acme"], capsys)
        assert env["ok"] is False
        assert "knowledge" not in json.dumps(env)

    @pytest.mark.parametrize(
        "args",
        [
            ["ls", "--tag", "Lip Sync"],
            ["ls", "--model", "Testvid"],
            ["ls", "--name", "faceswap"],
            ["show", "video_acme_h3_i2v"],
        ],
    )
    def test_without_a_bundle_nothing_changes(self, gallery_file, args, capsys):
        env = _invoke(templates_cmd.app, [args[0], "--gallery", gallery_file, *args[1:]], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]
        _validate(env["data"], "templates.json")


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


class TestNodes:
    def test_search_query_hits_the_alias(self, bundle, object_info_file, capsys):
        env = _invoke(nodes_cmd.app, ["search", "--input", object_info_file, "testvid"], capsys)
        data = env["data"]
        _validate(data, "nodes.json")
        assert data["rows"][0]["name"] == "TestvidImage2VideoNode"
        assert data["knowledge"]["models"][0]["id"] == "testvid"
        assert data["knowledge"]["models"][0]["matched_on"] == "testvid"

    def test_search_zero_hit_gets_a_nudge(self, bundle, object_info_file, capsys):
        env = _invoke(nodes_cmd.app, ["search", "--input", object_info_file, "zzzz"], capsys)
        data = env["data"]
        _validate(data, "nodes.json")
        assert data["total"] == 0
        assert data["knowledge"]["zero_hit"] is True
        assert "'zzzz'" in data["knowledge"]["nudge"]

    def test_search_row_without_its_nodes_locally_is_annotated(self, bundle, tmp_path, capsys):
        # A catalog without any Testvid class: the alias hits, but the row's
        # resolves.nodes are absent here. The row still ships, marked unavailable —
        # reporting a miss would tell the caller nothing is curated, which is false.
        oi = {k: v for k, v in _object_info().items() if k != "TestvidImage2VideoNode"}
        path = tmp_path / "oi.json"
        path.write_text(json.dumps(oi))
        env = _invoke(nodes_cmd.app, ["search", "--input", str(path), "testvid"], capsys)
        assert env["ok"] is True
        k = env["data"]["knowledge"]
        _validate(env["data"], "nodes.json")
        assert [m["id"] for m in k["models"]] == ["testvid"]
        assert k["models"][0]["available_locally"] is False
        # The row cannot run here, so its capability's ranked picks come along.
        assert {p["capability"] for p in k["picks"]} == {"lipsync"}
        assert k["hit_ids"] == ["testvid", "cap:lipsync"]
        assert k["zero_hit"] is False
        assert "nudge" not in k

    def test_ls_reverse_resolves_listed_classes(self, bundle, object_info_file, capsys):
        env = _invoke(
            nodes_cmd.app,
            ["ls", "--input", object_info_file, "--category", "api node/video/Testvid"],
            capsys,
        )
        data = env["data"]
        _validate(data, "nodes.json")
        assert any(r["name"] == "TestvidImage2VideoNode" for r in data["rows"])
        assert [m["id"] for m in data["knowledge"]["models"]] == ["testvid"]
        assert data["knowledge"]["models"][0]["matched_on"] == "TestvidImage2VideoNode"

    def test_unfiltered_ls_is_never_enriched(self, bundle, object_info_file, capsys):
        """An unfiltered listing asked about nothing, so its rows are the catalog
        rather than an answer. Enriching it picked a curated row out of the pile
        and presented it as the reply to a question nobody put."""
        env = _invoke(nodes_cmd.app, ["ls", "--input", object_info_file], capsys)
        assert env["ok"] is True
        assert any(r["name"] == "TestvidImage2VideoNode" for r in env["data"]["rows"])
        assert "knowledge" not in env["data"]
        _validate(env["data"], "nodes.json")

    def test_ls_without_a_known_class_has_no_key(self, bundle, object_info_file, capsys):
        env = _invoke(nodes_cmd.app, ["ls", "--input", object_info_file, "--category", "sampling"], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]

    @pytest.mark.parametrize("args", [["search", "testvid"], ["search", "zzzz"], ["ls"]])
    def test_without_a_bundle_nothing_changes(self, object_info_file, args, capsys):
        env = _invoke(nodes_cmd.app, [args[0], "--input", object_info_file, *args[1:]], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]
        _validate(env["data"], "nodes.json")


# ---------------------------------------------------------------------------
# models search
# ---------------------------------------------------------------------------


ROW = {
    "name": "testvid_lora.safetensors",
    "display_name": "testvid_lora.safetensors",
    "type": "loras",
    "tags": ["loras"],
    "base_model": None,
    "trained_words": None,
    "source_url": None,
    "preview_url": None,
    "size": None,
    "is_public": False,
    "id": None,
}


@pytest.fixture
def local_target(monkeypatch):
    from comfy_cli.target import Target

    fake = Target(
        kind="local",
        base_url="http://127.0.0.1:8188",
        path_prefix="",
        history_path="history",
        host="127.0.0.1",
        port=8188,
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    return fake


class TestModelsSearch:
    def test_zero_results_get_a_nudge(self, bundle, local_target, monkeypatch, capsys):
        monkeypatch.setattr(models_search_cmd, "_local_search", lambda *a, **kw: ([], 0))
        env = _invoke(models_search_cmd.app, ["search", "--text", "faceswap", "--where", "local"], capsys)
        data = env["data"]
        _validate(data, "models.json")
        assert data["total"] == 0
        assert data["knowledge"]["zero_hit"] is True

    def test_text_hits_the_alias(self, bundle, local_target, monkeypatch, capsys):
        monkeypatch.setattr(models_search_cmd, "_local_search", lambda *a, **kw: ([ROW], 1))
        env = _invoke(models_search_cmd.app, ["search", "--text", "testvid", "--where", "local"], capsys)
        data = env["data"]
        _validate(data, "models.json")
        assert data["rows"] == [ROW]
        assert data["knowledge"]["models"][0]["id"] == "testvid"

    def test_no_text_means_no_key(self, bundle, local_target, monkeypatch, capsys):
        monkeypatch.setattr(models_search_cmd, "_local_search", lambda *a, **kw: ([], 0))
        env = _invoke(models_search_cmd.app, ["search", "--where", "local"], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]

    def test_without_a_bundle_nothing_changes(self, local_target, monkeypatch, capsys):
        monkeypatch.setattr(models_search_cmd, "_local_search", lambda *a, **kw: ([ROW], 1))
        env = _invoke(models_search_cmd.app, ["search", "--text", "testvid", "--where", "local"], capsys)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]
        _validate(env["data"], "models.json")


# ---------------------------------------------------------------------------
# generate (subprocess: the generate sub-app parses its own tail flags)
# ---------------------------------------------------------------------------


def _run_generate(args: list[str], *, tmp_path: Path, with_bundle: bool) -> dict:
    env = os.environ.copy()
    env.pop(knowledge.ENV_URL, None)
    env.pop(knowledge.ENV_TTL, None)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("COMFY_CLI_DISABLE_TELEMETRY", "1")
    if with_bundle:
        env[knowledge.ENV_FILE] = str(FIXTURE_KNOWLEDGE)
    else:
        env.pop(knowledge.ENV_FILE, None)
    result = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "--json", "generate", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.stdout.strip(), f"empty stdout. stderr={result.stderr!r}"
    last = [line for line in result.stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


class TestGenerate:
    def test_schema_carries_the_model_row_and_params_are_unchanged(self, tmp_path):
        # "kling" is a real generate alias the fixture keys to the synthetic row.
        enriched = _run_generate(["schema", "kling"], tmp_path=tmp_path, with_bundle=True)
        baseline = _run_generate(["schema", "kling"], tmp_path=tmp_path, with_bundle=False)
        assert enriched["ok"] is True and baseline["ok"] is True
        data = enriched["data"]
        _validate(data, "generate_schema.json")
        assert data["knowledge"]["models"][0]["id"] == "testvid"
        assert data["knowledge"]["models"][0]["tier"] == "law"
        assert "knowledge" not in baseline["data"]
        assert {k: v for k, v in data.items() if k != "knowledge"} == baseline["data"]
        assert data["params"] == baseline["data"]["params"]

    def test_schema_for_an_unkeyed_variant_gets_no_family_row(self, tmp_path):
        """`generate schema kling-lipsync` used to attach the whole `kling` row,
        available and green, while the bundle keyed no such variant."""
        env = _run_generate(["schema", "kling-lipsync"], tmp_path=tmp_path, with_bundle=True)
        assert env["ok"] is True
        assert "knowledge" not in env["data"]

    def test_list_is_brief_and_byte_identical_without_a_bundle(self, tmp_path):
        enriched = _run_generate(["list", "--query", "kling"], tmp_path=tmp_path, with_bundle=True)
        baseline = _run_generate(["list", "--query", "kling"], tmp_path=tmp_path, with_bundle=False)
        data = enriched["data"]
        _validate(data, "generate_list.json")
        models = data["knowledge"]["models"]
        assert models and all("pitfalls" not in m for m in models)
        assert any(m["id"] == "testvid" for m in models)
        assert "knowledge" not in baseline["data"]
        assert {k: v for k, v in data.items() if k != "knowledge"} == baseline["data"]
        _validate(baseline["data"], "generate_list.json")

    def test_unfiltered_list_is_never_enriched(self, tmp_path):
        """52 endpoints is the whole catalog, not an answer; enriching it attached
        rows for models the caller never named."""
        env = _run_generate(["list"], tmp_path=tmp_path, with_bundle=True)
        assert env["ok"] is True and env["data"]["count"] > 1
        assert "knowledge" not in env["data"]

    def test_list_zero_match_with_query_gets_a_nudge(self, tmp_path):
        env = _run_generate(["list", "--query", "faceswapzzz"], tmp_path=tmp_path, with_bundle=True)
        assert env["ok"] is True
        assert env["data"]["count"] == 0
        assert env["data"]["knowledge"]["zero_hit"] is True

    def test_list_select_is_never_enriched(self, tmp_path):
        env = _run_generate(["list", "--select", "models.#.alias"], tmp_path=tmp_path, with_bundle=True)
        assert env["ok"] is True
        assert "knowledge" not in json.dumps(env)
