"""Tests for `comfy workflow compose / fragment {ls,show,validate}`."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.fragments import (
    BlueprintError,
    Fragment,
    FragmentError,
    compose_blueprint,
    compose_blueprints,
    load_fragment,
    parse_fragment,
)
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


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


def _run(args: list[str], capsys) -> dict:
    """Invoke `comfy workflow ...` and parse the trailing JSON envelope.

    Mirrors the helper in ``test_workflow_slots.py``: capsys catches the
    renderer's emit; CliRunner's stdout is the fallback. We parse the last
    JSON line of the combined output.
    """
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:500]!r})")


# ---------------------------------------------------------------------------
# Fixtures: well-formed and malformed fragments
# ---------------------------------------------------------------------------


def _text_encode_fragment() -> dict:
    """Minimal STRING-typed fragment for unit tests."""
    return {
        "_fragment": {
            "name": "text_encode",
            "version": "1",
            "description": "Encode a prompt.",
            "inputs": {"clip": {"type": "STRING", "binds": "10.clip"}},
            "outputs": {"conditioning": {"type": "STRING", "from": "10", "port": 0}},
            "params": {"text": {"type": "STRING", "binds": "10.text", "default": "default prompt"}},
        },
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "PLACEHOLDER", "clip": "PLACEHOLDER"}},
    }


def _save_still_fragment() -> dict:
    """Terminal fragment (self-saves)."""
    return {
        "_fragment": {
            "name": "save_still",
            "version": "1",
            "terminal": True,
            "inputs": {"images": {"type": "IMAGE", "binds": "10.images"}},
            "outputs": {},
            "params": {"prefix": {"type": "STRING", "binds": "10.filename_prefix", "default": "out"}},
        },
        "10": {"class_type": "SaveImage", "inputs": {"images": "PLACEHOLDER", "filename_prefix": "out"}},
    }


def _image_blend_fragment() -> dict:
    """Two-IMAGE-input fragment with a FLOAT param."""
    return {
        "_fragment": {
            "name": "image_blend",
            "version": "1",
            "inputs": {
                "image1": {"type": "IMAGE", "binds": "10.image1"},
                "image2": {"type": "IMAGE", "binds": "10.image2"},
            },
            "outputs": {"image": {"type": "IMAGE", "from": "10", "port": 0}},
            "params": {
                "blend_factor": {"type": "FLOAT", "binds": "10.blend_factor", "default": 0.5},
            },
        },
        "10": {"class_type": "ImageBlend", "inputs": {"image1": "P", "image2": "P", "blend_factor": 0.5}},
    }


def _av_mux_fragment() -> dict:
    """VIDEO + AUDIO path inputs — literal/$asset values must materialize the
    loaders with their REAL input keys (LoadVideo.file, LoadAudio.audio per
    the server's object_info), not invented ones."""
    return {
        "_fragment": {
            "name": "av_mux",
            "version": "1",
            "terminal": True,
            "inputs": {
                "clip": {"type": "VIDEO", "binds": "10.video"},
                "track": {"type": "AUDIO", "binds": "10.audio"},
            },
            "outputs": {},
            "params": {},
        },
        "10": {"class_type": "SaveVideo", "inputs": {"video": "P", "audio": "P"}},
    }


def _model_producer_fragment() -> dict:
    """Produces graph-typed outputs (MODEL/LATENT) — like a checkpoint loader."""
    return {
        "_fragment": {
            "name": "model_producer",
            "version": "1",
            "inputs": {},
            "outputs": {
                "model": {"type": "MODEL", "from": "10", "port": 0},
                "latent": {"type": "LATENT", "from": "11", "port": 0},
            },
            "params": {"ckpt": {"type": "STRING", "binds": "10.ckpt_name", "default": "x.safetensors"}},
        },
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "11": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    }


def _model_consumer_fragment() -> dict:
    """Consumes graph-typed inputs (MODEL/LATENT) via cross-step refs — like a sampler."""
    return {
        "_fragment": {
            "name": "model_consumer",
            "version": "1",
            "inputs": {
                "model": {"type": "MODEL", "binds": "20.model"},
                "latent": {"type": "LATENT", "binds": "20.latent_image"},
            },
            "outputs": {"latent": {"type": "LATENT", "from": "20", "port": 0}},
            "params": {},
        },
        "20": {"class_type": "KSampler", "inputs": {"model": "PLACEHOLDER", "latent_image": "PLACEHOLDER"}},
    }


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    """A `fragments/` library directory pre-populated with fragments."""
    d = tmp_path / "fragments"
    d.mkdir()
    (d / "text_encode.json").write_text(json.dumps(_text_encode_fragment()))
    (d / "save_still.json").write_text(json.dumps(_save_still_fragment()))
    (d / "image_blend.json").write_text(json.dumps(_image_blend_fragment()))
    (d / "model_producer.json").write_text(json.dumps(_model_producer_fragment()))
    (d / "model_consumer.json").write_text(json.dumps(_model_consumer_fragment()))
    (d / "av_mux.json").write_text(json.dumps(_av_mux_fragment()))
    return d


# ---------------------------------------------------------------------------
# parse_fragment / load_fragment
# ---------------------------------------------------------------------------


class TestParseFragment:
    def test_minimal_well_formed(self):
        frag = parse_fragment(_text_encode_fragment())
        assert isinstance(frag, Fragment)
        assert frag.name == "text_encode"
        assert frag.version == "1"
        assert "clip" in frag.inputs
        assert "conditioning" in frag.outputs
        assert "text" in frag.params
        assert frag.params["text"].has_default
        assert frag.params["text"].default == "default prompt"
        assert frag.nodes["10"]["class_type"] == "CLIPTextEncode"

    def test_terminal_flag_parsed(self):
        frag = parse_fragment(_save_still_fragment())
        assert frag.terminal is True

    def test_default_terminal_is_false(self):
        frag = parse_fragment(_text_encode_fragment())
        assert frag.terminal is False

    def test_missing_metadata_header_rejected(self):
        with pytest.raises(FragmentError, match="missing `_fragment` metadata header"):
            parse_fragment({"10": {"class_type": "Foo", "inputs": {}}})

    def test_missing_name_rejected(self):
        data = _text_encode_fragment()
        del data["_fragment"]["name"]
        with pytest.raises(FragmentError, match="name"):
            parse_fragment(data)

    def test_graph_socket_input_types_accepted(self):
        """Graph-only socket types (MODEL/CONDITIONING/LATENT/VAE, custom) are valid inputs."""
        for t in ("MODEL", "CONDITIONING", "LATENT", "VAE", "CLIP", "CONTROL_NET"):
            data = _text_encode_fragment()
            data["_fragment"]["inputs"]["clip"]["type"] = t
            frag = parse_fragment(data)
            assert frag.inputs["clip"].type == t

    def test_malformed_input_type_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["inputs"]["clip"]["type"] = "conditioning"  # lowercase = not a socket type
        with pytest.raises(FragmentError, match="socket type"):
            parse_fragment(data)

    def test_unknown_param_type_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["params"]["text"]["type"] = "TENSOR"
        with pytest.raises(FragmentError, match="TENSOR"):
            parse_fragment(data)

    def test_boolean_param_type_accepted(self):
        """Node schemas print BOOLEAN; fragments copied from `nodes show` must parse."""
        data = _text_encode_fragment()
        data["_fragment"]["params"]["text"]["type"] = "BOOLEAN"
        data["_fragment"]["params"]["text"]["default"] = True
        frag = parse_fragment(data)
        assert frag.params["text"].type == "BOOLEAN"

    def test_legacy_bool_param_type_rejected(self):
        """One name per concept: node schemas print BOOLEAN, so the fragment
        grammar speaks BOOLEAN only — the BOOL alias is gone, and the error
        lists the accepted set."""
        data = _text_encode_fragment()
        data["_fragment"]["params"]["text"]["type"] = "BOOL"
        with pytest.raises(FragmentError, match=r"'BOOL' not in .*BOOLEAN"):
            parse_fragment(data)

    def test_junk_param_type_still_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["params"]["text"]["type"] = "WIBBLE"
        with pytest.raises(FragmentError, match="WIBBLE"):
            parse_fragment(data)

    def test_binds_without_dot_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["inputs"]["clip"]["binds"] = "10"
        with pytest.raises(FragmentError, match="must be '<node_id>.<input_name>'"):
            parse_fragment(data)

    def test_dangling_binds_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["inputs"]["clip"]["binds"] = "999.clip"
        with pytest.raises(FragmentError, match="missing interior node '999'"):
            parse_fragment(data)

    def test_dangling_from_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["outputs"]["conditioning"]["from"] = "999"
        with pytest.raises(FragmentError, match="from"):
            parse_fragment(data)

    def test_no_interior_nodes_rejected(self):
        data = {
            "_fragment": {
                "name": "empty",
                "inputs": {},
                "outputs": {},
                "params": {},
            }
        }
        with pytest.raises(FragmentError, match="no interior nodes"):
            parse_fragment(data)

    def test_interior_node_without_class_type_rejected(self):
        data = _text_encode_fragment()
        data["10"] = {"inputs": {}}  # no class_type
        with pytest.raises(FragmentError, match="class_type"):
            parse_fragment(data)


class TestLoadFragment:
    def test_load_from_disk(self, tmp_path: Path):
        p = tmp_path / "f.json"
        p.write_text(json.dumps(_text_encode_fragment()))
        frag = load_fragment(p)
        assert frag.name == "text_encode"
        assert frag.source_path == str(p)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FragmentError, match="not found"):
            load_fragment(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{ this isnt valid")
        with pytest.raises(FragmentError, match="not valid JSON"):
            load_fragment(p)


# ---------------------------------------------------------------------------
# Pipeline / compose_blueprint — composition behavior
# ---------------------------------------------------------------------------


class TestCompose:
    def test_single_step_with_terminal_saves_nothing_extra(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {
                    "fragment": "save_still",
                    "alias": "save",
                    "inputs": {"images": "inputs/photo.png"},
                    "params": {"prefix": "demo"},
                }
            ]
        }
        wf, summary = compose_blueprint(blueprint, lib_dir=lib_dir)
        # Terminal fragment → no auto-save appended
        save_image_nodes = [n for n in wf.values() if n["class_type"] == "SaveImage"]
        assert len(save_image_nodes) == 1  # the one inside save_still
        assert save_image_nodes[0]["inputs"]["filename_prefix"] == "demo"
        # LoadImage was injected for the IMAGE input
        load_nodes = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(load_nodes) == 1
        assert load_nodes[0]["inputs"]["image"] == "inputs/photo.png"
        # save_action is None because the step was terminal
        assert summary["save_action"] is None
        assert summary["steps"] == 1
        assert summary["fragments_used"] == ["save_still"]

    def test_default_param_applied_when_omitted(self, lib_dir: Path):
        blueprint = {"pipeline": [{"fragment": "text_encode", "alias": "p", "inputs": {"clip": "fake_clip"}}]}
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        encode = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"][0]
        assert encode["inputs"]["text"] == "default prompt"

    def test_param_override(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {
                    "fragment": "text_encode",
                    "alias": "p",
                    "inputs": {"clip": "fake_clip"},
                    "params": {"text": "OVERRIDE"},
                }
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        encode = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"][0]
        assert encode["inputs"]["text"] == "OVERRIDE"

    def test_cross_step_ref_wires_to_prior_output(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "clip_a"}, "params": {"text": "first"}},
                {
                    "fragment": "text_encode",
                    "alias": "p2",
                    "inputs": {"clip": "$p1.conditioning"},
                    "params": {"text": "second"},
                },
            ]
        }
        wf, summary = compose_blueprint(blueprint, lib_dir=lib_dir)
        # The second CLIPTextEncode's `clip` input must be a node reference to the first
        encodes = sorted(
            [(nid, n) for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode"],
            key=lambda x: int(x[0]),
        )
        assert len(encodes) == 2
        p1_nid, _ = encodes[0]
        _, p2_node = encodes[1]
        assert p2_node["inputs"]["clip"] == [p1_nid, 0]

    def test_image_input_injects_loadimage(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {
                    "fragment": "image_blend",
                    "alias": "b",
                    "inputs": {"image1": "a.png", "image2": "b.png"},
                }
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        load_nodes = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(load_nodes) == 2
        loaded_paths = {n["inputs"]["image"] for n in load_nodes}
        assert loaded_paths == {"a.png", "b.png"}

    def test_video_input_injects_loadvideo_with_file_key(self, lib_dir: Path):
        """LoadVideo's only input is `file` (COMBO, per cloud object_info).
        Wiring `video` validates client-side then burns the cloud run
        server-side (ImageDownloadError) — fennec friction #7."""
        blueprint = {
            "pipeline": [
                {
                    "fragment": "av_mux",
                    "alias": "m",
                    "inputs": {"clip": "s1.mp4", "track": "score.flac"},
                }
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        load_videos = [n for n in wf.values() if n["class_type"] == "LoadVideo"]
        assert len(load_videos) == 1
        assert load_videos[0]["inputs"] == {"file": "s1.mp4"}

    def test_audio_input_injects_loadaudio_with_audio_key(self, lib_dir: Path):
        """LoadAudio's real input key is `audio` (COMBO, per cloud object_info)."""
        blueprint = {
            "pipeline": [
                {
                    "fragment": "av_mux",
                    "alias": "m",
                    "inputs": {"clip": "s1.mp4", "track": "score.flac"},
                }
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        load_audios = [n for n in wf.values() if n["class_type"] == "LoadAudio"]
        assert len(load_audios) == 1
        assert load_audios[0]["inputs"] == {"audio": "score.flac"}

    def test_asset_resolved_video_and_audio_use_real_loader_keys(self, lib_dir: Path):
        """$asset values fall through to the SAME loader materialization a
        literal filename gets — same real input keys."""
        blueprint = {
            "pipeline": [
                {
                    "fragment": "av_mux",
                    "alias": "m",
                    "inputs": {"clip": "$asset.clips/s1.mp4", "track": "$asset.score.flac"},
                }
            ]
        }
        wf, _ = compose_blueprint(
            blueprint,
            lib_dir=lib_dir,
            asset_resolver=lambda name: f"cloud-{Path(name).name}",
        )
        load_videos = [n for n in wf.values() if n["class_type"] == "LoadVideo"]
        load_audios = [n for n in wf.values() if n["class_type"] == "LoadAudio"]
        assert load_videos[0]["inputs"] == {"file": "cloud-s1.mp4"}
        assert load_audios[0]["inputs"] == {"audio": "cloud-score.flac"}

    def test_node_ids_remapped_no_collision(self, lib_dir: Path):
        """Two instances of the same fragment must not collide on interior node IDs."""
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "a", "inputs": {"clip": "ca"}, "params": {"text": "x"}},
                {"fragment": "text_encode", "alias": "b", "inputs": {"clip": "cb"}, "params": {"text": "y"}},
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        # Both fragments use interior id "10"; the merged workflow must have two
        # distinct CLIPTextEncode nodes with distinct IDs.
        encode_ids = [nid for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode"]
        assert len(encode_ids) == 2
        assert len(set(encode_ids)) == 2

    def test_non_terminal_final_auto_appends_save(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {
                    "fragment": "image_blend",
                    "alias": "b",
                    "inputs": {"image1": "a.png", "image2": "b.png"},
                }
            ],
            "output_prefix": "myprefix",
        }
        wf, summary = compose_blueprint(blueprint, lib_dir=lib_dir)
        saves = [n for n in wf.values() if n["class_type"] == "SaveImage"]
        assert len(saves) == 1
        assert saves[0]["inputs"]["filename_prefix"] == "myprefix"
        assert summary["save_action"] == {"type": "IMAGE", "prefix": "myprefix"}

    def test_missing_input_errors_with_step_alias(self, lib_dir: Path):
        blueprint = {"pipeline": [{"fragment": "text_encode", "alias": "only", "params": {"text": "x"}}]}
        with pytest.raises(BlueprintError) as exc:
            compose_blueprint(blueprint, lib_dir=lib_dir)
        assert exc.value.step_alias == "only"
        assert "missing required input" in str(exc.value)

    def test_unknown_input_key_errors(self, lib_dir: Path):
        blueprint = {"pipeline": [{"fragment": "text_encode", "alias": "x", "inputs": {"clip": "a", "typo": "b"}}]}
        with pytest.raises(BlueprintError, match="unknown inputs"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_unknown_param_key_errors(self, lib_dir: Path):
        blueprint = {
            "pipeline": [{"fragment": "text_encode", "alias": "x", "inputs": {"clip": "a"}, "params": {"typo": 1}}]
        }
        with pytest.raises(BlueprintError, match="unknown params"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_duplicate_alias_errors(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "dup", "inputs": {"clip": "a"}},
                {"fragment": "text_encode", "alias": "dup", "inputs": {"clip": "b"}},
            ]
        }
        with pytest.raises(BlueprintError, match="dup"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_unknown_alias_in_cross_ref(self, lib_dir: Path):
        blueprint = {"pipeline": [{"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$nope.conditioning"}}]}
        with pytest.raises(BlueprintError, match="unknown alias"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_unknown_output_name_in_cross_ref(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "x"}},
                {"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$p1.no_such_output"}},
            ]
        }
        with pytest.raises(BlueprintError, match="no output"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_empty_pipeline_errors(self, lib_dir: Path):
        with pytest.raises(BlueprintError, match="blueprint"):
            compose_blueprint({}, lib_dir=lib_dir)
        with pytest.raises(BlueprintError, match="blueprint"):
            compose_blueprint({"pipeline": []}, lib_dir=lib_dir)

    def test_graph_typed_inputs_wire_via_cross_ref(self, lib_dir: Path):
        """MODEL/LATENT inputs fed by `$alias.output` wire to the producer's nodes."""
        blueprint = {
            "pipeline": [
                {"fragment": "model_producer", "alias": "base", "params": {"ckpt": "m.safetensors"}},
                {
                    "fragment": "model_consumer",
                    "alias": "samp",
                    "inputs": {"model": "$base.model", "latent": "$base.latent"},
                },
            ]
        }
        wf, _ = compose_blueprint(blueprint, lib_dir=lib_dir)
        ckpt = next(nid for nid, n in wf.items() if n["class_type"] == "CheckpointLoaderSimple")
        empty = next(nid for nid, n in wf.items() if n["class_type"] == "EmptyLatentImage")
        ksampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
        assert ksampler["inputs"]["model"] == [ckpt, 0]
        assert ksampler["inputs"]["latent_image"] == [empty, 0]

    def test_graph_typed_input_with_path_literal_errors(self, lib_dir: Path):
        """A graph-only type can't be loaded from a path — must use a cross-step ref."""
        blueprint = {
            "pipeline": [
                {
                    "fragment": "model_consumer",
                    "alias": "samp",
                    "inputs": {"model": "some/model.safetensors", "latent": "$samp.latent"},
                },
            ]
        }
        with pytest.raises(BlueprintError, match="can't be loaded from a path"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_malformed_cross_step_ref_errors_clearly(self, lib_dir: Path):
        """A bad ref like `$ref:p1.x` reports a malformed ref, not a cryptic unknown alias."""
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "x"}},
                {"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$ref:p1.conditioning"}},
            ]
        }
        with pytest.raises(BlueprintError, match="malformed cross-step ref"):
            compose_blueprint(blueprint, lib_dir=lib_dir)

    def test_invalid_alias_rejected(self, lib_dir: Path):
        blueprint = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "bad:alias", "inputs": {"clip": "x"}},
            ]
        }
        with pytest.raises(BlueprintError, match="valid identifier"):
            compose_blueprint(blueprint, lib_dir=lib_dir)


# ---------------------------------------------------------------------------
# CLI integration tests via Typer's CliRunner
# ---------------------------------------------------------------------------


class TestComposeCmd:
    def test_compose_writes_compiled_json(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "demo.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            pipeline:
              - fragment: text_encode
                alias: p
                inputs: {clip: clip_a}
                params: {text: hello}
        """)
        )
        out = tmp_path / "built.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["nodes"] >= 1
        assert out.exists()
        wf = json.loads(out.read_text())
        encodes = [n for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
        assert encodes[0]["inputs"]["text"] == "hello"

    def test_compose_missing_blueprint(self, tmp_path: Path, capsys):
        envelope = _run(["compose", str(tmp_path / "nope.yaml")], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "blueprint_not_found"


class TestFragmentizeCmd:
    _RESTYLE_WF = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "base.png"}, "_meta": {"title": "Base"}},
        "2": {
            "class_type": "FluxKontextProImageNode",
            "inputs": {"prompt": "as an oil painting", "input_image": ["1", 0], "seed": 7},
            "_meta": {"title": "Kontext"},
        },
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "out"}},
    }

    def test_projects_api_workflow_to_valid_fragment_file(self, tmp_path: Path, capsys):
        wf = tmp_path / "restyle.json"
        wf.write_text(json.dumps(self._RESTYLE_WF))
        lib = tmp_path / "fragments"

        envelope = _run(["decompose", str(wf), "--name", "restyle", "--lib", str(lib)], capsys)

        assert envelope["ok"] is True, envelope
        data = envelope["data"]
        assert data["name"] == "restyle"
        assert data["ports"] == {"inputs": 1, "outputs": 1, "params": 2}  # prompt + seed

        # The written file is a real, loadable fragment.
        frag = load_fragment(lib / "restyle.json")
        assert any(p.binds == "2.prompt" for p in frag.params.values())
        assert "1" not in frag.nodes and "3" not in frag.nodes  # boundaries stripped

    def test_missing_workflow_file_errors(self, tmp_path: Path, capsys):
        envelope = _run(["decompose", str(tmp_path / "nope.json")], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "workflow_not_found"

    def test_compose_invalid_yaml(self, tmp_path: Path, capsys):
        blueprint = tmp_path / "bad.yaml"
        blueprint.write_text("pipeline: [ this is not balanced")
        envelope = _run(["compose", str(blueprint)], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "blueprint_invalid_yaml"

    def test_compose_missing_input_returns_blueprint_invalid(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "r.yaml"
        blueprint.write_text("pipeline:\n  - fragment: text_encode\n    alias: x\n    params: {text: y}\n")
        envelope = _run(["compose", str(blueprint), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "blueprint_invalid"
        assert "missing required input" in envelope["error"]["message"]

    def test_compose_foreach_single_graph(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            foreach:
              - {id: a, prompt: alpha}
              - {id: b, prompt: beta}
              - {id: c, prompt: gamma}
            pipeline:
              - fragment: text_encode
                alias: enc
                inputs: {clip: clip_a}
                params: {text: $item.prompt}
        """)
        )
        out = tmp_path / "fan.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["graphs"] == 1
        assert envelope["data"]["items"] == 3
        wf = json.loads(out.read_text())
        texts = {
            n["inputs"]["text"] for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        }
        assert texts == {"alpha", "beta", "gamma"}

    def test_compose_chunk_writes_multiple_files(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            chunk: 2
            foreach:
              - {id: a, prompt: alpha}
              - {id: b, prompt: beta}
              - {id: c, prompt: gamma}
            pipeline:
              - fragment: text_encode
                alias: enc
                inputs: {clip: clip_a}
                params: {text: $item.prompt}
        """)
        )
        out = tmp_path / "fan.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["graphs"] == 2
        assert len(envelope["data"]["written"]) == 2
        for path in envelope["data"]["written"]:
            assert Path(path).exists()

    def test_compose_chunk_clears_stale_unnumbered_file(self, lib_dir: Path, tmp_path: Path, capsys):
        """chunked compose must unlink any stale unnumbered base_out and set out=None."""
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            chunk: 2
            foreach:
              - {id: a, prompt: alpha}
              - {id: b, prompt: beta}
              - {id: c, prompt: gamma}
            pipeline:
              - fragment: text_encode
                alias: enc
                inputs: {clip: clip_a}
                params: {text: $item.prompt}
        """)
        )
        out = tmp_path / "fan.json"
        # Pre-write a stale unnumbered file (simulates a prior single-graph compose).
        out.write_text('{"stale": true}', encoding="utf-8")
        assert out.exists()  # confirm setup

        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        # The stale unnumbered file must have been removed.
        assert not out.exists(), "stale unnumbered base_out must be deleted on chunked compose"
        # out must be None for multi-graph (no single runnable file).
        assert envelope["data"]["out"] is None, "envelope.data.out must be None for chunked compose"
        # written must list both chunk files.
        assert len(envelope["data"]["written"]) == 2

    def test_compose_foreach_ref_resolves_relative_to_blueprint(self, lib_dir: Path, tmp_path: Path, capsys):
        (tmp_path / "items.yaml").write_text(
            textwrap.dedent("""\
            - {id: a, prompt: one}
            - {id: b, prompt: two}
        """)
        )
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            foreach: {$ref: items.yaml}
            pipeline:
              - fragment: text_encode
                alias: enc
                inputs: {clip: clip_a}
                params: {text: $item.prompt}
        """)
        )
        out = tmp_path / "fan.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["items"] == 2
        wf = json.loads(out.read_text())
        texts = {
            n["inputs"]["text"] for n in wf.values() if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
        }
        assert texts == {"one", "two"}


class TestFragmentCmds:
    def test_ls_lists_library(self, lib_dir: Path, capsys):
        envelope = _run(["fragment", "ls", "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        names = {f["name"] for f in envelope["data"]["fragments"]}
        assert names == {
            "text_encode",
            "save_still",
            "image_blend",
            "model_producer",
            "model_consumer",
            "av_mux",
        }

    def test_ls_missing_lib(self, tmp_path: Path, capsys):
        envelope = _run(["fragment", "ls", "--lib", str(tmp_path / "nope")], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "fragment_lib_not_found"

    def test_show_returns_full_schema(self, lib_dir: Path, capsys):
        envelope = _run(["fragment", "show", "text_encode", "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        d = envelope["data"]
        assert d["name"] == "text_encode"
        assert "clip" in d["inputs"]
        assert "conditioning" in d["outputs"]
        assert d["params"]["text"]["default"] == "default prompt"
        assert d["node_count"] == 1

    def test_validate_well_formed(self, lib_dir: Path, capsys):
        envelope = _run(["fragment", "validate", "text_encode", "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["valid"] is True


# ---------------------------------------------------------------------------
# foreach — fan-out a pipeline template over N items into ONE multi-branch graph
# ---------------------------------------------------------------------------


class TestForeach:
    def test_inline_list_compiles_to_one_graph_with_n_branches(self, lib_dir: Path):
        """A 3-item foreach instantiates the pipeline 3× as independent branches."""
        blueprint = {
            "foreach": [
                {"id": "a", "prompt": "alpha"},
                {"id": "b", "prompt": "beta"},
                {"id": "c", "prompt": "gamma"},
            ],
            "pipeline": [
                {
                    "fragment": "text_encode",
                    "alias": "enc",
                    "inputs": {"clip": "fake_clip"},
                    "params": {"text": "$item.prompt"},
                }
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir)
        assert len(graphs) == 1
        wf, summary = graphs[0]
        encodes = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
        # 3 independent branches, one per item, each bound to its own prompt.
        assert len(encodes) == 3
        assert {n["inputs"]["text"] for n in encodes} == {"alpha", "beta", "gamma"}
        assert summary["items"] == 3
        assert summary["graphs"] == 1

    def test_per_item_alias_namespacing_no_collision(self, lib_dir: Path):
        """The N copies must not collide on interior node IDs."""
        blueprint = {
            "foreach": [{"id": "x", "prompt": "p1"}, {"id": "y", "prompt": "p2"}],
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}}
            ],
        }
        wf, _ = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        encode_ids = [nid for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode"]
        assert len(encode_ids) == len(set(encode_ids)) == 2

    def test_cross_step_ref_stays_within_item_branch(self, lib_dir: Path):
        """A `$alias.output` ref must resolve to the SAME item's branch, not another item's."""
        blueprint = {
            "foreach": [{"id": "a", "prompt": "first"}, {"id": "b", "prompt": "second"}],
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}},
                {
                    "fragment": "text_encode",
                    "alias": "p2",
                    "inputs": {"clip": "$p1.conditioning"},
                    "params": {"text": "x"},
                },
            ],
        }
        wf, _ = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        # Group the p1 encode (has the item prompt) and p2 encode (text "x") per branch.
        # Each p2 must wire to a p1 in the same branch — i.e. exactly 2 distinct p1 targets.
        p2_targets = {
            tuple(n["inputs"]["clip"])
            for n in wf.values()
            if n["class_type"] == "CLIPTextEncode" and n["inputs"]["text"] == "x"
        }
        assert len(p2_targets) == 2  # each p2 wires to a distinct p1 in its own branch

    def test_item_whole_value_substitution(self, lib_dir: Path):
        """`$item` (no field) substitutes the whole item value."""
        blueprint = {
            "foreach": ["red", "green"],
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item"}}
            ],
        }
        wf, _ = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        texts = {n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"}
        assert texts == {"red", "green"}

    def test_per_item_output_prefix_uses_item_id(self, lib_dir: Path):
        """Auto-saved terminal output prefix incorporates the item id."""
        blueprint = {
            "output_prefix": "outputs",
            "foreach": [{"id": "shotA"}, {"id": "shotB"}],
            "pipeline": [{"fragment": "image_blend", "alias": "b", "inputs": {"image1": "a.png", "image2": "b.png"}}],
        }
        wf, _ = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        prefixes = {n["inputs"]["filename_prefix"] for n in wf.values() if n["class_type"] == "SaveImage"}
        assert prefixes == {"outputs/shotA", "outputs/shotB"}

    def test_ref_loads_items_from_external_yaml(self, lib_dir: Path, tmp_path: Path):
        """foreach: {$ref: items.yaml} resolves relative to the blueprint dir."""
        import yaml

        items_file = tmp_path / "items.yaml"
        items_file.write_text(yaml.safe_dump([{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two"}]))
        blueprint = {
            "foreach": {"$ref": "items.yaml"},
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}}
            ],
        }
        wf, summary = compose_blueprints(blueprint, lib_dir=lib_dir, blueprint_dir=tmp_path)[0]
        texts = {n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"}
        assert texts == {"one", "two"}
        assert summary["items"] == 2

    def test_chunk_splits_into_multiple_graphs(self, lib_dir: Path):
        """chunk: 2 over 3 items → ceil(3/2) = 2 graphs (2 items, then 1 item)."""
        blueprint = {
            "chunk": 2,
            "foreach": [
                {"id": "a", "prompt": "alpha"},
                {"id": "b", "prompt": "beta"},
                {"id": "c", "prompt": "gamma"},
            ],
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}}
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir)
        assert len(graphs) == 2
        counts = [len([n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]) for wf, _ in graphs]
        assert counts == [2, 1]
        # Every item's prompt appears exactly once across all graphs.
        all_texts = set()
        for wf, _ in graphs:
            all_texts |= {n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"}
        assert all_texts == {"alpha", "beta", "gamma"}
        for _, summary in graphs:
            assert summary["graphs"] == 2

    def test_no_foreach_returns_single_graph(self, lib_dir: Path):
        """compose_blueprints without foreach behaves like a single linear graph."""
        blueprint = {
            "pipeline": [{"fragment": "text_encode", "alias": "p", "inputs": {"clip": "c"}, "params": {"text": "hi"}}]
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir)
        assert len(graphs) == 1
        wf, summary = graphs[0]
        assert [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"][0]["inputs"]["text"] == "hi"
        assert summary.get("items") is None or summary["items"] == 1

    def test_empty_foreach_errors(self, lib_dir: Path):
        blueprint = {
            "foreach": [],
            "pipeline": [{"fragment": "text_encode", "alias": "p", "inputs": {"clip": "c"}}],
        }
        with pytest.raises(BlueprintError, match="foreach"):
            compose_blueprints(blueprint, lib_dir=lib_dir)

    def test_missing_item_field_errors(self, lib_dir: Path):
        blueprint = {
            "foreach": [{"id": "a"}],
            "pipeline": [
                {"fragment": "text_encode", "alias": "p", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}}
            ],
        }
        with pytest.raises(BlueprintError, match="prompt"):
            compose_blueprints(blueprint, lib_dir=lib_dir)

    def test_ref_without_blueprint_dir_errors(self, lib_dir: Path):
        blueprint = {
            "foreach": {"$ref": "items.yaml"},
            "pipeline": [{"fragment": "text_encode", "alias": "p", "inputs": {"clip": "c"}}],
        }
        with pytest.raises(BlueprintError, match="\\$ref"):
            compose_blueprints(blueprint, lib_dir=lib_dir)


# ---------------------------------------------------------------------------
# item_map — per-item provenance in compose summaries and _meta embedding
# ---------------------------------------------------------------------------


class TestItemMap:
    def _foreach_blend_blueprint(self) -> dict:
        return {
            "output_prefix": "story",
            "foreach": [{"id": "s1"}, {"id": "s2"}],
            "pipeline": [{"fragment": "image_blend", "alias": "b", "inputs": {"image1": "a.png", "image2": "b.png"}}],
        }

    def test_item_map_disjoint_nodes_with_save_inside(self, lib_dir: Path):
        wf, summary = compose_blueprints(self._foreach_blend_blueprint(), lib_dir=lib_dir)[0]
        im = summary["item_map"]
        assert set(im) == {"s1", "s2"}
        s1, s2 = set(im["s1"]["nodes"]), set(im["s2"]["nodes"])
        # Node sets are disjoint and together cover the whole graph.
        assert s1.isdisjoint(s2)
        assert s1 | s2 == set(wf)
        for key in ("s1", "s2"):
            entry = im[key]
            # Node lists are sorted numerically.
            assert entry["nodes"] == sorted(entry["nodes"], key=int)
            # The auto-appended save node belongs to its item's node set.
            assert entry["save_node"] in entry["nodes"]
            assert wf[entry["save_node"]]["class_type"] == "SaveImage"
            # The per-item prefix follows the `_item_prefix` rule (base/id).
            assert entry["prefix"] == f"story/{key}"
            assert wf[entry["save_node"]]["inputs"]["filename_prefix"] == entry["prefix"]

    def test_item_map_keys_fall_back_to_index_for_scalar_items(self, lib_dir: Path):
        blueprint = {
            "foreach": ["red", "green"],
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item"}}
            ],
        }
        _, summary = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        assert set(summary["item_map"]) == {"0", "1"}

    def test_item_map_terminal_branch_has_no_save_node(self, lib_dir: Path):
        blueprint = {
            "foreach": [{"id": "s1"}],
            "pipeline": [{"fragment": "save_still", "alias": "save", "inputs": {"images": "inputs/p.png"}}],
        }
        wf, summary = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        entry = summary["item_map"]["s1"]
        assert entry["save_node"] is None
        assert set(entry["nodes"]) == set(wf)

    def test_chunked_item_map_covers_only_its_batch(self, lib_dir: Path):
        blueprint = {
            "chunk": 2,
            "foreach": [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}, {"id": "c", "prompt": "z"}],
            "pipeline": [
                {"fragment": "text_encode", "alias": "enc", "inputs": {"clip": "c"}, "params": {"text": "$item.prompt"}}
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir)
        assert len(graphs) == 2
        assert set(graphs[0][1]["item_map"]) == {"a", "b"}
        assert set(graphs[1][1]["item_map"]) == {"c"}

    def test_non_foreach_summary_has_no_item_map(self, lib_dir: Path):
        blueprint = {
            "pipeline": [{"fragment": "text_encode", "alias": "p", "inputs": {"clip": "c"}, "params": {"text": "hi"}}]
        }
        _, summary = compose_blueprints(blueprint, lib_dir=lib_dir)[0]
        assert not summary.get("item_map")


class TestComposeCmdMeta:
    """compose_cmd embeds a `_meta` provenance block in every written workflow."""

    def test_written_workflow_carries_meta_with_items(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            output_prefix: story
            foreach:
              - {id: s1}
              - {id: s2}
            pipeline:
              - fragment: image_blend
                alias: b
                inputs: {image1: a.png, image2: b.png}
        """)
        )
        out = tmp_path / "fan.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        wf = json.loads(out.read_text())
        meta = wf["_meta"]
        assert meta["schema"] == "compose/1"
        assert meta["blueprint"] == str(blueprint.resolve())
        assert set(meta["items"]) == {"s1", "s2"}
        for key, entry in meta["items"].items():
            assert entry["save_node"] in entry["nodes"]
            assert entry["prefix"] == f"story/{key}"
        # The same map rides the envelope payload.
        assert set(envelope["data"]["item_map"]) == {"s1", "s2"}

    def test_non_foreach_blueprint_still_gets_meta(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "single.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            pipeline:
              - fragment: text_encode
                alias: p
                inputs: {clip: clip_a}
                params: {text: hello}
        """)
        )
        out = tmp_path / "single.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        wf = json.loads(out.read_text())
        meta = wf["_meta"]
        assert meta["schema"] == "compose/1"
        assert meta["blueprint"] == str(blueprint.resolve())
        assert "items" not in meta
        assert "item_map" not in envelope["data"]

    def test_chunked_files_carry_only_their_batch_items(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "fan.yaml"
        blueprint.write_text(
            textwrap.dedent("""\
            chunk: 2
            foreach:
              - {id: a, prompt: alpha}
              - {id: b, prompt: beta}
              - {id: c, prompt: gamma}
            pipeline:
              - fragment: text_encode
                alias: enc
                inputs: {clip: clip_a}
                params: {text: $item.prompt}
        """)
        )
        out = tmp_path / "fan.json"
        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        written = envelope["data"]["written"]
        assert len(written) == 2
        items_per_file = [set(json.loads(Path(p).read_text())["_meta"]["items"]) for p in written]
        assert items_per_file == [{"a", "b"}, {"c"}]
        # Envelope-level map is the union across batches.
        assert set(envelope["data"]["item_map"]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# _substitute_item — alias_refs flag prevents namespacing literal string params
# ---------------------------------------------------------------------------


def test_foreach_literal_string_param_not_namespaced():
    from comfy_cli.fragments import _substitute_item

    # params path: alias-looking literals must NOT be rewritten
    assert _substitute_item("$brand.tagline", {"x": 1}, ns="i0", alias_refs=False) == "$brand.tagline"
    # inputs path: $alias.output cross-step refs ARE namespaced (unchanged behavior)
    assert _substitute_item("$brand.tagline", {"x": 1}, ns="i0", alias_refs=True) == "$i0__brand.tagline"
    # $item substitution still works on both paths
    assert _substitute_item("$item.x", {"x": 42}, ns="i0", alias_refs=False) == 42
    assert _substitute_item("$item.x", {"x": 42}, ns="i0", alias_refs=True) == 42


# ---------------------------------------------------------------------------
# compose journals into the governing project (best-effort)
# ---------------------------------------------------------------------------


class TestComposeJournal:
    """Inside a project/1 tree, compose appends one runs.jsonl line; outside,
    nothing is written; a journaling failure never fails the command."""

    BLUEPRINT = textwrap.dedent("""\
        pipeline:
          - fragment: text_encode
            alias: p
            inputs: {clip: clip_a}
            params: {text: hello}
    """)

    def _project(self, tmp_path: Path) -> Path:
        proj = tmp_path / "proj"
        (proj / "blueprints").mkdir(parents=True)
        (proj / "comfy.yaml").write_text("schema: project/1\ndefaults:\n  where: cloud\n")
        return proj

    def test_compose_inside_project_appends_journal_line(self, lib_dir: Path, tmp_path: Path, capsys):
        proj = self._project(tmp_path)
        blueprint = proj / "blueprints" / "demo.yaml"
        blueprint.write_text(self.BLUEPRINT)
        out = proj / "built.json"

        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True

        lines = (proj / ".comfy" / "runs.jsonl").read_text().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["cmd"] == "compose"
        assert ev["blueprint"] == str(blueprint)
        assert ev["written"] == [str(out)]
        assert "ts" in ev

    def test_compose_outside_project_writes_no_journal(self, lib_dir: Path, tmp_path: Path, capsys):
        blueprint = tmp_path / "demo.yaml"
        blueprint.write_text(self.BLUEPRINT)
        out = tmp_path / "built.json"

        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert not (tmp_path / ".comfy").exists()

    def test_journal_failure_does_not_fail_compose(self, lib_dir: Path, tmp_path: Path, capsys, monkeypatch):
        import comfy_cli.project as project_mod

        def _boom(*a, **kw):
            raise RuntimeError("journal exploded")

        monkeypatch.setattr(project_mod, "journal", _boom)
        proj = self._project(tmp_path)
        blueprint = proj / "blueprints" / "demo.yaml"
        blueprint.write_text(self.BLUEPRINT)
        out = proj / "built.json"

        envelope = _run(["compose", str(blueprint), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert out.exists()
