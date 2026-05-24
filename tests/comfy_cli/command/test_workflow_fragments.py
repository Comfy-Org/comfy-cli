"""Tests for `comfy workflow compose / fragment {ls,show,validate}`."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.command.workflow_fragments import (
    Fragment,
    FragmentError,
    Pipeline,
    RecipeError,
    compose_recipe,
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
    raise AssertionError(
        f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:500]!r})"
    )


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


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    """A `fragments/` library directory pre-populated with three fragments."""
    d = tmp_path / "fragments"
    d.mkdir()
    (d / "text_encode.json").write_text(json.dumps(_text_encode_fragment()))
    (d / "save_still.json").write_text(json.dumps(_save_still_fragment()))
    (d / "image_blend.json").write_text(json.dumps(_image_blend_fragment()))
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

    def test_unknown_input_type_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["inputs"]["clip"]["type"] = "CONDITIONING"
        with pytest.raises(FragmentError, match="type 'CONDITIONING' not in"):
            parse_fragment(data)

    def test_unknown_param_type_rejected(self):
        data = _text_encode_fragment()
        data["_fragment"]["params"]["text"]["type"] = "TENSOR"
        with pytest.raises(FragmentError, match="TENSOR"):
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
# Pipeline / compose_recipe — composition behavior
# ---------------------------------------------------------------------------


class TestCompose:
    def test_single_step_with_terminal_saves_nothing_extra(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {
                    "fragment": "save_still",
                    "alias": "save",
                    "inputs": {"images": "inputs/photo.png"},
                    "params": {"prefix": "demo"},
                }
            ]
        }
        wf, summary = compose_recipe(recipe, lib_dir=lib_dir)
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
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p", "inputs": {"clip": "fake_clip"}}
            ]
        }
        wf, _ = compose_recipe(recipe, lib_dir=lib_dir)
        encode = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"][0]
        assert encode["inputs"]["text"] == "default prompt"

    def test_param_override(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {
                    "fragment": "text_encode",
                    "alias": "p",
                    "inputs": {"clip": "fake_clip"},
                    "params": {"text": "OVERRIDE"},
                }
            ]
        }
        wf, _ = compose_recipe(recipe, lib_dir=lib_dir)
        encode = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"][0]
        assert encode["inputs"]["text"] == "OVERRIDE"

    def test_cross_step_ref_wires_to_prior_output(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "clip_a"}, "params": {"text": "first"}},
                {"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$p1.conditioning"}, "params": {"text": "second"}},
            ]
        }
        wf, summary = compose_recipe(recipe, lib_dir=lib_dir)
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
        recipe = {
            "pipeline": [
                {
                    "fragment": "image_blend",
                    "alias": "b",
                    "inputs": {"image1": "a.png", "image2": "b.png"},
                }
            ]
        }
        wf, _ = compose_recipe(recipe, lib_dir=lib_dir)
        load_nodes = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(load_nodes) == 2
        loaded_paths = {n["inputs"]["image"] for n in load_nodes}
        assert loaded_paths == {"a.png", "b.png"}

    def test_node_ids_remapped_no_collision(self, lib_dir: Path):
        """Two instances of the same fragment must not collide on interior node IDs."""
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "a", "inputs": {"clip": "ca"}, "params": {"text": "x"}},
                {"fragment": "text_encode", "alias": "b", "inputs": {"clip": "cb"}, "params": {"text": "y"}},
            ]
        }
        wf, _ = compose_recipe(recipe, lib_dir=lib_dir)
        # Both fragments use interior id "10"; the merged workflow must have two
        # distinct CLIPTextEncode nodes with distinct IDs.
        encode_ids = [nid for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode"]
        assert len(encode_ids) == 2
        assert len(set(encode_ids)) == 2

    def test_non_terminal_final_auto_appends_save(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {
                    "fragment": "image_blend",
                    "alias": "b",
                    "inputs": {"image1": "a.png", "image2": "b.png"},
                }
            ],
            "output_prefix": "myprefix",
        }
        wf, summary = compose_recipe(recipe, lib_dir=lib_dir)
        saves = [n for n in wf.values() if n["class_type"] == "SaveImage"]
        assert len(saves) == 1
        assert saves[0]["inputs"]["filename_prefix"] == "myprefix"
        assert summary["save_action"] == {"type": "IMAGE", "prefix": "myprefix"}

    def test_missing_input_errors_with_step_alias(self, lib_dir: Path):
        recipe = {"pipeline": [{"fragment": "text_encode", "alias": "only", "params": {"text": "x"}}]}
        with pytest.raises(RecipeError) as exc:
            compose_recipe(recipe, lib_dir=lib_dir)
        assert exc.value.step_alias == "only"
        assert "missing required input" in str(exc.value)

    def test_unknown_input_key_errors(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "x", "inputs": {"clip": "a", "typo": "b"}}
            ]
        }
        with pytest.raises(RecipeError, match="unknown inputs"):
            compose_recipe(recipe, lib_dir=lib_dir)

    def test_unknown_param_key_errors(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "x", "inputs": {"clip": "a"}, "params": {"typo": 1}}
            ]
        }
        with pytest.raises(RecipeError, match="unknown params"):
            compose_recipe(recipe, lib_dir=lib_dir)

    def test_duplicate_alias_errors(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "dup", "inputs": {"clip": "a"}},
                {"fragment": "text_encode", "alias": "dup", "inputs": {"clip": "b"}},
            ]
        }
        with pytest.raises(RecipeError, match="dup"):
            compose_recipe(recipe, lib_dir=lib_dir)

    def test_unknown_alias_in_cross_ref(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$nope.conditioning"}}
            ]
        }
        with pytest.raises(RecipeError, match="unknown alias"):
            compose_recipe(recipe, lib_dir=lib_dir)

    def test_unknown_output_name_in_cross_ref(self, lib_dir: Path):
        recipe = {
            "pipeline": [
                {"fragment": "text_encode", "alias": "p1", "inputs": {"clip": "x"}},
                {"fragment": "text_encode", "alias": "p2", "inputs": {"clip": "$p1.no_such_output"}},
            ]
        }
        with pytest.raises(RecipeError, match="no output"):
            compose_recipe(recipe, lib_dir=lib_dir)

    def test_empty_pipeline_errors(self, lib_dir: Path):
        with pytest.raises(RecipeError, match="pipeline"):
            compose_recipe({}, lib_dir=lib_dir)
        with pytest.raises(RecipeError, match="pipeline"):
            compose_recipe({"pipeline": []}, lib_dir=lib_dir)


# ---------------------------------------------------------------------------
# CLI integration tests via Typer's CliRunner
# ---------------------------------------------------------------------------


class TestComposeCmd:
    def test_compose_writes_compiled_json(self, lib_dir: Path, tmp_path: Path, capsys):
        recipe = tmp_path / "demo.yaml"
        recipe.write_text(textwrap.dedent("""\
            pipeline:
              - fragment: text_encode
                alias: p
                inputs: {clip: clip_a}
                params: {text: hello}
        """))
        out = tmp_path / "built.json"
        envelope = _run(["compose", str(recipe), "-o", str(out), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        assert envelope["data"]["nodes"] >= 1
        assert out.exists()
        wf = json.loads(out.read_text())
        encodes = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
        assert encodes[0]["inputs"]["text"] == "hello"

    def test_compose_missing_recipe(self, tmp_path: Path, capsys):
        envelope = _run(["compose", str(tmp_path / "nope.yaml")], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "recipe_not_found"

    def test_compose_invalid_yaml(self, tmp_path: Path, capsys):
        recipe = tmp_path / "bad.yaml"
        recipe.write_text("pipeline: [ this is not balanced")
        envelope = _run(["compose", str(recipe)], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "recipe_invalid_yaml"

    def test_compose_missing_input_returns_recipe_invalid(self, lib_dir: Path, tmp_path: Path, capsys):
        recipe = tmp_path / "r.yaml"
        recipe.write_text("pipeline:\n  - fragment: text_encode\n    alias: x\n    params: {text: y}\n")
        envelope = _run(["compose", str(recipe), "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "recipe_invalid"
        assert "missing required input" in envelope["error"]["message"]


class TestFragmentCmds:
    def test_ls_lists_three(self, lib_dir: Path, capsys):
        envelope = _run(["fragment", "ls", "--lib", str(lib_dir)], capsys)
        assert envelope["ok"] is True
        names = {f["name"] for f in envelope["data"]["fragments"]}
        assert names == {"text_encode", "save_still", "image_blend"}

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

    def test_validate_legacy_subgraph_format_fails_cleanly(self, tmp_path: Path, capsys):
        """Files using the old `_subgraph` key (not `_fragment`) must fail cleanly."""
        legacy = {
            "_subgraph": {"name": "legacy", "inputs": {}, "outputs": {}, "params": {}},
            "10": {"class_type": "Whatever", "inputs": {}},
        }
        d = tmp_path / "fragments"
        d.mkdir()
        (d / "legacy.json").write_text(json.dumps(legacy))
        envelope = _run(["fragment", "validate", "legacy", "--lib", str(d)], capsys)
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "fragment_invalid"
        assert "_fragment" in envelope["error"]["message"]
