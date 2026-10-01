"""Bounds on the agent-facing payloads that carried whole option lists.

Production (Langfuse, comfy-cloud-prod): one `validate` of a graph with eleven
unknown model filenames over a 377-file folder returned ~340K tokens, because
every `unknown_enum_value` error carried the folder's full listing twice
(`suggestions` and `valid_options`); `nodes show LoraLoader` was ~31KB of LoRA
filenames on every call. These tests pin the bound and the way back to the
full list.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.cql.engine import ENUM_INLINE_MAX, ENUM_SUGGEST_MAX, Graph, full_enum_options
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

FILES = [f"model_{i:03d}_v1.safetensors" for i in range(377)]


def _object_info() -> dict[str, Any]:
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [FILES]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "category": "loaders",
            "display_name": "Load Checkpoint",
            "output_node": False,
            "python_module": "nodes",
        },
        "SaveLatent": {
            "input": {"required": {"model": ["MODEL"], "mode": [["a", "b", "c"]]}},
            "input_order": {"required": ["model", "mode"]},
            "output": [],
            "output_name": [],
            "category": "latent",
            "display_name": "Save",
            "output_node": True,
            "python_module": "nodes",
        },
    }


def _workflow(n_bad: int = 11) -> dict[str, Any]:
    wf: dict[str, Any] = {}
    for i in range(n_bad):
        wf[str(i * 2 + 1)] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": f"model_{i:03d}_v2.safetensors"},
        }
        wf[str(i * 2 + 2)] = {"class_type": "SaveLatent", "inputs": {"model": [str(i * 2 + 1), 0], "mode": "a"}}
    return wf


class TestValidateEnumErrors:
    def test_an_unknown_enum_error_names_the_closest_options_not_the_folder(self):
        result = Graph.from_object_info(_object_info()).validate_workflow(_workflow())
        errors = [e for e in result["errors"] if e["code"] == "unknown_enum_value"]
        assert len(errors) == 11
        err = errors[0]
        assert err["option_count"] == len(FILES)
        assert "valid_options" not in err
        assert len(err["suggestions"]) <= ENUM_SUGGEST_MAX
        assert err["suggestions"][0] == "model_000_v1.safetensors"
        assert err["options_omitted"] == len(FILES) - len(err["suggestions"])
        assert "--full-options" in err["hint"] and "choices.#(%" in err["hint"]
        # Eleven bad filenames: ~0.8KB each (was ~22KB each, the folder twice).
        assert len(json.dumps(result["errors"])) < 11 * 1_000

    def test_full_options_restores_the_whole_typed_list(self):
        with full_enum_options():
            result = Graph.from_object_info(_object_info()).validate_workflow(_workflow(1))
        err = next(e for e in result["errors"] if e["code"] == "unknown_enum_value")
        assert err["valid_options"] == FILES

    def test_a_short_option_list_is_still_carried_whole(self):
        wf = {"1": {"class_type": "SaveLatent", "inputs": {"model": None, "mode": "zz"}}}
        result = Graph.from_object_info(_object_info()).validate_workflow(wf)
        err = next(e for e in result["errors"] if e["code"] == "unknown_enum_value")
        assert err["valid_options"] == ["a", "b", "c"]
        assert len(["a", "b", "c"]) <= ENUM_INLINE_MAX

    def test_the_edit_finding_is_bounded_too(self):
        port = Graph.from_object_info(_object_info()).node("CheckpointLoaderSimple").inputs[0]
        (finding,) = port.validate_catalog("model_007_v2.safetensors")
        assert finding["code"] == "unknown_enum_value"
        assert "valid_options" not in finding
        assert finding["option_count"] == len(FILES)
        assert finding["did_you_mean"][0] == "model_007_v1.safetensors"


@pytest.fixture(autouse=True)
def _renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _run(args: list[str], capsys, monkeypatch) -> dict[str, Any]:
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: Graph.from_object_info(_object_info()))
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(nodes_cmd.app, args, standalone_mode=False)
    out = capsys.readouterr().out or result.stdout or ""
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no envelope: {out[:400]} {result.exception}")


class TestNodesShowChoices:
    def test_show_caps_a_long_choice_list_with_a_count(self, capsys, monkeypatch):
        data = _run(["show", "CheckpointLoaderSimple"], capsys, monkeypatch)["data"]
        ckpt = data["inputs"][0]
        assert len(ckpt["choices"]) == nodes_cmd.CHOICES_INLINE_MAX
        assert ckpt["choices_total"] == len(FILES)
        assert ckpt["choices_truncated"] is True
        assert "--all-choices" in data["choices_note"]
        assert len(json.dumps(data)) < 4_000

    def test_all_choices_lists_every_choice(self, capsys, monkeypatch):
        data = _run(["show", "CheckpointLoaderSimple", "--all-choices"], capsys, monkeypatch)["data"]
        assert data["inputs"][0]["choices"] == FILES
        assert "choices_note" not in data

    def test_a_select_projects_the_full_schema_and_can_filter_it(self, capsys, monkeypatch):
        env = _run(
            ["show", "CheckpointLoaderSimple", "--select", 'inputs.#(name=="ckpt_name").choices.#(%"model_37?_*")#'],
            capsys,
            monkeypatch,
        )
        assert env["data"] == [f"model_{i}_v1.safetensors" for i in range(370, 377)]

    def test_a_short_choice_list_is_untouched(self, capsys, monkeypatch):
        data = _run(["show", "SaveLatent"], capsys, monkeypatch)["data"]
        mode = next(i for i in data["inputs"] if i["name"] == "mode")
        assert mode["choices"] == ["a", "b", "c"]
        assert "choices_total" not in mode and "choices_note" not in data

    def test_search_expand_top_caps_choices(self, capsys, monkeypatch):
        data = _run(["search", "Checkpoint", "--expand-top", "1"], capsys, monkeypatch)["data"]
        entry = data["expanded"][0]
        assert entry["inputs"][0]["choices_total"] == len(FILES)
        assert len(entry["inputs"][0]["choices"]) == nodes_cmd.CHOICES_INLINE_MAX
