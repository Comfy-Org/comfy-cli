"""Offline rules behind `comfy workflow check --where cloud` (comfy_cli.workflow_coverage)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from comfy_cli import workflow_coverage as cov
from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

H1 = "blake3:" + "1" * 64
H2 = "blake3:" + "2" * 64

OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["sd15.safetensors", "sdxl/base.safetensors"], {}]}}
    },
    # The newer COMBO form.
    "LoraLoader": {"input": {"required": {"lora_name": ["COMBO", {"options": ["styles/ink.safetensors"]}]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["a/vae.safetensors", "b/vae.safetensors"], {}]}}},
    "KSampler": {"input": {"required": {"seed": ["INT", {}]}}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}},
    # Not stock nodes: what makes an input a file is its spec, not the node type.
    "CaptionWriter": {"input": {"required": {"caption": ["STRING", {"multiline": True}]}}},
    "OddLoader": {"input": {"required": {"weights": [["odd/w.pkl"], {}]}}},
}


def _no_uploads(_name):
    return []


def _ui(*nodes, **extra):
    return {"nodes": list(nodes), "links": [], **extra}


def _subgraph(sg_id, *nodes):
    return {"subgraphs": [{"id": sg_id, "nodes": list(nodes), "links": [], "inputs": [], "outputs": []}]}


def _uses(wf, object_info=None):
    return cov.find_uses(wf, OBJECT_INFO if object_info is None else object_info)


def _node(node_id, type_, values=None, mode=0, **extra):
    return {"id": node_id, "type": type_, "mode": mode, "widgets_values": values or [], **extra}


# --- reading the workflow --------------------------------------------------


def test_format_detects_ui_api_and_neither():
    assert cov.workflow_format({"nodes": []}) == "ui"
    assert cov.workflow_format({"1": {"class_type": "KSampler", "inputs": {}}}) == "api"
    assert cov.workflow_format({}) is None
    assert cov.workflow_format([1, 2]) is None
    assert cov.workflow_format({"1": {"inputs": {}}}) is None


def test_find_uses_ui_skips_muted_bypassed_notes_and_prompts():
    wf = _ui(
        _node(1, "CheckpointLoaderSimple", ["sd15.safetensors"]),
        _node(2, "LoraLoader", ["muted.safetensors", 1.0, 1.0], mode=2),
        _node(3, "LoraLoader", ["bypassed.safetensors", 1.0, 1.0], mode=4),
        _node(4, "Note", ["read me.safetensors"]),
        _node(5, "CLIPTextEncode", ["a photo.safetensors"]),
    )
    uses = _uses(wf)
    assert uses["format"] == "ui"
    assert uses["nodes"] == {"CheckpointLoaderSimple": ["1"], "CLIPTextEncode": ["5"]}
    assert [m["value"] for m in uses["models"]] == ["sd15.safetensors"]
    assert uses["skipped"] == [
        {"node_id": "2", "class_type": "LoraLoader", "why": "muted"},
        {"node_id": "3", "class_type": "LoraLoader", "why": "bypassed"},
    ]


def test_find_uses_ui_reads_lora_stack_rows():
    wf = _ui(_node(1, "Power Lora Loader", [{}, {"on": True, "lora": "x.safetensors", "strength": 1}]))
    assert [m["value"] for m in _uses(wf)["models"]] == ["x.safetensors"]


def test_find_uses_walks_subgraphs_with_scoped_ids_and_skips_muted_instances():
    sg_id = "aaaaaaaa-0000-0000-0000-000000000001"
    wf = _ui(
        _node(10, sg_id),
        _node(11, sg_id),
        _node(12, sg_id, mode=2),
        definitions=_subgraph(sg_id, _node(1, "LoraLoader", ["ink.safetensors", 1, 1])),
    )
    uses = _uses(wf)
    # The converter's and the frontend's ids, so a row names the node that runs.
    assert uses["nodes"] == {"LoraLoader": ["10:1", "11:1"]}
    # Each instance can hold its own promoted value, so each gets its own row.
    assert [(m["node_id"], m["value"], m["widget"]) for m in uses["models"]] == [
        ("10:1", "ink.safetensors", "lora_name"),
        ("11:1", "ink.safetensors", "lora_name"),
    ]
    assert uses["skipped"] == [{"node_id": "12", "class_type": sg_id, "why": "muted"}]


def _promoted_music_workflow():
    """A shipped template whose subgraph instance (node 37) promotes the interior
    UNETLoader's ``unet_name``; the instance's own value is ``widgets_values[4]``."""
    wf = json.loads((_FIXTURES / "gallery" / "audio_minimax_music_3.json").read_text(encoding="utf-8"))
    object_info = json.loads((_FIXTURES / "object_info_subgraph_promoted.json").read_text(encoding="utf-8"))
    host = next(n for n in wf["nodes"] if n["id"] == 37)
    host["widgets_values"][4] = "mine.safetensors"
    return wf, host, object_info


def _unet_rows(wf, object_info):
    return [m for m in cov.find_uses(wf, object_info)["models"] if m["class_type"] == "UNETLoader"]


def test_find_uses_checks_the_instances_promoted_value_not_the_interior_default():
    wf, _host, object_info = _promoted_music_workflow()
    assert [(m["node_id"], m["value"], m["widget"]) for m in _unet_rows(wf, object_info)] == [
        ("37:6", "mine.safetensors", "unet_name")
    ]


def test_apply_remaps_rewrites_the_promoted_value_on_the_instance():
    wf, host, object_info = _promoted_music_workflow()
    [row] = _unet_rows(wf, object_info)
    row.update(status="remap", cloud_value="music/mine.safetensors")
    out, changed = cov.apply_remaps(wf, [row])
    assert changed == 1
    assert next(n for n in out["nodes"] if n["id"] == 37)["widgets_values"][4] == "music/mine.safetensors"
    assert host["widgets_values"][4] == "mine.safetensors"  # the original is untouched
    # The interior default is not what runs, and it is left as saved.
    interior = [n for sg in out["definitions"]["subgraphs"] for n in sg["nodes"] if n["type"] == "UNETLoader"]
    assert interior[0]["widgets_values"][0] == "minimax_music3_dit_fp16.safetensors"
    # The copy now submits the remapped value.
    assert [m["value"] for m in _unet_rows(out, object_info)] == ["music/mine.safetensors"]


def test_apply_remaps_rewrites_a_quarantined_host_value():
    wf, host, object_info = _promoted_music_workflow()
    # The frontend reads a quarantined legacy value before the positional slot.
    host["widgets_values"][4] = "slot.safetensors"
    quarantine = [{"originalEntry": ["-1", "unet_name"], "hostValue": "mine.safetensors"}]
    host["properties"]["proxyWidgetErrorQuarantine"] = quarantine
    [row] = _unet_rows(wf, object_info)
    assert row["value"] == "mine.safetensors"
    row.update(status="remap", cloud_value="music/mine.safetensors")
    out, changed = cov.apply_remaps(wf, [row])
    assert changed == 1
    out_host = next(n for n in out["nodes"] if n["id"] == 37)
    assert out_host["properties"]["proxyWidgetErrorQuarantine"][0]["hostValue"] == "music/mine.safetensors"
    assert out_host["widgets_values"][4] == "slot.safetensors"
    interior = [n for sg in out["definitions"]["subgraphs"] for n in sg["nodes"] if n["type"] == "UNETLoader"]
    assert interior[0]["widgets_values"][0] == "minimax_music3_dit_fp16.safetensors"
    assert quarantine[0]["hostValue"] == "mine.safetensors"
    assert [m["value"] for m in _unet_rows(out, object_info)] == ["music/mine.safetensors"]


def test_remapping_a_top_level_loader_leaves_an_equal_promoted_caption_alone():
    wf, host, object_info = _promoted_music_workflow()
    host["widgets_values"][0] = "ink.safetensors"  # the promoted caption: text that runs, not a model
    wf["nodes"].append(_node(999, "LoraLoader", ["ink.safetensors", 1, 1]))
    object_info = {**object_info, "LoraLoader": OBJECT_INFO["LoraLoader"]}
    assert convert_ui_to_api(wf, object_info)["37:13"]["inputs"]["caption"] == "ink.safetensors"
    rows = cov.resolve_models(cov.find_uses(wf, object_info), object_info, _no_uploads)
    rows = [r for r in rows if r["node_id"] == "999"]
    assert [(r["status"], r["cloud_value"]) for r in rows] == [("remap", "styles/ink.safetensors")]
    out, changed = cov.apply_remaps(wf, rows)
    assert changed == 1
    assert next(n for n in out["nodes"] if n["id"] == 999)["widgets_values"][0] == "styles/ink.safetensors"
    assert next(n for n in out["nodes"] if n["id"] == 37)["widgets_values"][0] == "ink.safetensors"
    assert convert_ui_to_api(out, object_info)["37:13"]["inputs"]["caption"] == "ink.safetensors"


def test_find_uses_api_reads_inputs_and_ignores_urls():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
        "2": {"class_type": "LoadFromUrl", "inputs": {"url": "https://example.com/m.safetensors"}},
    }
    uses = _uses(wf)
    assert uses["format"] == "api"
    assert uses["models"] == [
        {
            "node_id": "1",
            "class_type": "CheckpointLoaderSimple",
            "value": "sd15.safetensors",
            "widget": "ckpt_name",
        }
    ]


def test_find_uses_api_does_not_read_prompt_text_as_a_model():
    wf = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo.safetensors", "clip": ["2", 1]}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
    }
    assert [m["value"] for m in _uses(wf)["models"]] == ["sd15.safetensors"]


@pytest.mark.parametrize("fmt", ["ui", "api"])
def test_whether_an_input_names_a_file_comes_from_its_spec_not_the_node_type(fmt):
    """No list of prompt node types: a STRING input is text, a dropdown names a file."""
    if fmt == "ui":
        wf = _ui(_node(1, "CaptionWriter", ["a cat.ckpt"]), _node(2, "OddLoader", ["w.pkl"]))
    else:
        wf = {
            "1": {"class_type": "CaptionWriter", "inputs": {"caption": "a cat.ckpt"}},
            "2": {"class_type": "OddLoader", "inputs": {"weights": "w.pkl"}},
        }
    assert [(m["node_id"], m["value"], m["widget"]) for m in _uses(wf)["models"]] == [("2", "w.pkl", "weights")]


def test_names_a_file_reads_the_input_spec():
    assert cov.names_a_file(OBJECT_INFO["CheckpointLoaderSimple"], "ckpt_name")
    assert cov.names_a_file(OBJECT_INFO["LoraLoader"], "lora_name")  # the COMBO form
    assert not cov.names_a_file(OBJECT_INFO["CLIPTextEncode"], "text")
    assert not cov.names_a_file(OBJECT_INFO["KSampler"], "seed")
    assert cov.names_a_file(OBJECT_INFO["CLIPTextEncode"], "lora_1")  # undeclared: judged by its value
    assert cov.names_a_file(None, "anything")  # a node type not on Cloud


@pytest.mark.parametrize("name", ["m.pkl", "m.pt2", "m.torchscript", "M.SAFETENSORS"])
def test_every_model_format_comfyui_loads_is_found(name, tmp_path):
    uses = _uses(_ui(_node(1, "SomeCustomLoader", [name])))
    assert [m["value"] for m in uses["models"]] == [name]
    (tmp_path / name).write_bytes(b"x")
    assert list(cov.local_index([str(tmp_path)])) == [name]


def test_find_uses_passes_on_a_converter_refusal(monkeypatch):
    def refuse(_workflow, _object_info):
        raise WorkflowConversionError("subgraph expansion exceeded its safe limit")

    monkeypatch.setattr("comfy_cli.workflow_to_api.convert_ui_to_api", refuse)
    with pytest.raises(WorkflowConversionError):
        _uses(_ui(_node(1, "CheckpointLoaderSimple", ["sd15.safetensors"])))


def test_find_uses_rejects_non_workflows():
    with pytest.raises(ValueError):
        _uses({"hello": "world"})


# --- placing models on Cloud -----------------------------------------------


def _resolve(wf, own=_no_uploads):
    return cov.resolve_models(_uses(wf), OBJECT_INFO, own)


def test_combo_choices_reads_both_forms():
    assert cov.combo_choices(OBJECT_INFO["LoraLoader"]) == {"styles/ink.safetensors": "lora_name"}
    assert cov.combo_choices(OBJECT_INFO["CheckpointLoaderSimple"])["sd15.safetensors"] == "ckpt_name"
    assert cov.combo_choices(None) == {}


def test_resolve_present_remap_ambiguous_missing_and_not_checked():
    wf = _ui(
        _node(1, "CheckpointLoaderSimple", ["sd15.safetensors"]),
        _node(2, "LoraLoader", ["ink.safetensors", 1, 1]),
        _node(3, "VAELoader", ["vae.safetensors"]),
        _node(4, "CheckpointLoaderSimple", ["nowhere.safetensors"]),
        _node(5, "CustomLoader", ["custom.safetensors"]),
    )
    rows = {r["node_id"]: r for r in _resolve(wf)}
    assert rows["1"]["status"] == "present" and rows["1"]["source"] == "cloud"
    assert rows["1"]["widget"] == "ckpt_name"
    assert rows["2"]["status"] == "remap" and rows["2"]["cloud_value"] == "styles/ink.safetensors"
    assert rows["3"]["status"] == "missing"
    assert rows["3"]["candidates"] == ["a/vae.safetensors", "b/vae.safetensors"]
    assert rows["4"]["status"] == "missing" and "candidates" not in rows["4"]
    assert rows["5"]["status"] == "not_checked"


def test_resolve_finds_the_persons_uploads_and_asks_once_per_name():
    asked = []

    def own(name):
        asked.append(name)
        return [{"name": "mine.safetensors", "hash": "1" * 64 + ".safetensors", "size": 5, "tags": ["models"]}]

    wf = _ui(
        _node(1, "CheckpointLoaderSimple", ["mine.safetensors"]),
        _node(2, "CheckpointLoaderSimple", ["sub/mine.safetensors"]),
    )
    rows = cov.resolve_models(_uses(wf), OBJECT_INFO, own)
    assert asked == ["mine.safetensors"]
    assert rows[0]["status"] == "present" and rows[0]["source"] == "yours"
    assert rows[0]["cloud_hash"] == H1 and rows[0]["cloud_size"] == 5
    assert rows[1]["status"] == "remap" and rows[1]["cloud_value"] == "mine.safetensors"


def test_resolve_two_uploads_with_one_name_leave_the_hash_unknown():
    def own(_name):
        return [{"name": "m.safetensors", "hash": H1}, {"name": "m.safetensors", "hash": H2}]

    uses = _uses(_ui(_node(1, "CheckpointLoaderSimple", ["m.safetensors"])))
    rows = cov.resolve_models(uses, OBJECT_INFO, own)
    assert rows[0]["status"] == "present" and rows[0]["cloud_hash"] is None


def test_resolve_finds_an_upload_kept_under_a_folder_as_cloud_does():
    """Cloud runs a saved bare name against the one upload whose name ends in ``/<name>``."""

    def own(_name):
        return [{"name": "Wan22_FunReward/model.safetensors", "hash": H1, "size": 5}]

    rows = _resolve(_ui(_node(1, "CheckpointLoaderSimple", ["model.safetensors"])), own)
    assert rows[0]["status"] == "present" and rows[0]["source"] == "yours"
    assert rows[0]["cloud_value"] == "Wan22_FunReward/model.safetensors"
    assert rows[0]["cloud_hash"] == H1


def test_resolve_two_uploads_under_folders_are_ambiguous():
    def own(_name):
        return [{"name": "a/model.safetensors"}, {"name": "b/model.safetensors"}]

    uses = _uses(_ui(_node(1, "CheckpointLoaderSimple", ["model.safetensors"])))
    rows = cov.resolve_models(uses, OBJECT_INFO, own)
    assert rows[0]["status"] == "missing"
    assert rows[0]["candidates"] == ["a/model.safetensors", "b/model.safetensors"]
    assert [b["reason"] for b in cov.blocking(uses, OBJECT_INFO, rows)] == ["ambiguous"]


def test_pick_upload_prefers_the_exact_name_then_the_folder_suffix_then_a_remap():
    exact = [{"name": "model.safetensors"}, {"name": "a/model.safetensors"}]
    assert cov._pick_upload("model.safetensors", exact) == ("present", "model.safetensors", [])
    other_path = [{"name": "x/model.safetensors"}]
    assert cov._pick_upload("y/model.safetensors", other_path) == ("remap", "x/model.safetensors", [])
    assert cov._pick_upload("model.safetensors", []) == ("missing", None, [])


def test_content_hash_normalizes_both_spellings():
    assert cov.content_hash("1" * 64 + ".safetensors") == H1
    assert cov.content_hash(H1) == H1
    assert cov.content_hash("not-a-hash") is None
    # Cloud's library models carry their folder and file name in `hash`.
    assert cov.content_hash("checkpoints/x.safetensors") is None
    assert cov.content_hash("checkpoints/" + "1" * 64 + ".safetensors") is None
    assert cov.content_hash(None) is None


# --- same file or not ------------------------------------------------------


def test_identity_needs_a_hash_match_to_say_same():
    def hashes(path):
        return {"/a": H1, "/b": H2}[path]

    assert cov.identity(None, 5, H1, hashes) == ("unverified", None)
    assert cov.identity([("/a", 5)], 5, H1, hashes) == ("same", "/a")
    assert cov.identity([("/a", 5)], 5, H2, hashes) == ("different", "/a")
    assert cov.identity([("/a", 5)], 6, H1, hashes) == ("different", "/a")  # size alone proves different
    assert cov.identity([("/a", 5)], 5, None, hashes) == ("unverified", "/a")
    assert cov.identity([("/a", 5), ("/b", 6)], 5, H1, hashes) == ("unverified", None)
    assert cov.identity([("/a", 5), ("/b", 5)], 5, H1, hashes) == ("unverified", None)


def test_identity_is_unverified_when_the_local_file_cannot_be_read():
    assert cov.identity([("/a", 5)], 5, H1, lambda _path: None) == ("unverified", None)


def test_identity_without_a_cloud_hash_uses_size_and_never_hashes():
    def never(path):
        raise AssertionError(f"hashed {path} with nothing to compare it to")

    assert cov.identity([("/a", 5)], 5, None, never) == ("unverified", "/a")
    assert cov.identity([("/a", 5)], 6, None, never) == ("different", "/a")


def test_local_index_finds_model_files_by_name(tmp_path):
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "ink.safetensors").write_bytes(b"abc")
    (tmp_path / "notes.txt").write_text("x")
    index = cov.local_index([str(tmp_path)])
    assert index == {"ink.safetensors": [(str(tmp_path / "loras" / "ink.safetensors"), 3)]}


# --- where a missing file goes ---------------------------------------------


def test_add_sources_prefers_the_workflows_download_notes(tmp_path):
    wf = _ui(
        _node(
            1,
            "UNETLoader",
            ["flux.safetensors", "default"],
            properties={"models": [{"name": "flux.safetensors", "url": "https://x/flux", "directory": "unet"}]},
        ),
        _node(2, "LoraLoader", ["sub/ink.safetensors", 1, 1]),
        _node(3, "SomeCustomLoader", ["odd.safetensors"]),
    )
    rows = cov.resolve_models(_uses(wf), OBJECT_INFO, _no_uploads)
    cov.add_sources(rows, wf, {"ink.safetensors": [("/m/loras/ink.safetensors", 3)]})
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["1"]["folder"] == "unet" and by_id["1"]["url"] == "https://x/flux"
    assert by_id["2"]["filename"] == "ink.safetensors" and by_id["2"]["folder"] == "loras"
    assert by_id["2"]["local_paths"] == ["/m/loras/ink.safetensors"]
    assert by_id["3"]["folder"] is None and by_id["3"]["local_paths"] == []


def test_add_sources_uses_the_api_input_name():
    wf = {"1": {"class_type": "Anything", "inputs": {"vae_name": "v.safetensors"}}}
    rows = cov.resolve_models(_uses(wf, {}), {}, _no_uploads)
    cov.add_sources(rows, wf, None)
    assert rows[0]["folder"] == "vae" and rows[0]["local_paths"] == []


# --- the answer -------------------------------------------------------------


def test_blocking_lists_missing_nodes_then_models_with_where_to_find_them():
    wf = _ui(
        _node(1, "CustomLoader", ["custom.safetensors"]),
        _node(2, "VAELoader", ["vae.safetensors"]),
        _node(3, "CheckpointLoaderSimple", ["gone.safetensors"]),
    )
    uses = _uses(wf)
    rows = cov.resolve_models(uses, OBJECT_INFO, _no_uploads)
    cov.add_sources(rows, wf, None)
    blocked = cov.blocking(uses, OBJECT_INFO, rows)
    assert blocked[0] == {"kind": "node", "class_type": "CustomLoader", "node_ids": ["1"], "reason": "not_on_cloud"}
    assert [(b["value"], b["reason"]) for b in blocked[1:]] == [
        ("vae.safetensors", "ambiguous"),
        ("gone.safetensors", "not_on_cloud"),
    ]
    assert blocked[2]["folder"] == "checkpoints" and blocked[2]["filename"] == "gone.safetensors"


def test_blocking_flags_a_proven_different_file():
    uses = _uses(_ui(_node(1, "CheckpointLoaderSimple", ["sd15.safetensors"])))
    rows = cov.resolve_models(uses, OBJECT_INFO, _no_uploads)
    rows[0]["identity"] = "different"
    assert [b["reason"] for b in cov.blocking(uses, OBJECT_INFO, rows)] == ["different_file"]


SAME = {"status": "present", "identity": "same"}
UNPROVEN = {"status": "present", "identity": "unverified"}
REMAP = {"status": "remap", "identity": "same"}


@pytest.mark.parametrize(
    ("blocked", "models", "expected"),
    [
        ([], [], "covered"),
        ([], [SAME], "covered"),
        ([], [SAME, UNPROVEN], "covered_needs_check"),
        ([], [REMAP, UNPROVEN], "covered_with_remaps"),
        ([{"kind": "node"}], [REMAP], "needs_build"),
    ],
)
def test_verdict_precedence(blocked, models, expected):
    assert cov.verdict(blocked, models) == expected


def test_apply_remaps_ui_changes_only_the_copy():
    sg_id = "aaaaaaaa-0000-0000-0000-000000000002"
    wf = _ui(
        _node(1, "LoraLoader", ["ink.safetensors", 1, 1]),
        _node(2, sg_id),
        definitions=_subgraph(sg_id, _node(7, "LoraLoader", ["ink.safetensors", 1, 1])),
    )
    before = copy.deepcopy(wf)
    rows = _resolve(wf)
    out, changed = cov.apply_remaps(wf, rows)
    assert wf == before
    assert changed == 2
    assert out["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"
    assert out["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"


def test_apply_remaps_writes_each_loader_its_own_destination_for_one_file_name():
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["checkpoints/shared.safetensors"], {}]}}},
        "LoraLoader": {"input": {"required": {"lora_name": [["loras/shared.safetensors"], {}]}}},
    }
    wf = _ui(
        _node(1, "CheckpointLoaderSimple", ["shared.safetensors"]),
        _node(2, "LoraLoader", ["shared.safetensors", 1, 1]),
    )
    rows = cov.resolve_models(_uses(wf, object_info), object_info, _no_uploads)
    assert [(r["status"], r["cloud_value"]) for r in rows] == [
        ("remap", "checkpoints/shared.safetensors"),
        ("remap", "loras/shared.safetensors"),
    ]
    not_applied: list = []
    out, changed = cov.apply_remaps(wf, rows, not_applied)
    assert changed == 2 and not_applied == []
    assert out["nodes"][0]["widgets_values"] == ["checkpoints/shared.safetensors"]
    assert out["nodes"][1]["widgets_values"] == ["loras/shared.safetensors", 1, 1]


def test_apply_remaps_writes_a_default_two_instances_run_once():
    sg_id = "aaaaaaaa-0000-0000-0000-000000000004"
    wf = _ui(
        _node(10, sg_id),
        _node(11, sg_id),
        definitions=_subgraph(sg_id, _node(1, "LoraLoader", ["ink.safetensors", 1, 1])),
    )
    rows = _resolve(wf)
    assert [r["node_id"] for r in rows if r["status"] == "remap"] == ["10:1", "11:1"]
    not_applied: list = []
    out, changed = cov.apply_remaps(wf, rows, not_applied)
    assert changed == 1 and not_applied == []
    assert out["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"


def test_apply_remaps_leaves_a_place_it_cannot_settle_and_says_so():
    wf = _ui(_node(1, "LoraLoader", ["shared.safetensors", 1, 1]))
    at = ("nodes", 0, "widgets_values", 0)
    row = {"node_id": "1", "class_type": "LoraLoader", "status": "remap", "value": "shared.safetensors"}
    rows = [
        # One saved value two rows need remapped two ways.
        {**row, "cloud_value": "a/shared.safetensors", "loc": at},
        {**row, "cloud_value": "b/shared.safetensors", "loc": at},
        # A value whose saved place could not be told.
        {**row, "node_id": "2", "value": "gone.safetensors", "cloud_value": "c/gone.safetensors", "loc": None},
    ]
    not_applied: list = []
    out, changed = cov.apply_remaps(wf, rows, not_applied)
    assert (out, changed) == (wf, 0)
    assert not_applied == rows


def test_apply_remaps_api_sets_the_input():
    wf = {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "ink.safetensors", "strength_model": 1}}}
    out, changed = cov.apply_remaps(wf, _resolve(wf))
    assert changed == 1
    assert out["1"]["inputs"] == {"lora_name": "styles/ink.safetensors", "strength_model": 1}
    assert wf["1"]["inputs"]["lora_name"] == "ink.safetensors"
