"""Offline rules behind `comfy workflow check --where cloud` (comfy_cli.workflow_coverage)."""

from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_coverage as cov

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
}


def _no_uploads(_name):
    return []


def _ui(*nodes, **extra):
    return {"nodes": list(nodes), "links": [], **extra}


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
    uses = cov.find_uses(wf)
    assert uses["format"] == "ui"
    assert uses["nodes"] == {"CheckpointLoaderSimple": ["1"], "CLIPTextEncode": ["5"]}
    assert [m["value"] for m in uses["models"]] == ["sd15.safetensors"]
    assert uses["skipped"] == [
        {"node_id": "2", "class_type": "LoraLoader", "why": "muted"},
        {"node_id": "3", "class_type": "LoraLoader", "why": "bypassed"},
    ]


def test_find_uses_ui_reads_lora_stack_rows():
    wf = _ui(_node(1, "Power Lora Loader", [{}, {"on": True, "lora": "x.safetensors", "strength": 1}]))
    assert [m["value"] for m in cov.find_uses(wf)["models"]] == ["x.safetensors"]


def test_find_uses_walks_subgraphs_with_scoped_ids_and_skips_muted_instances():
    sg_id = "aaaaaaaa-0000-0000-0000-000000000001"
    wf = _ui(
        _node(10, sg_id),
        _node(11, sg_id),
        _node(12, sg_id, mode=2),
        definitions={"subgraphs": [{"id": sg_id, "nodes": [_node(1, "LoraLoader", ["ink.safetensors", 1, 1])]}]},
    )
    uses = cov.find_uses(wf)
    assert uses["nodes"] == {"LoraLoader": ["10/1", "11/1"]}
    # Two instances of one definition share one place in the file: one model row.
    assert [(m["node_id"], m["value"]) for m in uses["models"]] == [("10/1", "ink.safetensors")]
    assert uses["skipped"] == [{"node_id": "12", "class_type": sg_id, "why": "muted"}]


def test_find_uses_api_reads_inputs_and_ignores_urls():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
        "2": {"class_type": "LoadFromUrl", "inputs": {"url": "https://example.com/m.safetensors"}},
    }
    uses = cov.find_uses(wf)
    assert uses["format"] == "api"
    assert uses["models"] == [
        {
            "node_id": "1",
            "class_type": "CheckpointLoaderSimple",
            "value": "sd15.safetensors",
            "widget": "ckpt_name",
            "loc": ("api", "1"),
        }
    ]


def test_find_uses_rejects_non_workflows():
    with pytest.raises(ValueError):
        cov.find_uses({"hello": "world"})


# --- placing models on Cloud -----------------------------------------------


def _resolve(wf, own=_no_uploads):
    return cov.resolve_models(cov.find_uses(wf), OBJECT_INFO, own)


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
    rows = cov.resolve_models(cov.find_uses(wf), OBJECT_INFO, own)
    assert asked == ["mine.safetensors"]
    assert rows[0]["status"] == "present" and rows[0]["source"] == "yours"
    assert rows[0]["cloud_hash"] == H1 and rows[0]["cloud_size"] == 5
    assert rows[1]["status"] == "remap" and rows[1]["cloud_value"] == "mine.safetensors"


def test_resolve_two_uploads_with_one_name_leave_the_hash_unknown():
    def own(_name):
        return [{"name": "m.safetensors", "hash": H1}, {"name": "m.safetensors", "hash": H2}]

    uses = cov.find_uses(_ui(_node(1, "CheckpointLoaderSimple", ["m.safetensors"])))
    rows = cov.resolve_models(uses, OBJECT_INFO, own)
    assert rows[0]["status"] == "present" and rows[0]["cloud_hash"] is None


def test_content_hash_normalizes_both_spellings():
    assert cov.content_hash("1" * 64 + ".safetensors") == H1
    assert cov.content_hash(H1) == H1
    assert cov.content_hash("not-a-hash") is None
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
    rows = cov.resolve_models(cov.find_uses(wf), OBJECT_INFO, _no_uploads)
    cov.add_sources(rows, wf, {"ink.safetensors": [("/m/loras/ink.safetensors", 3)]})
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["1"]["folder"] == "unet" and by_id["1"]["url"] == "https://x/flux"
    assert by_id["2"]["filename"] == "ink.safetensors" and by_id["2"]["folder"] == "loras"
    assert by_id["2"]["local_paths"] == ["/m/loras/ink.safetensors"]
    assert by_id["3"]["folder"] is None and by_id["3"]["local_paths"] == []


def test_add_sources_uses_the_api_input_name():
    wf = {"1": {"class_type": "Anything", "inputs": {"vae_name": "v.safetensors"}}}
    rows = cov.resolve_models(cov.find_uses(wf), {}, _no_uploads)
    cov.add_sources(rows, wf, None)
    assert rows[0]["folder"] == "vae" and rows[0]["local_paths"] == []


# --- the answer -------------------------------------------------------------


def test_blocking_lists_missing_nodes_then_models_with_where_to_find_them():
    wf = _ui(
        _node(1, "CustomLoader", ["custom.safetensors"]),
        _node(2, "VAELoader", ["vae.safetensors"]),
        _node(3, "CheckpointLoaderSimple", ["gone.safetensors"]),
    )
    uses = cov.find_uses(wf)
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
    uses = cov.find_uses(_ui(_node(1, "CheckpointLoaderSimple", ["sd15.safetensors"])))
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
        definitions={"subgraphs": [{"id": sg_id, "nodes": [_node(7, "LoraLoader", ["ink.safetensors", 1, 1])]}]},
    )
    before = copy.deepcopy(wf)
    rows = _resolve(wf)
    out, changed = cov.apply_remaps(wf, rows)
    assert wf == before
    assert changed == 2
    assert out["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"
    assert out["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"


def test_apply_remaps_api_sets_the_input():
    wf = {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "ink.safetensors", "strength_model": 1}}}
    out, changed = cov.apply_remaps(wf, _resolve(wf))
    assert changed == 1
    assert out["1"]["inputs"] == {"lora_name": "styles/ink.safetensors", "strength_model": 1}
    assert wf["1"]["inputs"]["lora_name"] == "ink.safetensors"
