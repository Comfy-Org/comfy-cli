from __future__ import annotations

import hashlib
from pathlib import Path

from comfy_cli.workflow_inventory import inventory, join_cloud, join_inputs, join_local, join_server


def _node(node_id, class_type, widgets=(), *, mode=0, **properties):
    return {"id": node_id, "type": class_type, "mode": mode, "widgets_values": list(widgets), "properties": properties}


def _workflow():
    return {
        "nodes": [
            _node(1, "CheckpointLoaderSimple", ["sdxl\\base_v1.safetensors"], cnr_id="comfy-core", ver="0.3.60"),
            _node(2, "LoraLoaderModelOnly", ["studio/hero_v5.safetensors", 0.8], cnr_id="comfy-core"),
            _node(3, "CLIPTextEncode", ["a portrait, embedding:EasyNegative, sharp"], cnr_id="comfy-core"),
            _node(4, "LoadImage", ["face.png", "image"], cnr_id="comfy-core"),
            _node(5, "UpscaleModelLoader", ["4x_unused.pth"], mode=4, cnr_id="comfy-core"),
            _node(6, "IPAdapterUnifiedLoader", ["PLUS (high strength)"], cnr_id="comfyui_ipadapter_plus", ver="2.0.0"),
            _node(7, "Note", ["remember to swap the LoRA"]),
            _node(8, "aaaaaaaa-0000-0000-0000-000000000001", []),
            _node(
                9,
                "VAELoader",
                ["sdxl_vae.safetensors"],
                cnr_id="comfy-core",
                models=[{"name": "sdxl_vae.safetensors", "directory": "vae", "url": "https://example.test/vae"}],
            ),
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "aaaaaaaa-0000-0000-0000-000000000001",
                    "nodes": [_node(20, "UltimateSDUpscale", [2, 42], cnr_id="comfyui_ultimatesdupscale", ver="1.6.0")],
                }
            ]
        },
        "extra": {"node_versions": {"comfy-core": "0.3.60"}},
    }


def test_inventory_scopes_to_active_nodes_and_keeps_subfolders():
    report = inventory(_workflow())

    assert report["format"] == "ui"
    assert set(report["classes"]) == {
        "CheckpointLoaderSimple",
        "LoraLoaderModelOnly",
        "CLIPTextEncode",
        "LoadImage",
        "IPAdapterUnifiedLoader",
        "VAELoader",
        "UltimateSDUpscale",
    }
    assert report["classes"]["IPAdapterUnifiedLoader"] == {
        "count": 1,
        "pack": "comfyui_ipadapter_plus",
        "ver": ["2.0.0"],
    }
    paths = [m["path"] for m in report["models"]]
    assert paths == ["sdxl/base_v1.safetensors", "studio/hero_v5.safetensors", "sdxl_vae.safetensors"]
    assert report["models"][0]["subfolder"] == "sdxl"
    assert any("Windows path" in w for w in report["warnings"])
    assert report["embeddings"] == [{"name": "EasyNegative", "node": "CLIPTextEncode", "node_id": 3}]
    assert report["inputs"] == [{"file": "face.png", "node": "LoadImage", "node_id": 4}]
    assert report["skipped"] == [
        {"id": 5, "class": "UpscaleModelLoader", "why": "bypassed", "models": ["4x_unused.pth"]}
    ]
    assert [i["class"] for i in report["implicit"]] == ["IPAdapterUnifiedLoader"]
    assert report["node_versions"] == {"comfy-core": "0.3.60"}
    assert report["packs"]["comfyui_ultimatesdupscale"] == ["UltimateSDUpscale"]


def test_inventory_reads_api_exports():
    api = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1}},
    }
    report = inventory(api)

    assert report["format"] == "api"
    assert set(report["classes"]) == {"CheckpointLoaderSimple", "KSampler"}
    assert [m["path"] for m in report["models"]] == ["base.safetensors"]


def test_join_local_matches_by_basename_across_roots(tmp_path: Path):
    first = tmp_path / "models"
    shared = tmp_path / "shared"
    (first / "checkpoints" / "sdxl").mkdir(parents=True)
    (shared / "loras").mkdir(parents=True)
    (first / "checkpoints" / "sdxl" / "base_v1.safetensors").write_bytes(b"BASE")
    (shared / "loras" / "hero_v5.safetensors").write_bytes(b"HERO")
    report = inventory(_workflow())

    join_local(report, [str(first), str(shared)], hash_files=True)

    base, lora, vae = report["models"]
    assert base["local"][0]["rel"] == "checkpoints/sdxl/base_v1.safetensors"
    assert base["local"][0]["type_dir"] == "checkpoints"
    assert base["local"][0]["sha256"] == hashlib.sha256(b"BASE").hexdigest()
    assert lora["local"][0]["root"] == str(shared)
    assert vae["local"] == []
    assert report["models_not_on_disk"] == ["sdxl_vae.safetensors"]


def test_join_server_maps_classes_to_local_pack_dirs():
    report = inventory(_workflow())
    object_info = {
        "CheckpointLoaderSimple": {"python_module": "nodes"},
        "UltimateSDUpscale": {"python_module": "custom_nodes.ComfyUI_UltimateSDUpscale"},
    }

    section = join_server(report, object_info, {"system": {"comfyui_version": "0.3.60"}})

    assert section["comfyui_version"] == "0.3.60"
    assert section["classes_by_local_pack_dir"] == {
        "core": ["CheckpointLoaderSimple"],
        "ComfyUI_UltimateSDUpscale": ["UltimateSDUpscale"],
    }
    assert "IPAdapterUnifiedLoader" in section["classes_missing_locally"]


def _catalog(**nodes):
    return lambda class_type: nodes.get(class_type)


def _combo(name, *choices, cloud_disabled=False):
    return {"inputs": [{"name": name, "choices": list(choices)}], "cloud_disabled": cloud_disabled}


def test_join_cloud_finds_remaps_and_blockers():
    report = inventory(_workflow())
    lookup = _catalog(
        CheckpointLoaderSimple=_combo("ckpt_name", "base_v1.safetensors", "other.safetensors"),
        LoraLoaderModelOnly=_combo("lora_name", "public.safetensors"),
        CLIPTextEncode={"inputs": []},
        LoadImage=_combo("image", "example.png"),
        VAELoader=_combo("vae_name", "sdxl_vae.safetensors"),
        UltimateSDUpscale={"inputs": []},
        IPAdapterUnifiedLoader=_combo("preset", "PLUS (high strength)", cloud_disabled=True),
    )

    cloud = join_cloud(report, lookup)

    statuses = {m["path"]: m for m in cloud["models"]}
    remap = statuses["sdxl/base_v1.safetensors"]
    assert remap["status"] == "remap"
    assert remap["set_widget"] == "1.ckpt_name"
    assert remap["to"] == "base_v1.safetensors"
    assert statuses["studio/hero_v5.safetensors"]["status"] == "absent"
    assert statuses["sdxl_vae.safetensors"]["status"] == "present"
    assert cloud["cloud_disabled"] == ["IPAdapterUnifiedLoader"]
    assert cloud["verdict"] == "needs_build_or_lossy_conversion"
    assert cloud["blocking"] == ["IPAdapterUnifiedLoader", "studio/hero_v5.safetensors"]
    assert any(u.startswith("embedding:EasyNegative") for u in cloud["unverified"])


def test_join_cloud_verdicts_without_blockers():
    plain = inventory({"nodes": [_node(1, "CheckpointLoaderSimple", ["base.safetensors"])]})
    nested = inventory({"nodes": [_node(1, "CheckpointLoaderSimple", ["sdxl/base.safetensors"])]})
    lookup = _catalog(CheckpointLoaderSimple=_combo("ckpt_name", "base.safetensors"))

    assert join_cloud(plain, lookup)["verdict"] == "covered"
    assert join_cloud(nested, lookup)["verdict"] == "covered_after_lossless_fixes"
    assert join_cloud(plain, _catalog())["missing_classes"] == ["CheckpointLoaderSimple"]


def test_join_cloud_treats_missing_perf_only_nodes_as_lossless():
    workflow = {
        "nodes": [
            _node(1, "WanVideoModelLoader", ["wan.safetensors"]),
            _node(2, "WanVideoBlockSwap", [10, False]),
            _node(3, "WanVideoSetBlockSwap"),
        ]
    }
    cloud = join_cloud(inventory(workflow), _catalog(WanVideoModelLoader=_combo("model", "wan.safetensors")))

    assert cloud["missing_classes"] == []
    assert [x["class"] for x in cloud["lossless_removable"]] == ["WanVideoBlockSwap", "WanVideoSetBlockSwap"]
    assert cloud["verdict"] == "covered_after_lossless_fixes"


def test_join_cloud_flags_a_same_named_file_with_different_weights(tmp_path: Path):
    (tmp_path / "upscale_models").mkdir()
    (tmp_path / "upscale_models" / "4x-UltraSharp.pth").write_bytes(b"x" * 10)
    report = inventory({"nodes": [_node(1, "UpscaleModelLoader", ["4x-UltraSharp.pth"])]})
    join_local(report, [str(tmp_path)])
    lookup = _catalog(UpscaleModelLoader=_combo("model_name", "4x-UltraSharp.pth"))

    differs = join_cloud(report, lookup, size_of=lambda name: 66_961_958)
    same = join_cloud(report, lookup, size_of=lambda name: 10)

    assert differs["models"][0]["weights"] == "different_size"
    assert differs["verdict"] == "needs_build_or_lossy_conversion"
    assert same["models"][0]["weights"] == "same_size"
    assert same["verdict"] == "covered"


def test_join_inputs_names_the_cloud_upload_and_finds_it(tmp_path: Path):
    (tmp_path / "face.png").write_bytes(b"PNG")
    report = inventory({"nodes": [_node(4, "LoadImage", ["face.png", "image"])]})

    join_inputs(report, [str(tmp_path)], lambda path: "blake3:abc123")
    cloud = join_cloud(report, _catalog(LoadImage=_combo("image", "abc123.png")))

    item = report["inputs"][0]
    assert item["local_path"] == str(tmp_path / "face.png")
    assert item["cloud_name"] == "abc123.png"
    assert item["on_cloud"] is True
    assert cloud["verdict"] == "covered"
