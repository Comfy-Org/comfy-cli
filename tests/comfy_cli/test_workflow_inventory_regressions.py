"""Dependency-scope and identity regressions from Run-on-Cloud validation."""

import pytest

from comfy_cli.workflow_inventory import inventory, join_cloud, join_inputs, join_local


def _node(identifier, kind, values=(), **extra):
    return {"id": identifier, "type": kind, "widgets_values": list(values), **extra}


def _catalog(kind):
    return {"inputs": [{"name": "ckpt_name", "choices": ["base.safetensors"]}]}


@pytest.mark.parametrize("api", [False, True])
def test_prompt_model_suffix_is_text_while_real_spaced_filename_is_retained(api):
    text = "portrait, sharp focus, embedding:EasyNegative.pt"
    workflow = (
        {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}},
            "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "My Model.safetensors"}},
        }
        if api
        else {
            "nodes": [_node(1, "CLIPTextEncode", [text]), _node(2, "CheckpointLoaderSimple", ["My Model.safetensors"])]
        }
    )
    report = inventory(workflow)
    assert report["embeddings"][0]["name"] == "EasyNegative.pt"
    assert [model["path"] for model in report["models"]] == ["My Model.safetensors"]


@pytest.mark.parametrize(
    "text,name", [("(embedding:BadDream:1.2)", "BadDream"), ("(embedding:BadDream.pt:0.75)", "BadDream.pt")]
)
def test_embedding_weight_is_not_part_of_filename(text, name):
    report = inventory({"nodes": [_node(1, "CLIPTextEncode", [text])]})
    assert report["embeddings"][0]["name"] == name


@pytest.mark.parametrize("mode,why", [(2, "muted"), (4, "bypassed")])
def test_inactive_subgraph_is_reported_without_expanding(mode, why):
    report = inventory(
        {
            "nodes": [_node(1, "sg", mode=mode)],
            "definitions": {"subgraphs": [{"id": "sg", "nodes": [_node(2, "PrivateLoader")]}]},
        }
    )
    assert report["classes"] == {}
    assert report["skipped"] == [{"id": 1, "class": "sg", "why": why, "models": []}]


@pytest.mark.parametrize("checked", [False, True])
def test_absent_local_identity_is_not_verified_coverage(tmp_path, checked):
    report = inventory({"nodes": [_node(1, "CheckpointLoaderSimple", ["base.safetensors"])]})
    if checked:
        join_local(report, [str(tmp_path)])
    cloud = join_cloud(report, _catalog)
    assert cloud["models"][0]["weights"] == ("no_local_copy" if checked else "not_checked")
    assert cloud["verdict"] == "covered_needs_check"
    assert cloud["unverified"]


@pytest.mark.parametrize("same_size,hash_files", [(False, False), (True, False), (True, True)])
def test_different_local_copies_are_ambiguous(tmp_path, same_size, hash_files):
    for name, payload in [("first", b"first"), ("second", b"other" if same_size else b"cloud-size")]:
        root = tmp_path / name
        root.mkdir()
        (root / "base.safetensors").write_bytes(payload)
    report = inventory({"nodes": [_node(1, "CheckpointLoaderSimple", ["base.safetensors"])]})
    join_local(report, [str(tmp_path / "first"), str(tmp_path / "second")], hash_files=hash_files)
    cloud = join_cloud(report, _catalog, lambda _: 5)
    assert cloud["models"][0]["weights"] == "ambiguous_local_copies"
    assert cloud["verdict"] == "covered_needs_check"


def test_verified_identical_local_copies_keep_size_evidence(tmp_path):
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "base.safetensors").write_bytes(b"same")
    report = inventory({"nodes": [_node(1, "CheckpointLoaderSimple", ["base.safetensors"])]})
    join_local(report, [str(tmp_path / "first"), str(tmp_path / "second")], hash_files=True)
    assert join_cloud(report, _catalog, lambda _: 4)["models"][0]["weights"] == "same_size"


@pytest.mark.parametrize("api", [False, True])
def test_unused_known_loader_is_excluded_but_unknown_output_is_preserved(api):
    workflow = (
        {
            "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            "3": {"class_type": "LoraLoader", "inputs": {"lora_name": "unused.safetensors"}},
            "4": {"class_type": "PrivateOutput", "inputs": {"model": ["5", 0]}},
            "5": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "needed.safetensors"}},
        }
        if api
        else {
            "nodes": [
                _node(1, "LoadImage", ["input.png"]),
                _node(2, "SaveImage"),
                _node(3, "LoraLoader", ["unused.safetensors"]),
                _node(4, "PrivateOutput"),
                _node(5, "CheckpointLoaderSimple", ["needed.safetensors"]),
            ],
            "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 5, 0, 4, 0, "MODEL"]],
        }
    )
    report = inventory(workflow)
    assert "LoraLoader" not in report["classes"]
    assert "PrivateOutput" in report["classes"]
    assert [model["path"] for model in report["models"]] == ["needed.safetensors"]
    assert any(str(node["id"]) == "3" and node["why"] == "disconnected" for node in report["skipped"])


def test_no_known_output_does_not_guess_which_nodes_execute():
    report = inventory({"nodes": [_node(1, "LoraLoader", ["keep.safetensors"])]})
    assert report["models"][0]["path"] == "keep.safetensors"


def test_known_text_widget_ending_with_model_suffix_is_not_a_path():
    report = inventory({"nodes": [_node(1, "CLIPTextEncode", ["a portrait titled model.pt"])]})
    assert report["models"] == []


def test_cli_preview_envelope_has_actionable_graph_extraction_error():
    with pytest.raises(ValueError, match=r"data\.prompt.*API graph"):
        inventory(
            {"schema": "envelope/1", "ok": True, "data": {"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}}}
        )


def test_literal_numeric_api_array_does_not_create_dependency_edge():
    report = inventory(
        {
            "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "unused.safetensors"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "range": [1, 0]}},
        }
    )
    assert "LoraLoader" not in report["classes"]


def test_bypass_dependencies_are_retained_for_reachable_output():
    report = inventory(
        {
            "nodes": [
                _node(1, "CheckpointLoaderSimple", ["keep.safetensors"]),
                _node(2, "LoraLoader", ["skip.safetensors"], mode=4),
                _node(3, "SaveImage"),
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 3, 0, "IMAGE"]],
        }
    )
    assert [model["path"] for model in report["models"]] == ["keep.safetensors"]
    assert report["skipped"][0]["why"] == "bypassed"


def test_incomplete_graph_does_not_hide_dependencies():
    report = inventory(
        {
            "nodes": [_node(1, "CheckpointLoaderSimple", ["keep.safetensors"]), _node(2, "SaveImage")],
            "links": [[1, 99, 0, 2, 0, "IMAGE"]],
        }
    )
    assert report["models"][0]["path"] == "keep.safetensors"


def test_broken_api_link_disables_pruning():
    report = inventory(
        {
            "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "other": ["missing-id", 0]}},
            "3": {"class_type": "LoraLoader", "inputs": {"lora_name": "keep.safetensors"}},
        }
    )
    assert report["models"][0]["path"] == "keep.safetensors"


def test_unknown_loader_model_paths_are_not_hidden_by_text_like_input_names():
    report = inventory(
        {
            "1": {
                "class_type": "CustomLoader",
                "inputs": {"positive": "positive.safetensors", "text": "text.safetensors"},
            }
        }
    )
    assert [model["path"] for model in report["models"]] == ["positive.safetensors", "text.safetensors"]


def test_subgraph_object_links_are_supported():
    report = inventory(
        {
            "nodes": [_node(1, "sg")],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "sg",
                        "nodes": [
                            _node(2, "LoadImage", ["input.png"]),
                            _node(3, "SaveImage"),
                            _node(4, "LoraLoader", ["unused.safetensors"]),
                        ],
                        "links": [
                            {
                                "id": 1,
                                "origin_id": 2,
                                "origin_slot": 0,
                                "target_id": 3,
                                "target_slot": 0,
                                "type": "IMAGE",
                            }
                        ],
                    }
                ]
            },
        }
    )
    assert report["models"] == []
    assert any(node["id"] == "1/4" and node["why"] == "disconnected" for node in report["skipped"])


def test_missing_input_is_separate_from_dependency_coverage(tmp_path):
    report = inventory({"nodes": [_node(1, "LoadImage", ["missing.png"])]})
    join_inputs(report, [str(tmp_path)], lambda _: pytest.fail("missing file cannot be hashed"))
    cloud = join_cloud(report, lambda _: {"inputs": [{"name": "image", "choices": []}]})
    assert cloud["verdict"] == "covered"
    assert cloud["readiness"] == {
        "status": "needs_inputs",
        "unresolved_inputs": [
            {"node_id": 1, "file": "missing.png", "reason": "local file not found and target availability unverified"}
        ],
    }


@pytest.mark.parametrize("output_class", ["SaveImage", "PrivateOutputNode"])
def test_untrusted_output_false_metadata_cannot_hide_an_executed_save_branch(output_class):
    from comfy_cli.workflow_inventory import inventory

    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "used.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "LoraLoader", "inputs": {"lora_name": "also-used.safetensors"}},
        "4": {"class_type": output_class, "output_node": False, "inputs": {"images": ["3", 0]}},
    }
    report = inventory(graph)
    assert {row["path"] for row in report["models"]} == {"used.safetensors", "also-used.safetensors"}
    assert not any(row["why"] == "disconnected" for row in report["skipped"])
