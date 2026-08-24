from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from comfy_cli.command import deploy_run, deploy_workflow
from comfy_cli.command.build_spec import JsonObject


def _workflow(inputs: JsonObject, *, class_type: str = "TestNode") -> JsonObject:
    return {"1": {"class_type": class_type, "inputs": inputs}}


def _write_workflow(path: Path, workflow: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow), encoding="utf-8")


def _rewritten_inputs(plan: deploy_run.WorkflowAssetPlan) -> JsonObject:
    node = plan.workflow["1"]
    assert isinstance(node, dict)
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    return inputs


def test_workflow_relative_image_is_rewritten_while_absent_checkpoint_stays_plain(tmp_path: Path) -> None:
    # Given
    workflow_dir = tmp_path / "workflow"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    image = workflow_dir / "input.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(
        workflow_path,
        _workflow({"image": "input.png", "ckpt_name": "sd15.safetensors"}, class_type="LoadImage"),
    )

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert len(plan.assets) == 1
    asset = plan.assets[0]
    assert asset.local_path == image.resolve()
    assert asset.file_path == "input.png"
    assert _rewritten_inputs(plan) == {
        "image": {"__type": "core/ASSET", "info": {"id": asset.marker, "file_path": "input.png"}},
        "ckpt_name": "sd15.safetensors",
    }


def test_workflow_directory_wins_before_cwd_for_same_filename(tmp_path: Path) -> None:
    # Given
    workflow_dir = tmp_path / "workflow"
    cwd = tmp_path / "cwd"
    workflow_dir.mkdir()
    cwd.mkdir()
    workflow_file = workflow_dir / "shared.bin"
    workflow_file.write_bytes(b"workflow")
    (cwd / "shared.bin").write_bytes(b"cwd")
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "shared.bin"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert plan.assets[0].local_path == workflow_file.resolve()


def test_cwd_file_is_used_as_the_fallback_resolution_root(tmp_path: Path) -> None:
    # Given
    workflow_dir = tmp_path / "workflow"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    cwd_file = cwd / "fallback.bin"
    cwd_file.write_bytes(b"cwd")
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "fallback.bin"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert plan.assets[0].local_path == cwd_file.resolve()


def test_made_up_custom_node_nested_input_is_rewritten_without_an_allowlist(tmp_path: Path) -> None:
    # Given
    local_file = tmp_path / "workflow" / "custom.payload"
    local_file.parent.mkdir()
    local_file.write_bytes(b"custom")
    workflow_path = local_file.parent / "workflow.json"
    _write_workflow(
        workflow_path,
        _workflow({"custom_loader_data": [{"source": "custom.payload"}]}, class_type="MadeUpAcmeLoader42"),
    )

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path / "cwd")

    # Then
    nested = _rewritten_inputs(plan)["custom_loader_data"]
    assert isinstance(nested, list)
    record = nested[0]
    assert isinstance(record, dict)
    assert record["source"] == {
        "__type": "core/ASSET",
        "info": {"id": plan.assets[0].marker, "file_path": "custom.payload"},
    }


def test_parent_traversal_to_real_file_outside_both_roots_is_not_an_asset(tmp_path: Path) -> None:
    # Given
    project = tmp_path / "project"
    workflow_dir = project / "workflows"
    cwd = project / "cwd"
    cwd.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "../../outside.txt"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert plan.assets == ()
    assert _rewritten_inputs(plan)["source"] == "../../outside.txt"


def test_absolute_file_outside_both_roots_is_not_an_asset(tmp_path: Path) -> None:
    # Given
    roots = tmp_path / "roots"
    workflow_path = roots / "workflow" / "workflow.json"
    cwd = roots / "cwd"
    cwd.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    _write_workflow(workflow_path, _workflow({"source": str(outside.resolve())}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert plan.assets == ()
    assert _rewritten_inputs(plan)["source"] == str(outside.resolve())


def test_symlink_inside_workflow_pointing_outside_is_not_an_asset(tmp_path: Path) -> None:
    # Given
    workflow_dir = tmp_path / "roots" / "workflow"
    workflow_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workflow_dir / "escape.txt").symlink_to(outside)
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "escape.txt"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path / "roots" / "cwd")

    # Then
    assert plan.assets == ()
    assert _rewritten_inputs(plan)["source"] == "escape.txt"


def test_cwd_fallback_parent_traversal_outside_cwd_is_not_an_asset(tmp_path: Path) -> None:
    # Given
    project = tmp_path / "project"
    workflow_dir = project / "workflow" / "nested"
    cwd = project / "cwd"
    cwd.mkdir(parents=True)
    outside = project / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "../outside.txt"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=cwd)

    # Then
    assert plan.assets == ()
    assert _rewritten_inputs(plan)["source"] == "../outside.txt"


def test_directory_matching_an_input_string_is_not_an_asset(tmp_path: Path) -> None:
    # Given
    workflow_dir = tmp_path / "workflow"
    (workflow_dir / "models").mkdir(parents=True)
    workflow_path = workflow_dir / "workflow.json"
    _write_workflow(workflow_path, _workflow({"source": "models"}))

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path / "cwd")

    # Then
    assert plan.assets == ()
    assert _rewritten_inputs(plan)["source"] == "models"


def test_ui_workflow_fails_before_scanning_or_any_plane_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    workflow_path = tmp_path / "ui.json"
    _write_workflow(workflow_path, {"nodes": [], "links": []})
    recorded_requests: dict[str, list[str]] = {"control": [], "data": [], "uploads": []}

    def scan_tripwire(*_args: object, **_kwargs: object) -> None:
        recorded_requests["uploads"].append("scan")

    monkeypatch.setattr(deploy_workflow, "_scan_api_workflow", scan_tripwire)

    # When
    with pytest.raises(deploy_run.DeployWorkflowFormatUIError) as exc_info:
        deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path)

    # Then
    assert exc_info.value.code == "deploy_workflow_format_ui"
    assert "File > Export (API)" in exc_info.value.hint
    assert "comfy run" in exc_info.value.hint
    assert recorded_requests == {"control": [], "data": [], "uploads": []}


def test_api_workflow_proceeds_normally(tmp_path: Path) -> None:
    # Given
    workflow_path = tmp_path / "api.json"
    workflow = _workflow({"prompt": "a red fox"})
    _write_workflow(workflow_path, workflow)

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path)

    # Then
    assert plan.workflow == workflow
    assert plan.assets == ()


def test_rewriting_never_modifies_the_workflow_file(tmp_path: Path) -> None:
    # Given
    local_file = tmp_path / "input.png"
    local_file.write_bytes(b"image")
    workflow_path = tmp_path / "workflow.json"
    _write_workflow(workflow_path, _workflow({"image": "input.png"}))
    before = hashlib.sha256(workflow_path.read_bytes()).digest()

    # When
    deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path)

    # Then
    assert hashlib.sha256(workflow_path.read_bytes()).digest() == before


def test_existing_core_asset_is_left_completely_untouched(tmp_path: Path) -> None:
    # Given
    (tmp_path / "photo.png").write_bytes(b"local shadow")
    existing: JsonObject = {
        "__type": "core/ASSET",
        "info": {"id": "asset-existing", "hash": "blake3:abc", "file_path": "photo.png"},
    }
    workflow_path = tmp_path / "workflow.json"
    _write_workflow(workflow_path, _workflow({"image": existing}))
    expected_bytes = json.dumps(existing, sort_keys=True, separators=(",", ":")).encode()

    # When
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path)

    # Then
    rewritten = _rewritten_inputs(plan)["image"]
    assert json.dumps(rewritten, sort_keys=True, separators=(",", ":")).encode() == expected_bytes
    assert plan.assets == ()


def test_uploaded_asset_ids_materialize_into_a_new_submit_ready_workflow(tmp_path: Path) -> None:
    # Given
    (tmp_path / "input.png").write_bytes(b"image")
    workflow_path = tmp_path / "workflow.json"
    _write_workflow(workflow_path, _workflow({"image": "input.png"}))
    plan = deploy_run.load_deploy_workflow(workflow_path, cwd=tmp_path)
    marker = plan.assets[0].marker

    # When
    rewritten = deploy_run.materialize_workflow_assets(plan, {marker: "asset-uuid"})

    # Then
    node = rewritten["1"]
    assert isinstance(node, dict)
    assert node["inputs"] == {"image": {"__type": "core/ASSET", "info": {"id": "asset-uuid", "file_path": "input.png"}}}
    assert _rewritten_inputs(plan)["image"] != node["inputs"]
