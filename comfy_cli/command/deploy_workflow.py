"""Prepare API-format workflows for deployment job submission."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.command.run.loader import _load_workflow_file, is_ui_workflow

_ASSET_TYPE: Final = "core/ASSET"
_LOCAL_ASSET_MARKER_PREFIX: Final = "local-asset:"


@dataclass(frozen=True, slots=True)
class WorkflowAsset:
    marker: str
    local_path: Path
    file_path: str


@dataclass(frozen=True, slots=True)
class WorkflowAssetPlan:
    workflow: JsonObject
    assets: tuple[WorkflowAsset, ...]


class DeployWorkflowFormatUIError(Exception):
    code = "deploy_workflow_format_ui"
    hint = (
        "use ComfyUI's 'File > Export (API)' to save as API format, or convert locally with "
        "`comfy run` against a running ComfyUI instance"
    )

    def __init__(self) -> None:
        super().__init__("deploy run accepts API-format workflows only")


def _resolve_local_file(value: str, roots: tuple[Path, Path]) -> Path | None:
    input_path = Path(value)
    for lookup_root in roots:
        candidate = input_path if input_path.is_absolute() else lookup_root / input_path
        try:
            resolved = candidate.resolve()
            contained = any(resolved.is_relative_to(root) for root in roots)
            if contained and resolved.is_file():
                return resolved
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _rewrite_input(
    value: JsonValue,
    roots: tuple[Path, Path],
    discovered: dict[Path, WorkflowAsset],
) -> JsonValue:
    match value:
        case str() as text:
            local_path = _resolve_local_file(text, roots)
            if local_path is None:
                return text
            asset = discovered.get(local_path)
            if asset is None:
                asset = WorkflowAsset(
                    marker=f"{_LOCAL_ASSET_MARKER_PREFIX}{len(discovered)}",
                    local_path=local_path,
                    file_path=local_path.name,
                )
                discovered[local_path] = asset
            return {
                "__type": _ASSET_TYPE,
                "info": {"id": asset.marker, "file_path": asset.file_path},
            }
        case dict() as mapping:
            if mapping.get("__type") == _ASSET_TYPE:
                return deepcopy(mapping)
            return {key: _rewrite_input(item, roots, discovered) for key, item in mapping.items()}
        case list() as items:
            return [_rewrite_input(item, roots, discovered) for item in items]
        case None | bool() | int() | float():
            return value
        case unreachable:
            assert_never(unreachable)


def _scan_api_workflow(workflow: JsonObject, roots: tuple[Path, Path]) -> WorkflowAssetPlan:
    discovered: dict[Path, WorkflowAsset] = {}
    rewritten = deepcopy(workflow)
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        rewritten_node = deepcopy(node)
        rewritten_node["inputs"] = _rewrite_input(inputs, roots, discovered)
        rewritten[node_id] = rewritten_node
    return WorkflowAssetPlan(workflow=rewritten, assets=tuple(discovered.values()))


def scan_workflow_inputs(workflow: JsonObject, *, workflow_dir: Path, cwd: Path) -> WorkflowAssetPlan:
    """Return a pure placeholder rewrite plan for local files in node inputs."""
    if is_ui_workflow(workflow):
        raise DeployWorkflowFormatUIError
    roots = (workflow_dir.resolve(), cwd.resolve())
    return _scan_api_workflow(workflow, roots)


def load_deploy_workflow(workflow_file: Path, *, cwd: Path | None = None) -> WorkflowAssetPlan:
    """Load an API workflow and return its contained local-file rewrite plan."""
    raw_workflow, absolute_path, _ = _load_workflow_file(str(workflow_file))
    if is_ui_workflow(raw_workflow):
        raise DeployWorkflowFormatUIError
    resolved_cwd = Path.cwd() if cwd is None else cwd
    roots = (Path(absolute_path).parent.resolve(), resolved_cwd.resolve())
    return _scan_api_workflow(raw_workflow, roots)


def _materialize_value(value: JsonValue, markers: frozenset[str], asset_ids: Mapping[str, str]) -> JsonValue:
    match value:
        case dict() as mapping:
            if mapping.get("__type") == _ASSET_TYPE:
                info = mapping.get("info")
                if isinstance(info, dict):
                    marker = info.get("id")
                    if isinstance(marker, str) and marker in markers:
                        rewritten = deepcopy(mapping)
                        rewritten_info = deepcopy(info)
                        rewritten_info["id"] = asset_ids[marker]
                        rewritten["info"] = rewritten_info
                        return rewritten
                return deepcopy(mapping)
            return {key: _materialize_value(item, markers, asset_ids) for key, item in mapping.items()}
        case list() as items:
            return [_materialize_value(item, markers, asset_ids) for item in items]
        case None | bool() | int() | float() | str():
            return value
        case unreachable:
            assert_never(unreachable)


def materialize_workflow_assets(plan: WorkflowAssetPlan, asset_ids: Mapping[str, str]) -> JsonObject:
    """Replace every local marker with the uploaded asset id in a new workflow."""
    markers = frozenset(asset.marker for asset in plan.assets)
    return {key: _materialize_value(value, markers, asset_ids) for key, value in plan.workflow.items()}
