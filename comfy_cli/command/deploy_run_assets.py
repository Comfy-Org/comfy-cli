"""Turn a workflow's local files into data-plane assets, and disclose each one.

The scanner picks these files out of a workflow the user may not have written,
so *which bytes left this machine* is a first-class result here rather than a
by-product of the upload totals: the totals alone never name a filename.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.command.deploy_types import server_shape_error
from comfy_cli.command.deploy_workflow import WorkflowAssetPlan, materialize_workflow_assets
from comfy_cli.deploy_assets import AssetResolveRequest, AssetResolveResult, DeployAssetClient
from comfy_cli.output.renderer import Renderer


@dataclass(frozen=True, slots=True)
class DisclosedAsset:
    """One local file this run handed to the data plane.

    ``bytes`` is the local file's size, reported even when the hash deduped and
    no body was actually sent — the caller still needs to know the file was read
    and its content fingerprinted.
    """

    local_path: Path
    file_path: str
    bytes: int
    uploaded: bool

    def payload(self) -> JsonObject:
        return {
            "local_path": str(self.local_path),
            "file_path": self.file_path,
            "bytes": self.bytes,
            "uploaded": self.uploaded,
        }


@dataclass(frozen=True, slots=True)
class ResolvedAssets:
    workflow: JsonObject
    uploaded: int
    deduped: int
    bytes: int
    files: tuple[DisclosedAsset, ...] = ()

    def payload(self) -> JsonObject:
        disclosed: list[JsonValue] = [file.payload() for file in self.files]
        return {"uploaded": self.uploaded, "deduped": self.deduped, "bytes": self.bytes, "files": disclosed}


@dataclass(frozen=True, slots=True)
class AssetResolveContext:
    """Everything asset resolution needs beyond the plan itself."""

    client: DeployAssetClient
    renderer: Renderer
    upload_enabled: bool


def resolve_assets(plan: WorkflowAssetPlan, context: AssetResolveContext) -> ResolvedAssets:
    asset_ids: dict[str, str] = {}
    results: list[AssetResolveResult] = []
    disclosed: list[DisclosedAsset] = []
    for asset in plan.assets:
        size = asset.local_path.stat().st_size
        if context.renderer.is_pretty():
            # Announced BEFORE the call, not after: a disclosure the user only
            # reads once the bytes are already gone is not a disclosure.
            context.renderer.info(f"Sending workflow asset {asset.local_path} ({size} bytes)")
        result = context.client.resolve_asset(
            AssetResolveRequest(asset.local_path, asset.file_path, context.upload_enabled)
        )
        asset_id = result.asset.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise server_shape_error("the data plane returned an asset without an id")
        asset_ids[asset.marker] = asset_id
        results.append(result)
        disclosed.append(DisclosedAsset(asset.local_path, asset.file_path, size, bool(result.uploaded)))
    return ResolvedAssets(
        workflow=materialize_workflow_assets(plan, asset_ids),
        uploaded=sum(result.uploaded for result in results),
        deduped=sum(result.deduped for result in results),
        bytes=sum(result.bytes for result in results),
        files=tuple(disclosed),
    )
