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
from comfy_cli.command.deploy_workflow import WorkflowAsset, WorkflowAssetPlan, materialize_workflow_assets
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.deploy_assets import (
    AssetResolveRequest,
    AssetResolveResult,
    DeployAssetClient,
    DeployAssetMissingError,
)
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
    allowed_hashes: frozenset[str] | None = None


def _missing_error(missing: list[JsonValue]) -> DeployAPIError:
    """One refusal naming every file the run may not upload.

    It lists them all, not just the first, so ``details.missing`` is the whole
    upload plan: each absolute ``local_path`` the scanner picked, its ``bytes``
    and its ``hash``. ``file_path`` and ``hash`` still name the first, as before.
    """
    first = missing[0]
    assert isinstance(first, dict)
    count = len(missing)
    return DeployAPIError(
        code="deploy_asset_missing",
        message=f"{count} local file{'' if count == 1 else 's'} the deployment does not have may not be uploaded",
        details={"file_path": first["file_path"], "hash": first["hash"], "missing": missing},
    )


def resolve_assets(plan: WorkflowAssetPlan, context: AssetResolveContext) -> ResolvedAssets:
    asset_ids: dict[str, str] = {}
    results: dict[int, tuple[AssetResolveResult, DisclosedAsset]] = {}
    missing: list[JsonValue] = []
    # With --allow-upload, every file's approval is settled before the first
    # upload: the first pass only probes, so a refused file fails the run
    # before an approved one is sent and left unused on the deployment.
    approved = context.allowed_hashes if context.upload_enabled else None
    to_upload: list[tuple[int, WorkflowAsset, int]] = []

    def record(index: int, asset: WorkflowAsset, size: int, result: AssetResolveResult) -> None:
        asset_id = result.asset.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise server_shape_error("the data plane returned an asset without an id")
        asset_ids[asset.marker] = asset_id
        results[index] = (result, DisclosedAsset(asset.local_path, asset.file_path, size, bool(result.uploaded)))

    for index, asset in enumerate(plan.assets):
        size = asset.local_path.stat().st_size
        if context.renderer.is_pretty():
            # Announced BEFORE the call, not after: a disclosure the user only
            # reads once the bytes are already gone is not a disclosure.
            context.renderer.info(f"Sending workflow asset {asset.local_path} ({size} bytes)")
        upload_now = context.upload_enabled and approved is None
        try:
            result = context.client.resolve_asset(
                AssetResolveRequest(asset.local_path, asset.file_path, upload_now, context.allowed_hashes)
            )
        except DeployAssetMissingError as error:
            if approved is not None and error.details["hash"] in approved:
                to_upload.append((index, asset, size))
                continue
            # Keep going, so the refusal names every file, not only the first.
            missing.append(error.details)
            continue
        record(index, asset, size, result)
    if missing:
        raise _missing_error(missing)
    for index, asset, size in to_upload:
        try:
            result = context.client.resolve_asset(
                AssetResolveRequest(asset.local_path, asset.file_path, True, context.allowed_hashes)
            )
        except DeployAssetMissingError as error:
            # The file changed after the first pass and no longer has an approved hash.
            raise _missing_error([error.details]) from error
        record(index, asset, size, result)
    ordered = [results[index] for index in sorted(results)]
    return ResolvedAssets(
        workflow=materialize_workflow_assets(plan, asset_ids),
        uploaded=sum(result.uploaded for result, _ in ordered),
        deduped=sum(result.deduped for result, _ in ordered),
        bytes=sum(result.bytes for result, _ in ordered),
        files=tuple(disclosed for _, disclosed in ordered),
    )
