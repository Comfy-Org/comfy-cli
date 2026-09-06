"""Render deployment reference catalogs without projecting server fields."""

from __future__ import annotations

import urllib.error
from typing import Final, Protocol

import typer

from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.command.deploy_types import required_string, server_shape_error
from comfy_cli.deploy_api import DeployClient
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer
from comfy_cli.output.renderer import Renderer

_AVAILABILITY_RANK: Final = {"high": 3, "medium": 2, "low": 1}


class ComputeCatalogClient(Protocol):
    def get_compute_catalog(self, /) -> JsonObject: ...


def _compute_client() -> ComputeCatalogClient:
    return DeployClient.from_session()


def _regions(catalog: JsonObject) -> list[JsonObject]:
    raw = catalog.get("regions")
    if not isinstance(raw, list):
        raise server_shape_error("the compute catalog has no regions array")
    regions: list[JsonObject] = []
    for region in raw:
        if not isinstance(region, dict):
            raise server_shape_error("the compute catalog contains a non-object region")
        regions.append(region)
    return regions


def _render_pretty(renderer: Renderer, regions: list[JsonObject]) -> None:
    from rich.table import Table

    rows: list[tuple[int, str, str, str, str, str, str]] = []
    for region in regions:
        region_id = required_string(region, "region")
        region_label_value = region.get("label")
        region_label = region_label_value if isinstance(region_label_value, str) else region_id
        gpus = region.get("gpus")
        if not isinstance(gpus, list):
            raise server_shape_error("a compute catalog region has no gpus array", region=region_id)
        for gpu in gpus:
            if not isinstance(gpu, dict):
                raise server_shape_error("a compute catalog region contains a non-object GPU", region=region_id)
            gpu_class = required_string(gpu, "gpuClass")
            gpu_label_value = gpu.get("label")
            gpu_label = gpu_label_value if isinstance(gpu_label_value, str) else gpu_class
            vram_value = gpu.get("vramGb")
            vram = str(vram_value) if isinstance(vram_value, int) and not isinstance(vram_value, bool) else ""
            availability_value = gpu.get("availability")
            if availability_value is not None and not isinstance(availability_value, str):
                raise server_shape_error(
                    "a compute catalog GPU has invalid availability",
                    region=region_id,
                    gpuClass=gpu_class,
                )
            availability = availability_value or ""
            rows.append(
                (
                    _AVAILABILITY_RANK.get(availability.casefold(), 0),
                    region_id,
                    region_label,
                    gpu_class,
                    gpu_label,
                    vram,
                    availability,
                )
            )
    rows.sort(key=lambda row: (-row[0], row[1], row[3]))
    table = Table(show_header=True, header_style="bold")
    for column in ("region", "region label", "gpu class", "gpu label", "VRAM (GB)", "availability"):
        table.add_column(column)
    for _, *cells in rows:
        table.add_row(*cells)
    if rows:
        renderer.console().print(table)
    else:
        renderer.info("No compute regions found.")


def run_compute(region: str | None) -> None:
    renderer = get_renderer()
    try:
        catalog = _compute_client().get_compute_catalog()
        regions = _regions(catalog)
        if region is None:
            result = catalog
            visible_regions = regions
        else:
            visible_regions = [item for item in regions if item.get("region") == region]
            region_values: list[JsonValue] = [*visible_regions]
            result = {**catalog, "regions": region_values}
        if renderer.is_pretty():
            _render_pretty(renderer, visible_regions)
        renderer.emit(result, command="deploy refs compute", changed=False)
    except DeployAPIError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
