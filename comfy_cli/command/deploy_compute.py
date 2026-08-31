"""Interactive compute-catalog pickers for deploy commands."""

from __future__ import annotations

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_types import DeployUpClient


def _catalog_regions(client: DeployUpClient) -> list[JsonObject]:
    regions = client.get_compute_catalog().get("regions")
    return [region for region in regions if isinstance(region, dict)] if isinstance(regions, list) else []


def prompt_gpu(client: DeployUpClient) -> str | None:
    from comfy_cli.ui import prompt_select

    choices: dict[str, str] = {}
    for region in _catalog_regions(client):
        gpus = region.get("gpus")
        for gpu in gpus if isinstance(gpus, list) else []:
            if isinstance(gpu, dict):
                gpu_class = gpu.get("gpuClass")
                label = gpu.get("label")
                if isinstance(gpu_class, str):
                    choices.setdefault(gpu_class, label if isinstance(label, str) else gpu_class)
    selected = prompt_select("GPU class", [{"name": label, "value": value} for value, label in choices.items()])
    return selected if isinstance(selected, str) else None


def prompt_region(client: DeployUpClient, gpu_class: str) -> str | None:
    from comfy_cli.ui import prompt_select

    choices = []
    for region in _catalog_regions(client):
        gpus = region.get("gpus")
        available = isinstance(gpus, list) and any(
            isinstance(gpu, dict) and gpu.get("gpuClass") == gpu_class for gpu in gpus
        )
        region_id = region.get("region")
        if available and isinstance(region_id, str):
            label = region.get("label")
            choices.append({"name": label if isinstance(label, str) else region_id, "value": region_id})
    selected = prompt_select("Region", choices)
    return selected if isinstance(selected, str) else None
