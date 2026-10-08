from __future__ import annotations

import copy
import importlib
import json

from deploy_up_support import option_names
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject

CATALOG: JsonObject = {
    "catalogVersion": "future-1",
    "regions": [
        {
            "region": "US-MO-2",
            "label": "Missouri",
            "futureField": "x",
            "gpus": [
                {"gpuClass": "rtx-4090", "label": "RTX 4090", "vramGb": 24, "availability": "Low"},
                {"gpuClass": "l4", "label": "NVIDIA L4", "vramGb": 24, "availability": "High"},
            ],
        },
        {
            "region": "EU-RO-1",
            "label": "Romania",
            "gpus": [{"gpuClass": "a100", "label": "NVIDIA A100", "vramGb": 80, "availability": "Medium"}],
        },
    ],
}


class ComputeClient:
    def __init__(self, catalog: JsonObject) -> None:
        self.catalog = copy.deepcopy(catalog)
        self.calls = 0

    def get_compute_catalog(self) -> JsonObject:
        self.calls += 1
        return copy.deepcopy(self.catalog)


def _install_client(monkeypatch, client: ComputeClient) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_refs")
    monkeypatch.setattr(module, "_compute_client", lambda: client)


def _invoke(*args: str, pretty: bool = False):
    output_flag = "--no-json" if pretty else "--json"
    return CliRunner().invoke(
        app,
        [output_flag, "deploy", "refs", "compute", *args],
        env={"COLUMNS": "400"},
    )


def _data(result) -> JsonObject:
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    data = envelope["data"]
    assert isinstance(data, dict)
    return data


def _regions(catalog: JsonObject) -> list[JsonObject]:
    value = catalog.get("regions")
    assert isinstance(value, list)
    regions: list[JsonObject] = []
    for region in value:
        assert isinstance(region, dict)
        regions.append(region)
    return regions


def test_refs_compute_is_registered_under_the_refs_group() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "refs", "compute", "--help"])

    # Then
    assert result.exit_code == 0
    assert "--region" in option_names("refs", "compute")


def test_json_passes_catalog_regions_through_byte_identically(monkeypatch) -> None:
    # Given
    client = ComputeClient(CATALOG)
    _install_client(monkeypatch, client)

    # When
    result = _invoke()

    # Then
    data = _data(result)
    actual_regions = _regions(data)
    expected_regions = _regions(CATALOG)
    assert result.exit_code == 0
    assert data == CATALOG
    assert actual_regions[0]["futureField"] == "x"
    assert json.dumps(actual_regions[0], separators=(",", ":")) == json.dumps(
        expected_regions[0], separators=(",", ":")
    )
    assert client.calls == 1


def test_pretty_table_renders_availability_in_descending_order(monkeypatch) -> None:
    # Given
    _install_client(monkeypatch, ComputeClient(CATALOG))

    # When
    result = _invoke(pretty=True)

    # Then
    assert result.exit_code == 0
    assert "availability" in result.stdout.lower()
    assert result.stdout.index("High") < result.stdout.index("Medium") < result.stdout.index("Low")


def test_region_filter_keeps_the_matching_region_verbatim(monkeypatch) -> None:
    # Given
    _install_client(monkeypatch, ComputeClient(CATALOG))

    # When
    result = _invoke("--region", "EU-RO-1")

    # Then
    data = _data(result)
    assert result.exit_code == 0
    assert data["catalogVersion"] == "future-1"
    assert _regions(data) == [_regions(CATALOG)[1]]


def test_unknown_region_returns_an_empty_region_list(monkeypatch) -> None:
    # Given
    _install_client(monkeypatch, ComputeClient(CATALOG))

    # When
    result = _invoke("--region", "UNKNOWN-1")

    # Then
    data = _data(result)
    assert result.exit_code == 0
    assert data == {"catalogVersion": "future-1", "regions": []}


TREE_CATALOG: JsonObject = {
    "regions": [
        {
            "region": "anywhere",
            "label": "Anywhere",
            "level": "global",
            "gpus": [{"gpuClass": "b200", "label": "B200", "vramGb": 180}],
        },
        {
            "region": "us",
            "label": "United States",
            "level": "country",
            "parent": "anywhere",
            "gpus": [{"gpuClass": "b200", "label": "B200", "vramGb": 180}],
        },
        {
            "region": "US-NE-1",
            "label": "US-NE-1",
            "level": "datacenter",
            "parent": "us",
            "gpus": [{"gpuClass": "rtx-pro-6000-server", "label": "RTX PRO 6000", "vramGb": 96}],
        },
    ],
}


def test_pretty_table_shows_each_location_level_and_parent(monkeypatch) -> None:
    # Given
    _install_client(monkeypatch, ComputeClient(TREE_CATALOG))

    # When
    result = _invoke(pretty=True)

    # Then
    assert result.exit_code == 0
    us_row = next(line for line in result.stdout.splitlines() if "United States" in line)
    datacenter_row = next(line for line in result.stdout.splitlines() if "RTX PRO 6000" in line)
    assert "country" in us_row and "anywhere" in us_row
    assert "datacenter" in datacenter_row and "us" in datacenter_row.split("US-NE-1", 2)[-1]
    assert "B200" in us_row
