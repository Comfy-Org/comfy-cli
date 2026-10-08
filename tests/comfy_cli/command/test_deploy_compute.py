from __future__ import annotations

from comfy_cli.command import deploy_compute
from comfy_cli.command.build_spec import JsonObject

TREE_CATALOG: JsonObject = {
    "regions": [
        {"region": "anywhere", "label": "Anywhere", "level": "global", "gpus": [{"gpuClass": "b200", "label": "B200"}]},
        {
            "region": "us",
            "label": "United States",
            "level": "country",
            "parent": "anywhere",
            "gpus": [
                {"gpuClass": "b200", "label": "B200"},
                {"gpuClass": "rtx-pro-6000-server", "label": "RTX PRO 6000"},
            ],
        },
        {
            "region": "US-NE-1",
            "label": "US-NE-1",
            "level": "datacenter",
            "parent": "us",
            "gpus": [{"gpuClass": "rtx-pro-6000-server", "label": "RTX PRO 6000"}],
        },
    ],
}


class _Client:
    def get_compute_catalog(self) -> JsonObject:
        return TREE_CATALOG


def _capture(monkeypatch) -> list[list[dict]]:
    offered: list[list[dict]] = []

    def choose(question, choices, default="", force_prompting=False):
        offered.append(choices)
        return choices[0]["value"]

    monkeypatch.setattr("comfy_cli.ui.prompt_select", choose)
    return offered


def test_region_picker_names_each_location_with_its_level(monkeypatch) -> None:
    # Given
    offered = _capture(monkeypatch)

    # When
    deploy_compute.prompt_region(_Client(), "rtx-pro-6000-server")

    # Then
    assert offered == [
        [
            {"name": "United States (country)", "value": "us"},
            {"name": "US-NE-1 (datacenter)", "value": "US-NE-1"},
        ]
    ]


def test_gpu_picker_offers_each_class_once_across_levels(monkeypatch) -> None:
    # Given
    offered = _capture(monkeypatch)

    # When
    deploy_compute.prompt_gpu(_Client())

    # Then
    assert offered == [
        [
            {"name": "B200", "value": "b200"},
            {"name": "RTX PRO 6000", "value": "rtx-pro-6000-server"},
        ]
    ]
