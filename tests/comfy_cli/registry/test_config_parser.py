from unittest.mock import mock_open, patch

import pytest

from comfy_cli.registry.config_parser import extract_node_configuration
from comfy_cli.registry.types import (
    License,
    Model,
    PyProjectConfig,
    URLs,
)


@pytest.fixture
def mock_toml_data():
    return {
        "project": {
            "name": "test-project",
            "description": "A test project",
            "version": "1.0.0",
            "requires-python": ">=3.7",
            "dependencies": ["requests"],
            "license": {"file": "LICENSE"},
            "urls": {
                "Homepage": "https://example.com",
                "Documentation": "https://docs.example.com",
                "Repository": "https://github.com/example/test-project",
                "Issues": "https://github.com/example/test-project/issues",
            },
        },
        "tool": {
            "comfy": {
                "PublisherId": "test-publisher",
                "DisplayName": "Test Project",
                "Icon": "icon.png",
                "Models": [
                    {
                        "location": "model1.bin",
                        "model_url": "https://example.com/model1",
                    },
                    {
                        "location": "model2.bin",
                        "model_url": "https://example.com/model2",
                    },
                ],
            }
        },
    }


def test_extract_node_configuration_success(mock_toml_data):
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_toml_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert isinstance(result, PyProjectConfig)
        assert result.project.name == "test-project"
        assert result.project.description == "A test project"
        assert result.project.version == "1.0.0"
        assert result.project.requires_python == ">=3.7"
        assert result.project.dependencies == ["requests"]
        assert result.project.license == License(file="LICENSE")
        assert result.project.urls == URLs(
            homepage="https://example.com",
            documentation="https://docs.example.com",
            repository="https://github.com/example/test-project",
            issues="https://github.com/example/test-project/issues",
        )
        assert result.tool_comfy.publisher_id == "test-publisher"
        assert result.tool_comfy.display_name == "Test Project"
        assert result.tool_comfy.icon == "icon.png"
        assert len(result.tool_comfy.models) == 2
        assert result.tool_comfy.models[0] == Model(location="model1.bin", model_url="https://example.com/model1")


def test_extract_node_configuration_license_text():
    mock_data = {
        "project": {
            "license": "MIT License",
        },
    }
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")
        assert result is not None, "Expected PyProjectConfig, got None"
        assert isinstance(result, PyProjectConfig)
        assert result.project.license == License(text="MIT License")


def test_extract_node_configuration_license_text_dict():
    mock_data = {
        "project": {
            "license": {"text": "MIT License\n\nCopyright (c) 2023 Example Corp\n\nPermission is hereby granted..."},
        },
    }
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None, "Expected PyProjectConfig, got None"
        assert isinstance(result, PyProjectConfig)
        assert result.project.license == License(
            text="MIT License\n\nCopyright (c) 2023 Example Corp\n\nPermission is hereby granted..."
        )


def test_extract_license_incorrect_format():
    mock_data = {
        "project": {"license": "MIT"},
    }
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None, "Expected PyProjectConfig, got None"
        assert isinstance(result, PyProjectConfig)
        assert result.project.license == License(text="MIT")
