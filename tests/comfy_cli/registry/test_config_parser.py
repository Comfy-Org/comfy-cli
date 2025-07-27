from unittest.mock import mock_open, patch

import pytest

from comfy_cli.registry.config_parser import (
    extract_node_configuration,
    validate_and_extract_accelerator_classifiers,
    validate_and_extract_os_classifiers,
    validate_version,
)
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
                "Banner": "https://example.com/banner.png",
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
        assert result.tool_comfy.banner_url == "https://example.com/banner.png"
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


def test_extract_node_configuration_with_os_classifiers():
    mock_data = {
        "project": {
            "classifiers": [
                "Operating System :: OS Independent",
                "Operating System :: Microsoft :: Windows",
                "Programming Language :: Python :: 3",
                "Topic :: Software Development",
            ]
        }
    }
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None
        assert len(result.project.supported_os) == 2
        assert "OS Independent" in result.project.supported_os
        assert "Microsoft :: Windows" in result.project.supported_os


def test_extract_node_configuration_with_accelerator_classifiers():
    mock_data = {
        "project": {
            "classifiers": [
                "Environment :: GPU :: NVIDIA CUDA",
                "Environment :: GPU :: AMD ROCm",
                "Environment :: GPU :: Intel Arc",
                "Environment :: NPU :: Huawei Ascend",
                "Environment :: GPU :: Apple Metal",
                "Programming Language :: Python :: 3",
                "Topic :: Software Development",
            ]
        }
    }
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None
        assert len(result.project.supported_accelerators) == 5
        assert "GPU :: NVIDIA CUDA" in result.project.supported_accelerators
        assert "GPU :: AMD ROCm" in result.project.supported_accelerators
        assert "GPU :: Intel Arc" in result.project.supported_accelerators
        assert "NPU :: Huawei Ascend" in result.project.supported_accelerators
        assert "GPU :: Apple Metal" in result.project.supported_accelerators


def test_extract_node_configuration_with_comfyui_version():
    mock_data = {"project": {"dependencies": ["packge1>=2.0.0", "comfyui-frontend-package>=1.2.3", "package2>=1.0.0"]}}
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None
        assert result.project.supported_comfyui_frontend_version == ">=1.2.3"
        assert len(result.project.dependencies) == 2
        assert "comfyui-frontend-package>=1.2.3" not in result.project.dependencies
        assert "packge1>=2.0.0" in result.project.dependencies
        assert "package2>=1.0.0" in result.project.dependencies


def test_extract_node_configuration_with_requires_comfyui():
    mock_data = {"project": {}, "tool": {"comfy": {"requires-comfyui": "2.0.0"}}}
    with (
        patch("os.path.isfile", return_value=True),
        patch("builtins.open", mock_open()),
        patch("tomlkit.load", return_value=mock_data),
    ):
        result = extract_node_configuration("fake_path.toml")

        assert result is not None
        assert result.project.supported_comfyui_version == "2.0.0"


def test_validate_and_extract_os_classifiers_valid():
    """Test OS validation with valid classifiers."""
    classifiers = [
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ]
    result = validate_and_extract_os_classifiers(classifiers)
    expected = ["Microsoft :: Windows", "POSIX :: Linux", "MacOS", "OS Independent"]
    assert result == expected


@patch("typer.echo")
def test_validate_and_extract_os_classifiers_invalid(mock_echo):
    """Test OS validation with invalid classifiers."""
    classifiers = [
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux",  # Invalid - should be "POSIX :: Linux"
        "Programming Language :: Python :: 3",
    ]
    result = validate_and_extract_os_classifiers(classifiers)
    assert result == []
    mock_echo.assert_called_once()
    assert "Invalid Operating System classifier found" in mock_echo.call_args[0][0]


def test_validate_and_extract_accelerator_classifiers_valid():
    """Test accelerator validation with valid classifiers."""
    classifiers = [
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: AMD ROCm",
        "Environment :: GPU :: Intel Arc",
        "Environment :: NPU :: Huawei Ascend",
        "Environment :: GPU :: Apple Metal",
        "Programming Language :: Python :: 3",
    ]
    result = validate_and_extract_accelerator_classifiers(classifiers)
    expected = [
        "GPU :: NVIDIA CUDA",
        "GPU :: AMD ROCm",
        "GPU :: Intel Arc",
        "NPU :: Huawei Ascend",
        "GPU :: Apple Metal",
    ]
    assert result == expected


@patch("typer.echo")
def test_validate_and_extract_accelerator_classifiers_invalid(mock_echo):
    """Test accelerator validation with invalid classifiers."""
    classifiers = [
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: Invalid GPU",  # Invalid
        "Programming Language :: Python :: 3",
    ]
    result = validate_and_extract_accelerator_classifiers(classifiers)
    assert result == []
    mock_echo.assert_called_once()
    assert "Invalid Environment classifier found" in mock_echo.call_args[0][0]


def test_validate_version_valid():
    """Test version validation with valid versions."""
    valid_versions = [
        "1.1.1",
        ">=1.0.0",
        "==2.1.0-beta",
        "1.5.2",
        "~=3.0.0",
        "!=1.2.3",
        ">2.0.0",
        "<3.0.0",
        "<=4.0.0",
        "<>1.0.0",
        "=1.0.0",
        "1.0.0-alpha1",
        ">=1.0.0,<2.0.0",
        "==1.2.3,!=1.2.4",
        ">=1.0.0,<=2.0.0,!=1.5.0",
        "1.0.0,2.0.0",
        ">1.0.0,<2.0.0,!=1.5.0-beta",
    ]

    for version in valid_versions:
        result = validate_version(version, "test_field")
        assert result == version, f"Version {version} should be valid"


@patch("typer.echo")
def test_validate_version_invalid(mock_echo):
    """Test version validation with invalid versions."""
    invalid_versions = [
        "1.0",  # Missing patch version
        ">=abc",  # Invalid version format
        "invalid-version",  # Completely invalid
        "1.0.0.0",  # Too many version parts
        ">>1.0.0",  # Invalid operator
        ">=1.0.0,invalid",
        "1.0,2.0.0",
        ">=1.0.0,>=abc",
    ]

    for version in invalid_versions:
        result = validate_version(version, "test_field")
        assert result == "", f"Version {version} should be invalid"

    assert mock_echo.call_count == len(invalid_versions)
