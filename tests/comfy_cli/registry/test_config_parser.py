import subprocess
from unittest.mock import mock_open, patch

import pytest
import tomlkit

from comfy_cli.registry.config_parser import (
    _strip_url_credentials,
    extract_node_configuration,
    initialize_project_config,
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


@pytest.mark.parametrize(
    "license_str",
    ["MIT", "Apache-2.0", "GPL-3.0-or-later", "MIT License"],
)
def test_extract_node_configuration_license_spdx_string(license_str):
    mock_data = {
        "project": {
            "license": license_str,
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
        assert result.project.license == License(text=license_str)


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


def _write_pyproject(tmp_path, body: str) -> str:
    """Write a pyproject.toml in tmp_path and return its absolute path as a string."""
    p = tmp_path / "pyproject.toml"
    p.write_text(body)
    return str(p)


def test_dynamic_version_resolved_from_double_quoted_literal(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text('__version__ = "1.2.3"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "pkg/__init__.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"


def test_dynamic_version_resolved_from_VERSION_name(tmp_path):
    (tmp_path / "_version.py").write_text('VERSION = "2.0.0"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_version.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "2.0.0"


def test_dynamic_version_resolved_from_single_quotes(tmp_path):
    (tmp_path / "_v.py").write_text("__version__ = '0.9.1'\n")
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "0.9.1"


def test_dynamic_version_resolved_with_type_annotation(tmp_path):
    (tmp_path / "_v.py").write_text('__version__: str = "3.4.5"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "3.4.5"


def test_dynamic_version_ignores_commented_line(tmp_path):
    # The `^` anchor ensures a commented-out line is not matched.
    (tmp_path / "_v.py").write_text('# __version__ = "9.9.9"\n__version__ = "1.0.0"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.0.0"


def test_dynamic_version_first_match_wins(tmp_path):
    (tmp_path / "_v.py").write_text('__version__ = "1.0.0"\n__version__ = "2.0.0"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.0.0"


def test_static_version_wins_over_tool_comfy_version(tmp_path):
    # Defensive: if a user accidentally has both, the static `project.version`
    # wins without ever reading the file (no warning, no resolution).
    (tmp_path / "_v.py").write_text('__version__ = "9.9.9"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\nversion = "1.0.0"\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.0.0"


@patch("typer.echo")
def test_dynamic_version_without_tool_comfy_version_warns(mock_echo, tmp_path):
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\ndynamic = ["version"]\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("[tool.comfy.version].path" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_absolute_path_rejected(mock_echo, tmp_path):
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "/etc/passwd"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("must be relative" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_windows_absolute_path_rejected(mock_echo, tmp_path):
    # Ensure a Windows-style absolute path is also rejected when tests run
    # on POSIX (and vice versa) — the check is OS-agnostic.
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = \'C:\\Windows\\version.py\'\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("must be relative" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_path_traversal_rejected(mock_echo, tmp_path):
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "../../etc/passwd"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("inside the project directory" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_missing_file_warns(mock_echo, tmp_path):
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "does_not_exist.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("could not read" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_no_match_warns(mock_echo, tmp_path):
    (tmp_path / "_v.py").write_text('other_var = "1.2.3"\nsome_other = 42\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("could not find" in str(c) for c in mock_echo.call_args_list)


def test_dynamic_version_handles_utf8_bom(tmp_path):
    # Windows editors that write a UTF-8 BOM must not defeat the `^` anchor.
    (tmp_path / "_v.py").write_bytes(b'\xef\xbb\xbf__version__ = "1.2.3"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"


@patch("typer.echo")
def test_dynamic_version_invalid_utf8_warns(mock_echo, tmp_path):
    # Non-UTF-8 content must not crash the parser (UnicodeDecodeError is a
    # ValueError, not an OSError — must be caught explicitly).
    (tmp_path / "_v.py").write_bytes(b"\xff\xfe\x00\x00garbage bytes")
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("could not read" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_scalar_tool_comfy_version_warns(mock_echo, tmp_path):
    # User misplaced a scalar version under [tool.comfy] instead of [project].
    # Warning should name the shape problem, not the path problem.
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy]\nversion = "1.2.3"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("must be a table" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_malformed_dynamic_scalar_string_warns(mock_echo, tmp_path):
    # User wrote `dynamic = "version"` (scalar) instead of `dynamic = ["version"]`.
    # Silent-skip would leave them confused; warn explicitly.
    (tmp_path / "_v.py").write_text('__version__ = "1.0.0"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = "version"\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("must be an array of strings" in str(c) for c in mock_echo.call_args_list)


def test_dynamic_version_indented_only_does_not_match(tmp_path):
    # Regex anchor `^` must reject indented `__version__` assignments (inside
    # classes/functions). File has ONLY the indented form — expect no match
    # and empty version.
    (tmp_path / "_v.py").write_text('class Foo:\n    __version__ = "1.2.3"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""


def test_dynamic_version_trailing_inline_comment_resolves(tmp_path):
    # `__version__ = "1.2.3"  # stable` must resolve to "1.2.3" (regex stops
    # capture at the closing quote, trailing comment ignored).
    (tmp_path / "_v.py").write_text('__version__ = "1.2.3"  # stable release\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"


@patch("typer.echo")
def test_dynamic_version_path_is_directory_warns(mock_echo, tmp_path):
    # `path` pointing at a directory must degrade gracefully (IsADirectoryError
    # is an OSError subclass) and surface a "could not read" warning.
    (tmp_path / "subdir").mkdir()
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "subdir"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("could not read" in str(c) for c in mock_echo.call_args_list)


def test_padded_static_version_is_stripped(tmp_path):
    # Static `version = "  1.0.0  "` must be normalized — registries should not
    # receive whitespace padding.
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\nversion = "  1.0.0  "\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.0.0"


@patch("typer.echo")
def test_dynamic_version_non_string_path_warns_as_type_error(mock_echo, tmp_path):
    # A non-string `path` value must produce a type warning, not a misleading
    # "could not read file `42`" OS error.
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = 42\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("must be a string" in str(c) for c in mock_echo.call_args_list)
    # Must NOT fall through to an OS "could not read" warning
    assert not any("could not read" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_static_version_happy_path_emits_no_version_warnings(mock_echo, tmp_path):
    # Regression guard for the common case: static version, no dynamic, no
    # [tool.comfy.version]. Must not emit ANY version/dynamic-related warning
    # (only unrelated warnings like the pre-existing "License..." one are allowed).
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\nversion = "1.2.3"\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"
    noisy = [
        str(c)
        for c in mock_echo.call_args_list
        if "version" in str(c).lower() or "dynamic" in str(c).lower() or "tool.comfy" in str(c).lower()
    ]
    assert noisy == [], f"Unexpected version/dynamic warnings on happy path: {noisy}"


# --- Fix K: non-dict `project` / `tool` degrade gracefully ---


def test_malformed_toml_does_not_crash(tmp_path):
    # Invalid TOML (syntax error) must not crash the parser — scanning
    # contexts would lose the whole pack inventory otherwise.
    (tmp_path / "pyproject.toml").write_text('[project\nname = "x"\n')  # missing `]`
    result = extract_node_configuration(str(tmp_path / "pyproject.toml"))
    assert result is None  # graceful None return, no exception


def test_pyproject_with_utf8_bom_parses_successfully(tmp_path):
    # Windows editors (e.g., Notepad with legacy settings, Visual Studio) write
    # a UTF-8 BOM on save. `encoding="utf-8-sig"` strips it transparently; without
    # this, tomlkit sees `﻿` as the first character and reports a cryptic
    # `Empty key at line 1 col 0`.
    (tmp_path / "pyproject.toml").write_bytes(b'\xef\xbb\xbf[project]\nname = "x"\nversion = "1.0.0"\n')
    result = extract_node_configuration(str(tmp_path / "pyproject.toml"))
    assert result is not None
    assert result.project.name == "x"
    assert result.project.version == "1.0.0"


def test_pyproject_with_invalid_utf8_returns_none_gracefully(tmp_path):
    # `UnicodeDecodeError` is a `ValueError`, not an `OSError`, so it must be in
    # the except tuple explicitly. Without it, a pyproject.toml with non-UTF-8
    # bytes raises a raw traceback instead of the friendly error shown for every
    # other file-read failure.
    (tmp_path / "pyproject.toml").write_bytes(b'[project]\nname = "x"\nx = "\xff\xfe garbage"\n')
    result = extract_node_configuration(str(tmp_path / "pyproject.toml"))
    assert result is None


@patch("typer.echo")
def test_static_version_non_string_scalar_rejected(mock_echo, tmp_path):
    # PEP 621 requires `version` to be a string. A non-string scalar must now
    # produce a typed warning, not be silently coerced via `str()`.
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\nversion = 1\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("`project.version` must be a string" in s for s in call_strs)


@patch("typer.echo")
def test_static_version_array_rejected(mock_echo, tmp_path):
    # Without the type check, `str(['1', '2'])` would POST `"['1', '2']"` to
    # the registry. Must be rejected.
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\nversion = ["1", "2"]\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("`project.version` must be a string" in s for s in call_strs)


@patch("typer.echo")
def test_static_version_inline_table_rejected(mock_echo, tmp_path):
    # Users conflating PEP 621 static and our `[tool.comfy.version]` might write
    # `version = { path = "_v.py" }`. Without the type check this POSTed as
    # `"{'path': '_v.py'}"`. Catch it up front.
    path = _write_pyproject(tmp_path, '[project]\nname = "x"\nversion = { path = "_v.py" }\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("`project.version` must be a string" in s for s in call_strs)


@patch("typer.echo")
def test_project_scalar_at_root_does_not_crash(mock_echo, tmp_path):
    # Malformed TOML: `project = "hello"` at root. Must not crash — used to
    # raise AttributeError: 'String' object has no attribute 'get'.
    path = _write_pyproject(tmp_path, 'project = "hello"\n[tool.comfy]\nPublisherId = "x"\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("`project` in pyproject.toml must be a table" in str(c) for c in mock_echo.call_args_list)


@pytest.mark.parametrize("value", ["0", "0.0", "false", "[]", "{}"])
@patch("typer.echo")
def test_static_version_falsy_non_string_rejected(mock_echo, tmp_path, value):
    # Regression guard: the type check must fire for FALSY non-strings too
    # (`version = 0`, `version = 0.0`, `version = false`, `version = []`,
    # `version = {}`). Earlier the truthy check (`if static_version:`) gated
    # the isinstance check, so these silently fell through to the dynamic
    # branch and the user only saw the downstream "project version is empty"
    # error.
    path = _write_pyproject(tmp_path, f'[project]\nname = "x"\nversion = {value}\n')
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("`project.version` must be a string" in s for s in call_strs), (
        f"value={value}: no type warning; saw {call_strs}"
    )


# --- Fix A (padded-static): strip semantics, already tested; add padded-dynamic variant ---


def test_dynamic_version_padded_literal_is_stripped(tmp_path):
    # `__version__ = "  1.2.3  "` — the `.strip()` in _resolve_dynamic_version
    # already handles this, pinning the behavior here for regression safety.
    (tmp_path / "_v.py").write_text('__version__ = "  1.2.3  "\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"


# --- Fix N: falsy-but-typed path values must trigger the type warning ---


@patch("typer.echo")
def test_dynamic_version_empty_path_string_warns_as_not_set(mock_echo, tmp_path):
    # `path = ""` explicitly set to empty string — equivalent to unset.
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = ""\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("path` is not set" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_dynamic_version_table_missing_path_key_warns_as_not_set(mock_echo, tmp_path):
    # `[tool.comfy.version]` table exists but has no `path` key.
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("path` is not set" in str(c) for c in mock_echo.call_args_list)


@patch("typer.echo")
def test_falsy_nonstring_path_values_warn_as_type_mismatch(mock_echo, tmp_path):
    # `path = 0 / false / [] / {}` are all truthy-falsy edge cases. They must
    # produce a "must be a string" warning, not the misleading "path is not set".
    for falsy in ["0", "false", "[]", "{}"]:
        mock_echo.reset_mock()
        path = _write_pyproject(
            tmp_path,
            f'[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = {falsy}\n',
        )
        result = extract_node_configuration(path)
        assert result is not None
        assert result.project.version == ""
        call_strs = [str(c) for c in mock_echo.call_args_list]
        assert any("must be a string" in s for s in call_strs), f"falsy={falsy}: no type warning"
        assert not any("path` is not set" in s for s in call_strs), f"falsy={falsy}: got misleading 'not set' warning"


# --- Backslash in value: regex rejects, surfaces as "could not find" ---


@patch("typer.echo")
def test_dynamic_version_backslash_in_value_not_matched(mock_echo, tmp_path):
    # The regex excludes `\` from the value class entirely, so any escape
    # sequence in the literal causes the regex to fail to match. Users get
    # a clear "could not find" warning rather than a silently misinterpreted
    # value. PEP 440 versions are ASCII-only so this is a clean fail-closed
    # contract; users with auto-generated `__version__` containing escapes
    # must clean up their source.
    (tmp_path / "_v.py").write_text('__version__ = "1.0\\n"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    assert any("could not find" in str(c) for c in mock_echo.call_args_list)


# --- H2 (Round 5): adjacent-string-literal concatenation detection ---


@patch("typer.echo")
def test_dynamic_version_adjacent_literals_double_quote_warns(mock_echo, tmp_path):
    # Python evaluates `"1." "2.3"` as `"1.2.3"` via implicit concatenation.
    # Without this check, the regex captures only `"1."` and we silently POST
    # the wrong version. The same-line look-ahead rejects concatenation so the
    # publish-layer guard exits 1 instead of shipping `"1."` to the registry.
    (tmp_path / "_v.py").write_text('__version__ = "1." "2.3"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("adjacent-string-literal concatenation" in s for s in call_strs), (
        f"no concatenation warning; saw {call_strs}"
    )


@patch("typer.echo")
def test_dynamic_version_adjacent_literals_no_whitespace_warns(mock_echo, tmp_path):
    # Python accepts `"1.""2.3"` (no whitespace between) — still concatenation.
    (tmp_path / "_v.py").write_text('__version__ = "1.""2.3"\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("adjacent-string-literal concatenation" in s for s in call_strs)


@patch("typer.echo")
def test_dynamic_version_adjacent_literals_single_quote_warns(mock_echo, tmp_path):
    # Detection must fire for single-quoted literals too.
    (tmp_path / "_v.py").write_text("__version__ = '1.' '2.3'\n")
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("adjacent-string-literal concatenation" in s for s in call_strs)


@patch("typer.echo")
def test_dynamic_version_adjacent_literals_mixed_quotes_warns(mock_echo, tmp_path):
    # Python allows mixing quote styles across adjacent literals: `"1." '2.3'`
    # evaluates to `"1.2.3"`. The check must inspect for ANY quote, not just
    # the matched quote, so a `"..." '...'` or `'...' "..."` pair still fires.
    (tmp_path / "_v.py").write_text("__version__ = \"1.\" '2.3'\n")
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == ""
    call_strs = [str(c) for c in mock_echo.call_args_list]
    assert any("adjacent-string-literal concatenation" in s for s in call_strs)


def test_dynamic_version_semicolon_after_literal_still_resolves(tmp_path):
    # Negative control: `; x = 1` on the same line must NOT trigger the
    # concatenation check (`;` isn't a quote). Pins the narrow look-ahead
    # so a future broadening to "any trailing content" can't silently
    # regress this case.
    (tmp_path / "_v.py").write_text('__version__ = "1.2.3"; x = 1\n')
    path = _write_pyproject(
        tmp_path,
        '[project]\nname = "x"\ndynamic = ["version"]\n\n[tool.comfy.version]\npath = "_v.py"\n',
    )
    result = extract_node_configuration(path)
    assert result is not None
    assert result.project.version == "1.2.3"


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


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/user/repo.git", "https://github.com/user/repo.git"),
        ("https://ghp_xxxx@github.com/user/repo.git", "https://github.com/user/repo.git"),
        ("https://user:ghp_xxxx@github.com/user/repo.git", "https://github.com/user/repo.git"),
        ("https://oauth2:token@gitlab.com:8443/user/repo.git", "https://gitlab.com:8443/user/repo.git"),
        ("git@github.com:user/repo.git", "git@github.com:user/repo.git"),
        ("https://user:@github.com/user/repo.git", "https://github.com/user/repo.git"),
        ("https://:pass@github.com/user/repo.git", "https://github.com/user/repo.git"),
        ("http://token@example.com/repo.git", "http://example.com/repo.git"),
        ("https://user:pass@[::1]:8080/repo.git", "https://[::1]:8080/repo.git"),
        ("git://github.com/user/repo.git", "git://github.com/user/repo.git"),
        ("https://github.com:443/user/repo.git", "https://github.com:443/user/repo.git"),
        ("ssh://git@github.com/user/repo.git", "ssh://git@github.com/user/repo.git"),
    ],
)
def test_strip_url_credentials(url, expected):
    assert _strip_url_credentials(url) == expected


def test_initialize_project_config_strips_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://ghp_secret@github.com/user/ComfyUI-MyNode.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    urls = data["project"]["urls"]
    assert urls["Repository"] == "https://github.com/user/ComfyUI-MyNode"
    assert urls["Documentation"] == "https://github.com/user/ComfyUI-MyNode/wiki"
    assert urls["Bug Tracker"] == "https://github.com/user/ComfyUI-MyNode/issues"
    assert "ghp_secret" not in tomlkit.dumps(data)


def test_initialize_project_config_clean_https(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/user/ComfyUI-MyNode.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    urls = data["project"]["urls"]
    assert urls["Repository"] == "https://github.com/user/ComfyUI-MyNode"
    assert urls["Documentation"] == "https://github.com/user/ComfyUI-MyNode/wiki"
    assert urls["Bug Tracker"] == "https://github.com/user/ComfyUI-MyNode/issues"


def test_initialize_project_config_ssh_remote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:user/ComfyUI-TestNode.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    urls = data["project"]["urls"]
    assert urls["Repository"] == "https://github.com/user/ComfyUI-TestNode"
    assert urls["Documentation"] == "https://github.com/user/ComfyUI-TestNode/wiki"
    assert urls["Bug Tracker"] == "https://github.com/user/ComfyUI-TestNode/issues"
    assert data["project"]["name"] == "testnode"
    assert data["tool"]["comfy"]["DisplayName"] == "ComfyUI-TestNode"


# Issue #431: requirements.txt → pyproject.toml migration must produce
# valid PEP 508 dependency specifiers. Inline comments, full-line comments,
# and pip-specific options (-r, -e, --index-url, ...) are not valid deps.


def _init_git_repo_with_reqs(tmp_path, requirements_content: str) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/user/ComfyUI-TestNode.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "requirements.txt").write_text(requirements_content)


def test_initialize_project_config_strips_inline_comments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_git_repo_with_reqs(
        tmp_path,
        "matplotlib>=3.3.0  # For visualization\nnumpy>=1.0 # trailing\n",
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert deps == ["matplotlib>=3.3.0", "numpy>=1.0"]


def test_initialize_project_config_skips_full_line_comments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_git_repo_with_reqs(
        tmp_path,
        "# heading comment\nfoo>=1.0\n  # indented comment\nbar\n",
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert deps == ["foo>=1.0", "bar"]


def test_initialize_project_config_skips_pip_options(tmp_path, monkeypatch, capsys):
    # `-r`, `-e`, `-c`, `--index-url`, `--extra-index-url`, `--find-links`
    # are pip-requirements-file syntax, not PEP 508 dep specifiers. They must
    # not land in [project.dependencies] where downstream build tools will
    # error trying to parse them. Each skipped line must also produce a
    # visible warning so silent data loss is avoided.
    monkeypatch.chdir(tmp_path)
    _init_git_repo_with_reqs(
        tmp_path,
        "-r other.txt\n"
        "-e .\n"
        "--index-url https://pypi.org/simple\n"
        "--extra-index-url https://example.com/simple\n"
        "--find-links ./local-wheels\n"
        "foo>=1.0\n",
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert deps == ["foo>=1.0"]
    out = capsys.readouterr().out
    for dropped in ["-r other.txt", "-e .", "--index-url", "--extra-index-url", "--find-links"]:
        assert dropped in out, f"missing skip warning for {dropped!r}"


def test_initialize_project_config_preserves_vcs_subdirectory_fragment(tmp_path, monkeypatch):
    # Regression guard against a naive `split("#")[0]` fix — VCS fragments
    # must survive because `#` is only a comment marker when preceded by
    # whitespace (pip's rule).
    monkeypatch.chdir(tmp_path)
    _init_git_repo_with_reqs(
        tmp_path,
        "git+https://github.com/org/mono.git#subdirectory=pkg\n",
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert deps == ["git+https://github.com/org/mono.git#subdirectory=pkg"]


def test_initialize_project_config_vcs_with_inline_comment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_git_repo_with_reqs(
        tmp_path,
        "git+https://github.com/org/mono.git#subdirectory=pkg  # monorepo dep\n",
    )
    initialize_project_config()
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    deps = [str(d) for d in data["project"]["dependencies"]]
    assert deps == ["git+https://github.com/org/mono.git#subdirectory=pkg"]
