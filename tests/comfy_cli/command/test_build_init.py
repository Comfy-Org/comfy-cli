from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, read_build_spec


@pytest.fixture(autouse=True)
def stable_command_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build,
        "capture_pip_provenance",
        lambda python: {
            "pipDependencies": "example==1.0.0\n",
            "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
        },
    )


@pytest.fixture
def install(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "checkpoints" / "base.safetensors").write_bytes(b"MODEL")
    (root / "custom_nodes" / "local-node").mkdir(parents=True)
    (root / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"NODE")
    return root


def _invoke(runner: CliRunner, args: list[str]):
    return runner.invoke(
        cli_app,
        ["build", "init", *args],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )


def _success_args(root: Path, *, name: str = "Foo") -> list[str]:
    return [
        "--name",
        name,
        "--python",
        sys.executable,
        "--comfy-version",
        "0.3.0",
        str(root),
    ]


def _envelope(result) -> dict:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    return json.loads(lines[-1])


def _definition(path: Path) -> JsonObject:
    definition = read_build_spec(path)["definition"]
    assert isinstance(definition, dict)
    return definition


def _first_entry(definition: JsonObject, collection: str) -> JsonObject:
    entries = definition[collection]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    return entry


def test_plain_init_writes_canonical_local_spec_without_builder_client(
    install: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))

    # When
    result = _invoke(CliRunner(), _success_args(install))

    # Then
    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["command"] == "build init"
    assert envelope["changed"] is True
    spec = read_build_spec(install / "comfy-build.yaml")
    assert spec["id"] is None
    assert spec["name"] == "Foo"
    assert spec["syncedRevision"] is None
    definition = _definition(install / "comfy-build.yaml")
    models = definition["models"]
    nodes = definition["customNodes"]
    assert isinstance(models, list)
    assert isinstance(nodes, list)
    assert models == build.scan_models(install / "models")
    assert nodes == build.scan_custom_nodes(install / "custom_nodes")
    node = _first_entry(definition, "customNodes")
    assert node["source"] == "local"
    local_digest = node["localDigest"]
    local_size = node["localSizeBytes"]
    assert isinstance(local_digest, str)
    assert isinstance(local_size, int)
    assert len(local_digest) == 64
    assert local_size > 0


def test_init_refuses_to_clobber_an_existing_spec(install: Path) -> None:
    # Given
    runner = CliRunner()
    first = _invoke(runner, _success_args(install))
    assert first.exit_code == 0
    spec_path = install / "comfy-build.yaml"
    before = spec_path.read_bytes()

    # When
    result = _invoke(runner, _success_args(install, name="Changed"))

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_spec_exists"
    assert spec_path.read_bytes() == before


def test_init_force_overwrites_deterministically(install: Path) -> None:
    # Given
    runner = CliRunner()
    assert _invoke(runner, _success_args(install)).exit_code == 0

    # When
    first = _invoke(runner, ["--force", *_success_args(install, name="Changed")])
    first_bytes = (install / "comfy-build.yaml").read_bytes()
    second = _invoke(runner, ["--force", *_success_args(install, name="Changed")])

    # Then
    assert first.exit_code == second.exit_code == 0
    assert read_build_spec(install / "comfy-build.yaml")["name"] == "Changed"
    assert (install / "comfy-build.yaml").read_bytes() == first_bytes


def test_output_elsewhere_does_not_change_local_paths(install: Path, tmp_path: Path) -> None:
    # Given
    output = tmp_path / "elsewhere" / "build.yaml"

    # When
    result = _invoke(CliRunner(), ["--output", str(output), *_success_args(install)])

    # Then
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert not (install / "comfy-build.yaml").exists()
    definition = _definition(output)
    assert _first_entry(definition, "models")["localPath"] == "checkpoints/base.safetensors"
    assert _first_entry(definition, "customNodes")["localPath"] == "local-node"


def test_split_layout_uses_each_entry_scan_root(tmp_path: Path) -> None:
    # Given
    install_root = tmp_path / "spec-home"
    models_root = tmp_path / "model-storage"
    nodes_root = tmp_path / "node-storage"
    (models_root / "loras").mkdir(parents=True)
    (models_root / "loras" / "x.safetensors").write_bytes(b"MODEL")
    (nodes_root / "pack").mkdir(parents=True)
    (nodes_root / "pack" / "node.py").write_bytes(b"NODE")

    # When
    result = _invoke(
        CliRunner(),
        [
            "--models-dir",
            str(models_root),
            "--custom-nodes-dir",
            str(nodes_root),
            *_success_args(install_root),
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    definition = _definition(install_root / "comfy-build.yaml")
    assert _first_entry(definition, "models")["localPath"] == "loras/x.safetensors"
    assert _first_entry(definition, "customNodes")["localPath"] == "pack"


def test_externally_symlinked_models_root_is_supported(tmp_path: Path) -> None:
    # Given
    install_root = tmp_path / "install"
    external = tmp_path / "external-models"
    (external / "vae").mkdir(parents=True)
    (external / "vae" / "ae.safetensors").write_bytes(b"MODEL")
    install_root.mkdir()
    linked_root = install_root / "linked-models"
    os.symlink(external, linked_root)

    # When
    result = _invoke(CliRunner(), ["--models-dir", str(linked_root), *_success_args(install_root)])

    # Then
    assert result.exit_code == 0, result.output
    definition = _definition(install_root / "comfy-build.yaml")
    assert _first_entry(definition, "models")["localPath"] == "vae/ae.safetensors"


def test_nested_model_directory_symlink_keeps_lexical_path(tmp_path: Path) -> None:
    # Given
    install_root = tmp_path / "install"
    models_root = install_root / "models"
    external = tmp_path / "external-loras"
    models_root.mkdir(parents=True)
    external.mkdir()
    (external / "x.safetensors").write_bytes(b"MODEL")
    os.symlink(external, models_root / "loras")

    # When
    result = _invoke(CliRunner(), _success_args(install_root))

    # Then
    assert result.exit_code == 0, result.output
    definition = _definition(install_root / "comfy-build.yaml")
    assert _first_entry(definition, "models")["localPath"] == "loras/x.safetensors"


@pytest.mark.parametrize("local_path", [r"C:\evil", r"\\host\share\evil", r"a\..\..\evil", "../evil"])
def test_local_path_rejects_cross_platform_traversal(tmp_path: Path, local_path: str) -> None:
    # Given / When / Then
    with pytest.raises(BuildSpecInvalidError, match="model entry") as error:
        build.resolve_local_path(tmp_path, local_path, entry="model entry")
    assert error.value.code == "build_spec_invalid"


def test_missing_name_is_agentic_build_missing_input(install: Path) -> None:
    # Given
    args = ["--python", sys.executable, "--comfy-version", "0.3.0", str(install)]

    # When
    result = _invoke(CliRunner(), args)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["missing"] == ["--name"]


def test_unresolvable_python_is_agentic_build_missing_input(install: Path) -> None:
    # Given
    missing_python = install / "missing-python"

    # When
    result = _invoke(
        CliRunner(),
        ["--name", "Foo", "--python", str(missing_python), "--comfy-version", "0.3.0", str(install)],
    )

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["missing"] == ["--python"]
    assert not (install / "comfy-build.yaml").exists()
