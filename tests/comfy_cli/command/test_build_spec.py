from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from comfy_cli.command import build_spec
from comfy_cli.command.build_spec import (
    BuildSpecInvalidError,
    BuildSpecWriteError,
    canonicalize_build_spec,
    read_build_spec,
    serialize_build_spec,
    write_build_spec,
)


def _spec() -> dict:
    return {
        "schema": "comfy-build/1",
        "id": "bld_123",
        "name": "Studio build",
        "description": "Reference pipeline",
        "syncedRevision": "revision-7",
        "definition": {
            "schema": "distribution-definition/0",
            "baseComfyVersion": "v0.3.40",
            "models": [
                {"type": "loras", "filename": "z.safetensors", "sha256": "bbb", "sizeBytes": 2},
                {"type": "checkpoints", "filename": "a.safetensors", "sha256": "aaa", "sizeBytes": 1},
            ],
            "customNodes": [
                {"name": "Zeta", "id": "zeta", "repository": "https://example.com/zeta"},
                {"name": "Alpha", "id": "alpha", "repository": "https://example.com/alpha"},
            ],
            "pipDependencies": "torch==2.4.0\ntorchvision==0.19.0",
            "environment": {"python": "3.12", "platform": "linux-x86_64"},
        },
    }


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_round_trip_reproduces_canonical_yaml_bytes(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    canonical = serialize_build_spec(_spec(), format="yaml")
    path.write_text(canonical, encoding="utf-8")

    # When
    write_build_spec(path, read_build_spec(path))

    # Then
    assert path.read_bytes() == canonical.encode()


def test_permutations_at_nested_mapping_and_entry_depths_are_byte_identical() -> None:
    # Given
    first = _spec()
    first["definition"]["future"] = {"zeta": 2, "alpha": {"right": 2, "left": 1}}
    first["definition"]["models"][0]["future"] = [{"z": 2, "a": 1}]
    second = {
        "definition": {
            "future": {"alpha": {"left": 1, "right": 2}, "zeta": 2},
            "environment": {"platform": "linux-x86_64", "python": "3.12"},
            "pipDependencies": "torch==2.4.0\ntorchvision==0.19.0",
            "customNodes": [
                {"repository": "https://example.com/alpha", "id": "alpha", "name": "Alpha"},
                {"repository": "https://example.com/zeta", "id": "zeta", "name": "Zeta"},
            ],
            "models": [
                {"sizeBytes": 1, "sha256": "aaa", "filename": "a.safetensors", "type": "checkpoints"},
                {
                    "future": [{"a": 1, "z": 2}],
                    "sizeBytes": 2,
                    "sha256": "bbb",
                    "filename": "z.safetensors",
                    "type": "loras",
                },
            ],
            "baseComfyVersion": "v0.3.40",
            "schema": "distribution-definition/0",
        },
        "syncedRevision": "revision-7",
        "description": "Reference pipeline",
        "name": "Studio build",
        "id": "bld_123",
        "schema": "comfy-build/1",
    }

    # When
    first_bytes = serialize_build_spec(first, format="yaml")
    second_bytes = serialize_build_spec(second, format="yaml")

    # Then
    assert first_bytes == second_bytes


def test_pip_dependency_line_permutation_produces_different_output() -> None:
    # Given
    first = _spec()
    second = deepcopy(first)
    second["definition"]["pipDependencies"] = "torchvision==0.19.0\ntorch==2.4.0"

    # When
    first_bytes = serialize_build_spec(first, format="yaml")
    second_bytes = serialize_build_spec(second, format="yaml")

    # Then
    assert first_bytes != second_bytes
    assert "torch==2.4.0\n    torchvision==0.19.0" in first_bytes


def test_unknown_nested_mapping_survives_and_is_recursively_ordered(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    spec = _spec()
    spec["definition"]["futureBuilderField"] = {"z": {"two": 2, "one": 1}, "a": 0}
    _write_yaml(path, spec)

    # When
    loaded = read_build_spec(path)
    write_build_spec(path, loaded)

    # Then
    output = path.read_text(encoding="utf-8")
    assert read_build_spec(path)["definition"]["futureBuilderField"] == {
        "a": 0,
        "z": {"one": 1, "two": 2},
    }
    assert output.index("  a: 0") < output.index("  z:")
    assert output.index("    one: 1") < output.index("    two: 2")


def test_metadata_is_preserved_when_only_definition_changes(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    write_build_spec(path, _spec())
    loaded = read_build_spec(path)
    metadata = {key: loaded[key] for key in ("id", "name", "description", "syncedRevision")}

    # When
    loaded["definition"] = {"schema": "distribution-definition/0", "models": [], "customNodes": []}
    write_build_spec(path, loaded)

    # Then
    rewritten = read_build_spec(path)
    assert {key: rewritten[key] for key in metadata} == metadata


@pytest.mark.parametrize(
    ("collection", "entries"),
    [
        ("models", [{"type": None, "filename": "same"}, {"filename": "same", "type": None}]),
        ("customNodes", [{"name": "same", "id": None}, {"id": None, "name": "same"}]),
    ],
)
def test_duplicate_full_sort_tuple_is_rejected(collection: str, entries: list[dict]) -> None:
    # Given
    spec = _spec()
    spec["definition"][collection] = entries

    # When / Then
    with pytest.raises(BuildSpecInvalidError, match=collection) as raised:
        canonicalize_build_spec(spec)
    assert raised.value.code == "build_spec_invalid"


def test_missing_and_null_sort_components_share_empty_string_ordering() -> None:
    # Given
    spec = _spec()
    spec["definition"]["models"] = [
        {"type": None, "filename": "z"},
        {"filename": "a", "sha256": None},
        {"type": "checkpoints", "filename": "m"},
    ]

    # When
    canonical = canonicalize_build_spec(spec)

    # Then
    assert [entry["filename"] for entry in canonical["definition"]["models"]] == ["a", "z", "m"]


def test_unsupported_schema_is_rejected_with_build_spec_invalid(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    path.write_text("schema: comfy-build/2\ndefinition: {}\n", encoding="utf-8")

    # When / Then
    with pytest.raises(BuildSpecInvalidError, match="comfy-build/2") as raised:
        read_build_spec(path)
    assert raised.value.code == "build_spec_invalid"


def test_yaml_anchor_or_alias_is_rejected_on_read(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    path.write_text(
        "schema: comfy-build/1\nid: null\nname: anchored\ndescription: test\nsyncedRevision: null\n"
        "definition:\n  schema: distribution-definition/0\n  environment: &env {python: '3.12'}\n  copied: *env\n",
        encoding="utf-8",
    )

    # When / Then
    with pytest.raises(BuildSpecInvalidError, match="anchor|alias"):
        read_build_spec(path)


def test_writer_emits_no_aliases_for_shared_python_values() -> None:
    # Given
    spec = _spec()
    shared = {"python": "3.12"}
    spec["definition"]["one"] = shared
    spec["definition"]["two"] = shared

    # When
    output = serialize_build_spec(spec, format="yaml")

    # Then
    assert "&id" not in output
    assert "*id" not in output


def test_empty_collections_are_not_elided_to_null() -> None:
    # Given
    spec = _spec()
    spec["definition"]["models"] = []
    spec["definition"]["environment"] = {}
    spec["definition"]["pipDependencies"] = ""

    # When
    output = serialize_build_spec(spec, format="yaml")

    # Then
    assert "models: []" in output
    assert "environment: {}" in output
    assert "pipDependencies: |-" in output


def test_unicode_is_emitted_as_utf8_without_ascii_escaping() -> None:
    # Given
    spec = _spec()
    spec["name"] = "构建 café"
    spec["description"] = "モデル"

    # When
    output = serialize_build_spec(spec, format="yaml")

    # Then
    assert "构建 café" in output
    assert "モデル" in output
    assert "\\u" not in output


def test_lf_single_trailing_newline_and_literal_chomping_are_stable(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    spec = _spec()
    spec["definition"]["pipDependencies"] = "first==1\r\nsecond==2"

    # When
    first = serialize_build_spec(spec, format="yaml")
    path.write_text(first, encoding="utf-8")
    second = serialize_build_spec(read_build_spec(path), format="yaml")

    # Then
    assert "\r" not in first
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert not first.startswith("---")
    assert "pipDependencies: |-" in first
    assert second == first


def test_absent_id_and_synced_revision_are_written_as_null() -> None:
    # Given
    spec = _spec()
    del spec["id"]
    del spec["syncedRevision"]

    # When
    output = serialize_build_spec(spec, format="yaml")

    # Then
    parsed = yaml.safe_load(output)
    assert parsed["id"] is None
    assert parsed["syncedRevision"] is None


def test_json_path_uses_same_known_key_order_and_canonical_newline(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.json"

    # When
    write_build_spec(path, _spec())
    output = path.read_text(encoding="utf-8")
    parsed = json.loads(output)

    # Then
    assert list(parsed)[:6] == ["schema", "id", "name", "description", "syncedRevision", "definition"]
    assert list(parsed["definition"])[:6] == [
        "schema",
        "baseComfyVersion",
        "models",
        "customNodes",
        "pipDependencies",
        "environment",
    ]
    assert output.endswith("\n") and not output.endswith("\n\n")


def test_failed_write_raises_build_spec_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"

    def fail_write(_path: Path, _content: str, *, fsync: bool) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(build_spec, "atomic_write_text", fail_write)

    # When / Then
    with pytest.raises(BuildSpecWriteError, match="disk full") as raised:
        write_build_spec(path, _spec())
    assert raised.value.code == "build_spec_write_error"
