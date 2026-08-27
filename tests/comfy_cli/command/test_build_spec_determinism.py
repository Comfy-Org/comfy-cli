"""Cross-machine no-churn proof for the ``comfy-build/1`` serializer (plan decision D-C).

Todo 6's ``test_build_spec.py`` pins the reader/writer's *behaviour*. This file pins the
one property that makes ``comfy build update`` trustworthy in a shared repository: two
developers with different absolute install roots, usernames and hostnames must produce
byte-identical spec files for the same logical install, so ``git diff`` only ever shows
real content changes.
"""

from __future__ import annotations

import getpass
import re
import socket
from itertools import permutations
from pathlib import Path

import pytest

from comfy_cli.command.build_spec import (
    SpecFormat,
    canonicalize_build_spec,
    read_build_spec,
    serialize_build_spec,
    write_build_spec,
)

# A real-world-shaped requirements body: a PEP-508 pin, a multi-line ``--hash=`` block whose
# continuations are indented, an editable VCS install, and a trailing plain pin. Reordering or
# reflowing any of these lines corrupts the value, so the serializer must treat it as opaque.
_PIP_DEPENDENCIES = (
    "torch==2.4.0\n"
    "some-pkg==1.0.0 \\\n"
    "    --hash=sha256:3b8f1c2d4e5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e \\\n"
    "    --hash=sha256:9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7\n"
    "-e git+https://github.com/example-org/example-node.git@v1.2.3#egg=example-node\n"
    "aiohttp==3.10.5\n"
)

# ``syncedRevision`` is a server-issued opaque token that happens to be timestamp-shaped
# (build design lines 180-198). Keeping it timestamp-shaped is deliberate: it is what makes
# the "no timestamps anywhere else" assertion non-vacuous.
_SYNCED_REVISION = "2026-08-23T12:34:56.789012Z"

_ISO8601 = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'(\[=])(?:[A-Za-z]:)?[\\/][A-Za-z0-9_.\-]+[\\/]")

_SENTINEL_USER = "sentinel-user-9d2f"
_SENTINEL_HOST = "sentinel-host-4c7b"
_SENTINEL_HOME = "/sentinel-home-1a3e"


def _model_sort_key(entry: dict) -> tuple[str, str, str]:
    return (entry.get("type") or "", entry.get("filename") or "", entry.get("sha256") or "")


def _node_sort_key(entry: dict) -> tuple[str, str, str]:
    return (entry.get("name") or "", entry.get("id") or "", entry.get("repository") or "")


def _model(models_root: Path, kind: str, filename: str, sha256: str) -> dict:
    absolute = models_root / kind / filename
    return {
        "type": kind,
        "filename": filename,
        "sha256": sha256,
        "sizeBytes": 4096,
        "source": "local",
        "localPath": absolute.relative_to(models_root).as_posix(),
    }


def _node(nodes_root: Path, name: str, node_id: str | None, repository: str, dirname: str) -> dict:
    absolute = nodes_root / dirname
    return {
        "name": name,
        "id": node_id,
        "repository": repository,
        "source": "local",
        "localPath": absolute.relative_to(nodes_root).as_posix(),
    }


def _nodes(nodes_root: Path) -> list[dict]:
    # "Essentials" appears twice with distinct ids - a legitimate shape per decision D-I that
    # the serializer must order deterministically rather than reject.
    return [
        _node(nodes_root, "Essentials", "acme-essentials", "https://github.com/acme/essentials", "acme-essentials"),
        _node(nodes_root, "Zeta", None, "https://github.com/acme/zeta", "zeta"),
        _node(
            nodes_root, "Essentials", "cubiq-essentials", "https://github.com/cubiq/essentials", "ComfyUI_essentials"
        ),
        _node(nodes_root, "Alpha", "alpha", "https://github.com/acme/alpha", "alpha"),
    ]


def _spec_for_root(root: Path) -> dict:
    """Build the one logical install as an author working under the absolute ``root`` would."""
    models_root = root / "models"
    nodes_root = root / "custom_nodes"
    return {
        "schema": "comfy-build/1",
        "id": "bld_determinism",
        "name": "Determinism fixture",
        "description": "One logical install, two absolute roots",
        "syncedRevision": _SYNCED_REVISION,
        "definition": {
            "schema": "distribution-definition/0",
            "baseComfyVersion": "v0.3.40",
            "models": [
                _model(models_root, "loras", "z.safetensors", "b" * 64),
                _model(models_root, "checkpoints", "a.safetensors", "a" * 64),
                _model(models_root, "checkpoints", "b.safetensors", "c" * 64),
            ],
            "customNodes": _nodes(nodes_root),
            "pipDependencies": _PIP_DEPENDENCIES,
            "environment": {"python": "3.12", "platform": "linux-x86_64"},
        },
    }


def _pip_block(output: str) -> str:
    body = output.split("  pipDependencies: |", 1)[1]
    return body.split("\n  environment:", 1)[0]


@pytest.fixture
def two_roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two genuinely different absolute install roots, differing in name and in depth."""
    first = tmp_path_factory.mktemp("alpha")
    second = tmp_path_factory.mktemp("beta-deeper") / "nested" / "install"
    second.mkdir(parents=True)
    return first, second


@pytest.fixture
def masked_machine_identity(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """Force every machine-identity lookup to a sentinel so a leak is visible on any host."""
    monkeypatch.setattr(getpass, "getuser", lambda: _SENTINEL_USER)
    monkeypatch.setattr(socket, "gethostname", lambda: _SENTINEL_HOST)
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: _SENTINEL_HOST)
    for name in ("USER", "USERNAME", "LOGNAME"):
        monkeypatch.setenv(name, _SENTINEL_USER)
    monkeypatch.setenv("HOSTNAME", _SENTINEL_HOST)
    monkeypatch.setenv("HOME", _SENTINEL_HOME)
    return (_SENTINEL_USER, _SENTINEL_HOST, _SENTINEL_HOME)


def test_the_same_logical_install_serializes_identically_from_two_absolute_roots(
    two_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    root_a, root_b = two_roots
    path_a, path_b = root_a / "comfy-build.yaml", root_b / "comfy-build.yaml"

    # When
    monkeypatch.chdir(root_a)
    write_build_spec(path_a, _spec_for_root(root_a))
    monkeypatch.chdir(root_b)
    write_build_spec(path_b, _spec_for_root(root_b))

    # Then
    assert root_a != root_b
    assert path_a.read_bytes() == path_b.read_bytes()


def test_rendered_bytes_carry_no_absolute_path_or_machine_identity(
    two_roots: tuple[Path, Path], masked_machine_identity: tuple[str, ...]
) -> None:
    # Given
    root_a, root_b = two_roots

    # When
    output = serialize_build_spec(_spec_for_root(root_a), format="yaml")

    # Then
    assert _ABSOLUTE_PATH.search(f"localPath: {root_a.as_posix()}/models") is not None
    assert _ABSOLUTE_PATH.search(output) is None
    for leak in (str(root_a), str(root_b), root_a.name, root_b.parent.parent.name, *masked_machine_identity):
        assert leak not in output


def test_no_iso8601_timestamp_appears_outside_the_synced_revision_value(two_roots: tuple[Path, Path]) -> None:
    # Given
    root_a, _ = two_roots
    output = serialize_build_spec(_spec_for_root(root_a), format="yaml")

    # When
    masked = re.sub(r"(?m)^syncedRevision:.*$", "syncedRevision:", output)

    # Then
    assert _ISO8601.search(output) is not None, "fixture must keep syncedRevision timestamp-shaped"
    assert _SYNCED_REVISION not in masked
    assert _ISO8601.search(masked) is None


def test_model_and_node_order_equal_their_total_sort_keys(two_roots: tuple[Path, Path]) -> None:
    # Given
    root_a, _ = two_roots
    spec = _spec_for_root(root_a)
    models, nodes = spec["definition"]["models"], spec["definition"]["customNodes"]

    # When
    definition = canonicalize_build_spec(spec)["definition"]

    # Then
    assert models != sorted(models, key=_model_sort_key), "fixture must start unsorted"
    assert nodes != sorted(nodes, key=_node_sort_key), "fixture must start unsorted"
    assert definition["models"] == sorted(models, key=_model_sort_key)
    assert definition["customNodes"] == sorted(nodes, key=_node_sort_key)


def test_duplicate_display_names_with_distinct_ids_are_stable_across_every_permutation(
    two_roots: tuple[Path, Path],
) -> None:
    # Given
    root_a, _ = two_roots
    spec = _spec_for_root(root_a)
    orderings = list(permutations(spec["definition"]["customNodes"]))

    # When
    rendered = set()
    for ordering in orderings:
        spec["definition"]["customNodes"] = list(ordering)
        rendered.add(serialize_build_spec(spec, format="yaml"))

    # Then
    assert len(orderings) == 24
    assert len(rendered) == 1
    names = [entry["name"] for entry in canonicalize_build_spec(spec)["definition"]["customNodes"]]
    assert names.count("Essentials") == 2


def test_pip_dependencies_keep_their_original_line_order_through_a_round_trip(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "comfy-build.yaml"
    write_build_spec(path, _spec_for_root(tmp_path))
    first = path.read_text(encoding="utf-8")

    # When
    reloaded = read_build_spec(path)
    write_build_spec(path, reloaded)

    # Then
    assert reloaded["definition"]["pipDependencies"] == _PIP_DEPENDENCIES
    assert path.read_text(encoding="utf-8") == first
    offsets = [first.index(f"    {line}") for line in _PIP_DEPENDENCIES.splitlines()]
    assert offsets == sorted(offsets)


def test_two_independent_constructions_emit_the_same_pip_dependencies(two_roots: tuple[Path, Path]) -> None:
    # Given
    root_a, root_b = two_roots

    # When
    first = _pip_block(serialize_build_spec(_spec_for_root(root_a), format="yaml"))
    second = _pip_block(serialize_build_spec(_spec_for_root(root_b), format="yaml"))

    # Then
    assert first == second
    assert "-e git+https://github.com/example-org/example-node.git@v1.2.3#egg=example-node" in first
    assert first.count("--hash=sha256:") == 2


@pytest.mark.parametrize("spec_format", ["yaml", "json"])
def test_serializing_the_same_spec_twice_in_one_process_is_byte_stable(
    two_roots: tuple[Path, Path], spec_format: SpecFormat
) -> None:
    # Given
    root_a, _ = two_roots
    spec = _spec_for_root(root_a)

    # When
    first = serialize_build_spec(spec, format=spec_format)
    second = serialize_build_spec(spec, format=spec_format)

    # Then
    assert first == second
