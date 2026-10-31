#!/usr/bin/env python3
"""Set the matching PyPI versions for comfy-cli and its optional docs pack."""

from __future__ import annotations

import sys
from pathlib import Path

import tomlkit

ROOT = Path(__file__).resolve().parents[1]
PACK_PROJECT = ROOT / "packages" / "comfy-cli-docs-search" / "pyproject.toml"


def set_release_versions(root: Path, version: str) -> None:
    root_path = root / "pyproject.toml"
    pack_path = root / "packages" / "comfy-cli-docs-search" / "pyproject.toml"
    root_doc = tomlkit.parse(root_path.read_text(encoding="utf-8"))
    root_doc["project"]["version"] = version
    dependencies = root_doc["project"]["optional-dependencies"]["docs-search"]
    for index, dependency in enumerate(dependencies):
        if dependency.startswith("comfy-cli-docs-search=="):
            dependencies[index] = f"comfy-cli-docs-search=={version}"
            break
    else:
        raise ValueError("docs-search extra is missing its companion package pin")
    pack_doc = tomlkit.parse(pack_path.read_text(encoding="utf-8"))
    pack_doc["project"]["version"] = version
    root_path.write_text(tomlkit.dumps(root_doc), encoding="utf-8")
    pack_path.write_text(tomlkit.dumps(pack_doc), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/set_release_version.py VERSION", file=sys.stderr)
        return 2
    version = sys.argv[1].removeprefix("v")
    if not version or any(char.isspace() for char in version):
        print("release version must be a non-empty version token", file=sys.stderr)
        return 2

    set_release_versions(ROOT, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
