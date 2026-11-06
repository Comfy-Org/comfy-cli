"""Tests that release versions stay aligned across the optional docs pack."""

from __future__ import annotations

import runpy
from pathlib import Path

import tomlkit

ROOT = Path(__file__).resolve().parents[2]
SET_VERSION = runpy.run_path(str(ROOT / "scripts" / "set_release_version.py"))["set_release_versions"]


def test_release_version_updates_cli_pack_and_extra_pin(tmp_path):
    package = tmp_path / "packages" / "comfy-cli-docs-search"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.0.0"\n[project.optional-dependencies]\n'
        'docs-search = ["lancedb==0.25.0", "comfy-cli-docs-search==0.0.0"]\n',
        encoding="utf-8",
    )
    (package / "pyproject.toml").write_text('[project]\nversion = "0.0.0"\n', encoding="utf-8")

    SET_VERSION(tmp_path, "1.2.3")

    cli = tomlkit.parse((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pack = tomlkit.parse((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert cli["version"] == pack["version"] == "1.2.3"
    assert "comfy-cli-docs-search==1.2.3" in cli["optional-dependencies"]["docs-search"]
