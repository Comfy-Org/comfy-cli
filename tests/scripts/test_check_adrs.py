from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_adrs.py"
_SPEC = importlib.util.spec_from_file_location("check_adrs", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_SCRIPT}")
check_adrs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_adrs
_SPEC.loader.exec_module(check_adrs)

find_legacy_adr_references = check_adrs.find_legacy_adr_references
validate_adr_directory = check_adrs.validate_adr_directory
validate_references = check_adrs.validate_references

ADR_IDENTIFIER = "-".join(("CLI", "CONFIG", "0001"))
ADR_FILENAME = f"{ADR_IDENTIFIER}-configuration-authority.md"


def write_adr(directory: Path, filename: str = ADR_FILENAME) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    identifier = filename.removesuffix(".md").rsplit("-", maxsplit=2)[0]
    (directory / filename).write_text(
        f"# ADR-{identifier}: Configuration Authority\n\nDate: 2026-10-17\n\n## Status\n\nProposed\n",
        encoding="utf-8",
    )


def test_validate_adr_directory_accepts_minimal_adr_without_recommended_sections(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "TEMPLATE.md").write_text("template", encoding="utf-8")

    validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_numbered_filenames(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "0002-new-decision.md").write_text("# 2. New decision\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ADR filenames"):
        validate_adr_directory(tmp_path)


@pytest.mark.parametrize(
    "filename",
    ["CLI-CONFIG-STORAGE-0001-title.md", "CLI-CONFIG-\u0660\u0660\u0660\u0661-title.md"],
)
def test_validate_adr_directory_rejects_invalid_identifier_shapes(tmp_path: Path, filename: str) -> None:
    write_adr(tmp_path)
    (tmp_path / filename).write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ADR filenames"):
        validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_orphaned_companion_directory(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "-".join(("CLI", "CONFIG", "0002", "missing"))).mkdir()

    with pytest.raises(ValueError, match="ADR companion directory has no matching ADR"):
        validate_adr_directory(tmp_path)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("# Wrong heading\n\nDate: 2026-10-17\n\n## Status\n\nProposed\n", "expected heading"),
        (f"# ADR-{ADR_IDENTIFIER}: Title\n\n## Status\n\nProposed\n", "expected heading"),
        (f"# ADR-{ADR_IDENTIFIER}: Title\n\nDate: 2026-10-17\n\n## Status\n\nDraft\n", "expected heading"),
        (f"# ADR-{ADR_IDENTIFIER}: Title\n\nDate: 2026-13-45\n\n## Status\n\nProposed\n", "invalid Date"),
        (
            f"# ADR-{ADR_IDENTIFIER}: Title\n\nDate: 2026-10-17\n\n## Status\n\nAccepted conditionally\n",
            "expected heading",
        ),
    ],
)
def test_validate_adr_directory_requires_metadata(tmp_path: Path, contents: str, error: str) -> None:
    tmp_path.joinpath(ADR_FILENAME).write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    write_adr(tmp_path)
    write_adr(tmp_path, f"{ADR_IDENTIFIER}-another-title.md")

    with pytest.raises(ValueError, match="Duplicate ADR identifiers"):
        validate_adr_directory(tmp_path)


def test_validate_adr_directory_accepts_status_without_blank_line(tmp_path: Path) -> None:
    tmp_path.joinpath(ADR_FILENAME).write_text(
        f"# ADR-{ADR_IDENTIFIER}: Title\nDate: 2026-10-17\n## Status\n  Accepted\n",
        encoding="utf-8",
    )

    validate_adr_directory(tmp_path)


@pytest.mark.parametrize(
    "reference",
    [
        "See ADR-" + "0008 for details",
        "This ADR (" + "0008) defines the contract",
        "See docs/adr/" + "0008-old-name.md",
    ],
)
def test_find_legacy_adr_references(reference: str) -> None:
    assert find_legacy_adr_references(reference) == [(1, reference)]


def test_find_legacy_adr_references_ignores_amendment_dates() -> None:
    assert find_legacy_adr_references(f"ADR-{ADR_IDENTIFIER} amendment (2026-10-17)") == []


def test_validate_references_rejects_dangling_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "AGENTS.md"
    source.write_text(f"See docs/adr/{ADR_IDENTIFIER}-missing.md\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_files", lambda _root: [source])

    with pytest.raises(ValueError, match="Dangling docs/adr references"):
        validate_references(tmp_path)


def test_validate_references_accepts_existing_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adr = tmp_path / "docs" / "adr" / ADR_FILENAME
    adr.parent.mkdir(parents=True)
    adr.write_text("record", encoding="utf-8")
    source = tmp_path / "AGENTS.md"
    source.write_text(f"See {adr.relative_to(tmp_path)}\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_files", lambda _root: [source, adr])

    validate_references(tmp_path)


def test_validate_references_ignores_external_adr_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "README.md"
    external_path = "docs/" + "adr/SUBGRAPH-PROMOTION-0009-represent-promoted-widgets-as-linked-inputs.md"
    source.write_text(
        f"https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/{external_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_adrs, "tracked_files", lambda _root: [source])

    validate_references(tmp_path)


def test_validate_references_checks_relative_links_from_adrs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adr = tmp_path / "docs" / "adr" / ADR_FILENAME
    adr.parent.mkdir(parents=True)
    adr.write_text("See [missing](MISSING-0002-record.md)\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_files", lambda _root: [adr])

    with pytest.raises(ValueError, match="MISSING-0002-record.md"):
        validate_references(tmp_path)


def test_validate_references_rejects_adr_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "README.md"
    traversal = "docs/adr/" + "../../README.md"
    source.write_text(f"{traversal}\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_files", lambda _root: [source])

    with pytest.raises(ValueError, match=re.escape(traversal)):
        validate_references(tmp_path)
