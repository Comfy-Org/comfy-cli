from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_adrs
from scripts.check_adrs import find_legacy_adr_references, validate_adr_directory, validate_references


def write_adr(directory: Path, filename: str = "CLI-CONFIG-0001-configuration-authority.md") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    identifier = filename.removesuffix(".md").rsplit("-", maxsplit=2)[0]
    (directory / filename).write_text(
        f"# ADR-{identifier}: Configuration Authority\n\nDate: 2026-10-17\n\n## Status\n\nProposed\n",
        encoding="utf-8",
    )


def test_validate_adr_directory_accepts_identifier_based_adrs(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "TEMPLATE.md").write_text("template", encoding="utf-8")

    validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_numbered_filenames(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "0002-new-decision.md").write_text("# 2. New decision\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ADR filenames"):
        validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_orphaned_companion_directory(tmp_path: Path) -> None:
    write_adr(tmp_path)
    (tmp_path / "CLI-CONFIG-0002-missing").mkdir()

    with pytest.raises(ValueError, match="ADR companion directory has no matching ADR"):
        validate_adr_directory(tmp_path)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("# Wrong heading\n\nDate: 2026-10-17\n\n## Status\n\nProposed\n", "expected heading"),
        ("# ADR-CLI-CONFIG-0001: Title\n\n## Status\n\nProposed\n", "Date field"),
        ("# ADR-CLI-CONFIG-0001: Title\n\nDate: 2026-10-17\n\n## Status\n\nDraft\n", "Status value"),
    ],
)
def test_validate_adr_directory_requires_metadata(tmp_path: Path, contents: str, error: str) -> None:
    tmp_path.joinpath("CLI-CONFIG-0001-configuration-authority.md").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_adr_directory(tmp_path)


def test_validate_adr_directory_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    write_adr(tmp_path)
    write_adr(tmp_path, "CLI-CONFIG-0001-another-title.md")

    with pytest.raises(ValueError, match="Duplicate ADR identifiers"):
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
    assert find_legacy_adr_references("ADR-CLI-CONFIG-0001 amendment (2026-10-17)") == []


def test_validate_references_rejects_dangling_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "AGENTS.md"
    source.write_text("See docs/adr/" + "CLI-CONFIG-0001-missing.md\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_and_untracked_files", lambda _root: [source])

    with pytest.raises(ValueError, match="Dangling docs/adr references"):
        validate_references(tmp_path)


def test_validate_references_accepts_existing_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adr = tmp_path / "docs" / "adr" / "CLI-CONFIG-0001-configuration-authority.md"
    adr.parent.mkdir(parents=True)
    adr.write_text("record", encoding="utf-8")
    source = tmp_path / "AGENTS.md"
    source.write_text(f"See {adr.relative_to(tmp_path)}\n", encoding="utf-8")
    monkeypatch.setattr(check_adrs, "tracked_and_untracked_files", lambda _root: [source, adr])

    validate_references(tmp_path)
