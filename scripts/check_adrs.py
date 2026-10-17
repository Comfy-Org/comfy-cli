from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ADR_FILE_PATTERN = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{4})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
ADR_LINK_PATTERN = re.compile(r"docs/adr/[A-Za-z0-9._/-]+\.md")
LEGACY_REFERENCE_PATTERN = re.compile(
    r"ADR(?:[- ]?\d{4}(?!-\d{2})|(?:-[A-Z][A-Z0-9]*)?\s*(?:/\s*|\(\s*)\d{4}(?!-\d{2}))|(?:docs/)?adr/\d{4}-"
)
STATUS_PATTERN = re.compile(
    r"^## Status[ \t]*\n(?:[ \t]*\n)*(Proposed|Accepted|Rejected|Deprecated|Superseded)\b",
    re.MULTILINE,
)
SPECIAL_FILES = {"TEMPLATE.md"}


@dataclass(frozen=True)
class Adr:
    filename: str
    identifier: str


def find_legacy_adr_references(contents: str) -> list[tuple[int, str]]:
    return [
        (line_number, line)
        for line_number, line in enumerate(contents.splitlines(), start=1)
        if LEGACY_REFERENCE_PATTERN.search(line)
    ]


def _metadata(directory: Path, filename: str) -> Adr:
    match = ADR_FILE_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid ADR filename: {filename}")

    identifier = match.group(1)
    contents = (directory / filename).read_text(encoding="utf-8")
    if re.search(rf"^# ADR-{re.escape(identifier)}: .+$", contents, re.MULTILINE) is None:
        raise ValueError(f'{filename}: expected heading "# ADR-{identifier}: Title"')
    if re.search(r"^Date: \d{4}-\d{2}-\d{2}$", contents, re.MULTILINE) is None:
        raise ValueError(f"{filename}: missing a valid Date field")
    if STATUS_PATTERN.search(contents) is None:
        raise ValueError(f"{filename}: missing a valid Status value")
    return Adr(filename=filename, identifier=identifier)


def validate_adr_directory(directory: Path) -> None:
    entries = list(directory.iterdir())
    markdown_files = sorted(
        entry.name for entry in entries if entry.is_file() and entry.suffix == ".md" and entry.name not in SPECIAL_FILES
    )
    invalid_files = [filename for filename in markdown_files if ADR_FILE_PATTERN.fullmatch(filename) is None]
    if invalid_files:
        raise ValueError(f"Invalid ADR filenames: {', '.join(invalid_files)}")

    adrs = [_metadata(directory, filename) for filename in markdown_files]
    duplicate_ids = sorted(
        {adr.identifier for adr in adrs if sum(item.identifier == adr.identifier for item in adrs) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate ADR identifiers: {', '.join(duplicate_ids)}")

    for companion in sorted(entry.name for entry in entries if entry.is_dir()):
        if f"{companion}.md" not in markdown_files:
            raise ValueError(f"ADR companion directory has no matching ADR: {companion}")


def tracked_and_untracked_files(repository_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [repository_root / name.decode() for name in result.stdout.split(b"\0") if name]


def validate_references(repository_root: Path) -> None:
    legacy_matches: list[str] = []
    dangling_links: list[str] = []
    for path in tracked_and_untracked_files(repository_root):
        if not path.is_file():
            continue
        contents_bytes = path.read_bytes()
        if b"\0" in contents_bytes:
            continue
        contents = contents_bytes.decode("utf-8", errors="replace")
        relative_path = path.relative_to(repository_root)
        for line_number, line in find_legacy_adr_references(contents):
            legacy_matches.append(f"{relative_path}:{line_number}:{line.strip()}")
        for line_number, line in enumerate(contents.splitlines(), start=1):
            for link in ADR_LINK_PATTERN.findall(line):
                if not (repository_root / link).is_file():
                    dangling_links.append(f"{relative_path}:{line_number}:{link}")

    messages: list[str] = []
    if legacy_matches:
        messages.append("Legacy numbered ADR references found:\n" + "\n".join(legacy_matches))
    if dangling_links:
        messages.append("Dangling docs/adr references found:\n" + "\n".join(dangling_links))
    if messages:
        raise ValueError("\n".join(messages))


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    validate_adr_directory(repository_root / "docs" / "adr")
    validate_references(repository_root)
    print("ADR validation passed")


if __name__ == "__main__":
    main()
