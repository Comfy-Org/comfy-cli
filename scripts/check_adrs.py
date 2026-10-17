from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

ADR_FILE_PATTERN = re.compile(r"^([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)?-[0-9]{4})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
ADR_LINK_PATTERN = re.compile(r"(?<![/A-Za-z0-9._-])docs/adr/[A-Za-z0-9._/-]+\.md")
MARKDOWN_LINK_PATTERN = re.compile(r"\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")
LEGACY_REFERENCE_PATTERN = re.compile(
    r"ADR(?:[- ]?\d{4}(?!-\d{2})|(?:-[A-Z][A-Z0-9]*)?\s*(?:/\s*|\(\s*)\d{4}(?!-\d{2}))|(?:docs/)?adr/\d{4}-"
)
METADATA_PATTERN = re.compile(
    r"\A# ADR-(?P<identifier>[A-Z0-9-]+): (?P<title>[^\n]+)[ \t]*\n"
    r"(?:[ \t]*\n)*Date: (?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})[ \t]*\n"
    r"(?:[ \t]*\n)*## Status[ \t]*\n(?:[ \t]*\n)*"
    r"[ \t]*(?P<status>Proposed|Accepted|Rejected|Deprecated|Superseded)[ \t]*(?:\n|\Z)"
)
SPECIAL_FILES = {"TEMPLATE.md"}
MAX_SCANNED_FILE_BYTES = 5 * 1024 * 1024


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
    metadata = METADATA_PATTERN.match(contents)
    if metadata is None or metadata.group("identifier") != identifier:
        raise ValueError(f'{filename}: expected heading "# ADR-{identifier}: Title"')
    try:
        date.fromisoformat(metadata.group("date"))
    except ValueError as exc:
        raise ValueError(f"{filename}: invalid Date value") from exc
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


def tracked_files(repository_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [repository_root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name]


def _resolved_local_target(source: Path, target: str) -> Path | None:
    target = target.removeprefix("<").removesuffix(">")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (source.parent / unquote(parsed.path)).resolve()


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_references(repository_root: Path) -> None:
    legacy_matches: list[str] = []
    dangling_links: list[str] = []
    repository_root = repository_root.resolve()
    adr_directory = (repository_root / "docs" / "adr").resolve()
    for path in tracked_files(repository_root):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SCANNED_FILE_BYTES:
            continue
        try:
            contents_bytes = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Unable to read tracked file {path.relative_to(repository_root)}: {exc}") from exc
        if b"\0" in contents_bytes:
            continue
        contents = contents_bytes.decode("utf-8", errors="replace")
        relative_path = path.relative_to(repository_root)
        for line_number, line in find_legacy_adr_references(contents):
            legacy_matches.append(f"{relative_path}:{line_number}:{line.strip()}")
        for line_number, line in enumerate(contents.splitlines(), start=1):
            for link in ADR_LINK_PATTERN.findall(line):
                target = (repository_root / link).resolve()
                if not _inside(target, adr_directory) or not target.is_file():
                    dangling_links.append(f"{relative_path}:{line_number}:{link}")
            for match in MARKDOWN_LINK_PATTERN.finditer(line):
                target = _resolved_local_target(path, match.group("target"))
                if target is None:
                    continue
                if not (_inside(path.resolve(), adr_directory) or _inside(target, adr_directory)):
                    continue
                if not _inside(target, repository_root) or not target.is_file():
                    dangling_links.append(f"{relative_path}:{line_number}:{match.group('target')}")

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
