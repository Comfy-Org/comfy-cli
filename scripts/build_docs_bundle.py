#!/usr/bin/env python3
"""Build the deterministic Markdown corpus used by ``comfy docs``."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "search-sources.json"
OUTPUT = ROOT / "comfy_cli" / "docs" / "data" / "corpus.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
SLUG_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def _strip_frontmatter(lines: list[str]) -> tuple[list[str], int]:
    if not lines or lines[0].strip() != "---":
        return lines, 0
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return lines[index + 1 :], index + 1
    raise ValueError("Markdown frontmatter is missing its closing delimiter")


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (level, zero-based line, title) for headings outside code fences."""
    found: list[tuple[int, int, str]] = []
    fence_char: str | None = None
    fence_size = 0
    for index, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char = None
                fence_size = 0
            continue
        if fence_char is not None:
            continue
        match = HEADING_RE.match(line)
        if match:
            found.append((len(match.group(1)), index, match.group(2).strip()))
    if fence_char is not None:
        raise ValueError("Markdown contains an unclosed fenced code block")
    return found


def _slug(value: str) -> str:
    slug = SLUG_RE.sub("-", value.lower()).strip("-.")
    return slug or "section"


def _sections(source: dict[str, Any]) -> list[dict[str, Any]]:
    relative_path = source["path"]
    path = ROOT / relative_path
    if not path.is_file():
        raise ValueError(f"Source does not exist: {relative_path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    line_offset = 0
    if path.name == "SKILL.md":
        lines, line_offset = _strip_frontmatter(lines)

    headings = _headings(lines)
    selected_roots = set(source.get("include_level_two_headings", []))
    if selected_roots:
        found_roots = {title for level, _, title in headings if level == 2}
        missing = selected_roots - found_roots
        if missing:
            raise ValueError(f"{relative_path} is missing selected headings: {', '.join(sorted(missing))}")

    source_path = Path(relative_path)
    document_id = _slug(source_path.parent.name if source_path.name == "SKILL.md" else source_path.stem)
    sections: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    ancestors: list[tuple[int, str]] = []

    for position, (level, line_index, title) in enumerate(headings):
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        ancestors.append((level, title))
        if selected_roots and not any(level == 2 and name in selected_roots for level, name in ancestors):
            continue

        next_line = headings[position + 1][1] if position + 1 < len(headings) else len(lines)
        content = "\n".join(lines[line_index + 1 : next_line]).strip()
        if not content:
            continue
        breadcrumb = [name for _, name in ancestors]
        base_id = f"{document_id}:{'-'.join(_slug(name) for name in breadcrumb)}"
        ids[base_id] = ids.get(base_id, 0) + 1
        section_id = base_id if ids[base_id] == 1 else f"{base_id}-{ids[base_id]}"
        sections.append(
            {
                "id": section_id,
                "title": source["title"],
                "headings": breadcrumb,
                "source": relative_path,
                "source_line": line_index + line_offset + 1,
                "content": content,
            }
        )

    if not sections:
        raise ValueError(f"No searchable headings found in {relative_path}")
    return sections


def build() -> bytes:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sources"), list):
        raise ValueError("Unsupported docs source manifest")
    sections = [section for source in manifest["sources"] for section in _sections(source)]
    canonical = json.dumps(sections, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    corpus = {
        "schema_version": 1,
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "sections": sections,
    }
    return (json.dumps(corpus, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in corpus is stale")
    args = parser.parse_args()
    try:
        content = build()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"docs bundle error: {error}", file=sys.stderr)
        return 1

    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != content:
            print("docs bundle is stale; run python scripts/build_docs_bundle.py", file=sys.stderr)
            return 1
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
