"""Read and search the documentation shipped with comfy-cli."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Sequence
from importlib import resources
from typing import Any

CORPUS_SCHEMA_VERSION = 1
MAX_QUERY_CHARS = 500
MAX_RESULTS = 20
DEFAULT_RESULTS = 5
MAX_EXCERPT_CHARS = 800
DEFAULT_SECTION_CHARS = 12_000
MAX_SECTION_CHARS = 50_000
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset("a an and are as at be by for from how i in is it of on or the to with".split())


class DocsUnavailableError(Exception):
    """The installed documentation corpus is absent or malformed."""


def _canonical_sections(sections: list[dict[str, Any]]) -> bytes:
    return json.dumps(sections, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_corpus() -> tuple[list[dict[str, Any]], str]:
    """Read and validate the packaged corpus without touching the network."""
    try:
        raw = (resources.files("comfy_cli.docs") / "data" / "corpus.json").read_text(encoding="utf-8")
        corpus = json.loads(raw)
        sections = corpus["sections"]
        content_hash = corpus["content_hash"]
        if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION or not isinstance(sections, list):
            raise ValueError("unsupported docs corpus schema")
        if not isinstance(content_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("invalid docs corpus hash")
        ids: set[str] = set()
        for section in sections:
            if not isinstance(section, dict) or not all(
                isinstance(section.get(key), expected)
                for key, expected in (
                    ("id", str),
                    ("title", str),
                    ("headings", list),
                    ("source", str),
                    ("source_line", int),
                    ("content", str),
                )
            ):
                raise ValueError("invalid docs section")
            if section["id"] in ids or not all(isinstance(heading, str) for heading in section["headings"]):
                raise ValueError("duplicate or malformed docs section")
            ids.add(section["id"])
        if not sections:
            raise ValueError("docs corpus is empty")
        if hashlib.sha256(_canonical_sections(sections)).hexdigest() != content_hash:
            raise ValueError("docs corpus hash does not match its content")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise DocsUnavailableError("the installed documentation corpus is missing or invalid") from error
    return sections, content_hash


def query_terms(query: str) -> list[str]:
    """Extract plain text search terms, dropping common grammatical filler."""
    return [term for term in _TOKEN_RE.findall(query.casefold()) if term not in _STOP_WORDS]


def _quote_fts_terms(terms: Sequence[str]) -> str:
    # Terms contain Unicode letters/numbers only, but quote them explicitly so
    # FTS always sees ordinary words rather than a caller-supplied expression.
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _stem(token: str) -> str:
    """Apply small Porter-style suffix rules for the no-FTS fallback."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("e"):
        return token[:-1]
    return token


def _fallback_search(
    sections: list[dict[str, Any]], terms: Sequence[str], limit: int
) -> list[tuple[dict[str, Any], float]]:
    terms = [_stem(term) for term in terms]
    ranked: list[tuple[dict[str, Any], float]] = []
    for section in sections:
        title_text = " ".join([section["title"], *section["headings"]]).casefold()
        title_tokens = {_stem(token) for token in _TOKEN_RE.findall(title_text)}
        body_tokens = {_stem(token) for token in _TOKEN_RE.findall(section["content"].casefold())}
        score = sum(5 for term in terms if term in title_tokens) + sum(1 for term in terms if term in body_tokens)
        if score:
            ranked.append((section, -float(score)))
    ranked.sort(key=lambda item: (item[1], item[0]["id"]))
    return ranked[:limit]


def _fts_search(sections: list[dict[str, Any]], terms: Sequence[str], limit: int) -> list[tuple[dict[str, Any], float]]:
    db = sqlite3.connect(":memory:")
    try:
        db.execute(
            "CREATE VIRTUAL TABLE doc_search USING fts5(id UNINDEXED, title, headings, content, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as error:
        if "no such module: fts5" in str(error).casefold():
            raise NotImplementedError("SQLite was built without FTS5") from error
        raise
    try:
        db.executemany(
            "INSERT INTO doc_search (id, title, headings, content) VALUES (?, ?, ?, ?)",
            [
                (section["id"], section["title"], " ".join(section["headings"]), section["content"])
                for section in sections
            ],
        )
        rows = db.execute(
            "SELECT id, bm25(doc_search, 0.0, 8.0, 5.0, 1.0) AS score "
            "FROM doc_search WHERE doc_search MATCH ? ORDER BY score, id LIMIT ?",
            (_quote_fts_terms(terms), limit),
        ).fetchall()
    finally:
        db.close()
    by_id = {section["id"]: section for section in sections}
    return [(by_id[section_id], float(score)) for section_id, score in rows]


def _excerpt(section: dict[str, Any], terms: Sequence[str]) -> str:
    content = section["content"].strip()
    if len(content) <= MAX_EXCERPT_CHARS:
        return content
    earliest: int | None = None
    for term in terms:
        match = re.search(re.escape(term), content, re.IGNORECASE)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    start = max(0, (earliest or 0) - MAX_EXCERPT_CHARS // 4)
    prefix = start > 0
    end = min(len(content), start + MAX_EXCERPT_CHARS - int(prefix))
    suffix = end < len(content)
    if suffix:
        end -= 1
    excerpt = content[start:end].strip()
    if prefix:
        excerpt = "…" + excerpt
    if suffix:
        excerpt += "…"
    return excerpt


def search(query: str, *, limit: int = DEFAULT_RESULTS) -> dict[str, Any]:
    """Return ranked section previews for a natural-language query."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")

    sections, content_hash = load_corpus()
    terms = query_terms(query)
    if not terms:
        ranked: list[tuple[dict[str, Any], float]] = []
        backend = "none"
    else:
        try:
            ranked = _fts_search(sections, terms, limit + 1)
            backend = "fts5"
        except NotImplementedError:
            ranked = _fallback_search(sections, terms, limit + 1)
            backend = "token-scan"

    results = [
        {
            **{key: section[key] for key in ("id", "title", "headings", "source", "source_line")},
            "excerpt": _excerpt(section, terms),
        }
        for section, _score in ranked[:limit]
    ]
    zero_hit = not results
    payload: dict[str, Any] = {
        "query": query,
        "results": results,
        "zero_hit": zero_hit,
        "has_more": len(ranked) > limit,
        "search_backend": backend,
        "corpus_hash": content_hash,
    }
    if zero_hit:
        payload["hint"] = "No documentation matched; try a command name or a more specific topic."
    return payload


def show(section_id: str, *, max_chars: int = DEFAULT_SECTION_CHARS, offset: int = 0) -> dict[str, Any] | None:
    """Return a section page, or None when the ID is not in this corpus."""
    if not 1 <= max_chars <= MAX_SECTION_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_SECTION_CHARS}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    sections, content_hash = load_corpus()
    section = next((item for item in sections if item["id"] == section_id), None)
    if section is None:
        return None
    content = section["content"]
    if offset > len(content):
        raise ValueError(f"offset must be between 0 and {len(content)} for this section")
    end = min(offset + max_chars, len(content))
    return {
        **{key: section[key] for key in ("id", "title", "headings", "source", "source_line")},
        "content": content[offset:end],
        "total_chars": len(content),
        "offset": offset,
        "truncated": end < len(content),
        "next_offset": end if end < len(content) else None,
        "corpus_hash": content_hash,
    }
