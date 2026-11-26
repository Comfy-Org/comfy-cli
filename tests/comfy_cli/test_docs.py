"""Tests for the packaged docs corpus and local search behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comfy_cli import docs
from comfy_cli.docs import lancedb_search
from scripts import build_docs_bundle

ROOT = Path(__file__).resolve().parents[2]
QUERY_FIXTURE = Path(__file__).parent / "fixtures" / "docs" / "search_queries.json"


def _search_with_fallback(monkeypatch, query: str) -> dict:
    def no_fts(*_args, **_kwargs):
        raise NotImplementedError("forced fallback")

    _force_core_search(monkeypatch)
    monkeypatch.setattr(docs, "_fts_search", no_fts)
    return docs.search(query, mode="bm25")


def _force_core_search(monkeypatch) -> None:
    monkeypatch.setattr(
        lancedb_search,
        "inspect_pack",
        lambda **_kwargs: {
            "installed": False,
            "compatible": False,
            "reason": "forced core retrieval",
            "hint": None,
            "available_modes": ["auto", "bm25"],
            "pack_version": None,
            "lancedb_version": None,
            "model": None,
        },
    )


def _matches_target(case: dict, results: list[dict]) -> bool:
    targets = case.get("targets", [case.get("target", "")])
    headings = [" ".join(result["headings"]).casefold() for result in results]
    return any(target.casefold() in heading for target in targets for heading in headings)


def test_packaged_corpus_is_current_and_has_valid_source_locations():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_docs_bundle.py"), "--check"], check=True)
    sections, content_hash = docs.load_corpus()

    assert len(content_hash) == 64
    assert len({section["id"] for section in sections}) == len(sections)
    assert all((ROOT / section["source"]).is_file() for section in sections)
    assert all(section["source_line"] > 0 for section in sections)
    assert any("```" in section["content"] for section in sections)


def test_corpus_freshness_check_accepts_windows_checkout_newlines(tmp_path, monkeypatch):
    expected = build_docs_bundle.build()
    output = tmp_path / "corpus.json"
    output.write_bytes(expected.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(build_docs_bundle, "OUTPUT", output)

    assert build_docs_bundle._current_corpus(expected)


def test_search_relevance_for_common_agent_queries(monkeypatch):
    cases = json.loads(QUERY_FIXTURE.read_text(encoding="utf-8"))
    _force_core_search(monkeypatch)
    for case in cases:
        fts_results = docs.search(case["query"], mode="bm25")["results"][:5]
        for result in fts_results:
            assert len(result["excerpt"]) <= docs.MAX_EXCERPT_CHARS
        assert _matches_target(case, fts_results), case

    for case in cases:
        fallback = _search_with_fallback(monkeypatch, case["query"])
        assert fallback["search_backend"] == "token-scan"
        assert _matches_target(case, fallback["results"][:5]), case


def test_search_is_bounded_deterministic_and_safe_for_plain_text_queries(monkeypatch):
    _force_core_search(monkeypatch)
    first = docs.search('"OR" --json (FTS)', mode="bm25")
    assert first["results"] == docs.search('"OR" --json (FTS)', mode="bm25")["results"]
    assert len(first["results"]) <= docs.DEFAULT_RESULTS
    assert first["search_backend"] == "fts5"

    assert docs.search("and the for", mode="bm25") == {
        "query": "and the for",
        "results": [],
        "zero_hit": True,
        "has_more": False,
        "search_backend": "none",
        "retrieval_mode": "none",
        "query_mode": "bm25",
        "corpus_hash": docs.load_corpus()[1],
        "fallback_reason": None,
        "hint": "No searchable terms; try a command name or a more specific topic.",
    }
    with pytest.raises(ValueError, match="query must not be empty"):
        docs.search("  ", mode="bm25")
    with pytest.raises(ValueError, match="at most"):
        docs.search("x" * (docs.MAX_QUERY_CHARS + 1), mode="bm25")
    with pytest.raises(ValueError, match="limit"):
        docs.search("workflow", limit=docs.MAX_RESULTS + 1, mode="bm25")


def test_search_defaults_to_core_bm25_when_the_optional_pack_is_missing(monkeypatch):
    _force_core_search(monkeypatch)
    result = docs.search("stuck job wait timeout")
    assert result["retrieval_mode"] == "bm25"
    assert result["search_backend"] == "fts5"
    assert result["fallback_reason"] == "forced core retrieval"


def test_show_pages_can_be_joined_to_reconstruct_a_section():
    sections, _ = docs.load_corpus()
    section = max(sections, key=lambda item: len(item["content"]))
    parts = []
    offset = 0
    while True:
        page = docs.show(section["id"], max_chars=317, offset=offset)
        assert page is not None
        assert len(page["content"]) <= 317
        parts.append(page["content"])
        if not page["truncated"]:
            assert page["next_offset"] is None
            break
        assert page["next_offset"] == offset + len(page["content"])
        offset = page["next_offset"]
    assert "".join(parts) == section["content"]


def test_show_validates_offset_and_returns_none_for_unknown_id():
    section = docs.load_corpus()[0][0]
    assert docs.show("not-a-section") is None
    with pytest.raises(ValueError, match="offset must be between"):
        docs.show(section["id"], offset=len(section["content"]) + 1)


def test_token_fallback_ranks_matching_headings(monkeypatch):
    payload = _search_with_fallback(monkeypatch, "install custom nodes")
    assert payload["results"]
    assert payload["search_backend"] == "token-scan"
    assert payload["results"][0]["headings"][-1] in {"Managing Custom Nodes", "`install` vs `registry-install`"}
