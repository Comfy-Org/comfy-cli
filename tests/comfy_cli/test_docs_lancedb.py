"""Integration checks for the optional, generated LanceDB search pack."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from comfy_cli import docs
from comfy_cli.docs import lancedb_search
from comfy_cli.docs.pack_constants import EMBEDDING_MODEL_REVISION


def _pack_available() -> bool:
    try:
        import lancedb  # noqa: F401
    except ImportError:
        return False
    return bool(docs.status()["pack_compatible"])


pytestmark = pytest.mark.skipif(not _pack_available(), reason="docs-search extra and generated pack are not installed")


def test_status_advertises_hybrid_modes_without_initializing_the_encoder(monkeypatch):
    monkeypatch.setattr(
        lancedb_search,
        "_query_encoder",
        lambda *_args: (_ for _ in ()).throw(AssertionError("status loaded the encoder")),
    )
    status = docs.status()
    assert status["pack_compatible"] is True
    assert status["available_modes"] == ["auto", "bm25", "semantic", "hybrid"]
    assert status["lancedb_version"] == lancedb_search.SUPPORTED_LANCEDB_VERSION


@pytest.mark.parametrize(
    ("mode", "query", "expected_backend", "expected_heading"),
    [
        ("bm25", "cloud_timeout", "lancedb-bm25", "`cloud_timeout`"),
        ("semantic", "hang up when task takes too long", "lancedb-vector", "Job-stuck triage"),
        ("hybrid", "job cloud_timeout expired progress", "lancedb-hybrid-rrf", "`cloud_timeout`"),
        ("auto", "stuck job wait timeout", "lancedb-hybrid-rrf", "Job-stuck triage"),
    ],
)
def test_lance_retrieval_modes_return_unique_sourceable_sections(mode, query, expected_backend, expected_heading):
    result = docs.search(query, mode=mode, limit=5)
    assert result["retrieval_mode"] == ("hybrid" if mode == "auto" else mode)
    assert result["search_backend"] == expected_backend
    ids = [item["id"] for item in result["results"]]
    assert len(ids) == len(set(ids))
    assert any(expected_heading.casefold() in " ".join(item["headings"]).casefold() for item in result["results"])
    for section_id in ids:
        section = docs.show(section_id)
        search_result = next(item for item in result["results"] if item["id"] == section_id)
        assert section is not None
        assert search_result["source_line"] == section["source_line"]


def test_lance_bm25_or_query_matches_individual_words():
    result = docs.search("local checkpoint", mode="bm25", limit=10)
    assert result["results"]
    assert any("checkpoint" in item["excerpt"].casefold() for item in result["results"])


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_lance_semantic_modes_return_zero_hit_without_evidence(mode):
    result = docs.search("what is the legal status of the moon", mode=mode, limit=5)
    assert result["zero_hit"] is True
    assert result["results"] == []


def test_lance_search_uses_packaged_assets_without_network(monkeypatch):
    import socket

    def deny_network(*_args, **_kwargs):
        raise AssertionError("docs search unexpectedly opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    result = docs.search("hang up when task takes too long", mode="semantic")
    assert result["results"]
    assert result["search_backend"] == "lancedb-vector"


def test_auto_falls_back_when_query_encoder_initialization_fails(monkeypatch):
    def broken_encoder(*_args, **_kwargs):
        raise RuntimeError("ONNX runtime could not initialize")

    monkeypatch.setattr(lancedb_search, "_query_encoder", broken_encoder)
    result = docs.search("hang up when task takes too long", mode="auto")

    assert result["search_backend"] == "fts5"
    assert result["query_mode"] == "auto"
    assert "ONNX runtime could not initialize" in result["fallback_reason"]


def test_explicit_hybrid_reports_encoder_initialization_failure(monkeypatch):
    def broken_encoder(*_args, **_kwargs):
        raise RuntimeError("ONNX runtime could not initialize")

    monkeypatch.setattr(lancedb_search, "_query_encoder", broken_encoder)
    with pytest.raises(docs.DocsSearchPackError, match="ONNX runtime could not initialize"):
        docs.search("hang up when task takes too long", mode="hybrid")


def test_semantic_and_hybrid_recall_held_out_agent_queries():
    query_path = Path(__file__).parent / "fixtures" / "docs" / "semantic_queries.json"
    cases = json.loads(query_path.read_text(encoding="utf-8"))
    for mode in ("semantic", "hybrid"):
        hits = 0
        for case in cases:
            result = docs.search(case["query"], mode=mode, limit=5)
            targets = case.get("targets", [case.get("target", "")])
            headings = [" ".join(item["headings"]).casefold() for item in result["results"]]
            hits += any(target.casefold() in heading for target in targets for heading in headings)
        assert hits / len(cases) >= 0.9, f"{mode} Recall@5 was {hits}/{len(cases)}"


def test_pack_metadata_matches_the_core_corpus():
    manifest = resources.files("comfy_cli_docs_search") / "assets" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["corpus_hash"] == docs.load_corpus()[1]
    assert data["model"]["dimension"] == 384
    assert data["model"]["revision"] == EMBEDDING_MODEL_REVISION
