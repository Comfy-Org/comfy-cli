"""Load the optional version-matched LanceDB pack and retrieve doc chunks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from comfy_cli.file_utils import cache_dir

from . import _TOKEN_RE, DocsSearchPackError, _excerpt, _stem, query_terms
from .encoder import LocalQueryEncoder
from .pack_constants import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_ONNX_SHA256,
    EMBEDDING_MODEL_REVISION,
    LANCEDB_PACK_VERSION,
    MIN_FTS_TERM_COVERAGE,
    MIN_VECTOR_SIMILARITY,
    PACK_SCHEMA_VERSION,
)

SUPPORTED_LANCEDB_VERSION = LANCEDB_PACK_VERSION
MAX_PACK_ARCHIVE_BYTES = 150 * 1024 * 1024
MAX_PACK_EXPANDED_BYTES = 180 * 1024 * 1024
MAX_PACK_FILES = 10000
_MODEL_FILES = frozenset(
    {
        "config.json",
        "model_optimized.onnx",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "MODEL-LICENSE.txt",
    }
)


@lru_cache(maxsize=2)
def _query_encoder(model_dir: str) -> LocalQueryEncoder:
    return LocalQueryEncoder(Path(model_dir))


@lru_cache(maxsize=2)
def _docs_table(database_dir: str):
    from lancedb import connect

    return connect(database_dir).open_table("docs")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pack_module():
    try:
        return __import__("comfy_cli_docs_search", fromlist=["pack_resource", "read_pack_manifest"])
    except (ImportError, ModuleNotFoundError):
        return None


def inspect_pack(*, corpus_hash: str) -> dict[str, Any]:
    """Check package metadata without importing LanceDB or loading the encoder."""
    pack_module = _pack_module()
    if pack_module is None:
        return {
            "installed": False,
            "compatible": False,
            "reason": "The optional docs-search pack is not installed.",
            "hint": "Install the matching extra with `pip install 'comfy-cli[docs-search]'`.",
            "available_modes": ["auto", "bm25"],
            "pack_version": None,
            "lancedb_version": None,
            "model": None,
        }

    try:
        manifest = pack_module.read_pack_manifest()
        archive = pack_module.pack_resource()
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {
            "installed": True,
            "compatible": False,
            "reason": f"The docs-search pack manifest could not be read: {error}",
            "hint": "Reinstall comfy-cli[docs-search] for this comfy-cli release.",
            "available_modes": ["auto", "bm25"],
            "pack_version": _distribution_version("comfy-cli-docs-search"),
            "lancedb_version": None,
            "model": None,
        }

    pack_version = _distribution_version("comfy-cli-docs-search")
    cli_version = _distribution_version("comfy-cli")
    installed_lancedb = _distribution_version("lancedb")
    reason = _manifest_problem(manifest)
    if reason:
        pass
    elif manifest.get("schema_version") != PACK_SCHEMA_VERSION:
        reason = "The installed docs-search pack uses an unsupported schema."
    elif manifest.get("corpus_hash") != corpus_hash:
        reason = "The installed docs-search pack was built for a different docs corpus."
    elif (
        manifest.get("model", {}).get("name") != EMBEDDING_MODEL_NAME
        or manifest.get("model", {}).get("revision") != EMBEDDING_MODEL_REVISION
        or manifest.get("model", {}).get("dimension") != EMBEDDING_DIMENSION
        or manifest.get("model", {}).get("files", {}).get("model_optimized.onnx", {}).get("sha256")
        != EMBEDDING_MODEL_ONNX_SHA256
    ):
        reason = "The installed docs-search pack uses an unsupported query encoder."
    elif not cli_version or pack_version != cli_version or manifest.get("comfy_cli_version") != cli_version:
        reason = "The installed docs-search pack does not match the comfy-cli version."
    elif manifest.get("lancedb_version") != SUPPORTED_LANCEDB_VERSION:
        reason = "The docs-search pack requires an unsupported LanceDB format version."
    elif manifest.get("min_vector_similarity") != MIN_VECTOR_SIMILARITY:
        reason = "The docs-search pack uses an unsupported semantic no-hit threshold."
    elif manifest.get("min_fts_term_coverage") != MIN_FTS_TERM_COVERAGE:
        reason = "The docs-search pack uses an unsupported lexical evidence threshold."
    elif installed_lancedb != manifest.get("lancedb_version"):
        reason = "The installed LanceDB version does not match the docs-search pack."
    elif not archive.is_file():
        reason = "The docs-search pack archive is missing."
    elif _distribution_version("onnxruntime") is None or _distribution_version("tokenizers") is None:
        reason = "The local query-encoder runtime is incomplete."

    if reason:
        return {
            "installed": True,
            "compatible": False,
            "reason": reason,
            "hint": "Reinstall the matching `comfy-cli[docs-search]` extra.",
            "available_modes": ["auto", "bm25"],
            "pack_version": pack_version,
            "lancedb_version": installed_lancedb,
            "model": manifest.get("model", {}).get("name"),
        }

    return {
        "installed": True,
        "compatible": True,
        "reason": None,
        "hint": None,
        "available_modes": ["auto", "bm25", "semantic", "hybrid"],
        "pack_version": pack_version,
        "lancedb_version": installed_lancedb,
        "model": manifest.get("model", {}).get("name"),
    }


def _manifest_problem(manifest: Any) -> str | None:
    if not isinstance(manifest, dict):
        return "The docs-search pack manifest is not an object."
    if manifest.get("schema_version") != PACK_SCHEMA_VERSION:
        return "The installed docs-search pack uses an unsupported schema."
    if not isinstance(manifest.get("corpus_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["corpus_hash"]):
        return "The docs-search pack has an invalid corpus hash."
    if not isinstance(manifest.get("comfy_cli_version"), str):
        return "The docs-search pack does not declare its comfy-cli version."
    if not isinstance(manifest.get("lancedb_version"), str):
        return "The docs-search pack does not declare its LanceDB format version."
    if not isinstance(manifest.get("model"), dict) or not isinstance(manifest["model"].get("files"), dict):
        return "The docs-search pack has invalid encoder metadata."
    model_files = manifest["model"]["files"]
    if any(
        not isinstance(model_files.get(name), dict)
        or not isinstance(model_files[name].get("bytes"), int)
        or isinstance(model_files[name].get("bytes"), bool)
        or not isinstance(model_files[name].get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", model_files[name]["sha256"])
        for name in _MODEL_FILES
    ):
        return "The docs-search pack has incomplete encoder file metadata."
    size = manifest.get("archive_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_PACK_ARCHIVE_BYTES:
        return "The docs-search pack archive has an invalid size."
    if not isinstance(manifest.get("archive_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest["archive_sha256"]
    ):
        return "The docs-search pack archive has an invalid SHA-256."
    if not isinstance(manifest.get("database_files"), list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or PurePosixPath(record["path"]).is_absolute()
        or ".." in PurePosixPath(record["path"]).parts
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or not isinstance(record.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        for record in manifest.get("database_files", [])
    ):
        return "The docs-search pack has invalid LanceDB file metadata."
    return None


def status(*, corpus_hash: str) -> dict[str, Any]:
    info = inspect_pack(corpus_hash=corpus_hash)
    return {
        "corpus_hash": corpus_hash,
        "pack_installed": info["installed"],
        "pack_compatible": info["compatible"],
        "pack_version": info["pack_version"],
        "lancedb_version": info["lancedb_version"],
        "model": info["model"],
        "available_modes": info["available_modes"],
        "unavailable_reason": info["reason"],
        "hint": info["hint"],
    }


def _copy_archive_safely(archive_bytes: bytes, destination: Path) -> None:
    if len(archive_bytes) > MAX_PACK_ARCHIVE_BYTES:
        raise ValueError("docs-search pack exceeds the archive size limit")
    destination.mkdir(parents=True, exist_ok=True)
    total_size = 0
    file_count = 0
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or ":" in member.name
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("docs-search archive contains an unsafe path or link")
            total_size += member.size
            file_count += member.isfile()
            if total_size > MAX_PACK_EXPANDED_BYTES or file_count > MAX_PACK_FILES:
                raise ValueError("docs-search archive exceeds supported expansion limits")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("docs-search archive contains an unreadable file")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _materialize_pack(*, corpus_hash: str) -> tuple[dict[str, Any], Path]:
    pack_module = _pack_module()
    state = inspect_pack(corpus_hash=corpus_hash)
    if not state["compatible"] or pack_module is None:
        code = "docs_pack_incompatible" if state["installed"] else "docs_search_unavailable"
        raise DocsSearchPackError(
            code,
            state["reason"] or "The docs-search pack is unavailable.",
            state["hint"] or "Install the matching comfy-cli[docs-search] extra.",
        )

    manifest = pack_module.read_pack_manifest()
    manifest_problem = _manifest_problem(manifest)
    if manifest_problem:
        raise DocsSearchPackError(
            "docs_pack_incompatible", manifest_problem, "Reinstall the matching comfy-cli[docs-search] extra."
        )

    pack_hash = manifest["archive_sha256"]
    cache_root = cache_dir() / "docs-search"
    final_dir = cache_root / pack_hash
    database_dir = final_dir / "database"
    model_dir = final_dir / "model"
    marker_path = final_dir / ".verified"
    if final_dir.is_symlink():
        raise DocsSearchPackError(
            "docs_pack_incompatible",
            "The docs-search cache path is a symbolic link.",
            "Remove the invalid docs-search cache entry and reinstall the matching extra.",
        )
    if final_dir.is_dir():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            marker = None
        if (
            database_dir.is_dir()
            and model_dir.is_dir()
            and marker
            == {"archive_sha256": pack_hash, "corpus_hash": corpus_hash, "schema_version": PACK_SCHEMA_VERSION}
        ):
            return manifest, final_dir
        try:
            shutil.rmtree(final_dir)
        except OSError as error:
            raise DocsSearchPackError(
                "docs_pack_incompatible",
                f"The incomplete docs-search cache could not be removed: {error}",
                "Remove the invalid docs-search cache entry and reinstall the matching extra.",
            ) from error

    archive_bytes = pack_module.pack_resource().read_bytes()
    if len(archive_bytes) != manifest["archive_bytes"] or _hash_bytes(archive_bytes) != pack_hash:
        raise DocsSearchPackError(
            "docs_pack_incompatible",
            "The docs-search archive failed its integrity check.",
            "Reinstall the matching comfy-cli[docs-search] extra.",
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{pack_hash[:12]}-", dir=cache_root))
    try:
        _copy_archive_safely(archive_bytes, temporary)
        internal_path = temporary / "manifest.json"
        internal = json.loads(internal_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "comfy_cli_version",
            "corpus_hash",
            "model",
            "lancedb_version",
            "min_vector_similarity",
            "min_fts_term_coverage",
            "chunk_count",
            "database_files",
        ):
            if internal.get(key) != manifest.get(key):
                raise ValueError(f"docs-search archive manifest mismatch in {key}")
        expected = {item["path"]: item for item in manifest["database_files"]}
        for relative_path, record in expected.items():
            path = temporary / relative_path
            if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
                raise ValueError(f"docs-search database file failed integrity check: {relative_path}")
        expected_model = manifest["model"].get("files", {})
        for name, record in expected_model.items():
            path = temporary / "model" / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256_file(path) != record["sha256"]:
                raise ValueError(f"docs-search model file failed integrity check: {name}")
        (temporary / ".verified").write_text(
            json.dumps(
                {"archive_sha256": pack_hash, "corpus_hash": corpus_hash, "schema_version": PACK_SCHEMA_VERSION}
            ),
            encoding="utf-8",
        )
        try:
            os.replace(temporary, final_dir)
        except OSError:
            if not final_dir.is_dir():
                raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as error:
        raise DocsSearchPackError(
            "docs_pack_incompatible",
            f"The installed docs-search pack could not be opened: {error}",
            "Reinstall the matching comfy-cli[docs-search] extra.",
        ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return manifest, final_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def search(
    query: str,
    *,
    mode: str,
    limit: int,
    sections: list[dict[str, Any]],
    corpus_hash: str,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    try:
        manifest, root = _materialize_pack(corpus_hash=corpus_hash)
        from lancedb.rerankers import RRFReranker

        vector = None
        if mode in {"semantic", "hybrid"}:
            encoder = _query_encoder(str(root / "model"))
            vector = encoder.embed_query(query)
        table = _docs_table(str(root / "database"))
        terms = query_terms(query)
        # query_terms removes punctuation and boolean operators, so these plain
        # Tantivy terms cannot add FTS syntax.
        text_query = " OR ".join(terms)
        candidate_limit = min(int(manifest["chunk_count"]), max(limit * 20, limit + 1))

        if mode == "bm25":
            ranked = (
                table.search(text_query, query_type="fts", fts_columns=manifest["fts_column"])
                .limit(candidate_limit)
                .to_list()
            )
            backend = "lancedb-bm25"
        elif mode == "semantic":
            ranked = (
                table.search(vector, query_type="vector", vector_column_name=manifest["vector_column"])
                .limit(candidate_limit)
                .to_list()
            )
            backend = "lancedb-vector"
        else:
            ranked = (
                table.search(
                    query_type="hybrid",
                    vector_column_name=manifest["vector_column"],
                    fts_columns=manifest["fts_column"],
                )
                .vector(vector)
                .text(text_query)
                .rerank(RRFReranker(return_score="all"))
                .limit(candidate_limit)
                .to_list()
            )
            backend = "lancedb-hybrid-rrf"
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise DocsSearchPackError(
            "docs_pack_incompatible",
            f"The installed LanceDB index could not serve this search: {error}",
            "Rebuild or reinstall the matching comfy-cli[docs-search] pack.",
        ) from error

    vector_similarities = [1.0 - float(row["_distance"]) / 2.0 for row in ranked if row.get("_distance") is not None]
    fts_scores = [row for row in ranked if row.get("_score") is not None]
    fts_coverage = 0.0
    if terms and fts_scores:
        for row in fts_scores:
            tokens = {_stem(token) for token in _TOKEN_RE.findall(row["search_text"].casefold())}
            matched = sum(_stem(term) in tokens for term in terms)
            fts_coverage = max(fts_coverage, matched / len(terms))

    best_vector_similarity = max(vector_similarities, default=None)
    low_evidence = mode == "semantic" and (
        best_vector_similarity is None or best_vector_similarity < MIN_VECTOR_SIMILARITY
    )
    if mode == "hybrid":
        low_evidence = (
            best_vector_similarity is None or best_vector_similarity < MIN_VECTOR_SIMILARITY
        ) and fts_coverage < MIN_FTS_TERM_COVERAGE

    if low_evidence:
        return {
            "query": query,
            "results": [],
            "zero_hit": True,
            "has_more": False,
            "hint": "No documentation had enough semantic or keyword evidence to support a result.",
            "search_backend": backend,
            "retrieval_mode": mode,
            "corpus_hash": corpus_hash,
            "pack_version": manifest["comfy_cli_version"],
            "model": manifest["model"]["name"],
            "fallback_reason": fallback_reason,
            "highest_vector_similarity": best_vector_similarity,
            "fts_term_coverage": fts_coverage,
        }

    by_id = {section["id"]: section for section in sections}
    results: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    for row in ranked:
        section_id = row["section_id"]
        if section_id in seen_sections or section_id not in by_id:
            continue
        seen_sections.add(section_id)
        section = by_id[section_id]
        result = {
            "id": section_id,
            "title": section["title"],
            "headings": section["headings"],
            "source": section["source"],
            "source_line": row["source_line"],
            "excerpt": _excerpt({"content": row["content"]}, terms),
        }
        if mode == "semantic" and row.get("_distance") is not None:
            result["score"] = 1.0 - float(row["_distance"]) / 2.0
            result["score_type"] = "cosine_similarity"
        elif mode == "hybrid" and row.get("_relevance_score") is not None:
            result["score"] = float(row["_relevance_score"])
            result["score_type"] = "rrf"
        elif mode == "bm25" and row.get("_score") is not None:
            result["score"] = float(row["_score"])
            result["score_type"] = "bm25"
        results.append(result)
        if len(results) > limit:
            break

    return {
        "query": query,
        "results": results[:limit],
        "zero_hit": not results,
        "has_more": len(results) > limit or len(ranked) == candidate_limit,
        "search_backend": backend,
        "retrieval_mode": mode,
        "corpus_hash": corpus_hash,
        "pack_version": manifest["comfy_cli_version"],
        "model": manifest["model"]["name"],
        "fallback_reason": fallback_reason,
        "highest_vector_similarity": best_vector_similarity,
        "fts_term_coverage": fts_coverage,
    }
