#!/usr/bin/env python3
"""Build and validate the version-matched LanceDB docs-search pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comfy_cli.docs import CORPUS_SCHEMA_VERSION, _canonical_sections  # noqa: E402
from comfy_cli.docs.encoder import LocalQueryEncoder  # noqa: E402
from comfy_cli.docs.pack_constants import (  # noqa: E402
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_ONNX_SHA256,
    EMBEDDING_MODEL_REPOSITORY,
    EMBEDDING_MODEL_REVISION,
    LANCEDB_PACK_VERSION,
    MIN_FTS_TERM_COVERAGE,
    MIN_VECTOR_SIMILARITY,
    PACK_SCHEMA_VERSION,
)

CORPUS_PATH = ROOT / "comfy_cli" / "docs" / "data" / "corpus.json"
ASSET_DIR = ROOT / "packages" / "comfy-cli-docs-search" / "src" / "comfy_cli_docs_search" / "assets"
ARCHIVE_NAME = "docs-search-pack.tar.gz"
LANCEDB_VERSION = LANCEDB_PACK_VERSION
MODEL_NAME = EMBEDDING_MODEL_NAME
MODEL_REPO = EMBEDDING_MODEL_REPOSITORY
MODEL_REVISION = EMBEDDING_MODEL_REVISION
MODEL_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_ONNX_SHA256 = EMBEDDING_MODEL_ONNX_SHA256
MODEL_NOTICE = ROOT / "packages" / "comfy-cli-docs-search" / "MODEL-LICENSE.txt"
MAX_CHUNK_TOKENS = 420
CHUNK_OVERLAP_TOKENS = 40
BATCH_SIZE = 48


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_model(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "comfy-cli-docs-pack-builder"})
        target = destination / name
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    if _sha256(destination / "model_optimized.onnx") != MODEL_ONNX_SHA256:
        raise ValueError("downloaded BGE ONNX file did not match its pinned SHA-256")
    return destination


def _validate_model(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve(strict=True)
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"model directory is missing: {', '.join(missing)}")
    model_hash = _sha256(model_dir / "model_optimized.onnx")
    if model_hash != MODEL_ONNX_SHA256:
        raise ValueError("BGE ONNX file does not match the pinned model revision")
    return {
        "name": MODEL_NAME,
        "repository": MODEL_REPO,
        "revision": MODEL_REVISION,
        "dimension": 384,
        "max_input_tokens": 512,
        "normalization": "l2",
        "query_instruction": None,
        "files": {
            name: {"bytes": (model_dir / name).stat().st_size, "sha256": _sha256(model_dir / name)}
            for name in MODEL_FILES
        },
    }


def _split_section(section: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    content = section["content"]
    encoded = tokenizer.encode(content, add_special_tokens=False)
    if len(encoded.ids) <= MAX_CHUNK_TOKENS:
        windows = [(0, len(encoded.ids))] if encoded.ids else []
    else:
        step = MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
        windows = [
            (start, min(start + MAX_CHUNK_TOKENS, len(encoded.ids))) for start in range(0, len(encoded.ids), step)
        ]

    chunks: list[dict[str, Any]] = []
    headings = " › ".join(section["headings"])
    for ordinal, (first_token, end_token) in enumerate(windows):
        start_char = encoded.offsets[first_token][0]
        end_char = encoded.offsets[end_token - 1][1]
        body = content[start_char:end_char].strip()
        if not body:
            continue
        search_text = f"{section['title']}\n{headings}\n{body}"
        chunk_id = hashlib.sha256(f"{section['id']}\0{ordinal}".encode()).hexdigest()[:24]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "section_id": section["id"],
                "ordinal": ordinal,
                "title": section["title"],
                "headings": headings,
                "source": section["source"],
                "source_line": section["source_line"] + 1 + content[:start_char].count("\n"),
                "start_char": start_char,
                "end_char": end_char,
                "content": body,
                "search_text": search_text,
            }
        )
    return chunks


def _embed_chunks(chunks: list[dict[str, Any]], encoder: LocalQueryEncoder) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors.extend(encoder.embed_batch([chunk["search_text"] for chunk in batch]))
    return vectors


def _create_lance_table(database_dir: Path, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    import lancedb
    import pyarrow as pa

    if len(chunks) != len(vectors) or not chunks:
        raise ValueError("cannot build an empty or mismatched docs table")
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("section_id", pa.string()),
            pa.field("ordinal", pa.int32()),
            pa.field("title", pa.string()),
            pa.field("headings", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_line", pa.int32()),
            pa.field("start_char", pa.int32()),
            pa.field("end_char", pa.int32()),
            pa.field("content", pa.string()),
            pa.field("search_text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
        ]
    )
    rows = [{**chunk, "vector": vector} for chunk, vector in zip(chunks, vectors, strict=True)]
    table_data = pa.Table.from_pylist(rows, schema=schema)
    db = lancedb.connect(str(database_dir))
    table = db.create_table("docs", data=table_data, mode="overwrite")
    table.create_fts_index("search_text", replace=True, language="English", with_position=True)


def _build(output_dir: Path, model_dir: Path | None = None) -> dict[str, Any]:
    import tomlkit
    from tokenizers import Tokenizer

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    sections = corpus["sections"]
    if corpus.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported source docs corpus schema")
    if hashlib.sha256(_canonical_sections(sections)).hexdigest() != corpus["content_hash"]:
        raise ValueError("source docs corpus hash does not match its content")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comfy-docs-lance-build-") as temp:
        temp_root = Path(temp)
        resolved_model = model_dir.resolve(strict=True) if model_dir else _download_model(temp_root / "model")
        model_metadata = _validate_model(resolved_model)
        tokenizer = Tokenizer.from_file(str(resolved_model / "tokenizer.json"))
        chunks = [chunk for section in sections for chunk in _split_section(section, tokenizer)]
        encoder = LocalQueryEncoder(resolved_model)
        vectors = _embed_chunks(chunks, encoder)
        if len(vectors) != len(chunks) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise ValueError("encoder returned an unexpected number or dimension of vectors")

        pack_root = temp_root / "pack"
        pack_root.mkdir()
        model_target = pack_root / "model"
        model_target.mkdir()
        for name in MODEL_FILES:
            shutil.copyfile(resolved_model / name, model_target / name)
        shutil.copyfile(MODEL_NOTICE, model_target / "MODEL-LICENSE.txt")
        model_metadata["files"]["MODEL-LICENSE.txt"] = {
            "bytes": (model_target / "MODEL-LICENSE.txt").stat().st_size,
            "sha256": _sha256(model_target / "MODEL-LICENSE.txt"),
        }
        database_dir = pack_root / "database"
        _create_lance_table(database_dir, chunks, vectors)

        database_files = sorted(path for path in database_dir.rglob("*") if path.is_file())
        internal_manifest = {
            "schema_version": PACK_SCHEMA_VERSION,
            "comfy_cli_version": _comfy_cli_version(tomlkit),
            "corpus_hash": corpus["content_hash"],
            "model": model_metadata,
            "lancedb_version": LANCEDB_VERSION,
            "min_vector_similarity": MIN_VECTOR_SIMILARITY,
            "min_fts_term_coverage": MIN_FTS_TERM_COVERAGE,
            "table_name": "docs",
            "vector_column": "vector",
            "fts_column": "search_text",
            "chunker": {"version": 1, "max_tokens": MAX_CHUNK_TOKENS, "overlap_tokens": CHUNK_OVERLAP_TOKENS},
            "section_count": len(sections),
            "chunk_count": len(chunks),
            "database_files": [
                {"path": path.relative_to(pack_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in database_files
            ],
        }
        (pack_root / "manifest.json").write_text(
            json.dumps(internal_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        archive_path = output_dir / "docs-search-pack.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
            archive.add(pack_root / "manifest.json", arcname="manifest.json")
            archive.add(database_dir, arcname="database")
            archive.add(model_target, arcname="model")

    public_manifest = {
        **internal_manifest,
        "archive": ARCHIVE_NAME,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _validate_pack(output_dir / "manifest.json", archive_path, expected_corpus_hash=corpus["content_hash"])
    return public_manifest


def _comfy_cli_version(tomlkit) -> str:
    project = tomlkit.parse((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError("comfy-cli version is missing from pyproject.toml")
    return version


def _validate_pack(
    manifest_path: Path, archive_path: Path, *, expected_corpus_hash: str | None = None
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (
        "schema_version",
        "comfy_cli_version",
        "corpus_hash",
        "model",
        "lancedb_version",
        "min_vector_similarity",
        "min_fts_term_coverage",
        "table_name",
        "vector_column",
        "fts_column",
        "chunker",
        "section_count",
        "chunk_count",
        "database_files",
        "archive_bytes",
        "archive_sha256",
    )
    if not isinstance(manifest, dict) or any(key not in manifest for key in required):
        raise ValueError("docs-search pack manifest is missing required fields")
    if manifest["schema_version"] != PACK_SCHEMA_VERSION or not isinstance(manifest["model"].get("files"), dict):
        raise ValueError("docs-search pack manifest uses an unsupported schema")
    if not isinstance(manifest["corpus_hash"], str) or len(manifest["corpus_hash"]) != 64:
        raise ValueError("docs-search pack manifest has an invalid corpus hash")
    if expected_corpus_hash and manifest["corpus_hash"] != expected_corpus_hash:
        raise ValueError("docs-search pack was built from a different docs corpus")
    if manifest["archive_bytes"] != archive_path.stat().st_size:
        raise ValueError("docs-search archive size does not match its manifest")
    if manifest["archive_sha256"] != _sha256(archive_path):
        raise ValueError("docs-search archive SHA-256 does not match its manifest")
    if manifest["lancedb_version"] != LANCEDB_VERSION or manifest["model"].get("revision") != MODEL_REVISION:
        raise ValueError("docs-search pack uses an unsupported LanceDB or encoder revision")
    if not isinstance(manifest["database_files"], list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("bytes"), int)
        or not isinstance(item.get("sha256"), str)
        for item in manifest["database_files"]
    ):
        raise ValueError("docs-search pack manifest has invalid database file records")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Use an existing directory containing the pinned BGE ONNX model and tokenizer.",
    )
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--check", action="store_true", help="Validate an existing pack without rebuilding it.")
    args = parser.parse_args()
    try:
        if args.check:
            manifest_path = args.output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
            _validate_pack(
                manifest_path,
                args.output_dir / ARCHIVE_NAME,
                expected_corpus_hash=corpus["content_hash"],
            )
        else:
            manifest = _build(args.output_dir, args.model_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"LanceDB docs pack error: {error}", file=sys.stderr)
        return 1
    print(
        f"docs pack ready: {manifest['chunk_count']} chunks, "
        f"{manifest['archive_bytes'] / (1024 * 1024):.1f} MiB, "
        f"corpus {manifest['corpus_hash'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
