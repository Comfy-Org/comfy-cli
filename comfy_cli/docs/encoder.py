"""Local ONNX query encoder used only by the optional docs-search extra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalQueryEncoder:
    """BGE small encoder with the CLS pooling and normalization used by FastEmbed."""

    def __init__(self, model_dir: Path, *, threads: int = 2) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import AddedToken, Tokenizer
        except ImportError as error:
            raise RuntimeError("install comfy-cli[docs-search] to enable semantic docs search") from error

        self._np = np
        self.model_dir = model_dir.resolve(strict=True)
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((self.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        special_tokens = json.loads((self.model_dir / "special_tokens_map.json").read_text(encoding="utf-8"))
        max_length = min(
            int(config.get("max_position_embeddings", 512)),
            int(tokenizer_config.get("model_max_length", 512)),
        )
        if max_length <= 2 or max_length > 512:
            raise ValueError(f"unsupported BGE tokenizer context size: {max_length}")

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_token = tokenizer_config.get("pad_token")
        if not isinstance(pad_token, str):
            raise ValueError("BGE tokenizer has no string pad_token")
        self.tokenizer.enable_padding(pad_id=int(config.get("pad_token_id", 0)), pad_token=pad_token)
        for token in special_tokens.values():
            if isinstance(token, str):
                self.tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                self.tokenizer.add_special_tokens([AddedToken(**token)])

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = max(1, threads)
        session_options.inter_op_num_threads = max(1, threads)
        self.session = ort.InferenceSession(
            str(self.model_dir / "model_optimized.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.embedding_size = int(config.get("hidden_size", 0))
        if not self.embedding_size:
            raise ValueError("BGE model config has no hidden_size")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encodings = self.tokenizer.encode_batch(texts)
        np = self._np
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        inputs: dict[str, Any] = {"input_ids": input_ids}
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        embeddings = self.session.run(None, inputs)[0]
        if embeddings.ndim == 3:
            embeddings = embeddings[:, 0]
        elif embeddings.ndim != 2:
            raise ValueError(f"unsupported BGE embedding shape: {embeddings.shape}")
        norms = np.maximum(np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True), 1e-12)
        normalized = embeddings / norms
        if normalized.shape[1] != self.embedding_size:
            raise ValueError(f"unexpected BGE embedding dimension: {normalized.shape[1]}")
        return normalized.astype(np.float32).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]
