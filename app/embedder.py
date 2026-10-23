"""Stage 4 - embedding: texts -> L2-normalized float32 vectors."""
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class FastEmbedder:
    """all-MiniLM-L6-v2 through ONNX Runtime (no torch), fully local once cached."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.array(list(self._model.embed(texts, batch_size=self.batch_size)))
        return _normalize(vecs)


class HashEmbedder:
    """Deterministic bag-of-words hashing. No model, no download: for tests only."""

    def __init__(self, dim: int = 384):
        self.name = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(out)


def make_embedder(kind: str, model_name: str, cache_dir: str | None, batch_size: int) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        return FastEmbedder(model_name, cache_dir, batch_size)
    raise ValueError(f"Unknown EMBEDDER: {kind}")
