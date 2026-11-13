"""Persistent cosine-similarity vector index with a FAISS backend.

FAISS is used when installed.  A NumPy implementation is intentionally retained
as a portable fallback for development and CI; both backends use inner product
over L2-normalized vectors, which is cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    faiss = None


class FaissVectorIndex:
    """An exact dense-vector index that can be saved and loaded from disk."""

    def __init__(self, embeddings: np.ndarray | None = None) -> None:
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._index = None
        if embeddings is not None:
            self.build(embeddings)

    @property
    def backend(self) -> str:
        """The active backend, useful for observability and reports."""
        return "faiss" if faiss is not None else "numpy"

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def build(self, embeddings: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if len(vectors) and not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain only finite values")

        self.embeddings = vectors
        self._index = None
        if faiss is not None and self.dimension:
            self._index = faiss.IndexFlatIP(self.dimension)
            if len(vectors):
                self._index.add(vectors)

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(scores, indices)` ordered from most to least similar."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        queries = np.ascontiguousarray(np.asarray(query_vectors, dtype=np.float32))
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or (self.dimension and queries.shape[1] != self.dimension):
            raise ValueError("query vector dimension does not match the index")
        k = min(top_k, len(self.embeddings))
        if k == 0:
            return np.empty((len(queries), 0), dtype=np.float32), np.empty((len(queries), 0), dtype=np.int64)
        if self._index is not None:
            scores, indices = self._index.search(queries, k)
            return scores, indices
        scores = queries @ self.embeddings.T
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, indices, axis=1), indices

    def save(self, directory: str | Path) -> None:
        """Save vectors and backend metadata.  No document text is stored here."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.embeddings)
        (path / "index.json").write_text(
            json.dumps({"dimension": self.dimension, "count": len(self.embeddings), "backend": self.backend}, indent=2),
            encoding="utf-8",
        )
        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorIndex":
        path = Path(directory)
        vectors_path = path / "vectors.npy"
        if not vectors_path.exists():
            raise FileNotFoundError(f"Vector data not found: {vectors_path}")
        result = cls(np.load(vectors_path))
        faiss_path = path / "index.faiss"
        if faiss is not None and faiss_path.exists():
            result._index = faiss.read_index(str(faiss_path))
        return result
