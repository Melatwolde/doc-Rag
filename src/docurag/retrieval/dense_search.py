"""docurag/retrieval/dense_search.py

Dense Semantic Vector Retriever using MiniLM embeddings.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..embeddings.encoder import MiniLMEncoder
from .result import SearchResult


class DenseRetriever:
    """Dense semantic vector retriever using MiniLM embeddings and cosine similarity.

    Parameters
    ----------
    chunks : Optional[List[Union[Dict[str, Any], Any]]]
        List of Chunk objects or dictionaries.
    encoder : Optional[MiniLMEncoder]
        Encoder instance. If None, default MiniLMEncoder is initialized.
    model_name : str
        Model name if encoder is not provided (default: "all-MiniLM-L6-v2").
    """

    def __init__(
        self,
        chunks: Optional[List[Union[Dict[str, Any], Any]]] = None,
        encoder: Optional[MiniLMEncoder] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.encoder = encoder or MiniLMEncoder(model_name=model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        if chunks:
            self.index(chunks)

    def _normalize_chunk(self, chunk: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Convert pydantic Chunk model or dict into standardized dict."""
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump()
        elif hasattr(chunk, "dict"):
            return chunk.dict()
        elif isinstance(chunk, dict):
            return chunk.copy()
        else:
            return {
                "chunk_id": getattr(chunk, "chunk_id", str(id(chunk))),
                "doc_id": getattr(chunk, "doc_id", "unknown"),
                "content": getattr(chunk, "content", str(chunk)),
                "source_url": getattr(chunk, "source_url", None),
                "section_title": getattr(chunk, "section_title", None),
                "char_count": getattr(chunk, "char_count", len(str(chunk))),
                "token_count": getattr(chunk, "token_count", len(str(chunk).split())),
                "metadata": getattr(chunk, "metadata", {}),
            }

    def index(self, chunks: List[Union[Dict[str, Any], Any]]) -> None:
        """Encode chunks and store their dense vectors."""
        self.chunks = [self._normalize_chunk(c) for c in chunks]
        texts = [c.get("content", "") for c in self.chunks]
        if texts:
            self.embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        else:
            self.embeddings = np.empty((0, self.encoder.embedding_dim), dtype=np.float32)

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, pathlib.Path],
        encoder: Optional[MiniLMEncoder] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> DenseRetriever:
        """Initialize DenseRetriever directly from chunks.json."""
        path = pathlib.Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        return cls(chunks=chunks, encoder=encoder, model_name=model_name)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve top-k chunks with highest cosine similarity to query.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Number of top results to return.

        Returns
        -------
        List[SearchResult]
            Ranked list of results with dense similarity scores.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vector = self.encoder.encode(query, normalize_embeddings=True)
        # Cosine similarity via dot product of normalized embeddings
        similarities = np.dot(self.embeddings, query_vector.T).flatten()

        ranked_indices = np.argsort(similarities)[::-1]

        results: List[SearchResult] = []
        for rank_idx, idx in enumerate(ranked_indices[:top_k]):
            score = float(similarities[idx])
            chunk_data = self.chunks[idx]
            metadata = {
                "doc_id": chunk_data.get("doc_id"),
                "source_url": chunk_data.get("source_url"),
                "section_title": chunk_data.get("section_title"),
                "char_count": chunk_data.get("char_count"),
                "token_count": chunk_data.get("token_count"),
                "retriever": "dense",
                "dense_score": score,
                "dense_rank": rank_idx + 1,
                **(chunk_data.get("metadata") or {}),
            }

            results.append(
                SearchResult(
                    chunk_id=chunk_data.get("chunk_id", str(idx)),
                    content=chunk_data.get("content", ""),
                    score=score,
                    metadata=metadata,
                )
            )

        return results
