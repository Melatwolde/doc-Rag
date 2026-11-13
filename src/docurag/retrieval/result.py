"""docurag/retrieval/result.py

Unified search result data structure for retrieval and reranking pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SearchResult:
    """Represents a retrieved document chunk with its retrieval score and metadata.

    Designed for seamless consumption by downstream components such as
    cross-encoder rerankers and generator LLMs. Supports both attribute
    access and dict-style indexing.
    """

    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chunk_id = str(chunk_id)
        self.content = content
        self.score = float(score)
        self.metadata = metadata.copy() if metadata else {}

    def __getitem__(self, key: str) -> Any:
        if key == "chunk_id":
            return self.chunk_id
        elif key == "content":
            return self.content
        elif key == "score":
            return self.score
        elif key == "metadata":
            return self.metadata
        elif key in self.metadata:
            return self.metadata[key]
        raise KeyError(f"Key '{key}' not found in SearchResult")

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "chunk_id":
            self.chunk_id = str(value)
        elif key == "content":
            self.content = value
        elif key == "score":
            self.score = float(value)
        elif key == "metadata":
            self.metadata = value
        else:
            self.metadata[key] = value

    def __contains__(self, key: str) -> bool:
        return key in {"chunk_id", "content", "score", "metadata"} or key in self.metadata

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to standard dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        section = self.metadata.get("section_title") or "N/A"
        return (
            f"SearchResult(chunk_id={self.chunk_id[:8]}..., "
            f"score={self.score:.4f}, section={section!r})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SearchResult):
            return False
        return self.chunk_id == other.chunk_id and self.score == other.score
