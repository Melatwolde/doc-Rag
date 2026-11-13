"""docurag/retrieval/bm25_search.py

BM25 Lexical Keyword Retrieval Engine.

Features:
- Technical tokenization preserving exact identifiers, snake_case, camelCase,
  CLI flags (--flag), error codes (ERR_*), and dotted config keys.
- Indexed over processed document chunks from Beza's pipeline.
- Returns ranked SearchResult objects preserving chunk IDs and full metadata for citations.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, List, Optional, Union

from rank_bm25 import BM25Okapi

from .result import SearchResult

# Regular expression pattern for technical tokens:
# 1. CLI flags or dotted keys or snake_case with dashes/underscores/dots:
#    e.g., --token-limit, auth.jwt.secret_key, ERR_AUTH_TOKEN_EXPIRED, refresh_access_token
# 2. CamelCase words or alphanumeric tokens: e.g. JWTAuthenticator, poolSize, 512
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?:--?)?[a-zA-Z0-9]+(?:[._\-][a-zA-Z0-9]+)+|[a-zA-Z0-9]+"
)


def tokenize_technical_text(text: str) -> List[str]:
    """Tokenize technical text while strictly preserving code identifiers, flags, and error codes.

    Handles:
    - Lowercasing for case-insensitive matching
    - CLI flags: `--token-limit`, `--db-replica`
    - Function names and snake_case: `refresh_access_token`, `verify_signature`
    - Dotted paths and config keys: `database.pool_size`, `auth.jwt.secret_key`
    - Error codes: `ERR_AUTH_TOKEN_EXPIRED`, `ERR_CONN_TIMEOUT`
    - Dual emission: Emits both compound tokens and component subwords so
      queries can match either the exact symbol or subparts.
    """
    if not text:
        return []

    # Strip markdown code fence markers (```python -> python) and punctuation brackets
    cleaned = text.replace("`", " ").replace("(", " ").replace(")", " ")
    cleaned = cleaned.replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")
    cleaned = cleaned.replace(":", " ").replace(";", " ").replace('"', " ").replace("'", " ")

    matches = TECHNICAL_TOKEN_PATTERN.findall(cleaned)
    tokens: List[str] = []

    for match in matches:
        lower_match = match.lower()
        if lower_match.startswith("--"):
            clean = "--" + lower_match[2:].strip(".,;?! ")
        elif lower_match.startswith("-"):
            clean = "-" + lower_match[1:].strip(".,;?! ")
        else:
            clean = lower_match.strip("._-.,;?! ")

        if not clean:
            continue
        tokens.append(clean)

        # If token has internal delimiters ('_', '-', '.'), also emit sub-tokens
        if any(sep in clean for sep in ("_", "-", ".")):
            subparts = re.split(r"[-_.]+", clean.lstrip("-"))
            for sub in subparts:
                if len(sub) > 1 and sub not in tokens:
                    tokens.append(sub)

    return tokens


class BM25Retriever:
    """Lexical retriever utilizing BM25Okapi over processed document chunks.

    Attributes
    ----------
    k1 : float
        BM25 term frequency saturation parameter (default 1.5).
    b : float
        BM25 document length normalization parameter (default 0.75).
    """

    def __init__(
        self,
        chunks: Optional[List[Union[Dict[str, Any], Any]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

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
        """Build the BM25 index over the provided document chunks."""
        self.chunks = [self._normalize_chunk(c) for c in chunks]
        self.tokenized_corpus = [
            tokenize_technical_text(c.get("content", "")) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b, epsilon=0.25)

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, pathlib.Path],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Initialize BM25Retriever directly from a chunks.json file."""
        path = pathlib.Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        return cls(chunks=chunks, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve the top-k chunks matching the query using BM25 scoring.

        Parameters
        ----------
        query : str
            User search query string.
        top_k : int
            Number of top results to return.

        Returns
        -------
        List[SearchResult]
            Ranked list of results with BM25 scores and preserved chunk metadata.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_technical_text(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        query_set = set(query_tokens)

        # Count token overlap as secondary ranking key / tiebreaker
        overlap_counts = [
            sum(1 for t in self.tokenized_corpus[i] if t in query_set)
            for i in range(len(self.chunks))
        ]

        # Sort indices by score first, then overlap count
        ranked_indices = sorted(
            range(len(doc_scores)),
            key=lambda idx: (float(doc_scores[idx]), overlap_counts[idx]),
            reverse=True,
        )

        results: List[SearchResult] = []
        for rank_idx, idx in enumerate(ranked_indices[:top_k]):
            score = float(doc_scores[idx])
            overlap = overlap_counts[idx]

            # If no score and no overlap, do not return irrelevant documents if we already have matches
            if score <= 0.0 and overlap == 0 and len(results) > 0:
                break

            chunk_data = self.chunks[idx]
            metadata = {
                "doc_id": chunk_data.get("doc_id"),
                "source_url": chunk_data.get("source_url"),
                "section_title": chunk_data.get("section_title"),
                "char_count": chunk_data.get("char_count"),
                "token_count": chunk_data.get("token_count"),
                "retriever": "bm25",
                "raw_bm25_score": score,
                "term_overlap_count": overlap,
                "lexical_rank": rank_idx + 1,
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
