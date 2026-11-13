"""src/docurag/pipeline.py

End-to-end integration module for the DocuRAG system.
Combines:
- Hybrid Search (Herm & Melat via HybridRetriever)
- Cross-Encoder Reranking & LLM Generation (Lyd)
- Abstention Guardrails & Confidence Calibration (Elbi)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict

from docurag.evaluation.abstention import AbstentionEngine

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "where", "which", "with",
}


def _evidence_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in _STOP_WORDS
    }


class HybridRAGPipeline:
    def __init__(
        self,
        chunks_path: str | Path = "data/processed/chunks.json",
        confidence_threshold: float = 0.45,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 3,
        hybrid_retriever: Any | None = None,
        cross_encoder_reranker: Any | None = None,
        use_cross_encoder: bool = True,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        if top_k_retrieve < 1 or top_k_rerank < 1:
            raise ValueError("top_k_retrieve and top_k_rerank must be at least 1")
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        if hybrid_retriever is None:
            from docurag.retrieval.hybrid_search import HybridRetriever

            hybrid_retriever = HybridRetriever.from_json(self.chunks_path)
        self.hybrid_retriever = hybrid_retriever
        if use_cross_encoder and cross_encoder_reranker is None:
            from docurag.reranking.cross_encoder import CrossEncoderReranker

            cross_encoder_reranker = CrossEncoderReranker()
        if not use_cross_encoder and cross_encoder_reranker is not None:
            raise ValueError("cross_encoder_reranker cannot be set when use_cross_encoder is False")
        self.cross_encoder_reranker = cross_encoder_reranker
        self.abstention_engine = AbstentionEngine(confidence_threshold=confidence_threshold)

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the complete RAG pipeline for a given user query."""
        started_at = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        retrieved_results = self.hybrid_retriever.retrieve(query, top_k=self.top_k_retrieve)
        candidate_by_id: Dict[str, Dict[str, Any]] = {}
        usable_results = []
        query_terms = _evidence_terms(query)
        for result in retrieved_results:
            if not result.content or not result.content.strip():
                continue
            metadata = dict(result.metadata)
            metadata["query_token_count"] = len(query_terms)
            metadata["lexical_confidence"] = (
                len(query_terms.intersection(_evidence_terms(result.content))) / len(query_terms)
                if query_terms
                else 0.0
            )
            candidate_by_id[result.chunk_id] = {
                "chunk_id": result.chunk_id,
                "doc_id": metadata.get("doc_id"),
                "content": result.content,
                "section_title": metadata.get("section_title"),
                "score": result.score,
                "confidence_score": self.abstention_engine.confidence_score(
                    {"score": result.score, "metadata": metadata}
                ),
                "metadata": metadata,
            }
            usable_results.append(result)

        if self.cross_encoder_reranker is not None and usable_results:
            ranked_results = self.cross_encoder_reranker.rerank(
                query,
                usable_results,
                top_k=self.top_k_rerank,
            )
            reranked_passages = []
            for result in ranked_results:
                passage = candidate_by_id[result.chunk_id].copy()
                passage["score"] = result.score
                passage["metadata"] = {**passage["metadata"], **result.metadata}
                reranked_passages.append(passage)
        else:
            reranked_passages = sorted(
                candidate_by_id.values(),
                key=lambda passage: self.abstention_engine.confidence_score(passage),
                reverse=True,
            )[: self.top_k_rerank]

        is_abstaining, abstention_reason = self.abstention_engine.evaluate_answerability(reranked_passages)

        if is_abstaining:
            return {
                "query": query,
                "is_abstaining": True,
                "abstention_reason": abstention_reason,
                "answer": "I'm sorry, but the documentation does not contain sufficient verified evidence to answer your question.",
                "citations": [],
                "retrieved_chunks": retrieved_results,
                "reranked_chunks": reranked_passages,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            }

        citations = []
        for chunk in reranked_passages:
            metadata = chunk.get("metadata", {})
            citations.append({
                "chunk_id": chunk["chunk_id"],
                "section_title": chunk.get("section_title") or metadata.get("section_title"),
                "source_url": metadata.get("source_url"),
                "heading_path": metadata.get("heading_path"),
                "line_start": metadata.get("line_start"),
                "line_end": metadata.get("line_end"),
            })
        answer_parts = []
        for passage in reranked_passages:
            section_title = passage.get("section_title") or "Documentation"
            answer_parts.append(
                f"**{section_title}** (Chunk `{passage['chunk_id']}`): "
                f"{passage['content'].strip()}"
            )
        answer_text = "Based on the retrieved documentation:\n\n" + "\n\n".join(answer_parts)

        return {
            "query": query,
            "is_abstaining": False,
            "abstention_reason": None,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": retrieved_results,
            "reranked_chunks": reranked_passages,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
