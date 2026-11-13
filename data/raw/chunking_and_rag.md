# Document Chunking and Hybrid Retrieval Pipeline

## Structure-Aware Chunking Strategy
Naïve fixed-character chunking algorithms frequently truncate code blocks in half, separating function headers from their implementation logic. The DocuRAG pipeline implements a structure-aware chunker that parses markdown headers (H1 through H3) and enforces atomic boundaries on fenced code blocks (`CODE_FENCE_REGEX`).

The maximum tokens allocated per chunk is configured using `--token-limit` (default: 512 tokens), while consecutive chunk overlap is governed by `--overlap-ratio` (default: 0.10, representing a 10% sliding token window).

```python
def chunk_document(doc: Document, token_limit: int = 512, overlap_ratio: float = 0.1) -> List[Chunk]:
    """Split structured document into token-bounded chunks while keeping code blocks intact."""
    # Enforces boundary preservation for code blocks
    return _split_section_into_chunks(doc.doc_id, doc.section_title, doc.content, token_limit, overlap_ratio)
```

## Hybrid Search Fusion Architecture
Dense semantic models excel at understanding paraphrased questions and conceptual similarity, but struggle with out-of-vocabulary technical tokens, rare error strings, and CLI flags. Conversely, BM25 excels at exact keyword and identifier matching but fails on conceptual synonyms.

DocuRAG bridges this semantic-lexical divide by fusing candidate rank lists through Reciprocal Rank Fusion (RRF). RRF scores each chunk according to `1 / (60 + rank)`, deduplicating common chunk IDs and producing a balanced top-k candidate set for Lyd's cross-encoder reranker.
