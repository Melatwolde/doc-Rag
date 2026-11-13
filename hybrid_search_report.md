# Comparative Retrieval Performance Report
## BM25 Lexical vs. MiniLM Dense vs. Hybrid Reciprocal Rank Fusion (RRF)

This evaluation benchmarks the three retrieval mechanisms on the processed document chunks produced by the `DocuRAG` data pipeline. The test set comprises two distinct query profiles: **Exact Technical Queries** (targeting specific concepts, syntax statements, and language features) and **Conceptual / Semantic Queries** (targeting natural language explanations and how-to questions).

---
## 1. Executive Summary & Aggregate Metrics

These metrics describe this labelled benchmark and should not be read as a general guarantee that one retrieval mode is always best. Compare the aggregate metrics with the per-query ranks and category breakdown.

### Overall Benchmark Performance (All 13 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 46.2% | 53.8% | 69.2% | 0.5256 |
| Dense (MiniLM-L6-v2) | 53.8% | 69.2% | 69.2% | 0.6154 |
| **Hybrid (RRF k=60)** | 61.5% | 69.2% | 69.2% | 0.6538 |

### Category Breakdown

#### Exact Technical Queries (7 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 42.9% | 42.9% | 42.9% | 0.4286 |
| Dense (MiniLM-L6-v2) | 42.9% | 42.9% | 42.9% | 0.4286 |
| **Hybrid (RRF k=60)** | 42.9% | 42.9% | 42.9% | 0.4286 |

#### Conceptual / Semantic Queries (6 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 50.0% | 66.7% | 100.0% | 0.6389 |
| Dense (MiniLM-L6-v2) | 66.7% | 100.0% | 100.0% | 0.8333 |
| **Hybrid (RRF k=60)** | 83.3% | 100.0% | 100.0% | 0.9167 |

---
## 2. Detailed Per-Query Retrieval & Rank Shift Log

The table below details the 1-based rank at which the ground-truth relevant chunk was retrieved by each mode (`-` indicates not retrieved in Top 5).

| Query | Category | BM25 Rank | Dense Rank | Hybrid Rank | Hybrid Top-1 Section |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `if statements` | Exact Technical | - | - | - | None |
| `for statements` | Exact Technical | 1 | 1 | **1** | None |
| `defining functions` | Exact Technical | - | - | - | write the result |
| `dictionaries` | Exact Technical | 1 | 1 | **1** | letters in a or b but not both |
| `Reading and Writing Files` | Exact Technical | - | - | - | Note use of 'end' on previous line |
| `Handling Exceptions` | Exact Technical | 1 | 1 | **1** | unpack args |
| `Inheritance` | Exact Technical | - | - | - | Function defined outside the class |
| `How do you make decisions and branch your ...` | Conceptual / Semantic | 3 | 1 | **1** | None |
| `How can I loop over items in a list or a s...` | Conceptual / Semantic | 1 | 2 | **1** | Strategy:  Create a new collection |
| `What is the best way to organize code into...` | Conceptual / Semantic | 4 | 2 | **2** | None |
| `How do I catch errors so my script doesn't...` | Conceptual / Semantic | 4 | 1 | **1** | None |
| `How do classes and object-oriented program...` | Conceptual / Semantic | 1 | 1 | **1** | None |
| `How do you store collections of key-value ...` | Conceptual / Semantic | 1 | 1 | **1** | letters in a or b but not both |

---
## 3. Interpreting the Comparison

- BM25 can help with exact terms and identifiers in the indexed documentation.
- Dense retrieval can help when a query paraphrases the source wording.
- RRF combines ranks; use the benchmark rows above to judge its effect on this corpus.

---
## 4. Downstream Handoff to the Cross-Encoder Reranker
1. **Contract Stability**: `HybridRetriever.retrieve(query, top_k=5)` returns `List[SearchResult]` containing `chunk_id`, `content`, `score`, and `metadata`. This fulfills the input contract for `CrossEncoderReranker.rerank(query, candidates, top_k=3)`.
2. **Candidate Pool**: Candidate count is a recall/latency trade-off. Tune it on labelled queries for the target corpus; a fixed candidate count does not guarantee a particular recall.
3. **Provenance**: Retrieved chunks retain `source_url`, `doc_id`, section, and retrieval metadata when supplied. Source URLs require manifest entries. The answer stage is extractive, not LLM-generated.