# Comparative Retrieval Performance Report
## BM25 Lexical vs. MiniLM Dense vs. Hybrid Reciprocal Rank Fusion (RRF)

This evaluation benchmarks the three retrieval mechanisms on the processed document chunks produced by the `DocuRAG` data pipeline. The test set comprises two distinct query profiles: **Exact Technical Queries** (targeting specific identifiers, CLI flags, configuration parameters, and error constants) and **Conceptual / Semantic Queries** (targeting natural language explanations, architectural patterns, and conceptual how-to questions).

---
## 1. Executive Summary & Aggregate Metrics

These metrics describe this labelled benchmark and should not be read as a general guarantee that one retrieval mode is always best. Compare the aggregate metrics with the per-query ranks and category breakdown.

### Overall Benchmark Performance (All 13 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 92.3% | 100.0% | 100.0% | 0.9615 |
| Dense (MiniLM-L6-v2) | 84.6% | 100.0% | 100.0% | 0.9103 |
| **Hybrid (RRF k=60)** | 92.3% | 100.0% | 100.0% | 0.9615 |

### Category Breakdown

#### Exact Technical Queries (7 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 100.0% | 100.0% | 100.0% | 1.0000 |
| Dense (MiniLM-L6-v2) | 71.4% | 100.0% | 100.0% | 0.8333 |
| **Hybrid (RRF k=60)** | 100.0% | 100.0% | 100.0% | 1.0000 |

#### Conceptual / Semantic Queries (6 Queries)

| Retrieval Mode | Recall@1 (%) | Recall@3 (%) | Recall@5 (%) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 (Lexical) | 83.3% | 100.0% | 100.0% | 0.9167 |
| Dense (MiniLM-L6-v2) | 100.0% | 100.0% | 100.0% | 1.0000 |
| **Hybrid (RRF k=60)** | 83.3% | 100.0% | 100.0% | 0.9167 |

---
## 2. Detailed Per-Query Retrieval & Rank Shift Log

The table below details the 1-based rank at which the ground-truth relevant chunk was retrieved by each mode (`-` indicates not retrieved in Top 5).

| Query | Category | BM25 Rank | Dense Rank | Hybrid Rank | Hybrid Top-1 Section |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `ERR_AUTH_TOKEN_EXPIRED` | Exact Technical | 1 | 1 | **1** | Authentication and Session Management Service > JWT Architecture and Token Lifecycle |
| `--token-limit` | Exact Technical | 1 | 3 | **1** | Document Chunking and Hybrid Retrieval Pipeline > Structure-Aware Chunking Strategy |
| `refresh_access_token` | Exact Technical | 1 | 2 | **1** | Authentication and Session Management Service > Refresh Token Exchange Endpoint |
| `database.pool_size` | Exact Technical | 1 | 1 | **1** | Database Connection Management and High Availability > Connection Pooling Strategy |
| `ERR_CONN_TIMEOUT` | Exact Technical | 1 | 1 | **1** | Database Connection Management and High Availability > Connection Pooling Strategy |
| `--db-replica-url` | Exact Technical | 1 | 1 | **1** | Database Connection Management and High Availability > Read Replica Routing and Load Distribution |
| `auth.jwt.secret_key` | Exact Technical | 1 | 1 | **1** | Authentication and Session Management Service > Security Configuration and Secrets |
| `How do client apps renew user credentials ...` | Conceptual / Semantic | 1 | 1 | **1** | Authentication and Session Management Service > Refresh Token Exchange Endpoint |
| `Strategies for managing multiple database ...` | Conceptual / Semantic | 2 | 1 | **2** | Database Connection Management and High Availability > Read Replica Routing and Load Distribution |
| `How does the chunker prevent splitting cod...` | Conceptual / Semantic | 1 | 1 | **1** | Document Chunking and Hybrid Retrieval Pipeline > Structure-Aware Chunking Strategy |
| `Alleviating primary server load by directi...` | Conceptual / Semantic | 1 | 1 | **1** | Database Connection Management and High Availability > Read Replica Routing and Load Distribution |
| `Why combine keyword search with semantic e...` | Conceptual / Semantic | 1 | 1 | **1** | Document Chunking and Hybrid Retrieval Pipeline > Hybrid Search Fusion Architecture |
| `What algorithm and encryption keys should ...` | Conceptual / Semantic | 1 | 1 | **1** | Authentication and Session Management Service > Security Configuration and Secrets |

---
## 3. Interpreting the Comparison

- BM25 is useful for exact identifiers, error codes, configuration paths, and CLI flags.
- Dense retrieval can find relevant passages when a question uses different wording from the documentation.
- RRF combines ranks from both methods; whether that improves a query is determined by the measured per-query results above.

---
## 4. Downstream Handoff to Lyd's Cross-Encoder Reranker

1. **Contract Stability**: `HybridRetriever.retrieve(query, top_k=5)` returns `List[SearchResult]` containing `chunk_id`, `content`, `score`, and `metadata`. This fulfills the input contract for `CrossEncoderReranker.rerank(query, candidates, top_k=3)`.
2. **Candidate Pool**: Candidate count is a recall/latency trade-off. Tune it against labelled queries on the target corpus; no recall guarantee is implied by a fixed candidate count.
3. **Provenance**: Retrieved chunks retain `source_url`, `doc_id`, section, and retrieval metadata when supplied. Source URLs require manifest entries; this project does not currently implement an LLM answer or citation renderer.