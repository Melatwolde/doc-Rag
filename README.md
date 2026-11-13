# DocuRAG

## Dense semantic retrieval

Document chunks are embedded with `all-MiniLM-L6-v2` and searched using cosine
similarity (inner product over L2-normalized vectors). FAISS provides the
persisted exact-vector index in deployed environments; the same files can be
searched through the built-in NumPy fallback during lightweight development.

```sh
pip install -r requirements.txt
python scripts/build_index.py
python scripts/evaluate_dense_retrieval.py
```

The index is written to `data/processed/dense_index/`. The evaluation produces
`dense_retrieval_report.md` and reports whether the labelled paragraph is found
at top-3, top-5, and top-10.
