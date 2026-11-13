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

## Documentation provenance and citations

The repository deliberately does not treat the sample files as official source
material. To collect a chosen project's official pages, copy
`data/source_manifest.example.json`, replace the placeholder with verified URLs
and section anchors, then run:

```sh
python scripts/collect_documents.py path/to/your-manifest.json
python -m data_pipeline.run_pipeline
```

The pipeline copies the manifest's source URL and section-specific URL to every
chunk, together with its heading path and source line range. It also saves the
chunk-size/overlap experiment to `data/processed/chunking_experiment.json`.
