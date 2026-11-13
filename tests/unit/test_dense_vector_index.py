from __future__ import annotations

import numpy as np

from docurag.retrieval.dense_search import DenseRetriever
from docurag.retrieval.vector_index import FaissVectorIndex


class FakeEncoder:
    embedding_dim = 2

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        values = {"cats": [1.0, 0.0], "dogs": [0.0, 1.0], "feline query": [0.9, 0.1]}
        if isinstance(texts, str):
            texts = [texts]
        vectors = np.asarray([values[text] for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True) if normalize_embeddings else vectors


def test_vector_index_returns_cosine_rank_order_and_persists(tmp_path):
    index = FaissVectorIndex(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    scores, ids = index.search(np.asarray([[0.8, 0.2]], dtype=np.float32), top_k=10)
    assert ids.tolist() == [[0, 1]]
    assert scores[0, 0] > scores[0, 1]
    index.save(tmp_path)
    restored = FaissVectorIndex.load(tmp_path)
    assert restored.search(np.asarray([0.8, 0.2], dtype=np.float32), 2)[1].tolist() == [[0, 1]]


def test_dense_retriever_uses_persistent_index_and_honours_top_k(tmp_path):
    chunks = [{"chunk_id": "cat", "content": "cats"}, {"chunk_id": "dog", "content": "dogs"}]
    retriever = DenseRetriever(chunks, encoder=FakeEncoder())
    assert [item.chunk_id for item in retriever.retrieve("feline query", top_k=1)] == ["cat"]
    retriever.save_index(tmp_path)
    restored = DenseRetriever.load_index(tmp_path, encoder=FakeEncoder())
    results = restored.retrieve("feline query", top_k=10)
    assert [item.chunk_id for item in results] == ["cat", "dog"]
    assert results[0].metadata["vector_backend"] in {"faiss", "numpy"}
    assert restored.retrieve("feline query", top_k=0) == []
