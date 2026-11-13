"""docurag/embeddings/encoder.py

Dense vector embedding encoder using SentenceTransformers with all-MiniLM-L6-v2.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class MiniLMEncoder:
    """MiniLM text embedding encoder.

    Parameters
    ----------
    model_name : str
        Pretrained SentenceTransformer model identifier (default: "all-MiniLM-L6-v2").
    device : Optional[str]
        Device to run the model on ('cpu', 'cuda', etc.). If None, automatically selected.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text strings into dense embedding vectors.

        Parameters
        ----------
        texts : Union[str, List[str]]
            Single string or list of text strings to encode.
        normalize_embeddings : bool
            Whether to L2-normalize embeddings for cosine similarity via dot product.
        batch_size : int
            Batch size for inference.

        Returns
        -------
        np.ndarray
            2D numpy array of shape (N, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()
