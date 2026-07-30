from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_project.embeddings.base import BaseEmbeddingModel


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            embedding = self.model.encode_query(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(embedding, dtype=np.float32)
