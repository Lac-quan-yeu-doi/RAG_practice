from __future__ import annotations

from abc import ABC, abstractmethod

from rag_project.embeddings.base import BaseEmbeddingModel
from rag_project.schemas import Chunk, RetrievalResult


class BaseVectorStore(ABC):
    @abstractmethod
    def rebuild(self, chunks: list[Chunk], embedding_model: BaseEmbeddingModel) -> None:
        raise NotImplementedError

    @abstractmethod
    def all_chunks(self, document_id: str | None = None) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        embedding_model: BaseEmbeddingModel,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError
