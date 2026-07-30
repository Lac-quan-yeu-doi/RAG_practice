from __future__ import annotations

from rag_project.embeddings.base import BaseEmbeddingModel
from rag_project.retrieval.base import BaseRetriever
from rag_project.schemas import RetrievalResult
from rag_project.storage.base import BaseVectorStore


class DenseRetriever(BaseRetriever):
    def __init__(self, store: BaseVectorStore, embedding_model: BaseEmbeddingModel):
        self.store = store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.store.similarity_search(
            query=query,
            embedding_model=self.embedding_model,
            top_k=top_k,
        )
