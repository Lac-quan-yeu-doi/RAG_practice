from __future__ import annotations

from abc import ABC, abstractmethod

from rag_project.schemas import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError
