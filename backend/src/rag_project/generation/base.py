from __future__ import annotations

from abc import ABC, abstractmethod

from rag_project.schemas import RAGResponse, RetrievalResult


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved: list[RetrievalResult]) -> RAGResponse:
        raise NotImplementedError
