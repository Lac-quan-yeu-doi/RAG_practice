from __future__ import annotations

from time import perf_counter

from rag_project.generation.base import BaseGenerator
from rag_project.retrieval.base import BaseRetriever
from rag_project.schemas import RAGResponse


class RAGPipeline:
    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        started = perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_seconds = perf_counter() - started
        response = self.generator.generate(question, retrieved)
        response.metadata["retrieval_seconds"] = retrieval_seconds
        response.metadata["top_k"] = self.top_k
        return response
