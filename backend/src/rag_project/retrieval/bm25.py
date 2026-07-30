from __future__ import annotations

import math
from collections import Counter, defaultdict

from rag_project.retrieval.base import BaseRetriever
from rag_project.schemas import RetrievalResult
from rag_project.storage.base import BaseVectorStore
from rag_project.text import tokenise


class BM25Retriever(BaseRetriever):
    def __init__(self, store: BaseVectorStore, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.store = store
        self.k1 = k1
        self.b = b
        self.chunks = store.all_chunks()
        if not self.chunks:
            raise ValueError("The vector store is empty. Run scripts/build_index.py first.")
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: dict[str, int] = {}
        self.document_lengths: list[int] = []
        self.average_document_length = 0.0
        self._fit()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        query_terms = tokenise(query)
        if not query_terms:
            return []
        scores = [self._score(query_terms, index) for index in range(len(self.chunks))]
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever="bm25",
            )
            for rank, index in enumerate(ranked_indices[:top_k], start=1)
        ]

    def _fit(self) -> None:
        document_frequencies: defaultdict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            tokens = tokenise(chunk.search_text)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            for term in frequencies:
                document_frequencies[term] += 1
        self.document_frequencies = dict(document_frequencies)
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)

    def _score(self, query_terms: list[str], index: int) -> float:
        score = 0.0
        frequencies = self.term_frequencies[index]
        document_length = self.document_lengths[index]
        number_of_documents = len(self.chunks)
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (number_of_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalisation = 1.0 - self.b + self.b * document_length / max(
                self.average_document_length,
                1.0,
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (self.k1 + 1.0)
                / (term_frequency + self.k1 * length_normalisation)
            )
        return score
