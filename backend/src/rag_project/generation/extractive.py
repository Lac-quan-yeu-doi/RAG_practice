from __future__ import annotations

from rag_project.generation.base import BaseGenerator
from rag_project.schemas import Citation, RAGResponse, RetrievalResult
from rag_project.text import split_sentences, tokenise


class ExtractiveGenerator(BaseGenerator):
    def __init__(self, max_sentences: int = 4, minimum_overlap: int = 1):
        self.max_sentences = max_sentences
        self.minimum_overlap = minimum_overlap

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> RAGResponse:
        query_tokens = set(tokenise(question))
        candidates: list[tuple[int, int, str, RetrievalResult]] = []
        for result in retrieved:
            for sentence in split_sentences(result.chunk.text):
                overlap = len(query_tokens.intersection(tokenise(sentence)))
                if overlap >= self.minimum_overlap:
                    candidates.append((overlap, -result.rank, sentence, result))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = candidates[:self.max_sentences]
        if not selected:
            return RAGResponse(
                question=question,
                answer="The indexed Python documentation does not contain enough information to answer reliably.",
                answerable=False,
                citations=[],
                retrieved=retrieved,
                metadata={"generator": "extractive"},
            )
        answer = " ".join(sentence for _, _, sentence, _ in selected)
        citations: list[Citation] = []
        seen: set[str] = set()
        for _, _, sentence, result in selected:
            chunk = result.chunk
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    relative_path=chunk.relative_path,
                    section_path=chunk.section_path,
                    excerpt=sentence,
                )
            )
        return RAGResponse(
            question=question,
            answer=answer,
            answerable=True,
            citations=citations,
            retrieved=retrieved,
            metadata={"generator": "extractive"},
        )
