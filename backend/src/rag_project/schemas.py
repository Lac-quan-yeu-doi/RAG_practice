from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceFile:
    document_id: str
    document_title: str
    source_path: str
    relative_path: str
    text: str


@dataclass(slots=True)
class Section:
    document_id: str
    document_title: str
    source_path: str
    relative_path: str
    section_path: list[str]
    text: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    source_path: str
    relative_path: str
    section_path: list[str]
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def section_title(self) -> str:
        return self.section_path[-1] if self.section_path else self.relative_path

    @property
    def search_text(self) -> str:
        heading = " > ".join(self.section_path)
        return (
            f"Document: {self.document_title}\n"
            f"Source: {self.relative_path}\n"
            f"Section: {heading}\n\n"
            f"{self.text}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "retriever": self.retriever,
        }


@dataclass(slots=True)
class Citation:
    chunk_id: str
    document_id: str
    document_title: str
    relative_path: str
    section_path: list[str]
    excerpt: str


@dataclass(slots=True)
class RAGResponse:
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]
    retrieved: list[RetrievalResult]
    metadata: dict[str, Any] = field(default_factory=dict)
