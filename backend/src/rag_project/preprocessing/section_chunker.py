from __future__ import annotations

import hashlib
import re

from rag_project.schemas import Chunk, Section
from rag_project.text import normalise_text


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class SectionChunker:
    def __init__(self, max_chars: int = 2400, overlap_paragraphs: int = 1, minimum_chars: int = 120):
        if max_chars < 300:
            raise ValueError("max_chars must be at least 300")
        if overlap_paragraphs < 0:
            raise ValueError("overlap_paragraphs cannot be negative")
        if minimum_chars < 1:
            raise ValueError("minimum_chars must be positive")
        self.max_chars = max_chars
        self.overlap_paragraphs = overlap_paragraphs
        self.minimum_chars = minimum_chars

    def chunk(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        per_file_index: dict[str, int] = {}
        for section in sections:
            for text in self._split_section(section.text):
                if len(text.strip()) < self.minimum_chars:
                    continue
                index = per_file_index.get(section.relative_path, 0)
                per_file_index[section.relative_path] = index + 1
                chunks.append(self._make_chunk(section, index, text))
        return chunks

    def _split_section(self, text: str) -> list[str]:
        text = normalise_text(text)
        if len(text) <= self.max_chars:
            return [text]
        paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
        if not paragraphs:
            return self._split_long_block(text)
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                if current:
                    chunks.append(normalise_text("\n\n".join(current)))
                    current = []
                    current_length = 0
                chunks.extend(self._split_long_block(paragraph))
                continue
            projected = current_length + len(paragraph) + (2 if current else 0)
            if current and projected > self.max_chars:
                chunks.append(normalise_text("\n\n".join(current)))
                overlap = current[-self.overlap_paragraphs :] if self.overlap_paragraphs else []
                current = list(overlap)
                current_length = sum(len(item) for item in current) + max(0, len(current) - 1) * 2
            current.append(paragraph)
            current_length += len(paragraph) + (2 if len(current) > 1 else 0)
        if current:
            chunks.append(normalise_text("\n\n".join(current)))
        return chunks

    def _split_long_block(self, text: str) -> list[str]:
        lines = text.splitlines()
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for line in lines:
            if len(line) > self.max_chars:
                if current:
                    chunks.append(normalise_text("\n".join(current)))
                    current = []
                    current_length = 0
                for start in range(0, len(line), self.max_chars):
                    chunks.append(line[start:start + self.max_chars].strip())
                continue
            additional = len(line) + (1 if current else 0)
            if current and current_length + additional > self.max_chars:
                chunks.append(normalise_text("\n".join(current)))
                current = []
                current_length = 0
            current.append(line)
            current_length += additional
        if current:
            chunks.append(normalise_text("\n".join(current)))
        return chunks

    @staticmethod
    def _make_chunk(section: Section, chunk_index: int, text: str) -> Chunk:
        identity = (
            f"{section.document_id}|{section.relative_path}|"
            f"{' > '.join(section.section_path)}|{chunk_index}|{text}"
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:14]
        return Chunk(
            chunk_id=f"{section.document_id}_{digest}",
            document_id=section.document_id,
            document_title=section.document_title,
            source_path=section.source_path,
            relative_path=section.relative_path,
            section_path=section.section_path,
            chunk_index=chunk_index,
            text=text,
        )
