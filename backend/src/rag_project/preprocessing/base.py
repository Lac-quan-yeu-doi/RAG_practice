from __future__ import annotations

from abc import ABC, abstractmethod

from rag_project.schemas import Section, SourceFile


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, source: SourceFile) -> list[Section]:
        raise NotImplementedError
