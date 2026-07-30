from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError
