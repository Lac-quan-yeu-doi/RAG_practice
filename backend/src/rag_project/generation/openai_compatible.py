from __future__ import annotations

import json
import os
import re

import requests

from rag_project.generation.base import BaseGenerator
from rag_project.schemas import Citation, RAGResponse, RetrievalResult


JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


class OpenAICompatibleGenerator(BaseGenerator):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout_seconds = timeout_seconds or int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        if not self.base_url:
            raise ValueError("LLM_BASE_URL is required")
        if not self.model:
            raise ValueError("LLM_MODEL is required")

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> RAGResponse:
        contexts = self._format_contexts(retrieved)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied Python documentation context. "
                        "If the context is insufficient, set answerable to false. "
                        "Return valid JSON with keys answerable, answer, and cited_chunk_ids. "
                        "cited_chunk_ids may contain only IDs present in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\\n{question}\\n\\n"
                        f"Context:\\n{contexts}\\n\\n"
                        "Give a concise technical answer and cite the relevant chunks."
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = self._parse_json(content)
        citations = self._build_citations(parsed.get("cited_chunk_ids", []), retrieved)
        return RAGResponse(
            question=question,
            answer=str(parsed.get("answer", "")).strip(),
            answerable=bool(parsed.get("answerable", False)),
            citations=citations,
            retrieved=retrieved,
            metadata={"generator": "openai-compatible", "model": self.model},
        )

    @staticmethod
    def _format_contexts(retrieved: list[RetrievalResult]) -> str:
        blocks: list[str] = []
        for result in retrieved:
            chunk = result.chunk
            metadata = {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document_title,
                "source_file": chunk.relative_path,
                "section": chunk.section_path,
            }
            blocks.append(
                f"METADATA: {json.dumps(metadata, ensure_ascii=False)}\\n"
                f"TEXT:\\n{chunk.text}"
            )
        return "\\n\\n---\\n\\n".join(blocks)

    @staticmethod
    def _parse_json(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = JSON_BLOCK_RE.search(content)
            if not match:
                raise ValueError(f"Model did not return JSON: {content}")
            return json.loads(match.group(0))

    @staticmethod
    def _build_citations(chunk_ids: list[str], retrieved: list[RetrievalResult]) -> list[Citation]:
        by_id = {result.chunk.chunk_id: result.chunk for result in retrieved}
        citations: list[Citation] = []
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    relative_path=chunk.relative_path,
                    section_path=chunk.section_path,
                    excerpt=chunk.text[:500].strip(),
                )
            )
        return citations
