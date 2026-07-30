from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urlparse
from difflib import SequenceMatcher

import requests

from rag_project.text import normalise_text


JSON_ARRAY_RE = re.compile(r"\[.*\]", flags=re.DOTALL)


class LocalQuestionGenerator:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        max_output_tokens: int = 700,
        context_size: int = 8192,
        keep_alive: str = "15m",
        retries: int = 2,
    ):
        configured_url = base_url or os.getenv("LLM_BASE_URL", "")
        self.base_url = self._normalise_ollama_url(configured_url)
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("BENCHMARK_LLM_MODEL") or os.getenv("LLM_MODEL", "")
        self.timeout_seconds = timeout_seconds or int(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
        self.max_output_tokens = max_output_tokens
        self.context_size = context_size
        self.keep_alive = keep_alive
        self.retries = retries
        if not self.base_url:
            raise ValueError("LLM_BASE_URL is required. For Ollama use http://localhost:11434")
        if not self.model:
            raise ValueError("BENCHMARK_LLM_MODEL or LLM_MODEL is required")

    def generate(self, context: str, source_metadata: dict, num_questions: int = 2) -> list[dict]:
        if num_questions <= 0:
            return []
        context = context.strip()
        if not context:
            return []

        schema = {
            "type": "array",
            "minItems": num_questions,
            "maxItems": num_questions,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reference_answer": {"type": "string"},
                    "evidence_text": {"type": "string"},
                },
                "required": ["question", "reference_answer", "evidence_text"],
                "additionalProperties": False,
            },
        }

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "format": schema,
            "options": {
                "temperature": 0.1,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.context_size,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate benchmark questions from one Python documentation section. "
                        "Return only the requested JSON array. Each item must contain question, "
                        "reference_answer, and evidence_text. Each question must be answerable "
                        "only from the supplied context. The question should paraphrase the "
                        "documentation rather than copy it. The reference answer must be concise. "
                        "The evidence_text value must be copied exactly from the supplied context "
                        "and should contain the minimum passage required to support the answer. "
                        "Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source metadata:\n"
                        f"{json.dumps(source_metadata, ensure_ascii=False)}\n\n"
                        f"Generate exactly {num_questions} questions from this context:\n\n"
                        f"{context}"
                    ),
                },
            ],
        }

        response_data = self._post_with_retry(payload)
        message = response_data.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError(f"Ollama returned an empty response: {response_data}")
        generated = self._parse_json_array(content)
        return self._validate(generated, context, num_questions)

    def warm_up(self) -> None:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 8,
                "num_ctx": 2048,
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with OK.",
                }
            ],
        }
        self._post_with_retry(payload)

    def _post_with_retry(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload,
                    timeout=(10, self.timeout_seconds),
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ReadTimeout as error:
                last_error = error
                if attempt > self.retries:
                    break
                time.sleep(attempt * 2)
            except requests.exceptions.ConnectionError as error:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Check that Ollama is running with `ollama serve`."
                ) from error
            except requests.exceptions.HTTPError as error:
                body = error.response.text if error.response is not None else ""
                raise RuntimeError(f"Ollama request failed: {error}\n{body}") from error

        raise TimeoutError(
            f"Ollama did not finish within {self.timeout_seconds} seconds after "
            f"{self.retries + 1} attempts. Reduce --max-context-chars, use fewer "
            "questions per section, or use a faster instruct model."
        ) from last_error

    @staticmethod
    def _normalise_ollama_url(url: str) -> str:
        url = url.strip().rstrip("/")
        if not url:
            return ""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")

    @staticmethod
    def _parse_json_array(content: str) -> list[dict]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = JSON_ARRAY_RE.search(content)
            if not match:
                raise ValueError(f"Question generator did not return a JSON array: {content}")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            raise ValueError("Question generator output must be a JSON array")
        return parsed

    @staticmethod
    def _resolve_evidence(
        generated_evidence: str,
        context: str,
        normalised_context: str,
        minimum_similarity: float = 0.65,
    ) -> str | None:
        normalised_generated = " ".join(
            normalise_text(generated_evidence).lower().split()
        )

        if normalised_generated in normalised_context:
            return generated_evidence

        candidates = [
            candidate.strip()
            for candidate in re.split(r"\n\s*\n", context)
            if candidate.strip()
        ]

        if not candidates:
            return None

        best_candidate: str | None = None
        best_score = 0.0

        for candidate in candidates:
            normalised_candidate = " ".join(
                normalise_text(candidate).lower().split()
            )

            score = SequenceMatcher(
                None,
                normalised_generated,
                normalised_candidate,
            ).ratio()

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score < minimum_similarity:
            return None

        return best_candidate

    @staticmethod
    def _validate(items: list[dict], context: str, limit: int) -> list[dict]:
        normalised_context = " ".join(normalise_text(context).lower().split())
        valid: list[dict] = []
        seen_questions: set[str] = set()

        for item in items:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            answer = str(item.get("reference_answer", "")).strip()
            generated_evidence = str(item.get("evidence_text", "")).strip()
            question_key = question.lower()

            if not question or not answer or not generated_evidence:
                continue

            if question_key in seen_questions:
                continue

            evidence = LocalQuestionGenerator._resolve_evidence(
                generated_evidence=generated_evidence,
                context=context,
                normalised_context=normalised_context,
            )

            if evidence is None:
                print(f"Rejected question because evidence could not be resolved: {question}")
                continue

            valid.append(
                {
                    "question": question,
                    "reference_answer": answer,
                    "evidence_text": evidence,
                }
            )
            seen_questions.add(question_key)

            if len(valid) == limit:
                break

        return valid


