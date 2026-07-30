from __future__ import annotations

from collections import Counter

from rag_project.text import tokenise


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(tokenise(prediction)) == " ".join(tokenise(reference)))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = tokenise(prediction)
    reference_tokens = tokenise(reference)
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)
