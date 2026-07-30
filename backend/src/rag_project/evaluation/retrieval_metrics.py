from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalMetrics:
    hit_rate: float
    recall: float
    mean_reciprocal_rank: float
    evaluated_questions: int


def evaluate_retrieval(
    predictions: list[list[str]],
    ground_truth: list[list[str]],
) -> RetrievalMetrics:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")
    hits = 0.0
    recalls = 0.0
    reciprocal_ranks = 0.0
    evaluated = 0
    for predicted_ids, relevant_ids in zip(predictions, ground_truth):
        relevant = set(relevant_ids)
        if not relevant:
            continue
        evaluated += 1
        matched = relevant.intersection(predicted_ids)
        hits += float(bool(matched))
        recalls += len(matched) / len(relevant)
        first_rank = next(
            (index + 1 for index, chunk_id in enumerate(predicted_ids) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks += 1.0 / first_rank if first_rank else 0.0
    if not evaluated:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0)
    return RetrievalMetrics(
        hit_rate=hits / evaluated,
        recall=recalls / evaluated,
        mean_reciprocal_rank=reciprocal_ranks / evaluated,
        evaluated_questions=evaluated,
    )
