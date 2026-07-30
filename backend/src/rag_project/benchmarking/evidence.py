from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rag_project.schemas import Chunk
from rag_project.text import normalise_text, tokenise


@dataclass(slots=True)
class EvidenceMapping:
    relevant_chunk_ids: list[str]
    evidence_chunk_ids: list[list[str]]
    evidence_coverages: list[float]
    resolved: bool


def token_coverage(evidence_text: str, candidate_text: str) -> float:
    evidence_tokens = Counter(tokenise(evidence_text))
    if not evidence_tokens:
        return 0.0
    candidate_tokens = Counter(tokenise(candidate_text))
    overlap = sum((evidence_tokens & candidate_tokens).values())
    return overlap / sum(evidence_tokens.values())


def combined_token_coverage(evidence_text: str, chunks: list[Chunk]) -> float:
    combined_text = "\n".join(chunk.text for chunk in chunks)
    return token_coverage(evidence_text, combined_text)


def map_record_evidence(record: dict, chunks: list[Chunk], threshold: float = 0.8) -> EvidenceMapping:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    evidence_items = record.get("evidence", [])
    if not evidence_items:
        return EvidenceMapping([], [], [], False)
    all_relevant_ids: list[str] = []
    evidence_chunk_ids: list[list[str]] = []
    evidence_coverages: list[float] = []
    all_resolved = True
    for evidence in evidence_items:
        candidates = _filter_candidates(evidence, chunks)
        selected, coverage = _map_single_evidence(evidence["evidence_text"], candidates, threshold)
        selected_ids = [chunk.chunk_id for chunk in selected]
        evidence_chunk_ids.append(selected_ids)
        evidence_coverages.append(coverage)
        if coverage < threshold or not selected_ids:
            all_resolved = False
        for chunk_id in selected_ids:
            if chunk_id not in all_relevant_ids:
                all_relevant_ids.append(chunk_id)
    return EvidenceMapping(
        relevant_chunk_ids=all_relevant_ids,
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_coverages=evidence_coverages,
        resolved=all_resolved,
    )


def retrieved_evidence_scores(record: dict, retrieved_chunks: list[Chunk], threshold: float = 0.8) -> tuple[float, float]:
    evidence_items = record.get("evidence", [])
    if not evidence_items:
        return 0.0, 0.0
    coverages: list[float] = []
    hits = 0
    for evidence in evidence_items:
        coverage = combined_token_coverage(evidence["evidence_text"], retrieved_chunks)
        coverages.append(coverage)
        hits += int(coverage >= threshold)
    return hits / len(evidence_items), sum(coverages) / len(coverages)


def _filter_candidates(evidence: dict, chunks: list[Chunk]) -> list[Chunk]:
    document_id = evidence.get("document_id")
    relative_path = evidence.get("relative_path")
    expected_section = evidence.get("section_path")
    candidates: list[Chunk] = []
    for chunk in chunks:
        if document_id and chunk.document_id != document_id:
            continue
        if relative_path and chunk.relative_path != relative_path:
            continue
        if expected_section and chunk.section_path != expected_section:
            continue
        candidates.append(chunk)
    return sorted(candidates, key=lambda chunk: chunk.chunk_index)


def _map_single_evidence(evidence_text: str, candidates: list[Chunk], threshold: float) -> tuple[list[Chunk], float]:
    normalised_evidence = _normalise_match_text(evidence_text)
    exact = [chunk for chunk in candidates if normalised_evidence in _normalise_match_text(chunk.text)]
    if exact:
        return exact, 1.0
    individually_relevant = [
        chunk for chunk in candidates if token_coverage(evidence_text, chunk.text) >= threshold
    ]
    if individually_relevant:
        coverage = combined_token_coverage(evidence_text, individually_relevant)
        return individually_relevant, coverage
    selected = _greedy_cover(evidence_text, candidates, threshold)
    return selected, combined_token_coverage(evidence_text, selected)


def _greedy_cover(evidence_text: str, candidates: list[Chunk], threshold: float) -> list[Chunk]:
    evidence_tokens = set(tokenise(evidence_text))
    if not evidence_tokens:
        return []
    uncovered = set(evidence_tokens)
    remaining = list(candidates)
    selected: list[Chunk] = []
    while remaining and 1.0 - len(uncovered) / len(evidence_tokens) < threshold:
        best = max(remaining, key=lambda chunk: len(uncovered.intersection(tokenise(chunk.text))))
        gain = uncovered.intersection(tokenise(best.text))
        if not gain:
            break
        selected.append(best)
        uncovered.difference_update(gain)
        remaining.remove(best)
    return selected


def _normalise_match_text(text: str) -> str:
    return " ".join(normalise_text(text).lower().split())
