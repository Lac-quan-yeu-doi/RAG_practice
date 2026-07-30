from rag_project.benchmarking.evidence import map_record_evidence, retrieved_evidence_scores
from rag_project.schemas import Chunk


def chunk(chunk_id: str, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="asyncio",
        document_title="Asyncio",
        source_path="asyncio-task.txt",
        relative_path="library/asyncio-task.txt",
        section_path=["Coroutines and Tasks", "Running Tasks Concurrently"],
        chunk_index=index,
        text=text,
    )


def test_exact_evidence_maps_without_manual_chunk_ids():
    chunks = [
        chunk("a", 0, "Run awaitable objects in the aws sequence concurrently."),
        chunk("b", 1, "Results are returned in input order."),
    ]
    record = {
        "evidence": [
            {
                "document_id": "asyncio",
                "relative_path": "library/asyncio-task.txt",
                "section_path": ["Coroutines and Tasks", "Running Tasks Concurrently"],
                "evidence_text": "Run awaitable objects in the aws sequence concurrently.",
            }
        ]
    }
    mapping = map_record_evidence(record, chunks)
    assert mapping.resolved
    assert mapping.relevant_chunk_ids == ["a"]


def test_retrieved_evidence_can_be_covered_by_multiple_chunks():
    chunks = [
        chunk("a", 0, "ThreadPoolExecutor uses worker threads."),
        chunk("b", 1, "ProcessPoolExecutor uses separate processes."),
    ]
    record = {
        "evidence": [
            {
                "evidence_text": (
                    "ThreadPoolExecutor uses worker threads. "
                    "ProcessPoolExecutor uses separate processes."
                )
            }
        ]
    }
    hit, coverage = retrieved_evidence_scores(record, chunks, threshold=0.8)
    assert hit == 1.0
    assert coverage == 1.0
