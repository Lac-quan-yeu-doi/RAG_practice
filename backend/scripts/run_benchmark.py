from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.benchmarking.evidence import map_record_evidence, retrieved_evidence_scores
from rag_project.config import load_config
from rag_project.evaluation import evaluate_retrieval, exact_match, token_f1
from rag_project.factory import create_generator, create_retriever, create_store
from rag_project.pipeline import RAGPipeline


def load_benchmark(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            required = {"question", "reference_answer", "evidence"}
            if not required.issubset(record):
                raise ValueError(f"Invalid benchmark record at line {line_number}: expected {sorted(required)}")
            records.append(record)
    if not records:
        raise ValueError("Benchmark is empty")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Python documentation RAG configuration")
    parser.add_argument("--config", default="configs/sources.json")
    parser.add_argument("--benchmark", default="benchmark/generated.jsonl")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--generator", choices=["extractive", "openai-compatible"], default="extractive")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-threshold", type=float, default=0.8)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = load_benchmark(args.benchmark)
    store = create_store(config)
    all_chunks = store.all_chunks()
    pipeline = RAGPipeline(
        retriever=create_retriever(args.retriever, config),
        generator=create_generator(args.generator),
        top_k=args.top_k,
    )

    predictions: list[list[str]] = []
    ground_truth: list[list[str]] = []
    exact_scores: list[float] = []
    f1_scores: list[float] = []
    evidence_hit_scores: list[float] = []
    evidence_coverage_scores: list[float] = []
    retrieval_latencies: list[float] = []
    total_latencies: list[float] = []
    rows: list[dict] = []
    unresolved_questions = 0

    for record in benchmark:
        mapping = map_record_evidence(record, all_chunks, threshold=args.evidence_threshold)
        if not mapping.resolved:
            unresolved_questions += 1

        started = perf_counter()
        response = pipeline.ask(record["question"])
        total_latencies.append(perf_counter() - started)
        retrieval_latencies.append(float(response.metadata["retrieval_seconds"]))

        predicted_ids = [result.chunk.chunk_id for result in response.retrieved]
        retrieved_chunks = [result.chunk for result in response.retrieved]
        prediction = response.answer
        reference = record["reference_answer"]
        evidence_hit, evidence_coverage = retrieved_evidence_scores(
            record,
            retrieved_chunks,
            threshold=args.evidence_threshold,
        )

        if mapping.resolved:
            predictions.append(predicted_ids)
            ground_truth.append(mapping.relevant_chunk_ids)
        exact_scores.append(exact_match(prediction, reference))
        f1_scores.append(token_f1(prediction, reference))
        evidence_hit_scores.append(evidence_hit)
        evidence_coverage_scores.append(evidence_coverage)

        rows.append(
            {
                "question_id": record.get("question_id"),
                "question": record["question"],
                "prediction": prediction,
                "reference_answer": reference,
                "predicted_chunk_ids": predicted_ids,
                "mapped_relevant_chunk_ids": mapping.relevant_chunk_ids,
                "evidence_chunk_ids": mapping.evidence_chunk_ids,
                "ground_truth_evidence_coverages": mapping.evidence_coverages,
                "evidence_mapping_resolved": mapping.resolved,
                "retrieved_evidence_hit": evidence_hit,
                "retrieved_evidence_token_coverage": evidence_coverage,
                "answer_exact_match": exact_scores[-1],
                "answer_token_f1": f1_scores[-1],
            }
        )

    retrieval_metrics = evaluate_retrieval(predictions, ground_truth)
    summary = {
        "retriever": args.retriever,
        "generator": args.generator,
        "top_k": args.top_k,
        "evidence_threshold": args.evidence_threshold,
        "questions": len(benchmark),
        "retrieval_evaluated_questions": retrieval_metrics.evaluated_questions,
        "unresolved_evidence_questions": unresolved_questions,
        "hit_rate": retrieval_metrics.hit_rate,
        "recall": retrieval_metrics.recall,
        "mean_reciprocal_rank": retrieval_metrics.mean_reciprocal_rank,
        "evidence_hit_rate": statistics.mean(evidence_hit_scores),
        "mean_evidence_token_coverage": statistics.mean(evidence_coverage_scores),
        "answer_exact_match": statistics.mean(exact_scores),
        "answer_token_f1": statistics.mean(f1_scores),
        "mean_retrieval_latency_ms": statistics.mean(retrieval_latencies) * 1000,
        "mean_end_to_end_latency_ms": statistics.mean(total_latencies) * 1000,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "predictions": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
