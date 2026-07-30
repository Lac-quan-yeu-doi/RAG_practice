from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.config import load_config
from rag_project.factory import create_generator, create_retriever
from rag_project.pipeline import RAGPipeline


def print_response(response) -> None:
    print("\nANSWER")
    print("-" * 90)
    print(response.answer)

    print("\nCITATIONS")
    print("-" * 90)
    if not response.citations:
        print("No citations")
    for citation in response.citations:
        print(
            f"- {citation.document_title} | {citation.relative_path} | "
            f"{' > '.join(citation.section_path)} | chunk_id={citation.chunk_id}"
        )

    print("\nRETRIEVED")
    print("-" * 90)
    for result in response.retrieved:
        print(
            f"{result.rank}. score={result.score:.5f} | {result.retriever} | "
            f"{result.chunk.relative_path} | {' > '.join(result.chunk.section_path)} | "
            f"{result.chunk.chunk_id}"
        )

    print("\nMETADATA")
    print("-" * 90)
    print(json.dumps(response.metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Python documentation RAG")
    parser.add_argument("--config", default="configs/sources.json")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--generator", choices=["extractive", "openai-compatible"], default="extractive")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(
        retriever=create_retriever(args.retriever, config),
        generator=create_generator(args.generator),
        top_k=args.top_k,
    )

    if args.interactive:
        while True:
            question = input("\nQuestion, or 'exit': ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                print_response(pipeline.ask(question))
        return

    if not args.question:
        parser.error("--question is required unless --interactive is used")
    print_response(pipeline.ask(args.question))


if __name__ == "__main__":
    main()
