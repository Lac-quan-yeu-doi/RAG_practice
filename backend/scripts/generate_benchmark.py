from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Add root directory, but because this codebased is already build with .toml with [tool.setuptools.packages.find] and where = ["src"], no need to add, the env path lib already has the link to src/
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.benchmarking import LocalQuestionGenerator
from rag_project.config import load_config
from rag_project.factory import create_store
from rag_project.schemas import Chunk


def group_sections(chunks: list[Chunk]) -> list[list[Chunk]]:
    groups: defaultdict[tuple, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        key = (chunk.document_id, chunk.relative_path, tuple(chunk.section_path))
        groups[key].append(chunk)
    return [sorted(group, key=lambda chunk: chunk.chunk_index) for group in groups.values()]


def merge_section_text(chunks: list[Chunk]) -> str:
    paragraphs: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            paragraph = paragraph.strip()
            key = " ".join(paragraph.lower().split())
            if paragraph and key not in seen:
                paragraphs.append(paragraph)
                seen.add(key)
    return "\n\n".join(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded synthetic benchmark")
    parser.add_argument("--config", default="configs/sources.json")
    parser.add_argument("--output", default="benchmark/generated.jsonl")
    parser.add_argument("--sections", type=int, default=20)
    parser.add_argument("--questions-per-section", type=int, default=2)
    parser.add_argument("--min-section-chars", type=int, default=500)
    parser.add_argument("--max-context-chars", type=int, default=7000)
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    chunks = create_store(config).all_chunks()
    if args.document_id:
        allowed = set(args.document_id)
        chunks = [chunk for chunk in chunks if chunk.document_id in allowed]
    sections = []
    for group in group_sections(chunks):
        context = merge_section_text(group)
        if len(context) >= args.min_section_chars:
            sections.append((group, context[:args.max_context_chars]))
    if not sections:
        raise ValueError("No eligible sections found. Build the index or lower --min-section-chars.")

    random.Random(args.seed).shuffle(sections)
    selected_sections = sections[:min(args.sections, len(sections))]
    generator = LocalQuestionGenerator()
    records: list[dict] = []
    failures = 0

    for section_index, (group, context) in enumerate(selected_sections, start=1):
        first = group[0]
        metadata = {
            "document_id": first.document_id,
            "relative_path": first.relative_path,
            "section_path": first.section_path,
        }
        try:
            generated = generator.generate(
                context=context,
                source_metadata=metadata,
                num_questions=args.questions_per_section,
            )
        except Exception as error:
            failures += 1
            print(f"Skipped section {section_index}: {error}")
            continue
        for item in generated:
            records.append(
                {
                    "question_id": f"auto_{len(records) + 1:04d}",
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "answerable": True,
                    "evidence": [
                        {
                            "document_id": first.document_id,
                            "relative_path": first.relative_path,
                            "section_path": first.section_path,
                            "evidence_text": item["evidence_text"],
                        }
                    ],
                }
            )
        print(f"Section {section_index}/{len(selected_sections)}: generated {len(generated)} questions")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} questions to {output_path}")
    print(f"Skipped sections: {failures}")


if __name__ == "__main__":
    main()
