from __future__ import annotations

import argparse
import sys
from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.config import load_config
from rag_project.factory import create_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect chunks in ChromaDB")
    parser.add_argument("--config", default="configs/sources.json")
    parser.add_argument("--document-id")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    chunks = create_store(config).all_chunks(document_id=args.document_id)
    print(f"Stored chunks: {len(chunks)}")

    for chunk in chunks[:args.limit]:
        print("=" * 100)
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"document: {chunk.document_id} | {chunk.document_title}")
        print(f"source: {chunk.relative_path}")
        print(f"section: {' > '.join(chunk.section_path)}")
        print("-" * 100)
        print(chunk.text[:1800])


if __name__ == "__main__":
    main()
