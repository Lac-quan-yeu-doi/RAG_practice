from __future__ import annotations

import argparse
import sys
from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_project.config import load_config
from rag_project.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from rag_project.loaders import TextDirectoryLoader
from rag_project.preprocessing import SectionChunker, SphinxTextPreprocessor
from rag_project.storage.chroma_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Python docs and build ChromaDB")
    parser.add_argument("--config", default="configs/sources.json")
    args = parser.parse_args()

    config = load_config(args.config)
    sources = TextDirectoryLoader().load(config)

    preprocessing_config = config.get("preprocessing", {})
    preprocessor = SphinxTextPreprocessor(
        remove_navigation_lines=bool(preprocessing_config.get("remove_navigation_lines", True))
    )

    chunking_config = config["chunking"]
    chunker = SectionChunker(
        max_chars=int(chunking_config.get("max_chars", 2400)),
        overlap_paragraphs=int(chunking_config.get("overlap_paragraphs", 1)),
        minimum_chars=int(chunking_config.get("minimum_chars", 120)),
    )

    sections = []
    for source in sources:
        source_sections = preprocessor.process(source)
        sections.extend(source_sections)
        print(f"{source.relative_path}: {len(source_sections)} sections")

    chunks = chunker.chunk(sections)
    print(f"Created {len(chunks)} chunks from {len(sources)} source files")

    embedding_config = config["embedding"]
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name=embedding_config["model_name"],
        batch_size=int(embedding_config.get("batch_size", 32)),
    )

    store_config = config["vector_store"]
    store = ChromaVectorStore(
        path=store_config["path"],
        collection_name=store_config["collection"],
    )
    store.rebuild(chunks, embedding_model)

    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1

    print(f"Indexed {len(chunks)} chunks in {store_config['path']}")
    for document_id, count in sorted(counts.items()):
        print(f"- {document_id}: {count} chunks")


if __name__ == "__main__":
    main()
