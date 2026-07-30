from __future__ import annotations

from dotenv import load_dotenv

from rag_project.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from rag_project.generation import ExtractiveGenerator, OpenAICompatibleGenerator
from rag_project.retrieval import BM25Retriever, DenseRetriever
from rag_project.storage.chroma_store import ChromaVectorStore


def create_store(config: dict) -> ChromaVectorStore:
    store_config = config["vector_store"]
    return ChromaVectorStore(
        path=store_config["path"],
        collection_name=store_config["collection"],
    )


def create_embedding_model(config: dict) -> SentenceTransformerEmbeddingModel:
    embedding_config = config["embedding"]
    return SentenceTransformerEmbeddingModel(
        model_name=embedding_config["model_name"],
        batch_size=int(embedding_config.get("batch_size", 32)),
    )


def create_retriever(name: str, config: dict):
    store = create_store(config)
    if name == "bm25":
        return BM25Retriever(store=store)
    if name == "dense":
        return DenseRetriever(
            store=store,
            embedding_model=create_embedding_model(config),
        )
    raise ValueError(f"Unknown retriever: {name}")


def create_generator(name: str):
    load_dotenv()
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "openai-compatible":
        return OpenAICompatibleGenerator()
    raise ValueError(f"Unknown generator: {name}")
