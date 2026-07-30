from __future__ import annotations

import json
from pathlib import Path

import chromadb

from rag_project.embeddings.base import BaseEmbeddingModel
from rag_project.schemas import Chunk, RetrievalResult
from rag_project.storage.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, path: str, collection_name: str, batch_size: int = 128):
        self.path = str(Path(path))
        self.collection_name = collection_name
        self.batch_size = batch_size
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def rebuild(self, chunks: list[Chunk], embedding_model: BaseEmbeddingModel) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty chunk list")
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            embeddings = embedding_model.encode_documents([chunk.search_text for chunk in batch])
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings.tolist(),
                metadatas=[self._chunk_to_metadata(chunk) for chunk in batch],
            )

    def all_chunks(self, document_id: str | None = None) -> list[Chunk]:
        arguments = {"include": ["documents", "metadatas"]}
        if document_id:
            arguments["where"] = {"document_id": document_id}
        result = self.collection.get(**arguments)
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            self._metadata_to_chunk(chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def similarity_search(
        self,
        query: str,
        embedding_model: BaseEmbeddingModel,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = embedding_model.encode_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievalResult(
                chunk=self._metadata_to_chunk(chunk_id, document, metadata),
                score=1.0 / (1.0 + float(distance)),
                rank=rank,
                retriever="dense",
            )
            for rank, (chunk_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            )
        ]

    @staticmethod
    def _chunk_to_metadata(chunk: Chunk) -> dict:
        return {
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "source_path": chunk.source_path,
            "relative_path": chunk.relative_path,
            "section_path_json": json.dumps(chunk.section_path, ensure_ascii=False),
            "chunk_index": int(chunk.chunk_index),
            "extra_metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
        }

    @staticmethod
    def _metadata_to_chunk(chunk_id: str, document: str, metadata: dict) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            document_id=str(metadata["document_id"]),
            document_title=str(metadata["document_title"]),
            source_path=str(metadata["source_path"]),
            relative_path=str(metadata["relative_path"]),
            section_path=json.loads(str(metadata["section_path_json"])),
            chunk_index=int(metadata["chunk_index"]),
            text=document or "",
            metadata=json.loads(str(metadata.get("extra_metadata_json", "{}"))),
        )
