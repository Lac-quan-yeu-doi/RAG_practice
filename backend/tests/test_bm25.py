from rag_project.retrieval import BM25Retriever
from rag_project.schemas import Chunk


class MemoryStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def all_chunks(self, document_id=None):
        if document_id is None:
            return self.chunks
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="docs",
        document_title="Python docs",
        source_path="file.txt",
        relative_path="file.txt",
        section_path=["Section"],
        chunk_index=0,
        text=text,
    )


def test_bm25_ranks_matching_chunk_first():
    chunks = [
        make_chunk("gather", "asyncio.gather runs awaitable objects concurrently"),
        make_chunk("executor", "ThreadPoolExecutor executes callables using worker threads"),
    ]
    retriever = BM25Retriever(store=MemoryStore(chunks))
    results = retriever.retrieve("run awaitables concurrently with gather", top_k=2)
    assert results[0].chunk.chunk_id == "gather"
    assert results[0].score > results[1].score
