from rag_project.preprocessing import SectionChunker
from rag_project.schemas import Section


def test_chunker_creates_stable_metadata():
    section = Section(
        document_id="asyncio",
        document_title="Asyncio",
        source_path="asyncio-task.txt",
        relative_path="library/asyncio-task.txt",
        section_path=["Coroutines and Tasks", "Running Tasks Concurrently"],
        text=("Paragraph about gather. " * 40) + "\n\n" + ("Another paragraph. " * 40),
    )
    chunks = SectionChunker(max_chars=500, overlap_paragraphs=1, minimum_chars=20).chunk([section])
    assert len(chunks) >= 2
    assert all(chunk.document_id == "asyncio" for chunk in chunks)
    assert all(chunk.section_path[-1] == "Running Tasks Concurrently" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
