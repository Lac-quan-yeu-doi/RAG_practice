from rag_project.preprocessing import SphinxTextPreprocessor
from rag_project.schemas import SourceFile


def test_sphinx_headings_become_sections():
    source = SourceFile(
        document_id="asyncio",
        document_title="Asyncio",
        source_path="asyncio.txt",
        relative_path="library/asyncio.txt",
        text=(
            "Asyncio\n"
            "=======\n\n"
            "Introduction text.\n\n"
            "Tasks\n"
            "-----\n\n"
            "Task content."
        ),
    )
    sections = SphinxTextPreprocessor().process(source)
    assert len(sections) == 2
    assert sections[0].section_path == ["Asyncio"]
    assert sections[1].section_path == ["Asyncio", "Tasks"]
    assert "Task content" in sections[1].text
