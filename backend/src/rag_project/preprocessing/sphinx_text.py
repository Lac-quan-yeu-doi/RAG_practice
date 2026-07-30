from __future__ import annotations

import re

from rag_project.preprocessing.base import BasePreprocessor
from rag_project.schemas import Section, SourceFile
from rag_project.text import normalise_text


UNDERLINE_RE = re.compile(r"^([=\-~^\"'`:#*+])\1{2,}\s*$")
NAVIGATION_LINE_RE = re.compile(
    r"^(Navigation|Previous topic|Next topic|This Page|Quick search|"
    r"Enter search terms or a module, class or function name\.|"
    r"© Copyright .*|Created using Sphinx .*)$",
    flags=re.IGNORECASE,
)
LEVEL_BY_MARKER = {
    "=": 1,
    "*": 1,
    "#": 1,
    "-": 2,
    "~": 3,
    "^": 4,
    '"': 5,
    "'": 5,
    "`": 5,
    ":": 5,
    "+": 5,
}


class SphinxTextPreprocessor(BasePreprocessor):
    def __init__(self, remove_navigation_lines: bool = True):
        self.remove_navigation_lines = remove_navigation_lines

    def process(self, source: SourceFile) -> list[Section]:
        lines = self._clean_lines(normalise_text(source.text).splitlines())
        sections: list[Section] = []
        path_stack: list[str] = []
        current_path = [source.relative_path]
        content_lines: list[str] = []

        def flush() -> None:
            nonlocal content_lines
            content = normalise_text("\n".join(content_lines))
            if content:
                sections.append(
                    Section(
                        document_id=source.document_id,
                        document_title=source.document_title,
                        source_path=source.source_path,
                        relative_path=source.relative_path,
                        section_path=list(current_path),
                        text=content,
                    )
                )
            content_lines = []

        index = 0
        while index < len(lines):
            if index + 1 < len(lines):
                title = lines[index].strip()
                underline = lines[index + 1].strip()
                match = UNDERLINE_RE.fullmatch(underline)
                if title and match:
                    flush()
                    level = LEVEL_BY_MARKER.get(match.group(1), 5)
                    path_stack = path_stack[: level - 1]
                    path_stack.append(title)
                    current_path = list(path_stack)
                    index += 2
                    continue
            content_lines.append(lines[index])
            index += 1
        flush()
        if not sections:
            sections.append(
                Section(
                    document_id=source.document_id,
                    document_title=source.document_title,
                    source_path=source.source_path,
                    relative_path=source.relative_path,
                    section_path=[source.relative_path],
                    text=normalise_text("\n".join(lines)),
                )
            )
        return sections

    def _clean_lines(self, lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if self.remove_navigation_lines and NAVIGATION_LINE_RE.fullmatch(stripped):
                continue
            cleaned.append(line.rstrip())
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return cleaned
