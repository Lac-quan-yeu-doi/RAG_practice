from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_project.schemas import SourceFile


class TextDirectoryLoader:
    def load(self, config: dict[str, Any]) -> list[SourceFile]:
        root = Path(config["docs_root"])
        if not root.exists():
            raise FileNotFoundError(
                f"Documentation directory not found: {root}\n"
                "Copy the extracted docs there or update docs_root in the configuration."
            )
        sources: list[SourceFile] = []
        seen_paths: set[Path] = set()
        for document in config["documents"]:
            document_id = document["document_id"]
            document_title = document["title"]
            included = self._expand_patterns(root, document.get("include", []))
            excluded = set(self._expand_patterns(root, document.get("exclude", [])))
            selected = [path for path in included if path not in excluded]
            if not selected:
                raise ValueError(f"No files matched document '{document_id}'")
            for path in selected:
                resolved = path.resolve()
                if resolved in seen_paths:
                    raise ValueError(f"Source file selected more than once: {path}")
                seen_paths.add(resolved)
                sources.append(
                    SourceFile(
                        document_id=document_id,
                        document_title=document_title,
                        source_path=str(path),
                        relative_path=path.relative_to(root).as_posix(),
                        text=self._read_text(path),
                    )
                )
        return sources

    @staticmethod
    def _expand_patterns(root: Path, patterns: list[str]) -> list[Path]:
        paths: set[Path] = set()
        for pattern in patterns:
            paths.update(path for path in root.glob(pattern) if path.is_file())
        return sorted(paths)

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode text file: {path}")
