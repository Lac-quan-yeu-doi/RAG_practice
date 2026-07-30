from __future__ import annotations

import re
import unicodedata


TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`])")


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.replace("\u00ad", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def tokenise(text: str) -> list[str]:
    return [cleaned for token in TOKEN_RE.findall(normalise_text(text)) if (cleaned := token.lower().strip("._-"))]


def split_sentences(text: str) -> list[str]:
    text = normalise_text(text)
    sentences = [part.strip() for part in SENTENCE_RE.split(text)]
    return [sentence for sentence in sentences if sentence]
