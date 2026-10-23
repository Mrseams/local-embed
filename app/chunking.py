"""Stage 3 - chunking: recursive, structure-aware splitting with overlap.

Text is split on the coarsest separator that works (paragraph, line,
sentence, word) until every piece fits the token budget, then pieces are
packed greedily into chunks with a token overlap between neighbours.
Chunks never cross Section boundaries, so page/heading metadata stays exact.
"""
import re
from dataclasses import dataclass

from .loaders import Section

# Approximates WordPiece counts: one token per word or punctuation mark.
# Real counts run a little higher, which the default 200-token budget absorbs.
_TOKEN = re.compile(r"\w+|[^\w\s]")
_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


@dataclass
class Chunk:
    index: int
    text: str
    tokens: int
    page: int | None
    heading: str | None

    @property
    def embed_text(self) -> str:
        # Heading context noticeably helps retrieval for short chunks.
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _split(text: str, max_tokens: int, seps: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    if not seps:
        words = text.split()
        # Hard split; only reached for a single pathological "word" run.
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    sep, rest = seps[0], seps[1:]
    parts = [p for p in re.split(sep, text) if p.strip()]
    if len(parts) == 1:
        return _split(text, max_tokens, rest)
    out = []
    for p in parts:
        out.extend(_split(p.strip(), max_tokens, rest))
    return out


def _tail(text: str, n_tokens: int) -> str:
    """Last ~n_tokens of text, cut at a word boundary."""
    if n_tokens <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    total = 0
    for w in reversed(words):
        total += count_tokens(w)
        if total > n_tokens:
            break
        tail.append(w)
    return " ".join(reversed(tail))


def chunk_sections(
    sections: list[Section],
    max_tokens: int = 200,
    overlap: int = 40,
    min_tokens: int = 8,
) -> list[Chunk]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    chunks: list[Chunk] = []

    for section in sections:
        pieces = _split(section.text, max_tokens - overlap, _SEPARATORS)
        current: list[str] = []
        current_tokens = 0

        def emit():
            text = "\n".join(current).strip()
            n = count_tokens(text)
            if n >= min_tokens or (n and not chunks):
                chunks.append(Chunk(len(chunks), text, n, section.page, section.heading))

        for piece in pieces:
            n = count_tokens(piece)
            if current and current_tokens + n > max_tokens:
                emit()
                carry = _tail("\n".join(current), overlap)
                current = [carry] if carry else []
                current_tokens = count_tokens(carry)
            current.append(piece)
            current_tokens += n
        if current:
            emit()

    return chunks
