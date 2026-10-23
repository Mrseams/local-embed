"""Stage 2 - cleaning: normalize extracted text before chunking."""
import re
import unicodedata
from collections import Counter

from .loaders import Section

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(/|of)\s*\d+)?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "embed-\nding" -> "embedding"
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def strip_repeated_lines(sections: list[Section], threshold: float = 0.5) -> list[Section]:
    """Drop running headers/footers and bare page numbers from paged documents.

    A line that shows up on more than `threshold` of the pages is boilerplate.
    It is kept once, on the first page, because running headers are often the
    document title ("Q3 Report") and dropping them everywhere loses context.
    """
    paged = [s for s in sections if s.page is not None]
    if len(paged) < 3:
        return sections
    counts = Counter()
    for s in paged:
        counts.update({ln.strip() for ln in s.text.split("\n") if ln.strip()})
    limit = threshold * len(paged)
    repeated = {ln for ln, n in counts.items() if n > limit}

    first_page = paged[0].page
    out = []
    for s in sections:
        lines = [
            ln for ln in s.text.split("\n")
            if (s.page == first_page or ln.strip() not in repeated) and not _PAGE_NUMBER.match(ln)
        ]
        out.append(Section("\n".join(lines), page=s.page, heading=s.heading))
    return out


def clean(sections: list[Section]) -> list[Section]:
    sections = strip_repeated_lines(sections)
    out = []
    for s in sections:
        text = clean_text(s.text)
        if text:
            out.append(Section(text, page=s.page, heading=s.heading))
    return out
