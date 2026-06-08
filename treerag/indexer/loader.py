from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    number: int
    text: str


class DocumentLoader:
    """
    Loads a plain-text document and splits it into pages.

    Pages are delimited by lines that contain only the word CHAPTER or by
    blank lines preceding a section heading. For simplicity in this project
    we treat each paragraph block separated by a double newline as one page.
    That maps well to the sample document structure and keeps pages small
    enough for the LLM to summarise cheaply.

    In a real project you would swap this class with one that uses PyMuPDF
    or pdfplumber to extract actual PDF pages — the rest of the system would
    not change at all.
    """

    def __init__(self, min_page_length: int = 80):
        self.min_page_length = min_page_length

    def load(self, filepath: str | Path) -> list[Page]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")

        raw = path.read_text(encoding="utf-8")

        # split on double newlines to get paragraph-level blocks
        blocks = [b.strip() for b in raw.split("\n\n")]

        # drop blocks that are too short to be meaningful (e.g. blank lines)
        blocks = [b for b in blocks if len(b) >= self.min_page_length]

        pages = [Page(number=i + 1, text=block) for i, block in enumerate(blocks)]
        return pages

    def as_lookup(self, pages: list[Page]) -> dict[int, str]:
        """Return a dict mapping page number to page text for fast retrieval later."""
        return {p.number: p.text for p in pages}
