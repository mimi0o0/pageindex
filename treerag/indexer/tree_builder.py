import json
from pathlib import Path

from treerag.indexer.loader import Page
from treerag.utils.llm import call_llm
from treerag.utils.json_parser import safe_parse_json


SUMMARISE_SYSTEM = (
    "You are a precise document analyst. Write a factual, specific summary of the "
    "document section given to you. Two to three sentences. No fluff."
)

SUMMARISE_PROMPT = """Summarise the following document section. Be specific about the main topic and any key points.

Page {number}:
{text}

Summary:"""


GROUP_SYSTEM = (
    "You are a document structure expert. Group the page summaries into logical "
    "top-level sections. Return only valid JSON with no explanation."
)

GROUP_PROMPT = """Below are summaries for each page in a document. Group them into logical top-level sections (like chapters or major topics).

{summaries}

Return a JSON array. Each element must have exactly these fields:
- "title": a short descriptive title for this section
- "summary": one sentence describing what the whole section covers
- "page_numbers": a list of integers for the pages in this group

Example shape (do not copy the values):
[
  {{"title": "Introduction", "summary": "Covers the basics.", "page_numbers": [1, 2]}},
  {{"title": "Advanced Topics", "summary": "Covers advanced material.", "page_numbers": [3, 4, 5]}}
]

JSON:"""


class TreeBuilder:
    """
    Builds a two-level hierarchical tree index over a document.

    Pass 1: Ask the LLM to write a short summary of every page individually.
    Pass 2: Ask the LLM to group those page summaries into top-level sections.

    The result is a JSON structure that looks like a smart table of contents.
    It gets saved to disk so it is only ever built once per document.
    """

    def build(self, pages: list[Page], verbose: bool = True) -> dict:
        leaf_nodes = self._build_leaf_nodes(pages, verbose)
        parent_nodes = self._build_parent_nodes(leaf_nodes, verbose)

        return {
            "total_pages": len(pages),
            "sections": parent_nodes,
        }

    def _build_leaf_nodes(self, pages: list[Page], verbose: bool) -> list[dict]:
        leaf_nodes = []
        for page in pages:
            if verbose:
                print(f"  Summarising page {page.number} of {len(pages)}...")
            summary = call_llm(
                prompt=SUMMARISE_PROMPT.format(number=page.number, text=page.text),
                system=SUMMARISE_SYSTEM,
                max_tokens=200,
            )
            leaf_nodes.append({
                "page_number": page.number,
                "summary": summary,
                "text": page.text,
            })
        return leaf_nodes

    def _build_parent_nodes(self, leaf_nodes: list[dict], verbose: bool) -> list[dict]:
        if verbose:
            print("  Grouping pages into sections...")

        summaries_text = "\n".join(
            f"Page {n['page_number']}: {n['summary']}" for n in leaf_nodes
        )

        raw = call_llm(
            prompt=GROUP_PROMPT.format(summaries=summaries_text),
            system=GROUP_SYSTEM,
            max_tokens=1024,
        )

        groups = safe_parse_json(raw)

        leaf_by_page = {n["page_number"]: n for n in leaf_nodes}
        for group in groups:
            group["children"] = [
                leaf_by_page[p]
                for p in group.get("page_numbers", [])
                if p in leaf_by_page
            ]

        return groups

    def save(self, tree: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
        print(f"Index saved to {path}")

    def load(self, path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))
