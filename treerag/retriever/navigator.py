from treerag.utils.llm import call_llm
from treerag.utils.json_parser import safe_parse_json


SECTION_SELECT_SYSTEM = (
    "You are a document navigation assistant. Return only valid JSON, nothing else."
)

SECTION_SELECT_PROMPT = """Question: {query}

Document sections:
{sections}

Return a JSON array of the section indices (0-based) that are relevant to the question.
Example: [0, 2]

JSON array only:"""


PAGE_SELECT_SYSTEM = (
    "You are a document navigation assistant. Return only valid JSON, nothing else."
)

PAGE_SELECT_PROMPT = """Question: {query}

Section: "{section_title}"
Pages:
{page_summaries}

Return a JSON array of the page numbers that are relevant to the question.
Example: [3, 5]

JSON array only:"""


ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the document excerpts "
    "provided. Be clear and specific."
)

ANSWER_PROMPT = """Question: {query}

Document excerpts:
{excerpts}

Answer:"""


class TreeNavigator:

    def query(self, question: str, tree: dict, page_lookup: dict, verbose: bool = True) -> dict:
        trace = []

        if verbose:
            print("Selecting relevant sections...")
        section_indices = self._select_sections(question, tree)
        if not section_indices:
            return {
                "answer": "No relevant sections found for this question.",
                "pages_used": [],
                "trace": ["No relevant sections identified."],
            }

        if verbose:
            print("Selecting relevant pages...")
        selected_pages = []
        for idx in section_indices:
            if idx >= len(tree["sections"]):
                continue
            section = tree["sections"][idx]
            trace.append(f"Section selected: '{section['title']}'")
            pages = self._select_pages(question, section)
            trace.append(f"  Pages picked: {pages}")
            selected_pages.extend(pages)

        seen = set()
        unique_pages = []
        for p in selected_pages:
            if p not in seen:
                seen.add(p)
                unique_pages.append(p)

        if not unique_pages:
            return {
                "answer": "Could not narrow down to specific pages.",
                "pages_used": [],
                "trace": trace,
            }

        if verbose:
            print(f"Reading pages {unique_pages} and generating answer...")
        excerpts = "\n\n".join(
            f"[Page {p}]\n{page_lookup[p]}"
            for p in unique_pages
            if p in page_lookup
        )

        answer = call_llm(
            prompt=ANSWER_PROMPT.format(query=question, excerpts=excerpts),
            system=ANSWER_SYSTEM,
            max_tokens=800,
        )

        return {
            "answer": answer,
            "pages_used": unique_pages,
            "trace": trace,
        }

    def _select_sections(self, question: str, tree: dict) -> list:
        sections_text = "\n".join(
            f"[{i}] {s['title']}: {s['summary']}"
            for i, s in enumerate(tree["sections"])
        )
        raw = call_llm(
            prompt=SECTION_SELECT_PROMPT.format(query=question, sections=sections_text),
            system=SECTION_SELECT_SYSTEM,
            max_tokens=500,

        )
        print(f"  Raw response: {repr(raw)}")

        result = safe_parse_json(raw)

        indices = result if isinstance(result, list) else result.get("relevant_indices", [])
        print(f"  Sections chosen: {indices}")
        return indices

    def _select_pages(self, question: str, section: dict) -> list:
        page_summaries = "\n".join(
            f"Page {child['page_number']}: {child['summary']}"
            for child in section.get("children", [])
        )
        if not page_summaries:
            return []
        raw = call_llm(
            prompt=PAGE_SELECT_PROMPT.format(
                query=question,
                section_title=section["title"],
                page_summaries=page_summaries,
            ),
            system=PAGE_SELECT_SYSTEM,
            max_tokens=500,

        )
        result = safe_parse_json(raw)
        return result if isinstance(result, list) else result.get("relevant_pages", [])