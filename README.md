# TreeRAG

Vectorless, reasoning-based RAG over plain-text documents.

No vector database. No chunking. No cosine similarity. The LLM builds a
tree index of your document and then navigates that tree to answer questions
— the same way a human expert uses a table of contents.

## Project structure

```
treerag_full/
├── treerag/
│   ├── __init__.py          exports TreeRAGPipeline
│   ├── pipeline.py          public-facing class that ties everything together
│   ├── indexer/
│   │   ├── loader.py        reads a text file and splits it into pages
│   │   └── tree_builder.py  two-pass LLM indexing (leaf summaries + grouping)
│   ├── retriever/
│   │   └── navigator.py     three-step LLM-driven retrieval
│   └── utils/
│       ├── llm.py           single function wrapping the Anthropic API
│       └── json_parser.py   robust JSON extraction from model responses
├── data/
│   └── ml_handbook.txt      sample document (ML engineering handbook)
├── tests/
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_json_parser.py
│   ├── test_tree_builder.py
│   └── test_navigator.py
├── main.py                  CLI entry point
└── requirements.txt
```

## How retrieval works

When you ask a question:

1. The LLM sees the top-level section summaries and decides which sections
   are relevant — like skimming a table of contents.
2. For each relevant section, the LLM sees page-level summaries and picks
   the specific pages most likely to answer the question.
3. Only the full text of those pages is read to generate the final answer.

The index (`.index.json`) is built once and reused. Rebuilding only happens
if you pass `--reindex` or call `load_document(..., force_reindex=True)`.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run from the command line

```bash
python main.py --doc data/ml_handbook.txt --query "What is the difference between training and inference?"
python main.py --doc data/ml_handbook.txt --query "How do I avoid overfitting?"
python main.py --doc data/ml_handbook.txt --query "What tools exist for experiment tracking?"
```

Add `--reindex` to force a fresh index build:

```bash
python main.py --doc data/ml_handbook.txt --query "What is MLOps?" --reindex
```

## Use as a library

```python
from treerag import TreeRAGPipeline

pipeline = TreeRAGPipeline(verbose=True)
pipeline.load_document("data/ml_handbook.txt")

result = pipeline.ask("What causes overfitting and how do I prevent it?")
print(result["answer"])
print("Pages read:", result["pages_used"])
```

## Run tests

```bash
pytest tests/
```

Tests mock the LLM so they run without an API key and finish in seconds.
