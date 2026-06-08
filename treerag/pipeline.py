from pathlib import Path

from treerag.indexer.loader import DocumentLoader
from treerag.indexer.tree_builder import TreeBuilder
from treerag.retriever.navigator import TreeNavigator


class TreeRAGPipeline:
    """
    The main entry point for the TreeRAG system.

    Usage:
        pipeline = TreeRAGPipeline()
        pipeline.load_document("data/ml_handbook.txt")
        result = pipeline.ask("What causes overfitting?")
        print(result["answer"])
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.loader = DocumentLoader()
        self.builder = TreeBuilder()
        self.navigator = TreeNavigator()

        self._tree = None
        self._page_lookup = None

    def load_document(self, filepath: str | Path, force_reindex: bool = False) -> None:
        """
        Load a document and build (or reuse) its tree index.

        The index is saved as a .json file next to the source document.
        If that file already exists the build step is skipped unless
        force_reindex=True is passed.
        """
        filepath = Path(filepath)
        index_path = filepath.with_suffix(".index.json")

        pages = self.loader.load(filepath)
        self._page_lookup = self.loader.as_lookup(pages)

        if index_path.exists() and not force_reindex:
            if self.verbose:
                print(f"Loading existing index from {index_path}")
            self._tree = self.builder.load(index_path)
        else:
            if self.verbose:
                print(f"Building index for {filepath.name} ({len(pages)} pages)...")
            self._tree = self.builder.build(pages, verbose=self.verbose)
            self.builder.save(self._tree, index_path)

    def ask(self, question: str) -> dict:
        """
        Ask a question against the loaded document.

        Returns a dict with:
          answer       — the final answer string
          pages_used   — list of page numbers that were read
          trace        — list of reasoning steps showing how retrieval worked
        """
        if self._tree is None or self._page_lookup is None:
            raise RuntimeError("Call load_document() before ask().")

        return self.navigator.query(
            question=question,
            tree=self._tree,
            page_lookup=self._page_lookup,
            verbose=self.verbose,
        )
