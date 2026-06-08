import argparse
import sys
from treerag import TreeRAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="TreeRAG — ask questions about a document without vectors"
    )
    parser.add_argument("--doc", required=True, help="Path to the text document")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the index even if one already exists",
    )
    args = parser.parse_args()

    pipeline = TreeRAGPipeline(verbose=True)

    try:
        pipeline.load_document(args.doc, force_reindex=args.reindex)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nQuestion: {args.query}\n")
    result = pipeline.ask(args.query)

    print("\nANSWER")
    print(result["answer"])

    print("\nPages read:", result["pages_used"])
    print("\nReasoning trace:")
    for step in result["trace"]:
        print(f"  {step}")


if __name__ == "__main__":
    main()
