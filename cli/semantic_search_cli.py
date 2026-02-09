#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the model")

    embed_parser = subparsers.add_parser("embed_text", help="embedding gen")
    embed_parser.add_argument("text", type=str, help="embed text")

    subparsers.add_parser("verify_emb", help="verify the model")

    search_parser = subparsers.add_parser("search", help="search from query")
    search_parser.add_argument("query", type=str, help="query")
    search_parser.add_argument("limit", type=int, help="limit")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "verify_emb":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()