#!/usr/bin/env python3
# .venv/Scripts/activate
# uv sync

import argparse
from lib.keyword_search import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index cache")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            res = search_command(args.query, 5)
            for i, re in enumerate(res):
                print(f"{i} {re['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()