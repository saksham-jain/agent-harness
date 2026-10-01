#!/usr/bin/env python3
"""Index a folder of documents into Qdrant.

A thin CLI over `rag_service` — all the logic lives there, shared with the MCP
server, so indexing and querying cannot drift apart.

Usage: python index_docs.py [docs_dir]     (default: $DOCS_DIR, else ./docs)

Indexing stores absolute paths as the payload filter key, so run it from the same
place the server runs in (inside the container, in other words) or you will get two
sets of points for the same files.
"""
import sys

import rag_service


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else None
    print(rag_service.index(docs_dir))


if __name__ == "__main__":
    main()
