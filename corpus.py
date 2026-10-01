#!/usr/bin/env python3
"""Shared document handling for the RAG layer.

Used by `rag_service` for both indexing and querying, so the two can never disagree
about which files are picked up, how they are split, or how they are embedded. The
MCP layer above knows nothing about any of this.
"""
import os

CHUNK_CHARS, OVERLAP, TOP_K, BATCH = 800, 100, 4, 32
EXTS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".pdf"}


def read_file(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader  # pip install pypdf

        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    with open(path, errors="ignore") as f:
        return f.read()


def list_files(docs_dir, absolute=False):
    """Files under `docs_dir` that we know how to index.

    `absolute` matters: Qdrant stores the path as the payload filter key, so changing
    how it is spelled orphans every point already in the collection. Keep it True there.
    """
    files = []
    for root, dirs, names in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        for n in names:
            if os.path.splitext(n)[1].lower() in EXTS:
                p = os.path.join(root, n)
                files.append(os.path.abspath(p) if absolute else p)
    return sorted(files)


def chunk(text):
    step = CHUNK_CHARS - OVERLAP
    return [text[i : i + CHUNK_CHARS] for i in range(0, len(text), step) if text[i : i + CHUNK_CHARS].strip()]


def embed(client, model, texts, prefix, progress=False):
    """Embed `texts` in batches, prefixing each for the embedding model's task."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = [prefix + t for t in texts[i : i + BATCH]]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend(d.embedding for d in resp.data)
        if progress:
            print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}", end="\r")
    return out
