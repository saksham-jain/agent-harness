#!/usr/bin/env python3
"""Minimal local RAG: embed a folder of docs with Ollama, retrieve top chunks, answer with qwen.

Usage: python rag.py path/to/docs
"""
import hashlib
import json
import os
import sys

import numpy as np
from openai import OpenAI

from corpus import TOP_K, chunk, embed, list_files, read_file

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
CHAT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DOCS_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs"
INDEX_FILE = ".rag_index.json"


def signature(files):
    parts = [f"{p}:{os.path.getmtime(p)}:{os.path.getsize(p)}" for p in files]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_index():
    files = list_files(DOCS_DIR)
    if not files:
        sys.exit(f"No supported files found in {DOCS_DIR!r}")
    sig = signature(files)
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            cached = json.load(f)
        if cached.get("sig") == sig and cached.get("model") == EMBED_MODEL:
            print(f"Loaded cached index ({len(cached['chunks'])} chunks).")
            return cached["chunks"]

    print(f"Indexing {len(files)} files...")
    chunks = []
    for path in files:
        try:
            for c in chunk(read_file(path)):
                chunks.append({"file": path, "text": c})
        except Exception as e:
            print(f"  skipped {path}: {e}")
    vecs = embed(client, EMBED_MODEL, [c["text"] for c in chunks], "search_document: ", progress=True)
    for c, v in zip(chunks, vecs):
        c["emb"] = v
    with open(INDEX_FILE, "w") as f:
        json.dump({"sig": sig, "model": EMBED_MODEL, "chunks": chunks}, f)
    print(f"\nIndexed {len(chunks)} chunks.")
    return chunks


def retrieve(query, chunks, matrix):
    q = np.array(embed(client, EMBED_MODEL, [query], "search_query: ")[0])
    q /= np.linalg.norm(q)
    scores = matrix @ q
    top = np.argsort(scores)[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top]


def answer(query, hits):
    context = "\n\n".join(f"[{i + 1}] ({c['file']})\n{c['text']}" for i, (c, _) in enumerate(hits))
    messages = [
        {
            "role": "system",
            "content": "Answer using ONLY the context below. Cite sources like [1]. "
            "If the context doesn't contain the answer, say you don't know.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    print(f"\n{resp.choices[0].message.content}\n")
    print("Sources:")
    for i, (c, s) in enumerate(hits):
        print(f"  [{i + 1}] {c['file']} (score {s:.2f})")


def main():
    chunks = build_index()
    matrix = np.array([c["emb"] for c in chunks], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    print("Ask a question about your docs. Ctrl+C to quit.")
    while True:
        try:
            q = input("\n? ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            answer(q, retrieve(q, chunks, matrix))


if __name__ == "__main__":
    main()
