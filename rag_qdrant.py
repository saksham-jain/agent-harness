#!/usr/bin/env python3
"""Local RAG with Qdrant as the vector store, Ollama for embeddings + chat.

Usage: python rag_qdrant.py path/to/docs
Requires: docker compose up -d   (Qdrant on :6333)  and Ollama running on the host.
"""
import hashlib
import os
import re
import sys
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient, models

from corpus import TOP_K, chunk, embed, list_files, read_file

OLLAMA_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHAT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DOCS_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs"
COLLECTION = "rag_" + re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL)  # one collection per embed model

llm = OpenAI(base_url=OLLAMA_URL, api_key=os.getenv("LLM_API_KEY", "ollama"))
qdrant = QdrantClient(url=QDRANT_URL)


def file_filter(path):
    return models.Filter(must=[models.FieldCondition(key="file", match=models.MatchValue(value=path))])


def stored_hash(path):
    points, _ = qdrant.scroll(COLLECTION, scroll_filter=file_filter(path), limit=1, with_payload=["hash"])
    return points[0].payload.get("hash") if points else None


def ensure_collection(dim):
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
        qdrant.create_payload_index(COLLECTION, "file", models.PayloadSchemaType.KEYWORD)


def index_docs():
    files = list_files(DOCS_DIR, absolute=True)
    if not files:
        sys.exit(f"No supported files found in {DOCS_DIR!r}")
    new = updated = skipped = 0
    for path in files:
        try:
            text = read_file(path)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        exists = qdrant.collection_exists(COLLECTION)
        old = stored_hash(path) if exists else None
        if old == h:
            skipped += 1
            continue
        chunks = chunk(text)
        if not chunks:
            continue
        vecs = embed(llm, EMBED_MODEL, chunks, "search_document: ")
        ensure_collection(len(vecs[0]))
        if old is not None:  # file changed: drop its old chunks
            qdrant.delete(COLLECTION, points_selector=models.FilterSelector(filter=file_filter(path)))
            updated += 1
        else:
            new += 1
        qdrant.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{i}")),
                    vector=v,
                    payload={"file": path, "hash": h, "chunk": i, "text": c},
                )
                for i, (c, v) in enumerate(zip(chunks, vecs))
            ],
        )
        print(f"  indexed {path} ({len(chunks)} chunks)")
    print(f"Index ready: {new} new, {updated} updated, {skipped} unchanged files.")


def answer(query):
    qvec = embed(llm, EMBED_MODEL, [query], "search_query: ")[0]
    hits = qdrant.query_points(COLLECTION, query=qvec, limit=TOP_K, with_payload=True).points
    context = "\n\n".join(f"[{i + 1}] ({h.payload['file']})\n{h.payload['text']}" for i, h in enumerate(hits))
    messages = [
        {
            "role": "system",
            "content": "Answer using ONLY the context below. Cite sources like [1]. "
            "If the context doesn't contain the answer, say you don't know.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = llm.chat.completions.create(model=CHAT_MODEL, messages=messages)
    print(f"\n{resp.choices[0].message.content}\n\nSources:")
    for i, h in enumerate(hits):
        print(f"  [{i + 1}] {h.payload['file']} (score {h.score:.2f})")


def main():
    index_docs()
    print("Ask a question about your docs. Ctrl+C to quit.")
    while True:
        try:
            q = input("\n? ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
