#!/usr/bin/env python3
"""RAG Service: embedding and retrieval over Qdrant.

The layer below the MCP server. Knows about documents, embeddings and vectors --
and nothing about MCP. Both `mcp_server.py` and `index_docs.py` call into it, so
indexing and querying can never drift apart.

Env:
  LLM_BASE_URL   default http://localhost:11434/v1
  QDRANT_URL     default http://localhost:6333
  EMBED_MODEL    default nomic-embed-text
  DOCS_DIR       default /docs
  MIN_SCORE      relevance floor; see README for why it is only a weak signal
"""
import hashlib
import os
import re
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient, models

from corpus import TOP_K, chunk, embed, list_files, read_file

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHAT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DOCS_DIR = os.getenv("DOCS_DIR", "/docs")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.5"))
SEARCH_PREFIX = "search_query: "  # nomic-embed-text requires the task prefix
COLLECTION = "rag_" + re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL)  # one per embed model

llm = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "ollama"))
_qdrant = None


def qdrant():
    """Lazily connect, so importing this module never fails on a cold Qdrant."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, timeout=15)
    return _qdrant


def available():
    """True when Qdrant is reachable and the collection has been built."""
    try:
        return qdrant().collection_exists(COLLECTION)
    except Exception:
        return False


def list_docs():
    """Map of file path -> number of chunks stored for it."""
    points, _ = qdrant().scroll(COLLECTION, limit=1000, with_payload=["file"])
    counts = {}
    for p in points:
        path = p.payload.get("file", "?")
        counts[path] = counts.get(path, 0) + 1
    return dict(sorted(counts.items()))


def _file_filter(path):
    return models.Filter(must=[models.FieldCondition(key="file", match=models.MatchValue(value=path))])


def _stored_hash(path):
    points, _ = qdrant().scroll(COLLECTION, scroll_filter=_file_filter(path), limit=1, with_payload=["hash"])
    return points[0].payload.get("hash") if points else None


def _ensure_collection(dim):
    q = qdrant()
    if not q.collection_exists(COLLECTION):
        q.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        q.create_payload_index(COLLECTION, "file", models.PayloadSchemaType.KEYWORD)


def index(docs_dir=None):
    """Embed changed files into Qdrant. Unchanged files are skipped by content hash.

    Paths are stored absolute and are the payload filter key, so the same docs
    indexed from the host and from a container produce two sets of points. Keep
    indexing in one place.
    """
    docs_dir = docs_dir or DOCS_DIR
    files = list_files(docs_dir, absolute=True)
    if not files:
        return f"No supported files found in {docs_dir!r}"

    new = updated = skipped = 0
    for path in files:
        try:
            text = read_file(path)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        exists = qdrant().collection_exists(COLLECTION)
        old = _stored_hash(path) if exists else None
        if old == h:
            skipped += 1
            continue
        chunks = chunk(text)
        if not chunks:
            continue
        vecs = embed(llm, EMBED_MODEL, chunks, "search_document: ")
        _ensure_collection(len(vecs[0]))
        if old is not None:  # file changed: drop its old chunks first
            qdrant().delete(COLLECTION, points_selector=models.FilterSelector(filter=_file_filter(path)))
            updated += 1
        else:
            new += 1
        qdrant().upsert(
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
    return f"Index ready: {new} new, {updated} updated, {skipped} unchanged files."


def retrieve(query, k=TOP_K):
    """Embed the query and return the chunks that clear MIN_SCORE."""
    emb = llm.embeddings.create(model=EMBED_MODEL, input=[SEARCH_PREFIX + query])
    points = qdrant().query_points(COLLECTION, query=emb.data[0].embedding, limit=k, with_payload=True).points
    return [p for p in points if p.score >= MIN_SCORE]


def answer(query):
    """Ground an answer in the corpus, or admit nothing was found.

    Generation stays in here on purpose: a caller that receives raw chunks has to
    do its own grounding, which is exactly where small models start inventing.
    """
    if not available():
        return (
            f"No document index found (collection {COLLECTION!r} at {QDRANT_URL}). "
            "Run the indexer first."
        )
    try:
        hits = retrieve(query)
    except Exception as e:
        return f"Document search unavailable: {e}"

    if not hits:
        return (
            f"No document matched {query!r} above the relevance threshold "
            f"(MIN_SCORE={MIN_SCORE}). Either the corpus does not cover it, or the "
            "threshold is too high for this embedding model."
        )

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
    sources = "\n".join(f"  [{i + 1}] {h.payload['file']} (score {h.score:.2f})" for i, h in enumerate(hits))
    return f"{resp.choices[0].message.content}\n\nSources:\n{sources}"
