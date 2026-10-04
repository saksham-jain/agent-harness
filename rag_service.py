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

from corpus import MAX_CHUNKS_PER_DOC, TOP_K, TOP_N, chunk, embed, list_files, read_file

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
    """(counts, sources) for the indexed documents.

    counts maps doc_id -> chunk count; sources maps doc_id -> absolute path, for display.
    """
    points, _ = qdrant().scroll(COLLECTION, limit=1000, with_payload=["doc_id", "source"])
    counts, sources = {}, {}
    for p in points:
        key = p.payload.get("doc_id") or p.payload.get("file") or "?"
        counts[key] = counts.get(key, 0) + 1
        sources[key] = p.payload.get("source") or p.payload.get("file") or key
    return dict(sorted(counts.items())), sources


def _doc_filter(doc_id):
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def _stored_hash(doc_id):
    points, _ = qdrant().scroll(COLLECTION, scroll_filter=_doc_filter(doc_id), limit=1, with_payload=["hash"])
    return points[0].payload.get("hash") if points else None


def _ensure_collection(dim):
    q = qdrant()
    if not q.collection_exists(COLLECTION):
        q.create_collection(COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        for field in ("doc_id", "source"):
            q.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)


def _prune(seen):
    """Delete documents that are in the index but no longer on disk.

    Without this a deleted file stays searchable forever. Names are reported rather than
    pruned silently, because indexing a *different* directory than last time will remove
    everything from the previous one -- a collection holds exactly one corpus.

    Points written before `doc_id` existed used an absolute path as the key and cannot be
    converted. They are always redundant: a run either writes a modern `doc_id` point for
    the document, or finds one already there, so anything left on the old key is a
    duplicate of the same text.
    """
    points, _ = qdrant().scroll(COLLECTION, limit=1000, with_payload=["doc_id", "file"])
    modern = {p.payload.get("doc_id") for p in points if p.payload.get("doc_id")}
    stale = {}
    for p in points:
        key = p.payload.get("doc_id") or p.payload.get("file")
        if key and key not in seen:
            stale[key] = "doc_id" if key in modern else "file"
    for key, field in sorted(stale.items()):
        qdrant().delete(
            COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=key))])
            ),
        )
    return sorted(stale)


def index(docs_dir=None):
    """Embed changed files into Qdrant. Unchanged files are skipped by content hash.

    Identity is `doc_id`: the path relative to `docs_dir`, so it does not depend on
    where the process runs. Indexing the same corpus from the host and from a container
    previously produced two sets of points, because the absolute path was the filter key.
    `source` keeps the absolute path for display only.
    """
    docs_dir = docs_dir or DOCS_DIR
    files = list_files(docs_dir, absolute=True)
    if not files:
        return f"No supported files found in {docs_dir!r}"

    new = updated = skipped = 0
    seen = set()
    for path in files:
        doc_id = os.path.relpath(path, os.path.abspath(docs_dir))
        seen.add(doc_id)
        try:
            text = read_file(path)
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        exists = qdrant().collection_exists(COLLECTION)
        old = _stored_hash(doc_id) if exists else None
        if old == h:
            skipped += 1
            continue
        chunks = chunk(text)
        if not chunks:
            continue
        vecs = embed(llm, EMBED_MODEL, chunks, "search_document: ")
        _ensure_collection(len(vecs[0]))

        # Upsert before deleting. Point ids are uuid5(doc_id:index), so an upsert
        # overwrites in place and the document is never momentarily unsearchable -- which
        # is what happened when the delete came first and an upsert could fail.
        # Only trailing chunks from a file that shrank need removing, and that is after.
        qdrant().upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{i}")),
                    vector=v,
                    payload={"doc_id": doc_id, "source": path, "hash": h, "chunk": i, "text": c},
                )
                for i, (c, v) in enumerate(zip(chunks, vecs))
            ],
        )
        if exists:
            qdrant().delete(
                COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                            models.FieldCondition(key="chunk", range=models.Range(gte=len(chunks))),
                        ]
                    )
                ),
            )
        if old is None:
            new += 1
        else:
            updated += 1
        print(f"  indexed {doc_id} ({len(chunks)} chunks)")

    pruned = _prune(seen) if qdrant().collection_exists(COLLECTION) else []
    report = f"Index ready: {new} new, {updated} updated, {skipped} unchanged files."
    if pruned:
        report += f" Pruned {len(pruned)} removed document(s): {', '.join(pruned)}."
    return report


def retrieve(query, k=TOP_K):
    """Embed the query, then spread the result slots across documents.

    The dedup is the point. `k` is the number of chunks the model sees, and four slots is
    four chances to be relevant. Querying Qdrant for exactly `k` and stopping there means a
    single long document can fill all four with adjacent passages -- near duplicates that
    add no evidence -- and crowd out every other candidate. So over-fetch, then keep the
    best few per document until `k` distinct-document slots are filled.

    Adjacent chunks of one document are also collapsed to their first occurrence, because
    consecutive chunks of the same text score near-identically and would otherwise consume
    the per-document allowance on their own.
    """
    emb = llm.embeddings.create(model=EMBED_MODEL, input=[SEARCH_PREFIX + query])
    # Ask for more than we need: the surplus is what makes room for other documents.
    fetch = max(k * 4, TOP_N) if k > TOP_K else max(k, TOP_N)
    points = qdrant().query_points(
        COLLECTION, query=emb.data[0].embedding, limit=fetch, with_payload=True
    ).points

    kept, per_doc, last_chunk = [], {}, {}
    for p in points:
        if p.score < MIN_SCORE:
            continue
        doc_id = p.payload.get("doc_id") or p.payload.get("file") or "?"
        idx = p.payload.get("chunk")
        # Skip a chunk immediately following one already kept from the same document.
        if idx is not None and last_chunk.get(doc_id) is not None and idx == last_chunk[doc_id] + 1:
            continue
        if per_doc.get(doc_id, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        if idx is not None:
            last_chunk[doc_id] = idx
        kept.append(p)
        if len(kept) == k:
            break
    return kept


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

    def _label(hit):
        return hit.payload.get("source") or hit.payload.get("doc_id") or hit.payload.get("file", "?")

    context = "\n\n".join(f"[{i + 1}] ({_label(h)})\n{h.payload['text']}" for i, h in enumerate(hits))
    messages = [
        {
            "role": "system",
            "content": "Answer using ONLY the context below. Cite sources like [1]. "
            "If the context doesn't contain the answer, say you don't know.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = llm.chat.completions.create(model=CHAT_MODEL, messages=messages)
    sources = "\n".join(f"  [{i + 1}] {_label(h)} (score {h.score:.2f})" for i, h in enumerate(hits))
    return f"{resp.choices[0].message.content}\n\nSources:\n{sources}"
