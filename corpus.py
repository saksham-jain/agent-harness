#!/usr/bin/env python3
"""Shared document handling for the RAG layer.

Used by `rag_service` for both indexing and querying, so the two can never disagree
about which files are picked up, how they are split, or how they are embedded. The
MCP layer above knows nothing about any of this.
"""
import os

# CHUNK_CHARS is a *target*, not a fixed width: `chunk()` splits on structure and only
# falls back to hard slicing for a run of text with no boundaries in it.
CHUNK_CHARS, OVERLAP, TOP_K, BATCH = 800, 100, 4, 32

# How many chunks to pull from Qdrant before deduping. TOP_K is the number that reaches
# the model, so this must be comfortably larger: four slots is four chances to be
# relevant, and one long document can otherwise fill all four with adjacent passages.
TOP_N = 40

# Keep at most this many chunks from any one document, so a single long file cannot
# crowd out every other candidate. Adjacent chunks of the same document are near
# duplicates and add no evidence.
MAX_CHUNKS_PER_DOC = 3

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


def _hard_split(text, size):
    """Slice a boundary-free run of text into overlapping fixed-width pieces."""
    step = size - OVERLAP
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def chunk(text):
    """Split text on structure, then pack the pieces up to roughly CHUNK_CHARS.

    Fixed-width slicing cuts mid-sentence and mid-clause, which costs retrieval accuracy
    because the embedding then describes half a thought. Splitting on blank lines, and
    treating a markdown heading as the start of a new unit, keeps a chunk readable on its
    own. Pieces are packed greedily up to the target so short paragraphs do not each
    become their own tiny chunk.
    """
    lines = text.splitlines()
    units, current = [], []
    # A unit is "anchored" if it begins at a top-level heading, which is a document
    # boundary rather than a section within it. Anchored units start a fresh chunk and are
    # never packed backwards into what came before.
    #
    # Sub-headings are deliberately *not* boundaries. Treating them as one produced 32
    # chunks of ~76 chars from a 2.4 KB document, and a 76-char chunk embeds badly. Because
    # the heading text stays in the chunk, several short sections can share one chunk and
    # the embedding still sees every title.
    anchored = False

    def flush():
        nonlocal anchored
        if current:
            joined = "\n".join(current).strip()
            if joined:
                units.append((anchored, joined))
            current.clear()
        anchored = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            flush()
            anchored = True
            current.append(line)
            continue
        current.append(line)
        # A blank line closes a paragraph. Acting on it immediately would make every
        # one-line paragraph its own chunk, so only cut once there is something to cut.
        if not stripped and len("\n".join(current)) >= CHUNK_CHARS:
            flush()
    flush()

    if not units:
        return _hard_split(text, CHUNK_CHARS)

    out, buf = [], ""
    for is_anchored, unit in units:
        # A single oversized unit (a long code block, a dense table) still has to be cut.
        if len(unit) > CHUNK_CHARS:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_hard_split(unit, CHUNK_CHARS))
            continue
        # Start a fresh chunk at a section heading, and never let one section's content
        # bleed into the previous section.
        if buf and (is_anchored or len(buf) + len(unit) + 1 > CHUNK_CHARS):
            out.append(buf)
            buf = unit
        else:
            buf = f"{buf}\n{unit}" if buf else unit
    if buf:
        out.append(buf)
    return out


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
