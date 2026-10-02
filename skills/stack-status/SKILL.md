---
name: stack-status
description: Check that the model, the vector store, and the document index are all reachable. Use when something looks broken, or before blaming retrieval for a wrong answer.
---

Run these in order, stopping at the first failure. Each one uses an HTTP endpoint that
works whether you are on the host or inside the client container.

1. `curl -s -m 5 localhost:11434/api/tags` — Ollama must answer. If it hangs rather than
   failing, a generation is already running: Ollama serves one request at a time, so
   anything else is queued behind it.
2. `curl -s -m 5 localhost:6333/collections` — Qdrant must list the collection. If the
   collection is missing the corpus was never indexed.
3. Call the `index_status` MCP tool. It reports the collection name, the URL, the
   documents directory, and the embed model the index was built with.

Then report which step failed, one line each, quoting what actually came back. Do not
speculate about causes — if a command returned nothing, say it returned nothing.

Two failures that look alike but are not the same bug:

- `index_status` says the index is fine, but the answer is wrong — that is retrieval, not
  infrastructure. Check whether the source chunk contained the answer at all.
- A question answered in far longer than usual — that is a slow generation on a CPU-bound
  model, roughly 37s for a document answer.

Note that `docker compose ps` will not work from inside the client container; there is no
Docker socket there. Use the endpoints above instead.
