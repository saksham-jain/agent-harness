# agent-harness

A local agent that acts on your files and answers from your document corpus. Ollama for
inference, Qdrant for vectors, MCP between the layers, Docker Compose to run it.

## What's used

| | |
| --- | --- |
| LLM | `qwen2.5:7b` (7.6B, Q4_K_M) via Ollama |
| Embeddings | `nomic-embed-text` (137M, nomic-bert, 768-dim) |
| Vector DB | Qdrant `1.19.1`, 768-dim Cosine, collection `rag_nomic_embed_text` |
| Protocol | MCP Python SDK `2.2.0`, Streamable HTTP |
| Runtime | Python 3.11 (`python:3.11-slim`), Docker Compose |
| Retrieval | 800-char chunks, 100 overlap, top-4 |

## Architecture

```
┌──────────────────┐
│    AI Harness    │   agent_harness_base.py
│  Agent + LLM     │   model loop, bash/read_file/write_file
└────────┬─────────┘
         │
┌────────▼─────────┐
│    MCP Client    │   mcp_client.py
└────────┬─────────┘   holds one connection open
         │  MCP protocol /mcp
┌────────▼─────────┐
│    MCP Server    │   mcp_server.py
│     RAG tools    │   answer_docs, list_docs, refresh_index, index_status
└────────┬─────────┘
         │
┌────────▼─────────┐
│   RAG Service    │   rag_service.py + corpus.py
│    embedding     │   embedding, retrieval, indexing
└────────┬─────────┘
         │
┌────────▼─────────┐
│     Qdrant       │   compose service :6333
└──────────────────┘
```

| Layer | File | Responsibility |
| --- | --- | --- |
| AI Harness | `agent_harness_base.py` | The agent loop. Never imports Qdrant |
| MCP Client | `mcp_client.py` | Tool discovery, schema conversion, dispatch |
| MCP Server | `mcp_server.py` | Protocol only. Tools delegate to the RAG service |
| RAG Service | `rag_service.py` | Embedding and retrieval over Qdrant |
| Corpus | `corpus.py` | File discovery, chunking, batched embedding |
| Indexer | `index_docs.py` | Thin CLI over the RAG service |

Each layer only knows the one below it. `rag_service` is testable without any MCP,
and the harness is usable by any MCP client.

**Ollama is not a compose service** — Metal acceleration needs it on the Mac, so the
containers reach it over `host.docker.internal:11434`.

## Run

```bash
docker compose up -d                                  # qdrant + mcp-server
docker compose --profile index run --rm index         # index docs
docker compose --profile client run --rm mcp-client   # the agent
```

Override `DOCS_DIR` (default `./docs`) and `WORKDIR` (default `.`).

Inside the agent:

```
> /tools
> /call answer_docs {"query": "how long is the trip?"}
> /help
```

Slash commands skip the model, so a tool always runs exactly as asked.

## Logs

```bash
docker compose logs -f mcp-server --no-log-prefix 2>&1 | grep mcp.tools
```

```
INFO    mcp.tools: answer_docs(query='how long is the trip?') -> 'The hotel is booked for 3 nights...' in 37000ms
WARNING mcp.tools: refresh_index() failed after 120ms: No supported files found in '/docs'
```

## Gotchas

- **Bind `0.0.0.0`.** `mcp.run()` defaults to `127.0.0.1`, unreachable from other containers.
- **Use `../.venv/bin/python3`, not bare `python3`.** Without the venv the agent dies at
  the `from openai import OpenAI` line and prints nothing at all.
- **`mcp` is pinned `>=2.2,<3`.** v1 and v2 APIs are incompatible (`FastMCP` → `MCPServer`,
  `ClientSession` → `Client`, transport options moved to `run()`). Most examples online are v1.
- **`MCP_TIMEOUT` (default 300s) must cover a whole tool call.** `answer_docs` embeds,
  searches, then generates — ~37s on a 7B model. A 30s budget truncates it mid-answer.
- **Index from one place.** Paths are the Qdrant payload filter key, so indexing on the
  host and in the container stores two sets of points for the same files.
- **`MIN_SCORE` is a weak signal.** Relevant and irrelevant top-1 scores overlap on this
  corpus (0.59 relevant vs 0.60–0.62 irrelevant), so no threshold separates them. The
  grounding prompt inside `answer_docs` is the actual guardrail. This is why the server
  returns a finished answer rather than raw chunks — handing chunks back would move the
  grounding rules into a 7B model's judgement, where they get lost.
- **Rebuild after changing the Dockerfile.** Compose keeps a per-service image, so a stale
  one can carry an old `ENTRYPOINT` that swallows your `command:` override.

## Local, no Docker

```bash
cd .. && source .venv/bin/activate && cd agent_harness

python3 mcp_server.py                                            # terminal 1
MCP_SERVER_URL=http://localhost:8000/mcp \
  python3 agent_harness_base.py                                  # terminal 2
```

Stop `mcp-server` first or move both to another port — 8000 is taken by the container.
Do not create a venv inside `agent_harness/`; it shadows the working one at `../.venv`.
