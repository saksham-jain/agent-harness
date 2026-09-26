# agent-harness

One agent with two halves: it can act on your files, and it can answer from your
document corpus. Ollama for inference, Qdrant for vectors, MCP for tools, Docker
Compose to run it.

## Files

| File | Role |
| --- | --- |
| `agent_harness_base.py` | The agent. Local tools + `answer_docs` + tools discovered over MCP |
| `mcp_server.py` | MCP server exposing tools over Streamable HTTP |
| `rag_qdrant.py` | Indexes `docs/` into Qdrant. Run once, then it's a store |
| `rag.py` | Older numpy-only RAG, kept for reference |

## Run

```bash
docker compose up -d                                     # qdrant + mcp-server
docker compose --profile rag run --rm rag                # index docs (once)
docker compose --profile client run --rm mcp-client      # the agent
```

Override with `DOCS_DIR` (default `./docs`) and `WORKDIR` (default `.`).

## How it fits together

```
                     ┌──────────────────────────┐
  mcp-client ──MCP──>│  mcp-server              │  add, echo, now, sqrt, word_count
         ──local────>│  bash, read_file,        │
         ──local────>│       write_file         │
         ──local────>│  answer_docs ──> Qdrant  │  static doc search
                     └──────────────────────────┘
                                └──> Ollama (embed + generate)
```

`answer_docs` deliberately talks to Qdrant **directly** rather than over MCP. The
MCP server stays a general tools server; retrieval is the agent's own capability.

## Why answer_docs and not search_docs

`search_docs` would hand raw chunks to the model and let it write the answer. That
moves the grounding rules out of a prompt and into a 7B model's judgement, which is
where they get lost. `answer_docs` keeps generation inside the tool, where the
grounding prompt and the "say you don't know" rule still apply.

A score threshold does **not** protect you. Measured on this corpus, relevant and
irrelevant top-1 scores overlap:

| Query | top-1 |
| --- | --- |
| *How long is the hotel booked?* (in corpus) | 0.763 |
| *What is on the resume?* (in corpus) | 0.590 |
| *Capital of France?* (not in corpus) | 0.600 |
| *Stock price of Apple?* (not in corpus) | 0.623 |

No threshold keeps all relevant hits and drops all irrelevant ones. `MIN_SCORE`
(default `0.5`) only starts to work once the corpus is large and varied enough for
similarity to spread out. Until then, the grounding prompt is the guardrail.

## Gotchas

- **The MCP server must bind `0.0.0.0`.** `run()` defaults to `127.0.0.1`, unreachable
  from other containers.
- **Rebuild after changing the Dockerfile.** Compose keeps a per-service image, so a
  stale one can carry an old `ENTRYPOINT` that swallows your `command:` override.
- **`mcp` is pinned `>=2.2,<3`.** v1 and v2 APIs are incompatible (`FastMCP` →
  `MCPServer`, `ClientSession` → `Client`, transport options moved to `run()`). Most
  examples online are still v1.
- **Raise `ToolError` for anything the model should read and retry.** Any other
  exception reaches it as a bare "Error executing tool X".
- **Slow, by design of the model.** A single `answer_docs` call measures **~37s**: an
  embedding call, a Qdrant search, then a full `qwen2.5:7b` generation. A turn using
  three tools is a minute and a half. Budget per tool call, not per turn — the 7b model
  is CPU-bound on Ollama.

## Local, no Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 mcp_server.py                                           # terminal 1
MCP_SERVER_URL=http://localhost:8000/mcp \
  python3 agent_harness_base.py                                 # terminal 2
```

macOS ships `python3`, not `python`. Everything defaults to Ollama on
`localhost:11434` and Qdrant on `localhost:6333`, so the containers aren't needed.

Both the server and the client default to port `8000`, which `mcp-server` also holds.
Either `docker compose stop mcp-server` first, or pick another port for both:

```bash
MCP_PORT=8100 python3 mcp_server.py                         # terminal 1
MCP_SERVER_URL=http://localhost:8100/mcp \
  python3 agent_harness_base.py                              # terminal 2
```

`answer_docs` appears only if the collection exists, so index with `rag_qdrant.py`
first or the tool is silently omitted.
