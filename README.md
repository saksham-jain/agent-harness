# agent-harness

One agent with two halves: it can act on your files, and it can answer from your
document corpus. Ollama for inference, Qdrant for vectors, MCP for tools, Docker
Compose to run it.

## What's used

| | |
| --- | --- |
| LLM | `qwen2.5:7b` (7.6B, Q4_K_M) via Ollama |
| Embeddings | `nomic-embed-text` (137M, nomic-bert, 768-dim) |
| Vector DB | Qdrant `1.19.1`, 768-dim Cosine, collection `rag_nomic_embed_text` |
| Tool protocol | MCP Python SDK `2.2.0`, Streamable HTTP |
| Clients | `openai` `3.16.2` (Ollama's OpenAI-compatible API), `qdrant-client` `1.19.1` |
| Runtime | Python 3.11 (`python:3.11-slim`), Docker Compose |
| Retrieval | 800-char chunks, 100 overlap, top-4 |

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

## Slash commands

Three shortcuts run before the model sees anything, so a tool always executes exactly
as asked:

```
> /tools
> /call word_count {"text": "haha how are you?"}
> /help
```

Useful when the model decides a task is too simple to need a tool — it got *"haha how
are you?"* wrong as **5 words** when `word_count` returned **4**.

## Prompting it

The model is `qwen2.5:7b`, so it needs a nudge toward tools it doesn't know it has.
Verified routing:

| Prompt | Calls |
| --- | --- |
| *What did my interview notes say about salary?* | `answer_docs` |
| *What is the square root of 144?* | `sqrt` |
| *Echo the word "compose" back to me three times* | `echo` |
| *Use the word_count tool on "the quick brown fox jumps"* | `word_count` |
| *Use the now tool to get the current time in +05:30* | `now` |
| *Who wrote the novel Dune?* | nothing, answers directly |

Two behaviours worth knowing. The model **will not** reach for `now` on *"what time
is it?"* — it guesses instead, so name the tool. And it **will** answer trivial
arithmetic itself rather than call `add`, which is faster and fine.

`answer_docs` is only chosen when the question signals private documents ("my
notes", "my resume", "the lease"). *"How long is the hotel booked?"* routes straight
to an answer, because nothing in the question says the hotel is in your corpus.

## Logs

`docker compose logs -f mcp-server` shows every tool call with its arguments, result
and duration, which uvicorn's access log alone cannot tell you:

```
INFO    mcp.tools: add(a=41, b=1) -> 42 in 0ms
INFO    mcp.tools: now(tz='+05:30') -> '2026-09-27T01:37:39+05:30' in 0ms
WARNING mcp.tools: sqrt(x=-9.0) failed after 0ms: Cannot take the square root of a negative number
```

Set `MCP_LOG_LEVEL=DEBUG` for more. If something on your host is polling
`GET /health` every few minutes and filling the log with 404s, that is not this
project — it is an external monitor pointed at port 8000.

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
- **Use the venv's python, not a bare `python3`.** The dependencies live in
  `../.venv`. With no venv active, `python3 agent_harness_base.py` dies at the
  `from openai import OpenAI` line with `ModuleNotFoundError` and prints nothing at all
  — no banner, no error that looks like a banner. Run `../.venv/bin/python3
  agent_harness_base.py`, or activate the venv first.
- **Slow, by design of the model.** A single `answer_docs` call measures **~37s**: an
  embedding call, a Qdrant search, then a full `qwen2.5:7b` generation. A turn using
  three tools is a minute and a half. Budget per tool call, not per turn — the 7b model
  is CPU-bound on Ollama.

## Local, no Docker

There is a venv at `../.venv`. Activate it, or call its interpreter directly:

```bash
cd .. && source .venv/bin/activate && cd agent_harness

python3 mcp_server.py                                           # terminal 1
MCP_SERVER_URL=http://localhost:8000/mcp \
  python3 agent_harness_base.py                                 # terminal 2
```

Or without activating, using `../.venv/bin/python3` for both. Everything else
defaults to Ollama on `localhost:11434` and Qdrant on `localhost:6333`, so the
containers aren't needed.

Do **not** create a venv inside `agent_harness/` — it shadows the working one at
`../.venv` and is a trap for anything that auto-detects `.venv` in the project root.

Both the server and the client default to port `8000`, which `mcp-server` also holds.
Either `docker compose stop mcp-server` first, or pick another port for both:

```bash
MCP_PORT=8100 python3 mcp_server.py                         # terminal 1
MCP_SERVER_URL=http://localhost:8100/mcp \
  python3 agent_harness_base.py                              # terminal 2
```

`answer_docs` appears only if the collection exists, so index with `rag_qdrant.py`
first or the tool is silently omitted.
