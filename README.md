# agent-harness

A local agent that acts on your files and answers from your document corpus. Ollama for
inference, Qdrant for vectors, MCP between the layers, Docker Compose to run it.

## What's used

| Component | Detail |
| --- | --- |
| LLM | `qwen2.5:7b` (7.6B, Q4_K_M) via Ollama |
| Router | `qwen2.5:1.5b` — decides whether a message needs the corpus |
| Embeddings | `nomic-embed-text` (137M, nomic-bert, 768-dim) |
| Vector DB | Qdrant `1.19.1`, 768-dim Cosine, collection `rag_nomic_embed_text` |
| Protocol | MCP Python SDK `2.2.0`, Streamable HTTP |
| Runtime | Python 3.11 (`python:3.11-slim`), Docker Compose |
| Retrieval | Structure-aware chunks (~800 char target, 100 overlap), top-4 after per-document dedup |
| Streaming | `answer_docs` streams tokens as MCP progress notifications. Ollama `stream: true` + `ctx.report_progress()` |
| Tracing | **planned — LangSmith.** The MCP SDK already emits an OpenTelemetry span per request, but router / retrieval / embedding / generation are one opaque span |
| Auth | OAuth 2.1 resource server. `TokenVerifier` + `AuthSettings`, scopes `docs:read`. Static bearer tokens — **not** production auth |
| Auth libs | `mcp.server.auth.*`, `pydantic` `2.13.5`, `httpx2` `2.13.0` (client side) |
| TLS | Real certificate, terminated by a **Tailscale Funnel** at `<machine>.<tailnet>.ts.net`. Origin is plain HTTP on loopback |
| Host protection | `TransportSecuritySettings` — DNS-rebinding. **Off** unless `MCP_ALLOWED_HOSTS` is set |

## Routing

A 7B choosing between nine tools misroutes — it answered *"how long is the hotel
booked?"* from memory instead of searching. So once per message a small model makes one
binary call, and the tool list is filtered on the answer:

```
> how long is the hotel booked?
[route] documents
[tool:mcp] answer_docs({"query": "how long is the hotel booked?"})
```

Measured warm: **~0.7s** for the routing call, against **~37s** for one `answer_docs`.

The two error directions are not symmetric, which is the design:

| | Effect |
| --- | --- |
| False negative — needs docs, routed no | `answer_docs` is hidden, so the model cannot search. **Damaging** |
| False positive — routed yes, did not need | only adds a tool back; the model still chooses. **Harmless** |

So it is deliberately permissive and never forces a call. On a labelled set of eight
questions `qwen2.5:1.5b` scored 6/8 with **no false negatives**; `qwen2.5:0.5b` scored
4/8 and routed arithmetic to the corpus, so it is not used. Set `ROUTER_MODEL=""` to
disable routing entirely.

Pinning matters. An unloaded model costs **~12s** to load, which would dwarf the 0.7s
saved, so the router is held resident with `keep_alive`.

## Architecture

```
┌────────────────────────┐
│      AI Harness        │   agent_harness_base.py
│     Agent + LLM        │   the loop + local file/shell tools;
│                        │   the model itself is the host Ollama box
└────────────┬───────────┘
             │
┌────────────▼───────────┐
│      MCP Client        │   mcp_client.py
│                        │   one connection held open, bearer token
└────────────┬───────────┘
             │  MCP over Streamable HTTP, Authorization: Bearer <token>
┌────────────▼───────────┐
│      MCP Server        │   mcp_server.py — protocol only
│       RAG tools        │   answer_docs, list_docs, refresh_index,
└────────────┬───────────┘   index_status, whoami
             │
┌────────────▼───────────┐
│        Auth            │   auth.py — verifies the token, checks scope
│   TokenVerifier        │   401 + RFC 9728 discovery when it fails
└────────────┬───────────┘
             │
┌────────────▼───────────┐
│     RAG Service        │   rag_service.py + corpus.py
│  embedding, retrieval  │   embedding, retrieval, indexing
└────────────┬───────────┘
             │
┌────────────▼───────────┐
│        Qdrant          │   compose service :6333
└────────────────────────┘

   called by AI Harness and RAG Service, not chained below Qdrant:
        ┌────────────────────────┐
        │       Ollama           │  HOST, not a container
        │  qwen2.5:7b    chat    │  100% GPU (Metal)
        │  qwen2.5:1.5b  router  │  served on :11434, reached over
        │  nomic-embed-text      │  host.docker.internal
        └────────────────────────┘
```

TLS is served by `mcp-server` itself — see **Auth and TLS** below.

| Layer | File | Responsibility |
| --- | --- | --- |
| AI Harness | `agent_harness_base.py` | The agent loop. Never imports Qdrant |
| MCP Client | `mcp_client.py` | Tool discovery, schema conversion, dispatch, bearer token |
| MCP Server | `mcp_server.py` | Protocol only. Tools delegate to the RAG service |
| Auth | `auth.py` | `TokenVerifier` + scope check, between the server and the network |
| RAG Service | `rag_service.py` | Embedding and retrieval over Qdrant |
| Corpus | `corpus.py` | File discovery, chunking, batched embedding |
| Ollama *(host)* | — | Chat, routing decisions, embeddings. **The GPU lives here** |
| Skills | `skills.py` + `skills/*/SKILL.md` | Playbooks the agent can load on demand |
| Indexer | `index_docs.py` | Thin CLI over the RAG service |
| Evals | `evals/` | Labelled cases and the scoring runner |

Each layer only knows the one below it. `rag_service` is testable without any MCP,
and the harness is usable by any MCP client. Ollama sits outside the chain — two layers
call it over HTTP, which is why the "Agent + LLM" box holds the loop, not the model.

**Ollama is not a compose service** — see **Where the GPU is used** below.

## Auth and TLS

**Status: both working, verified end to end.** Bearer tokens are required on every
request, and the public name is served by a Tailscale Funnel holding a real Let's Encrypt
certificate.

```
 any MCP client ──https──> Tailscale          TLS terminates here, real cert for
        │                    <machine>.<tailnet>.ts.net
        │                          │
        │              outbound Funnel — no inbound port,
        │              no router config
        │                          │
        │  Authorization: Bearer <token>
        └──────────────────────────┴──> 127.0.0.1:8000 ──> mcp-server
                                        plain HTTP,      auth.py verify_token()
                                        loopback only    401 if token unknown
```

Verified over that path: no token → `401`; valid token → all five tools; discovery
document's `resource` matching `MCP_RESOURCE_URL` exactly; `answer_docs` returning a
grounded answer in **3.3s** without the relay truncating it; direct loopback rejected
with `421`. `curl http://192.168.1.3:8000` refuses — the LAN has no route in.

The server is an OAuth 2.1 **resource server**: it verifies tokens, never issues them.
`auth.py` implements `TokenVerifier`, one async method — the SDK owns the 401, the
`WWW-Authenticate` pointer and the RFC 9728 discovery document.

Enable auth with `MCP_AUTH=1` in `.env`. It is currently a **static token table**:
possession is identity, no expiry, and revocation means editing the file and restarting.
Fine for a pilot with trusted users, not production auth.

### Why a tunnel rather than a certificate in the container

A `mkcert` certificate exists only where its CA is installed, so it could never let
anyone else connect. The Funnel holds a public certificate for the `ts.net` name, which
means **one URL for everything** — local and remote clients dial the same address, and
there is no second port and no CA to install.

That also retires the TLS code path. `MCPServer.run()` exposes no SSL options, so
certificate support meant building the ASGI app and handing it to uvicorn; with TLS
terminated at the edge, `MCP_TLS_CERT` / `MCP_TLS_KEY` are unset and the origin is plain
HTTP — safe because it is published as `127.0.0.1:8000` and never leaves the machine.

Three things cost time to find: `token_verifier` and `auth` must be passed together or
`MCPServer` raises at construction; `Client` takes no `headers` argument, so a bearer
token has to go on the HTTP client the transport wraps; and on port 443 the `Host`
header carries **no port**, so `MCP_ALLOWED_HOSTS` must list the public hostname bare.

Design, and what is still missing: **[AUTH-TLS.md](AUTH-TLS.md)**.
Roadmap: **[ENHANCEMENTS.md](ENHANCEMENTS.md)**.
How the project got here, mistakes included: **[BUILD-JOURNEY.md](BUILD-JOURNEY.md)**.
Rules for agents and contributors: **[AGENTS.md](AGENTS.md)**.

## Where the GPU is used

**Only on the host. The containers have no GPU and never will.**

```
┌── host ─────────────────────────────────────────────┐
│  Ollama ──> Apple M1 GPU (Metal 4)      ✅ 100% GPU │
│    ▲                                           │
│    │  HTTP, host.docker.internal:11434          │
│  ┌─┴───────────────────────────────┐            │
│  │ mcp-server   172.20.0.3         │  ❌ no GPU  │
│  │ qdrant                          │  ❌ no GPU  │
│  └─────────────────────────────────┘            │
└──────────────────────────────────────────────────┘
```

Verified inside the container: neither `/dev/dri` nor `Metal.framework` exists.

Docker Desktop runs containers in a Linux VM with no GPU passthrough, and Metal cannot be
exposed to a Linux guest. On Linux with NVIDIA you would add
`deploy.resources.reservations.devices`; there is no macOS equivalent. Containerising
Ollama would silently drop it to CPU — still working, just several times slower, with
nothing to warn you.

The containers are not idle, they just don't run models: protocol, auth, orchestration and
vector search. Search over a few hundred vectors is microseconds of CPU, and Qdrant's CPU
HNSW is the right tool at this scale.

That is why the network hop is invisible — **~0.2 ms against a 37-second generation.**

**Memory is shared.** Apple Silicon has one pool for CPU and GPU, so on 16 GB a resident 7B
plus Qdrant plus containers is tight. A container that dies with `exit 137` was SIGKILLed,
almost always by the OOM killer — check `docker compose ps -a` before assuming a code fault.

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
> /skills
> /call answer_docs {"query": "how long is the trip?"}
> /commit
```

Slash commands skip the model, so a tool or skill always runs exactly as asked.

## Skills

Playbooks in `skills/<name>/SKILL.md`, following the
[Agent Skills](https://agentskills.io) open format — an open spec, not one vendor's
convention. YAML frontmatter plus a markdown body:

```yaml
---
name: commit
description: Write and create a git commit. Use when the user asks to commit.
---

Run `git status --short` and `git diff`, then `git commit -F -`.
```

Only `name` and `description` go in the system prompt (~50 tokens each); the body loads
on demand via `load_skill`, so a long playbook costs nothing until it is used.

Invoke with `/<name>`, or let the model call `load_skill` when a description matches.
Expect the automatic path to be unreliable on `qwen2.5:7b` — the same model that
misroutes tools. Two skills ship: `commit` and `stack-status`.

## Evals

```bash
../.venv/bin/python3 evals/run.py --fast    # retrieval + routing, seconds
../.venv/bin/python3 evals/run.py --full    # also grades answers, one generation per case
```

Split by cost on purpose. Retrieval and routing need no generation, run in seconds, and
are where the failures have actually been — so they can gate every change. Answer grading
costs a generation per case and is only worth running when the answer path changes.

Prose quality is deliberately **not** graded: that needs a judge model, and with a 7B
judge that mostly measures the judge. Structure is graded instead — did it abstain, do
the citations resolve.

`cases.json` holds two sets. `cases` is a **tuning set**: the router prompt was last
changed against it, so its score is optimistic. `holdout` was written afterwards and is
never tuned against — that is the number to trust, and it is what the exit code gates on.

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
- **Identity is `doc_id`, not the path.** Document identity is the path *relative to the
  docs directory*, so the same corpus indexed from the host and from a container produces
  one set of points. Indexing also **prunes** documents that have disappeared — and since
  a collection holds exactly one corpus, indexing a different directory than last time
  removes the previous one's documents. That is reported, not silent.
- **`top-4` is deduped, not raw.** Four slots is four chances to be relevant, so `retrieve()`
  over-fetches from Qdrant and keeps the best few chunks per document. Without it, one long
  file fills all four with adjacent near-duplicate passages. Chunks also follow document
  structure rather than a fixed 800-character width, so a chunk is a readable passage instead
  of half a sentence.
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
