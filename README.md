# agent-harness

A local agent that acts on your files and answers from your document corpus. Ollama for
inference, Qdrant for vectors, MCP between the layers, Docker Compose to run it.

## Quickstart

Verified on a fresh clone. Needs [Ollama](https://ollama.com) running with
`ollama pull qwen2.5:7b && ollama pull nomic-embed-text`, and Docker.

```bash
cp tokens.example.json tokens.json              # 1. a token table; keys are the tokens
mkdir -p docs && cp -R docs.sample/. docs/      # 2. somewhere to keep documents
docker compose --profile index run --rm index    # 3. embed them
```

Step 2 copies the **21-document sample corpus** (markdown, CSV, JSON, Python, text and
PDF), so the agent has something real to answer from. Replace it with your own files, or
point `DOCS_DIR` somewhere else.

Everything binds to `127.0.0.1` — the MCP server, and Qdrant, which has no authentication and
so must not be reachable from the network. Its dashboard is still at
<http://localhost:6333/dashboard>.

Then either the terminal agent or the browser client:

```bash
docker compose --profile client run --rm mcp-client
# or open http://localhost:8000 and paste a token from tokens.json
```

```
> how long is the hotel booked for?
[route] documents
[tool:mcp] answer_docs(...)
The hotel is booked for 3 nights [1].
```

**Three things the clone does not contain, and why.** `tokens.json` and `docs/` are
gitignored — the first holds bearer tokens, the second is where *your* documents live, and
neither belongs in a public repository. `docs.sample/` is the committed copy the evals use.
A missing `docs/` is not an error state: the indexer refuses rather than pruning a corpus it
cannot see.

## Use it from your own documents

Drop anything into `docs/` — `.txt .md .pdf .csv .json .py .js .ts` — then
`docker compose --profile index run --rm index`. Files are chunked on structure, embedded, and
searchable. Delete a file and the next run prunes it; the index is derived from the directory,
so the files are the only copy.

## What's used

| Component | Detail |
| --- | --- |
| LLM | `qwen2.5:7b` (7.6B, Q4_K_M) via Ollama; optional `FAST_MODEL` (default `qwen2.5:1.5b`) used for short queries (≤ `FAST_MODEL_THRESHOLD` chars, default 80) |
| Router | `qwen2.5:1.5b` — one binary call per message. Few-shot prompt, **0 false negatives** on the tuning set |
| Embeddings | `nomic-embed-text` (137M, nomic-bert, 768-dim) |
| Vector DB | Qdrant `1.19.1`, 768-dim Cosine, collection `rag_nomic_embed_text` |
| Protocol | MCP Python SDK `2.2.0`, Streamable HTTP |
| Runtime | Python 3.11 (`python:3.11-slim`), Docker Compose |
| Retrieval | Structure-aware chunks (~800 char target, 100 overlap), top-4 after per-document dedup |
| Streaming | `answer_docs` streams tokens as MCP progress notifications. Ollama `stream: true` + `ctx.report_progress()` |
| Web client | `web/index.html` at `/` — same origin as `/mcp`, so a phone browser works with no app. Set `MCP_WEB=0` to disable |
| Tracing | **optional — LangSmith.** Set `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY`; the loop is wrapped in `@traceable` spans (router, turn, LLM calls, tools). Off by default, no-op when unset |
| Auth | OAuth 2.1 resource server. `TokenVerifier` + `AuthSettings`, scopes `docs:read`. Static bearer tokens — **not** production auth |
| Auth libs | `mcp.server.auth.*`, `pydantic` `2.13.5`, `httpx2` `2.13.0` (client side) |
| TLS | Real certificate, terminated by a **Tailscale Funnel** at `<machine>.<tailnet>.ts.net`. Origin is plain HTTP on loopback |
| Host protection | `TransportSecuritySettings` — DNS-rebinding. **Off** unless `MCP_ALLOWED_HOSTS` is set |
| Sample corpus | 21 documents, 6 formats, 35 chunks — `docs.sample/`, committed, and what the evals run against |

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
| False positive — routed yes, did not need | one extra retrieval on a question that did not need it |

So the router is deliberately permissive, and the prompt is written to match: it
**demonstrates** the decision with six worked examples rather than describing it, and it
**ends on "yes"** so the model's last impression is the cheap error.

### The prompt has to show, not tell

A prose prompt — *"answer yes for first person or possessive wording"* — scored **0.69**
on the labelled set with **nine false negatives**. It answered "no" to *"what did the
doctor prescribe?"* and to *"how many months notice do I have to leave the flat"*, because
neither contains a pronoun, so it read them as general knowledge.

| Router | Score | False negatives |
| --- | --- | --- |
| Prose prompt, `qwen2.5:1.5b` | 0.69 | 9 |
| Prose prompt + a list of the indexed filenames | 0.44 | 18 |
| Prose prompt, `qwen2.5:7b` | 0.38 | 20 |
| One-line prompt, `qwen2.5:1.5b` | 0.19 | 26 |
| **Few-shot, `qwen2.5:1.5b`** | **0.84** | **0** |

A bigger model is not the fix — the 7B is *worse* at this than the 1.5B, for the same
reason it declines `answer_docs`: it reasons about the question on its own terms rather
than pattern-matching the demonstrated task. And listing the indexed filenames made it
worse, not better: grounding helps a 7B, but 21 filenames in a preamble drown a 1.5B's
attention.

**The order of the examples is load-bearing.** Moving the "yes" examples to the front
dropped the score to 0.53 with six false negatives; eight examples were worse than six.
The last example sets the bias, so the prompt ends on "yes" — which is the right bias
here, because a false positive only adds an option the model can ignore while a false
negative means the documents can never be searched.

**And there is no prose in it at all.** The shipped prompt is six worked examples and one
neutral lead sentence. A clarifying lead — *"Answer no only for arithmetic, general
knowledge, coding questions, or a named file"* — collapsed it from **0.84 to 0.16** with
fifteen false negatives, because "Answer no only for…" primes the model to answer no and
overrides every demonstration after it. *"If you are unsure, answer yes"* cost 0.10 in
either position. A 1.5B pattern-matches; it does not read a rule and apply it to the
examples, so any instruction competes with the pattern.

The examples are deliberately generic and appear in **no eval case**, so the tuning set
stays a clean measure of the prompt. Set `ROUTER_MODEL=""` to disable routing entirely.

### A correct router is not enough — the router's decision is enforced

Filtering the tool list was the first half, and it turned out not to be sufficient. The
7B was offered `answer_docs` and **declined to call it**, answering from memory instead:

```
> when are we going to Hampi?
[route] documents        ← router correct
no tool                  ← 7B declines
```

Reproduced with a direct `chat.completions.create`, outside this codebase entirely, so it
was never a routing or MCP problem. Measured over three document questions:

| System prompt | Called `answer_docs` |
| --- | --- |
| Describes which tool to prefer | **0/3** |
| Also states the documents exist | **3/3** |
| …plus "call it even if you think you know" | 5/5 on document questions, but 2/5 on ones that should not retrieve |

The cause is not disobedience. The prompt said *which tool to prefer* without saying the
documents **existed**, so the model read "where are we going in November?" as unanswerable
and said so — while `test.md` contained exactly that. For a personal corpus the failure is
confidently wrong, not visibly wrong.

So when the router says `documents` and the model returns no tool call, the call is made
anyway:

```
> when are we going to Hampi?
[route] documents
[tool:mcp] answer_docs({'query': 'when are we going to Hampi?'}) (forced: router said documents)
You are going to visit Hampi in November. [1]
```

This is the one place the model gets no veto, and it is deliberate: a 1.5B making a binary
call is more reliable than a 7B deciding whether it needs to act on it. A forced call that
fails falls back to whatever the model said, so this can never replace a usable answer with
an error.

The cost is honest — a false-positive route now costs a real retrieval rather than nothing.
`MIN_SCORE` and the grounding prompt inside `answer_docs` are what stop the extra context
from becoming a wrong answer, and on this one-document corpus that path abstains
correctly.

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
| Ollama *(host)* | — | Chat, routing decisions, embeddings. **The GPU lives here** — and **is the inference engine** |
| Skills | `skills.py` + `skills/*/SKILL.md` | Playbooks the agent can load on demand |
| Indexer | `index_docs.py` | Thin CLI over the RAG service |
| Evals | `evals/` | Labelled cases and the scoring runner |

Each layer only knows the one below it. `rag_service` is testable without any MCP,
and the harness is usable by any MCP client. Ollama sits outside the chain — two layers
call it over HTTP, which is why the "Agent + LLM" box holds the loop, not the model.

**Ollama is not a compose service** — see **Where the GPU is used** below.

### Where inference happens

Ollama is the **inference engine**, not just a model host. The token loop —
logits → sample → append → repeat → stop — runs inside Ollama (through llama.cpp on the
GGUF weights). The harness builds the prompt and calls it; the loop itself is Ollama's.

```
AI Harness (agent_harness_base.py)
   │  builds prompt: system + retrieved context + question
   ▼
Ollama — HOST, the inference engine
   │  serves the token loop: logits → sample → append → repeat → stop
   ▼
answer → MCP progress notifications → answer_docs
```

Verified live: Ollama serving on `:11434`, `qwen2.5:7b` context length **32768**, and token
counters (`prompt_eval_count`, `eval_count`) present in real generation responses.

The harness now reads that usage back: `rag_service.answer_from` captures the OpenAI-style
`usage` object from the generation response and `answer_docs` returns it with the answer as a
trailing `[token usage: prompt_tokens=… completion_tokens=…]` line, so token measurement runs
through the real pipeline instead of by hand-calling Ollama per generation. `answer_docs` also
appends a `[response time: Xs]` line measured around the whole call. The web UI renders both
lines as small source-style hints under the answer, so the counts are visible in the panel
rather than only in logs.

**Answer cache.** `rag_service` keeps an in-memory LRU cache of generated answers keyed by the
normalized query (`ANSWER_CACHE_TTL` seconds, `ANSWER_CACHE_SIZE` entries; set TTL to 0 to
disable). A cache hit skips retrieval and generation entirely and restores the token stats.

**Fast-model fallback.** `rag_service._select_model` chooses the generation model per query:
queries no longer than `FAST_MODEL_THRESHOLD` characters (default 80) use `FAST_MODEL`
(default `qwen2.5:1.5b`) when configured; longer queries keep `CHAT_MODEL` (`qwen2.5:7b`).
The chosen model is appended to the answer as `[model: …]` and rendered in the UI. This trades
quality for latency on short questions; the evals gate whether the trade is acceptable.

### Other inference options

Ollama is the current choice, not the only one. The layer boundary is what matters — any
OpenAI-compatible endpoint can back the harness unchanged.

| Option | What it is | Status |
| --- | --- | --- |
| **Ollama** *(current)* | Local runtime; llama.cpp on GGUF; GPU on host | ✅ in use, verified |
| **llama.cpp** | The engine underneath Ollama — self-hosted directly | not chosen; same engine |
| vLLM / LocalAI / LM Studio / text-generation-webui | Other self-hosted OpenAI-compatible servers | not chosen (planned) |
| Hosted open models (Groq free tier, Hugging Face free inference, OpenRouter…) | Remote inference, free tiers where they exist | not chosen — local-first |

The swap is a `base_url` change, not a rewrite.

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

### Read and write are different scopes

| Scope | Grants |
|---|---|
| `docs:read` | `answer_docs`, `list_docs`, `index_status`, `whoami` |
| `docs:write` | `refresh_index` — it rewrites the index |

These are enforced differently, and that difference is the whole design. `docs:read` is
`MCP_REQUIRED_SCOPES`, a **floor** the SDK applies to every request with AND semantics.
Per-tool scopes are declared with `@needs_scope()` and checked against the token's own
scopes. Putting `docs:write` in the required list would force it on read-only tokens — the
opposite of the intent.

Verified with two real tokens over HTTPS:

```
answer_docs    saksham (read+write) -> cited answer
answer_docs    bob     (read only)   -> cited answer
refresh_index  saksham              -> Index ready: 0 new, 0 updated, 1 unchanged
refresh_index  bob                  -> This tool needs docs:write, which this token does not have.
```

The refusal is a `ToolError`, so the model reads the reason and can ask for the right token
— which a bare "Error executing tool" would not allow. A token with no `scopes` field falls
back to the required list, so an entry predating this is read-only rather than accidentally
privileged.

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

## Use it from a phone

`web/index.html` is served at `/`, same origin as `/mcp`. Open the hostname in any phone
browser, paste a bearer token once, and it works — no app, no build step, no framework.

```
https://sakshams-macbook-air.tailf61e07.ts.net/
```

The page is a real MCP client: it runs `initialize`, keeps the `mcp-session-id` it gets
back, and sends it on every later request. Skipping that handshake gets `400 Missing session
ID`, and replies arrive as SSE frames rather than one JSON body.

Same origin is the point — no CORS preflight, and no second hostname to remember.

**The token lives in `localStorage` on a public hostname.** That is the same trust boundary
as the API itself — the Funnel hostname is reachable by anyone, and the token never expires
— so this does not weaken anything, but it does mean a shared browser profile should not
keep it. See [AUTH-TLS.md](AUTH-TLS.md) for what would close that.

Anything else speaking MCP over HTTP works the same way:

```json
{
  "mcpServers": {
    "agent-harness": {
      "type": "http",
      "url": "https://sakshams-macbook-air.tailf61e07.ts.net/mcp",
      "headers": { "Authorization": "Bearer tok_saksham_..." }
    }
  }
}
```

Set `MCP_WEB=0` to turn the page off and keep the surface protocol-only.

### Which tools the page may call

Safe parity with the terminal agent, not full parity. The terminal has 10 tools; the page
gets the 5 that are safe to expose over a public URL.

| Tool | Page | |
|---|---|---|
| `answer_docs`, `index_status`, `whoami` | ✅ | read-only |
| `refresh_index` | ⚠️ | behind a `confirm()`, the one mutation |
| `bash`, `read_file`, `write_file`, `load_skill` | ❌ | not reachable, and deliberately so |
| arithmetic | ✅ | computed in the browser, ~0.04 ms |

The **Library** panel is not a tool — it is a view. It calls the REST routes directly
(`/docs`, `/uploads`, `/upload`, `/uploads/approve`, `/uploads/reject`), shows everything
indexed with chunk counts, and lists pending uploads with an **Approve** and a **Reject**
button each. Approving is what indexes the document; the panel says so, because a document
can contain text that tries to steer the model and the gate is the only control measured to
hold. A read-only token sees the list and gets `403` on the buttons.

The last row is not theatre. `10+1` sent to `answer_docs` costs **4.45s** and returns "no
document matched"; computed locally it is instant and exact. Picking the right tool is a
speed fix, not a feature.

**`bash` and `write_file` are excluded on purpose.** They live in the *client* process, so
the page cannot reach them anyway — but publishing a shell behind a URL that is public and a
token that never expires is not a demo. Per-tool scopes would be the prerequisite for even
`refresh_index` being comfortable.

Arithmetic is parsed with a regex, never `eval`. It takes `10+1`, `what is 10+1` and
`calculate 2.5*4`, and rejects `3-5 May` — a date range must not be read as subtraction.

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

## Speeding up responses

Baseline on this machine: `qwen2.5:7b` generates at ~9 tok/s on Metal, so a 100-token
answer takes ~11 s plus ~3 s prefill. The pieces below are already wired in; tune them
through `.env` / compose env without code changes.

### Already implemented

| Mechanism | Env / default | Effect |
| --- | --- | --- |
| Fast-model fallback | `FAST_MODEL=qwen2.5:1.5b`, `FAST_MODEL_THRESHOLD=80` | Short queries (≤ 80 chars) use the 1.5B model (~4× faster); longer ones keep 7B. Chosen model is appended as `[model: …]` |
| Answer cache | `ANSWER_CACHE_TTL=300`, `ANSWER_CACHE_SIZE=64` (0 disables) | Identical queries skip retrieval + generation entirely; token stats restored from cache |
| Output cap | `CHAT_MAX_TOKENS=512` | Stops runaway generation |
| Context trim | `RETRIEVE_K=4`, `MAX_CONTEXT_CHARS=12000` | Fewer/shorter passages → faster prefill |
| Keep-alive | `LLM_KEEP_ALIVE=30m` | Avoids ~12 s model reloads between calls |
| Streaming | (always on) | First token in ~3 s instead of waiting for the whole answer |

### Not yet wired — biggest remaining wins

| Change | How | Expected gain |
| --- | --- | --- |
| Disable the router | `ROUTER_MODEL=""` | ~2 s saved per query (routing accuracy drops; evals measure it) |
| Smaller context window | `OLLAMA_NUM_CTX=2048` (Ollama env) | Faster prefill, lower memory |
| Flash attention | `OLLAMA_FLASH_ATTENTION=1` (Ollama env) | ~10–30 % faster decode |
| Quantized KV cache | `OLLAMA_KV_CACHE_TYPE=q8_0` (Ollama env) | Lower memory, modest speedup |
| Smaller / more quantized model | `LLM_MODEL=qwen2.5:3b` or a Q3 variant | Large, at a quality cost |
| Hosted inference | point `LLM_BASE_URL` at Groq / OpenRouter | 10–20× decode speed, costs local-first |

### Measure before guessing

`evals/run.py --fast` scores retrieval + routing with no generation; `--full` grades answers
and citations. Any change to model, prompt, or retrieval knobs should be checked against the
holdout before it is trusted.

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
../.venv/bin/python3 evals/run.py --fast              # retrieval + routing, seconds
../.venv/bin/python3 evals/run.py --fast --no-model    # skip the 7B column: ~20s
../.venv/bin/python3 evals/run.py --full               # also grades answers, one generation per case
```

Split by cost on purpose. Retrieval and routing need no generation, run in seconds, and
are where the failures have actually been — so they can gate every change. Answer grading
costs a generation per case and is only worth running when the answer path changes.

**`--no-model` exists because the `model would call` column costs one 7B generation per
case** — it was most of a ten-minute run, and the 7B competes for memory with the MCP
server on the same machine. That column is the one that caught the model ignoring the
router, so it is worth keeping; it is just not worth paying for on every run.

Prose quality is deliberately **not** graded: that needs a judge model, and with a 7B
judge that mostly measures the judge. Structure is graded instead — did it abstain, do
the citations resolve.

`cases.json` holds two sets. `cases` is a **tuning set**: the router prompt was last
changed against it, so its score is optimistic. `holdout` was written afterwards and is
never tuned against — that is the number to trust, and it is what the exit code gates on.

The holdout has been **rewritten twice**, and both times for the same reason. The first
one revealed that the router was answering "no" to nine document questions; the second
was retired because the eval run that exposed a broken router prompt also showed its
score, so it could no longer be an untouched set. The current one has never been run
against a tuned system. It also includes two *near-miss* negatives: a relevant document
exists, so the router must still route, but the answer is not in it and the model must
abstain. Those catch a confident hallucination, which a plain negative cannot.

Measured on the current holdout (24 cases, never tuned against):

```
retrieval recall@4   0.83   (18 cases)
routing accuracy     0.83   (24 cases)   0 false negatives, 4 false positives
```

The four routing errors are all the cheap direction — general knowledge ("how many legs
does a spider have?") was routed to the corpus. The three recall errors are all the same
ranking failure: the correct document is in the corpus and scores just below the cut
(ranks 6, 12 and 6, within 0.03–0.09 of the winner). That is the hybrid-search problem,
not a broken pipeline.

The corpus is **21 documents across six formats** — markdown, CSV, JSON, Python, plain
text and PDF — committed as `docs.sample/`. That matters more than the case count: an
earlier `recall@4 = 0.00` was a scoring artefact of a one-document corpus, and the score
overlap that ruled out `MIN_SCORE` was measured on a 65-byte file.

`routing accuracy` and `model would call` are reported separately on purpose. Routing alone
is not enough: it read **1.00** while the agent was completely broken, because the router was
right every time and the 7B ignored it. The second column asks whether the model would
actually call the tool given that decision, which is the thing that failed.

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
- **`MIN_SCORE` is a weak signal, and no threshold can fix it.** On the 21-document corpus
  the correct document scores **0.491 at rank 4** while the wrong one scores **0.517** — the
  whole top-12 sits inside **0.06**. Any floor that admits rank 1 also admits rank 4, so
  there is no separating value; the grounding prompt inside `answer_docs` is the actual
  guardrail. This is why the server returns a finished answer rather than raw chunks —
  handing chunks back would move the grounding rules into a 7B model's judgement, where
  they get lost. The fix is hybrid search, not a better constant.
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
