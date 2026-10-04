# Build journey

Everything that happened to this project, in order, and what each step taught. Kept
because the mistakes are the useful part — the fixes are mostly one-liners.

## 1. A RAG over Qdrant, in Docker

Started with two scripts and a compose file: chunk documents, embed with Ollama, search
Qdrant, answer with citations. `rag_qdrant.py` skips unchanged files by content hash, so
re-indexing costs nothing.

**Learned:** incremental indexing by hash is the first thing worth building in any RAG —
without it every run re-embeds the corpus.

## 2. Added an MCP server and client

`mcp[cli]` was already sitting unused in `requirements.txt`. Added a server exposing
demo tools over Streamable HTTP, and made the agent discover and call them.

The instinct to check the SDK version first paid off: the installed package was **v2**,
where `FastMCP` → `MCPServer` and `ClientSession` → `Client`. Almost every example online
is v1. Hence the `mcp>=2.2,<3` pin.

Two traps that cost real time:

- `MCPServer.run()` defaults to `127.0.0.1`, unreachable from other containers. Needs
  `0.0.0.0`.
- `functools.wraps` on a tool decorator preserves the signature, and the SDK derives the
  JSON Schema from it. Decorating in the wrong order would silently produce empty
  schemas.

**Learned:** the sync REPL wrapping an async MCP connection needed a background event
loop — reconnecting per call would repeat the handshake every time. 5 calls then took
30ms total.

## 3. Tool-call tracing

The harness printed nothing during tool calls, because those turns have no `content`. It
was impossible to tell whether the model had called a tool or answered from memory.

**Learned:** an agent that hides its own actions is unauditable. Log before the call, not
only after.

## 4. `answer_docs`: the wrong-result problem

`search_docs` would hand raw chunks to the model and let it write the answer. That moves
the grounding rules out of a prompt and into a 7B model's judgement, where they get
lost. `answer_docs` keeps generation inside the tool, so the "say you don't know" rule
still applies.

I also assumed a score threshold would filter bad retrievals. Measured it:

| Query | top-1 |
| --- | --- |
| *How long is the hotel booked?* (in corpus) | 0.763 |
| *Capital of France?* (not in corpus) | 0.600 |

No threshold separates them — an out-of-corpus query outscores an in-corpus one. The
grounding prompt is the real guardrail, not `MIN_SCORE`.

**Learned:** measure before building the guardrail, or you ship the wrong one with
confidence.

## 5. Layered architecture

Restructured so each layer knows only the one below it:

```
agent → mcp_client → mcp_server → rag_service → qdrant
```

`rag_service` has no MCP import, so it is testable without a protocol. The harness no
longer imports `qdrant_client`.

**Learned:** the seam that pays off is the one you can test without a network. It also
made later changes cheap — RAG moved behind MCP without touching the layers above.

## 6. Agent Skills

Playbooks in `skills/<name>/SKILL.md`, following the Agent Skills open format. Only
`name` and `description` sit in the system prompt; the body loads on demand, so a long
playbook costs nothing until used. 396 characters for two skills.

Writing `stack-status` exposed a bug in it: it began with `docker compose ps`, which
cannot work from inside the client container — no Docker socket there. The model ran it,
got nothing, and invented *"Docker is not installed."*

**Learned:** a skill is only correct after you run it. Instructions fail in ways code
does not.

## 7. Small-model routing

A 7B choosing between nine tools misroutes constantly. Added a 1.5B model making one
binary call per message, then filtering the tool list on the answer.

The error directions are deliberately asymmetric: a false **negative** hides
`answer_docs` so the model cannot search (damaging), while a false **positive** only
adds a tool back (harmless). So it is permissive and never forces a call.

Tried `qwen2.5:0.5b` first: **4/8, coin-flip, and it routed arithmetic to the corpus.**
1.5b scored 6/8. Measured, not assumed.

**Learned:** the asymmetric-cost reasoning determined the design more than the model
choice did. Also: an unloaded model costs ~12s to load, more than the routing call saves
— pin it or the optimisation is a pessimisation.

## 8. Evaluation, which caught a lie

Built `evals/run.py`, split by cost: `--fast` scores retrieval and routing with no
generation; `--full` adds abstention and citation checks.

The first run falsified something I had reported: routing was **0.43**, not the 6/8 I had
claimed. The 6/8 was real arithmetic but **contaminated** — the prompt had been tuned
against those exact cases.

Two more findings from reading the output rather than the summary:

- Routing was **stochastic** — the same cases scored 0.86 then 0.71 on consecutive runs.
  Pinned `temperature=0`; two runs are now byte-identical.
- `--full` never graded the holdout, so the "honest" set was scored on routing alone.
  That is the exact flaw the two-set design exists to prevent.

**Learned:** an eval set is what tells a contaminated number from a real one. And two
defects sat underneath a full set of 1.00 scores.

## 9. Deduplication

`rag.py` and `rag_qdrant.py` each had their own copy of `read_file`, `list_files`,
`chunk`, `embed`. Extracted to `corpus.py`.

Two behavioural differences had to be preserved, and getting them wrong would have been
silent: `list_files` returns **absolute** paths for Qdrant (the path is the payload filter
key) and relative for numpy; `embed` prints progress in one and not the other.

**Learned:** extracting shared code is also an audit. It surfaced that documents were
keyed by filesystem path — harmless locally, a cross-tenant collision later.

## 10. Auth and TLS

Added bearer-token auth: the server is an OAuth 2.1 resource server, verifying tokens and
never issuing them. `auth.py` implements `TokenVerifier`, one async method, so a real
JWT or introspection verifier replaces the static table without touching anything else.

Verified: no token → 401, wrong token → 401, valid token → identity and scopes,
`Host: evil.example.com` → rejected, discovery document served.

Then TLS, in two stages. First encrypted localhost with `mkcert`. The unknown was whether
the client would trust it — `httpx2` validates through `truststore` against the **OS**
store, not certifi. Tested before building: it failed until `mkcert -install` completed,
then worked with no client change at all.

Serving it needed bypassing `run()` entirely, since it exposes no SSL options.

**Learned:** prove the trust assumption before building on it. And `Authorization` is an
HTTP header, so stdio and the in-process `Client(mcp)` never see auth — a test using
`Client(mcp)` proves nothing about authentication. Full design and the measured handshake:
[AUTH-TLS.md](AUTH-TLS.md).

## Mistakes worth listing

| Mistake | Cost |
|---|---|
| `git add -A` committed a 500-file venv | History rewritten, `.gitignore` fixed |
| Told the user to `python3 -m venv` when their venv was one level up | Agent looked dead; `ModuleNotFoundError` prints nothing |
| Dropped `mcp_tools` when adding `answer_docs` | Model couldn't call MCP tools at all |
| Advised "prefer answering directly" for routing | Overcorrected — `answer_docs` was never called |
| 30s MCP timeout vs a 37s tool | Every document call timed out |
| Two eval defects under an all-1.00 run | The harness was measuring the wrong thing |
| Duplicated layer table in the README | Caught only by auditing before a commit |

## What is still open

| | |
| --- | --- |
| **Port 8000 published on `0.0.0.0`** | Live now. TLS does not close it — a published port is a way *around* TLS |
| Multi-tenancy | Every tenant shares one collection |
| Document identity | Still keyed by filesystem path |
| Stale index points | Removed documents are never purged |
| Tests | Zero, though evals exist |
| Corpus | One 65-byte file, so retrieval numbers are an anecdote |
| Prompt injection | Document text enters prompts unescaped |
| Real auth | Static table: no expiry, revocation needs a restart |
| Public hostname | Needed before anyone else can connect |

## Reading

| | |
| --- | --- |
| *Building Effective Agents* | Workflows vs agents. The most useful thing here |
| The MCP specification | Would have avoided the v1/v2 API split |
| [agentskills.io](https://agentskills.io) | The format is implemented; the spec covers the rest |
| LangGraph / Vercel AI SDK | Reference implementations of streaming, retries, tool loops |
