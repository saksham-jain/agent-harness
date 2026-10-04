# Possible enhancements

Where this project could go, and what each step teaches. Ordered by value, not
difficulty.

## Where it stands

| Concern | Status |
| --- | --- |
| Architecture & layer separation | done |
| Protocol correctness (MCP v2) | done |
| Incremental indexing | done |
| Grounded answers with citations | done |
| Small-model routing | done |
| Auth (bearer tokens) | done, static token table |
| **TLS** | **not yet** — plaintext, localhost only |
| **A hostname and a publicly-trusted certificate** | **not yet** — needed before anyone else can connect |
| **Evaluation** | **nothing** |
| **Tests** | **none** |
| **Tracing / spans** | logs only |
| **Streaming** | 37s of silence per document answer |
| **Security** | document text goes into prompts unescaped; `bash` guarded by one prompt |
| Context management | history truncate cap |
| Concurrency | serial |

## 1. Evaluation — do this first

The gap that makes everything else safer. Twice now a measurement changed the design:
the score overlap that ruled out `MIN_SCORE`, and the routing score that ruled out
`qwen2.5:0.5b`. Both were done ad hoc. That is the pattern of someone who needs evals.

A labelled question set plus a scoring script:

```python
CASES = [
    ("how long is the hotel booked?", "3 nights",  must_cite=True),
    ("what is the capital of France?", "don't know", must_abstain=True),
    ("who wrote dune?", "Herbert", must_not_use_docs=True),
]
```

`must_not_use_docs` is the important field — it scores *routing*, not just answer
quality, which is the bug class that keeps recurring here.

With that in place, each later change becomes measurable instead of a guess.

## 2. Tests

MCP v2 connects a client to the server object in-process: `Client(mcp)`. No port, no
subprocess, no fixtures.

```python
from mcp import Client
from mcp_server import mcp

async def test_index_status() -> None:
    async with Client(mcp) as c:
        result = await c.call_tool("index_status", {})
        assert result.is_error is False
```

`rag_service` is already free of MCP imports, so it tests the same way.

## 3. Tracing

OpenTelemetry is **on by default** in the MCP SDK — every request already gets a
server span. Exporting one is a few lines of config. Latency is currently invisible
except in log text, and the interesting number (37s) is invisible entirely.

## 4. Streaming and progress

The largest thing a user feels. Two protocol features at once:

- Ollama `stream: true`, so tokens arrive as generated
- MCP `ctx.report_progress()`, so the client can show progress during a tool call

## 5. Security

- Document text is embedded into prompts verbatim. A PDF containing instructions is a
  live prompt-injection path. Needs delimiting and treating retrieved text as untrusted.
- `bash` asks `run? [y/N]` — one prompt for every command. Wants allow/deny per tool,
  and a read-only mode for the file tools.
## 5a. ~~A hostname and a publicly-trusted certificate~~ — done

**Shipped.** A Tailscale Funnel on `<machine>.<tailnet>.ts.net`, holding a Let's Encrypt
certificate. The origin is plain HTTP published as `127.0.0.1:8000`, so the LAN has no
route in and the old published port is gone.

Chosen over the alternatives for two reasons worth keeping on file:

- **Free.** Funnel is available on every Tailscale plan including Personal at $0. A
  Cloudflare *named* tunnel needs a domain (~$10/yr); its *quick* tunnel is free but hands
  out a random hostname on every restart, which is unusable as a token resource because
  `MCP_RESOURCE_URL` names the resource a token is issued for.
- **One URL for everything.** Because the certificate is publicly trusted, local and
  remote clients dial the same address. That retired the `mkcert` stage entirely — no
  second port, no CA to install, and the `Caddyfile` was deleted with it.

The `mkcert` path still works if you set `MCP_TLS_CERT` and `MCP_TLS_KEY`; it is just no
longer the recommended configuration.

Two operational notes, both learned the hard way:

- **`sudo brew services restart tailscale` is the reboot test.** The Funnel config lives in
  daemon state, not in a process, so it survives — but confirm it rather than assume.
- **The URL depends on this Mac's name.** `sakshams-macbook-air` comes from LocalHostName.
  Renaming the Mac changes the hostname, breaking every client and invalidating tokens
  issued for the old resource URL.

Remaining on the auth side, and these are the real limits:

- **The token table has no expiry and no revocation** short of editing the file and
  restarting. Possession is identity.
- **`refresh_index` is a mutation** and every token currently holds `docs:read`, which is
  all it checks. One token leaked is one leaked corpus *and* one writable index.
- **Per-tool scopes** are still on the roadmap: split read from write so `refresh_index`
  needs something a read-only client would not hold.

Whichever route is ever used, `MCP_RESOURCE_URL` must be the exact https URL clients connect to — it
names which resource a token is issued for, and where discovery lives. Getting it wrong
shows up as a token that verifies locally and is rejected remotely.

Design and staging: [AUTH-TLS.md](AUTH-TLS.md).

## 6. Hybrid search and reranking

The measured failure: relevant top-1 scores 0.59 against 0.60–0.62 for irrelevant, so
no threshold separates them. That is what pure dense retrieval does.

BM25 (sparse) + dense + reciprocal rank fusion fixes the overlap directly, and Qdrant
supports sparse vectors natively, so it needs no new service. Reranking with a
cross-encoder is the usual follow-up.

This is now the main retrieval limitation. Two structural fixes have landed — chunks follow
document structure, and `retrieve()` over-fetches then dedupes across documents so one long
file cannot fill all four result slots — but neither addresses *scoring*, and scoring is
where the measured failure is. `MAX_CHUNKS_PER_DOC=3` against `k=4` also still lets one
document take three slots; raising `TOP_K` is the knob, the cap is the trade.

None of it is measurable yet, because `docs/` holds one 65-byte file. `recall@4 = 0.00` on
the holdout is a scoring artefact, not a retrieval failure. The 20–30 document sample corpus
in section 1 is the prerequisite for measuring any of this.

## 7. Subagents and context isolation

Long tasks accumulate tool output in one window. Separate context per task is how the
frameworks handle it, and it is the natural next structural step after this.

## Reading

| | |
| --- | --- |
| Anthropic, *Building Effective Agents* | The workflows-vs-agents distinction. Short, and the most useful thing here |
| The MCP specification | Reading it top to bottom would have avoided the v1/v2 API split |
| [agentskills.io](https://agentskills.io) | The format is implemented; the spec explains the skipped parts |
| LangGraph / Vercel AI SDK source | Reference implementations of streaming, retries and tool loops |

## Honest note

The architecture is stronger than most agent repos, and the reasoning in the commit
history is the differentiator. What is missing is its counterpart: evidence that the
system works. That is item 1, and items 2–7 are polish until it exists.
