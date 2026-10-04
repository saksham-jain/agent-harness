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
- **Port 8000 is published on `0.0.0.0`** — the most exposed thing here, and it is live
  now. Anything that can reach the machine on the LAN can read the corpus. Unpublish the
  port or bind it to `127.0.0.1`. TLS does not fix this, because a published port is a
  way *around* TLS.
- The MCP server binds `0.0.0.0` with DNS-rebinding protection off.

## 5a. A hostname and a publicly-trusted certificate

The blocker for letting anyone but you connect. There are exactly three places a
certificate can come from, and nothing else is possible:

| Source | What it costs |
| --- | --- |
| Public CA (Caddy, Let's Encrypt) | Own a domain, prove control over DNS, forward ports, renew every 90 days |
| A tunnel — Tailscale Funnel, Cloudflare | They hold the certificate, you point at their hostname. Usually the best value |
| Distribute your own CA | What `mkcert` does locally, but every client must install it first |

Two things worth knowing before choosing:

- **A local `mkcert` certificate is not a substitute.** It exists only on machines where
  the CA is installed. Another client gets an untrusted certificate, which they will
  refuse or, worse, click past.
- **A tunnel usually makes Caddy unnecessary.** TLS is terminated before traffic reaches
  the container, so the `Caddyfile` in this repo is only needed on the public-CA path.

Whichever route, `MCP_RESOURCE_URL` must be the exact https URL clients connect to — it
names which resource a token is issued for, and where discovery lives. Getting it wrong
shows up as a token that verifies locally and is rejected remotely.

Design and staging: [AUTH-TLS.md](AUTH-TLS.md).

## 6. Hybrid search and reranking

The measured failure: relevant top-1 scores 0.59 against 0.60–0.62 for irrelevant, so
no threshold separates them. That is what pure dense retrieval does.

BM25 (sparse) + dense + reciprocal rank fusion fixes the overlap directly, and Qdrant
supports sparse vectors natively, so it needs no new service. Reranking with a
cross-encoder is the usual follow-up.

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
