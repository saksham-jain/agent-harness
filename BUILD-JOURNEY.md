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

*(Measured on one document, and §11 repeats it on 21 — the conclusion held, and got
sharper: the whole top-12 sits inside 0.06.)*

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
adds a tool back (cheap). So the router is permissive.

This reasoning was correct and still incomplete. Permissive was meant to mean "harmless",
and for a while it was — but the 7B was offered `answer_docs` and declined, answering from
memory instead: **0 of 3** document questions, with the router right every time. The cost
of a false positive stopped being harmless the moment the model ignored the decision anyway.
`run_turn` now makes the router's call binding when the model returns no tool call, which is
the one place the model has no veto. `README.md` has the measurements, and the eval scores
`model would call` separately from `routing accuracy` because routing accuracy alone reads
`1.00` on a completely broken path.

Tried `qwen2.5:0.5b` first: **4/8, coin-flip, and it routed arithmetic to the corpus.**
1.5b scored 6/8. Measured, not assumed.

*(§11 replaces this prompt. The prose version below turned out to have nine false
negatives once the corpus was big enough to tell — the 6/8 was measured on eight
questions, which is not enough to see them.)*

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

Then the hostname, via a Tailscale Funnel. The mkcert stage turned out to be unnecessary
the moment that landed: a publicly-trusted certificate means one URL for local and remote
clients, so there is no second port and no CA to install. The tunnel cost nothing —
Funnel is on every Tailscale plan including the free Personal one — and the `Caddyfile`
was deleted rather than maintained for a path nobody would take.

The saving was noticing the free option first. Cloudflare's named tunnel needs a domain
and its quick tunnel changes hostname on every restart, which is unusable when
`MCP_RESOURCE_URL` names the resource a token is issued for.

**Learned:** prove the trust assumption before building on it. And `Authorization` is an
HTTP header, so stdio and the in-process `Client(mcp)` never see auth — a test using
`Client(mcp)` proves nothing about authentication. Full design and the measured handshake:
[AUTH-TLS.md](AUTH-TLS.md).

## 11. A corpus, and what it exposed

Everything above was measured against a corpus of **one file, 65 bytes**. That is enough
to test plumbing and not enough to measure retrieval, and it had quietly become the
reason several numbers in this repository were anecdotes.

So: **21 documents, 35 chunks**, across six formats — markdown, CSV, JSON, Python, plain
text and PDF — with the three PDFs hand-written and then checked through `pypdf`, because
the last hand-written PDF in this project was rejected by `pypdf` and proved nothing. The
documents were written to have checkable facts in them, so a case can assert on a value
rather than on a topic.

Three things stopped being true the moment it existed.

### The eval was measuring nothing

`recall@4` had read **0.00** on every case since the payload key was renamed from `file`
to `doc_id`. The runner asked for `payload["file"]`, got `""` for every point, matched no
source, and reported a perfect zero. It looked exactly like a retrieval result.

Fixed, it reads **1.00** on both sets. Retrieval was working the whole time; the number
was a broken matcher wearing a measurement's clothes.

The lesson is not "add tests". It is that **a rename is not finished when the writer has
updated it** — every other reader of that payload has to be found, and the one that was
left behind produced a plausible-looking number for weeks.

### The score overlap was real, and worse than measured

The `MIN_SCORE` problem had been documented from a one-document corpus. On 21 documents
it is sharper:

```
the CORRECT document, insurance.md:  0.491, rank 4 of 35
the winner, bike-maintenance.md:     0.517
gap:                                  0.025

top-12: 0.517 0.502 0.497 0.491 0.489 0.482 0.480 0.478 0.473 0.466 0.463 0.457
```

The whole field is inside **0.06** of each other. Any floor that admits rank 1 also
admits rank 4, so **no threshold separates them** — not a badly-chosen one, none. The
correct answer lost by 0.025 on a question where the word "excess" appears literally in
the right document, because a dense vector scores a document about bicycles *above* a
document about car insurance.

That is the case for hybrid search, and it is now a measurement rather than a preference.

### The router was answering "no" to almost everything

With a real corpus and a real case set, the router's score was **0.54** on the holdout
with **nine false negatives** — the damaging kind, where `answer_docs` is hidden and the
model cannot search at all. It answered "no" to *"what did the doctor prescribe?"* and to
*"how many months notice do I have to leave the flat"*.

The cause is the one already recorded for the 7B in §7, one layer down: **the prompt
described the decision instead of demonstrating it.** It listed categories — "first person
or possessive", "their own plans" — and a question without a pronoun read as general
knowledge. A 7B with the same prompt was worse still: **0.38, twenty false negatives**.

Four fixes were tried and measured:

| Router | Result | False negatives |
| --- | --- | --- |
| Prose prompt, 1.5B | 0.69 | 9 |
| Prose prompt + corpus manifest | 0.44 | 18 |
| Prose prompt, **7B** | 0.38 | 20 |
| One-line prompt, 1.5B | 0.19 | 26 |
| **Few-shot, 1.5B** | **0.84** | **0** |

The manifest made it *worse*. Grounding a 7B is worth doing; grounding a 1.5B with 21
filenames is not.

What actually worked was showing six decisions instead of describing one, and the
**order** turned out to be load-bearing. Moving the "yes" examples to the front dropped
it to 0.53 with six false negatives; eight examples were worse than six. The last example
sets the bias, so the prompt ends on "yes" — which is the right bias here anyway, because
a false yes only adds an option the model can ignore while a false no means the documents
can never be searched.

The examples are deliberately generic and appear in **no eval case**, so the tuning set
stays a clean measure of the prompt.

### The trap: making the prompt clearer made it five times worse

The shipped prompt had one more sentence than the one I measured. It read:

> Decide if the message needs the user's own documents. **Answer no only for arithmetic,
> general knowledge, coding questions, or a named file in the working directory. If you
> are unsure, answer yes.**

It is a better prompt. It is also a **disaster**, and the eval said so:

```
the text I measured:   0.84   0 false negatives
the text I shipped:    0.16   15 false negatives
```

Same model, same temperature, same examples, one sentence different. "Answer no only
for…" primes the model to answer *no*, and the demonstrations after it cannot overcome
the instruction. Adding *"If you are unsure, answer yes"* to either end cost 0.10.

A 1.5B does not read a rule and then apply it to the examples. It pattern-matches, and
**any instruction in the prompt competes with the pattern**. The shipped string is now six
worked examples and one neutral lead sentence, and nothing else.

**Learned:** a small model does not need a better *description* of the decision, it needs
to see the decision made — and the least helpful thing you can do is explain it. And the
holdout was rewritten afterwards — the old one had earned its keep by revealing the bug,
but a set you have inspected cannot report an honest score.

### Where it landed

On a holdout written afterwards and never tuned against, 24 cases:

```
retrieval recall@4   0.83   (18 cases)
routing accuracy     0.83   (24 cases)   0 false negatives, 4 false positives
```

All four routing errors are the cheap direction — general knowledge routed to the corpus.
All three recall errors are one problem: the correct document is in the corpus and scores
just below the cut (ranks 6, 12 and 6; within 0.03–0.09 of the winner). That is the
hybrid-search limitation, not a broken pipeline, and it is the next thing to fix.

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
| Assumed a correct router meant the model would use the tool | The 7B was offered `answer_docs` and answered *"no tool"* — 0 of 3 document questions. Routing accuracy read **1.00** on a completely broken path, so the only metric watching this was blind to it |
| Wrote a `$(python3 ...)` token into `.env` expecting substitution | Compose does not run it. The container received a 119-character literal string as its bearer token |
| Pasted a token by hand and truncated it | 24 of 44 characters. Every request `401`, and `connect()` reported "unavailable" — which reads like a network problem |
| Put `MCP_SERVER_URL` as a literal in `docker-compose.yml` | `.env` cannot override a hardcoded value, so the client kept dialling an internal address that now answers `421` |
| Verified the web client with a Python client, then reported it working | The browser sends an `Origin` header and Python does not, so the test could not reproduce the failure. It shipped a `403` that named the hostname while the page blamed the token, and sent both of us after the wrong thing |
| Allowlisted `allowed_hosts` and assumed that covered browsers | It does not. The SDK validates `Origin` separately, and only a browser ever sends one — so the check was untested by every client used until then |
| Called a false-positive route "harmless" and left it that way | It stopped being harmless when the 7B ignored the router anyway. The asymmetry argument was right about the router and wrong about the model |
| Shipped a system prompt advertising a tool that did not exist | `- add : arithmetic` was in the prompt for the whole project with no `add` anywhere, so arithmetic silently fell through to retrieval. It also omitted three tools that *did* exist |
| Treated "empty directory" and "missing directory" as the same case | `index()` returned before `_prune()` ran, so deleting the last document left it searchable forever and the message read like a harmless no-op. Found only by testing the last file, not the middle one |
| Wrote a test PDF by hand and trusted it was valid | `pypdf` rejected it, so the file was never indexed and the "already indexed, now corrupt" case proved nothing. Needed `chmod 000` on a real file to make a genuine read failure |
| Diagnosed a "duplicate point" as a `_stored_hash` bug | It read `None` only because I had scrolled with `with_payload=['doc_id','source']`, so Qdrant returned nothing for `hash`. The real cause was a stale `index` image |
| Ran `docker compose build` and assumed every service was rebuilt | Per-service images. `agent_harness-index` stayed two days old and kept writing pre-`doc_id` payloads, so host and container runs pruned each other on every pass |
| Treated `#` headings as chunk boundaries everywhere | Produced 32 chunks of ~76 chars from a 2.4 KB file. Only `# ` is a document boundary; `##` sections pack together, since the heading text stays in the chunk |
| Published two host ports that were identical | Read the config, not the traffic. Both were TLS; the "http on 8000, https on 8443" comment described a distinction that did not exist |
| Basename fallback when handling legacy points | Kept a duplicate of a live document. The reasoning was wrong: a run either writes a modern point or finds one already there, so a legacy key is always redundant |
| `tailscale funnel --bg 443 <url>` | The port is a flag. It failed with `invalid argument format` while looking plausible |
| Raw `curl` for `tools/list` returned `Missing session ID` | Streamable HTTP needs an `initialize` handshake first. The tunnel was fine; the test was wrong |
| Renamed a payload key and left one reader behind | `payload["file"]` became `payload["doc_id"]`, and the eval runner still read the old name. It matched no source, so `recall@4` read **0.00** on every case — a number that looked like a retrieval result and was a broken matcher |
| Measured retrieval on a one-document corpus and called the numbers real | `recall@4 = 0.00` was a scoring artefact, and the top-1 score overlap that ruled out `MIN_SCORE` was measured on a 65-byte file. Both needed 21 documents to become facts |
| Described the decision to the router instead of demonstrating it | A prose prompt scored 0.69 with **nine** false negatives. The 1.5B answered "no" to "what did the doctor prescribe?" because the prompt never showed what a yes looks like |
| Put the corpus manifest in the router prompt | It got *worse* — 0.44. Twenty-one filenames in the preamble drown a 1.5B's attention. Grounding helps a 7B; it does not scale down |
| Reordered the router's examples and kept the score | Moving the "yes" examples to the front dropped accuracy to 0.53 with six false negatives. The last example sets the bias, and the prompt is only as good as its ordering |
| Clarified the router prompt and made it five times worse | One extra sentence — "Answer no only for arithmetic, general knowledge…" — took it from **0.84 to 0.16** with fifteen false negatives. It is a better-written prompt and a worse one, and the difference was invisible until the eval ran |
| Read the holdout's score, then used it to pick the fix | The holdout had to be retired and rewritten. A set you have looked at cannot report an honest number, and the second rewrite was needed even though the fix was chosen on the tuning set alone |
| Ran the "fast" eval in the foreground and waited 30 minutes | `--fast` was making one 7B generation per case for the `model would call` column. `--no-model` now runs the same checks in ~20s, and the full column is still one flag away |

## What is still open

| | |
| --- | --- |
| Multi-tenancy | Every tenant shares one collection |
| Document identity | **done** — `doc_id` is now relative to the docs directory |
| Index pruning | **done** — removed documents are pruned and reported |
| Public hostname + real certificate | **done** — Tailscale Funnel, Let's Encrypt |
| LAN exposure | **done** — `127.0.0.1:8000:8000`; `192.168.1.3:8000` refuses |
| Tests | Zero, though evals exist |
| Corpus | **done** — 21 documents, 6 formats, 35 chunks. Now a corpus worth measuring against |
| Retrieval at scale | `top-4` dedupes across documents and chunks follow structure. `recall@4` is 1.00, but **ranking** is not: the correct document loses by 0.025 on a 21-document corpus, so hybrid search is the open item |
| Prompt injection | Document text enters prompts unescaped |
| Real auth | Static table: no expiry, revocation needs a restart. Per-tool scopes now split read from write |
| Ran `DELETE` against the live Qdrant to prove it was exposed | It was — and it **deleted the index**. Rebuilt from the originals on disk in one command, which is the filesystem-as-source-of-truth design paying off, but the indexer had already been proven not to prune a corpus it cannot see |
| Cloned the repo only at the very end | Every earlier check ran against the working tree, which is precisely where the `_comment` crash did not exist. The quickstart was the first test of the repo as a stranger receives it |
| Injection via uploaded files | Not built. Whatever lands in `docs/` enters prompts unescaped, and the grounding prompt is the only guardrail |
| Tool choice | The 7B still declines `answer_docs` on its own — the router's decision is now enforced rather than advisory, which works but is a patch over a prompt that misstates what the model knows |
| Per-tool scopes | **done** — `docs:write` gates `refresh_index`. It was a live gap before any upload existed: a read-only token could rebuild the index |

### The exposure that outlived TLS

The port was published on `0.0.0.0` while TLS was being set up, and adding TLS did not
close it — a published port is a way *around* TLS. Worse, the instinct was to treat
encryption and exposure as one problem, so the port stayed open across the whole TLS
stage and only got noticed when hostname work began.

What fixed it was not a security argument but a routing one. Once the tunnel existed, the
public path went through Tailscale on the host, which can reach `127.0.0.1`. The LAN
listener stopped having a purpose, so binding it to loopback was the obvious change
rather than a defensive one. Two problems, one move.

The habit worth keeping: **a port is exposure regardless of what else is protecting
it.** Encryption is not a reason to leave a listener open.

## Reading

| | |
| --- | --- |
| *Building Effective Agents* | Workflows vs agents. The most useful thing here |
| The MCP specification | Would have avoided the v1/v2 API split |
| [agentskills.io](https://agentskills.io) | The format is implemented; the spec covers the rest |
| LangGraph / Vercel AI SDK | Reference implementations of streaming, retries, tool loops |
