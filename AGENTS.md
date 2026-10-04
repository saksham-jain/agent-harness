# AGENTS.md

Guidance for agents and contributors working in this repository. Repo-wide rules only —
put anything area-specific in a nested `AGENTS.md` next to the code it covers.

## Layout

Layers, each depending only on the one below:

```
agent_harness_base.py   the agent loop. must not import qdrant_client
mcp_client.py           tool discovery, schema conversion, dispatch, bearer token
mcp_server.py           protocol only. tools delegate to the RAG service
auth.py                 TokenVerifier. sits between the server and the network
rag_service.py          embedding and retrieval over Qdrant
corpus.py               file discovery, chunking, batched embedding
skills.py               SKILL.md discovery
evals/                  labelled cases and the scoring runner
```

## Rules

### Update the docs in the same change

If a change alters behaviour, architecture, configuration, or status, update the matching
markdown **in the same commit**. Nothing ships with docs describing the previous version.

| Change | Update |
| --- | --- |
| any behaviour, architecture, or status change | `README.md` |
| new idea worth doing later | `ENHANCEMENTS.md` |
| anything auth or TLS | `AUTH-TLS.md` |
| a mistake worth recording | `BUILD-JOURNEY.md` |

**Do not describe something as working unless it was verified in this session.** Mark
planned work as planned. A `Caddyfile` sitting in the repo reads as a working deployment
and is not one — say so next to it. Status claims that are wrong cost more than absent
ones.

### Keep the README matching reality

It drifts silently. Before committing a README change, confirm:

- every file named in a table exists (`ls`)
- every command shown is current (profiles get renamed, files get deleted)
- no stale references to removed files, renamed profiles, or renamed identifiers
- diagrams align — check box edges programmatically, not by eye

A duplicated table or a stale tool name has survived review more than once here.

### Layer discipline

- `mcp_server.py` must not import `openai` or `qdrant_client`. It calls `rag_service`.
- `rag_service.py` must not import `mcp`. That is what makes it testable without a protocol.
- `agent_harness_base.py` must not know a vector store exists. Retrieval arrives over MCP.

### Decorators

Apply `@logged` **beneath** `@mcp.tool()`. The SDK derives the JSON Schema from type
hints, so wrapping on the wrong side hands it `(*args, **kwargs)` and produces empty
schemas. `functools.wraps` makes the correct order safe.

### Errors the model should read

Raise `ToolError` when the reason should reach the model so it can retry. Any other
exception surfaces as a bare `Error executing tool X` with the traceback in the log.

## Gotchas

- **`../.venv/bin/python3`, not bare `python3`.** Dependencies live in the venv one level
  up. Without it the agent dies at `from openai import OpenAI` and prints nothing at all.
- **Rebuild after changing the Dockerfile.** Compose keeps a *separate image per service*,
  so a stale one can carry an old `ENTRYPOINT` that silently swallows a `command:`
  override. This caused a service to run with the wrong argv.
- **`tokens.json` and `.env` are gitignored.** Do not commit them. `.certs/` holds a
  private key.
- **`MCP_RESOURCE_URL` must equal the URL clients dial**, exactly. It names the resource a
  token is issued for; a mismatch verifies locally and is rejected remotely.
- **`MCP_ALLOWED_HOSTS` must include the port clients use**, or DNS-rebinding protection
  rejects every request.
- **`Answer` latency is ~17–37s**, dominated by a CPU-bound 7B generation. Budget per tool
  call, not per turn. `MCP_TIMEOUT` must cover a whole call.
- **Document identity is `doc_id`**, the path relative to the docs directory — never the
  absolute path. Absolute paths as keys are what made host and container indexing produce
  two sets of points.
- **Indexing prunes deleted documents** and reports what it removed. One collection holds
  one corpus, so indexing a *different* directory than last time removes the previous
  directory's documents. That is intentional, not a bug.
- **Port 8000 is published as `127.0.0.1` only.** The public path is the Tailscale Funnel,
  which runs on the host and dials loopback. Publishing on `0.0.0.0` would be a way
  *around* TLS, so it is not done.
- **`MCP_ALLOWED_HOSTS` lists the public hostname bare — no port.** On 443 the `Host`
  header omits the port. Adding `:443` does not match. The Funnel forwards the original
  `Host`, so that one entry covers tunnel traffic; direct loopback is rejected with 421.
- **The Funnel URL is stable, and depends on the Mac's name.** `sakshams-macbook-air` comes
  from LocalHostName and `tailf61e07` is fixed at tailnet creation. Renaming the Mac
  changes the URL, which breaks every client *and* invalidates tokens issued for the old
  resource URL.
- **Enabling the Funnel needs one admin click.** `tailscale funnel` prints a
  `login.tailscale.com/f/funnel?node=…` URL and stores nothing until it is visited. The
  port is a flag, not a positional: `tailscale funnel --bg --https=443 http://127.0.0.1:8000`.
- **On macOS use the Homebrew CLI, not the App Store app.** Funnel needs the open-source
  variant; `brew services start tailscale` also gives `RunAtLoad` + `KeepAlive`.

## Commands

```bash
docker compose up -d                                  # qdrant + mcp-server
docker compose --profile index run --rm index         # index docs
docker compose --profile client run --rm mcp-client   # the agent

tailscale funnel --bg --https=443 http://127.0.0.1:8000   # (re)publish the hostname
tailscale funnel status                                  # is it up?
tailscale funnel --https=443 off                         # take it down

../.venv/bin/python3 evals/run.py --fast              # retrieval + routing, seconds
../.venv/bin/python3 evals/run.py --full              # also grades answers, slow
docker compose logs -f mcp-server --no-log-prefix 2>&1 | grep mcp.tools
```

## Verify before claiming

Test the failure being fixed, not just the happy path. Three examples from this repo:

- a timeout was found by measuring a real call, not by reading the config
- routing turned out to be stochastic — identical inputs scored 0.86 then 0.71, so
  `temperature=0`
- two eval defects sat underneath a full set of `1.00` scores

Prefer a command that prints the result to a claim that it works.

## Evaluating a change

`evals/run.py --fast` is cheap enough to run before committing anything that touches
retrieval, routing, or the tool list.

`cases.json` has two sets. `cases` is a **tuning set** — if a change is tuned against it,
its score is optimistic and no longer means anything. `holdout` is written afterwards and
never tuned against; that is the honest number, and it is what the exit code gates on.
