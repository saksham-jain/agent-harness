# agent-harness

Local-first agent + RAG stack. Ollama for inference, Qdrant for vectors, MCP for tool
serving, all wired together with Docker Compose.

## Layout

| File | Role |
| --- | --- |
| `mcp_server.py` | MCP server exposing demo tools over Streamable HTTP |
| `agent_harness_base.py` | Coding agent. Local tools + tools discovered from the MCP server |
| `rag_qdrant.py` | RAG over Qdrant (chunk, embed, retrieve, answer with citations) |
| `rag.py` | Older numpy-only RAG, kept for comparison. Needs `numpy` |

## Running

```bash
docker compose up -d                      # qdrant + mcp-server
docker compose --profile client run --rm mcp-client    # the agent
docker compose --profile rag run --rm rag              # the RAG REPL
```

Ollama runs natively on the host and the containers reach it via
`host.docker.internal:11434`. If you change the port the MCP server binds, update
`MCP_SERVER_URL` on the `mcp-client` service to match.

Environment overrides: `DOCS_DIR` (default `./docs`) and `WORKDIR` (default `.`).

## How the MCP pieces connect

```
mcp-client ──HTTP──> mcp-server:8000/mcp        (Streamable HTTP, compose network)
       │
       └──HTTP──> host.docker.internal:11434    (Ollama: chat + tool calling)
```

`agent_harness_base.py` connects to the MCP server at startup, calls `list_tools()`, and
converts each tool's `input_schema` into an OpenAI tool schema. The model then sees one
flat tool list; when it calls something, the harness routes it to the local dispatcher or
back over MCP. Tool calls are traced in the output as `[tool:local]` or `[tool:mcp]`.

The REPL is synchronous but MCP is async, so `McpBridge` holds one connection open on a
background event loop rather than reconnecting on every call. If the server is
unreachable the agent logs it and continues with local tools only.

Inspect the server from your machine with the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector) against
`http://localhost:8000/mcp`.

## Gotchas

- **The server must bind `0.0.0.0`.** `MCPServer.run()` defaults to `127.0.0.1`, which is
  unreachable from other containers. `docker-compose.yml` sets `MCP_HOST=0.0.0.0`.
- **Rebuild after changing the Dockerfile.** Compose keeps a separate image per service,
  so a stale `agent_harness-rag` can still carry an old `ENTRYPOINT` that swallows the
  `command:` override. `docker compose build --no-cache` if a service behaves as if it
  ignored your changes.
- **`mcp` is pinned to `>=2.2,<3`.** v1 and v2 have incompatible APIs: `FastMCP` /
  `ClientSession` became `MCPServer` / `Client`, and transport options moved from the
  constructor to `run()`. Most examples online are still v1.
- **Server-side errors:** raise `ToolError` when the model should read the reason and
  retry. Any other exception reaches it as a bare "Error executing tool X" with the
  traceback in the server log.
- **DNS-rebinding protection is off by default**, so container hostnames are accepted. Turn
  it on with `TransportSecuritySettings(allowed_hosts=[...])` before exposing the port
  anywhere real.

## Local (no Docker)

```bash
pip install -r requirements.txt
python mcp_server.py                                    # terminal 1
MCP_SERVER_URL=http://localhost:8000/mcp python agent_harness_base.py   # terminal 2
```
