#!/usr/bin/env python3
"""MCP Client: the harness's connection to MCP servers.

Kept separate from the agent so the harness only has to deal with a list of tool
schemas and a `call`. One connection is held open on a background event loop,
because the agent's REPL is synchronous and reconnecting per call would repeat the
handshake every time.
"""
import asyncio
import json
import os
import threading


def _token_from_file():
    """First token in `tokens.json`, or None.

    Reads the same table the server authenticates against, which removes the need to paste
    a 44-character secret into `.env` before every run. Two failure modes worth knowing,
    because both were live during development:

    - a truncated token in `.env` fails auth with no hint about which half is missing
    - picking a token by position authenticates as whichever subject happens to be first,
      which silently changes identity when the file is reordered

    Position is acceptable *here* only because every current token is that user's own; the
    server's `whoami` reports the subject, so a mismatch is visible immediately.
    """
    path = os.getenv("MCP_TOKENS_FILE", "tokens.json")
    try:
        with open(path) as f:
            tokens = json.load(f)
    except (OSError, ValueError):
        return None
    for key in tokens:
        return key
    return None


def transport_for(url):
    """A Streamable HTTP transport, carrying a bearer token if one is configured.

    `Client` takes no headers argument -- headers go on the HTTP client the transport
    wraps, which is why this builds its own instead of passing the URL straight in.
    Returns the URL unchanged when no token is set, so an unauthenticated server is
    left alone.
    """
    token = os.getenv("MCP_BEARER_TOKEN", "").strip() or _token_from_file()
    if not token:
        return url

    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    http_client = httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=float(os.getenv("MCP_TIMEOUT", "300")),
    )
    return streamable_http_client(url, http_client=http_client)


def to_openai_tool(tool):
    """MCP tool definition -> OpenAI tool schema.

    `input_schema` is already a JSON Schema object, so only the envelope differs.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": tool.input_schema,
        },
    }


def result_to_text(result):
    """Flatten an MCP CallToolResult into the string the model reads back."""
    parts = [b.text for b in result.content if getattr(b, "text", None)]
    text = "\n".join(parts) if parts else json.dumps(result.structured_content)
    return f"[tool error] {text}" if result.is_error else text


class McpBridge:
    """One MCP server, connected for the life of the process.

    `timeout` has to cover a whole tool call. `answer_docs` embeds the query, searches
    and then runs a full generation, which measured ~37s on qwen2.5:7b -- a 30s budget
    cuts it off mid-answer.
    """

    def __init__(self, url, timeout=None):
        self.url = url
        self.timeout = float(timeout if timeout is not None else os.getenv("MCP_TIMEOUT", "300"))
        self.tools = []  # OpenAI-shaped tool schemas from the server
        self._ready = threading.Event()
        self._error = None
        self._loop = None
        self._client = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError(f"timed out after {timeout}s connecting to MCP server at {url}")
        if self._error:
            raise RuntimeError(f"could not connect to MCP server at {url}: {self._error}")

    def _serve(self):
        from mcp import Client

        async def main():
            self._loop = asyncio.get_running_loop()
            async with Client(transport_for(self.url), read_timeout_seconds=self.timeout) as client:
                self._client = client
                self.tools = [to_openai_tool(t) for t in (await client.list_tools()).tools]
                self._ready.set()
                while True:  # park; the agent drives us via _submit
                    await asyncio.sleep(3600)

        try:
            asyncio.run(main())
        except Exception as e:  # surface connection failures to __init__
            self._error = e
            self._ready.set()

    def _submit(self, coro):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("MCP connection is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout + 5)

    def call(self, name, args):
        """Call a tool on the already-connected server. Blocking, from the REPL thread."""
        return self._submit(self._client.call_tool(name, args))


def connect(url, local_names=frozenset()):
    """Return (bridge, tools).

    A server that is down is not fatal: the caller keeps its local tools and says
    so. Tools whose name collides with a local one are dropped rather than
    shadowing it, because dispatch would be ambiguous.
    """
    try:
        bridge = McpBridge(url)
    except Exception as e:
        print(f"MCP unavailable ({e}); continuing with local tools only.")
        return None, []

    tools = []
    for t in bridge.tools:
        name = t["function"]["name"]
        if name in local_names:
            print(f"  skipped MCP tool {name!r}: name collides with a local tool")
        else:
            tools.append(t)
    return bridge, tools
