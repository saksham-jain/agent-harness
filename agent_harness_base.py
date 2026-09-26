#!/usr/bin/env python3
"""Minimal agent harness for any OpenAI-compatible endpoint (default: local Ollama).

Local tools (bash, read_file, write_file) are defined in-process. Tools from an MCP
server are discovered at startup and exposed to the model alongside them, so the
agent can call either without knowing the difference.
"""
import asyncio
import json
import os
import re
import subprocess
import threading

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Static document search, straight against Qdrant. Deliberately not an MCP
# server: this is the agent's own capability, not a tool it discovers.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "rag_" + re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL)  # must match rag_qdrant.py
SEARCH_PREFIX = "search_query: "  # nomic-embed-text requires the task prefix
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.5"))
# Measured on this 6-chunk corpus, relevant and irrelevant top-1 scores overlap
# (0.74/0.76 relevant vs 0.52-0.62 irrelevant), so no threshold here separates them.
# The grounding prompt in answer_docs() is the real guardrail; this only helps once
# the corpus is large and varied enough for similarity to spread out. Tune per model.
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

SYSTEM = """You are a helpful coding agent in the user's current directory.

Route each question to the right tool:

- answer_docs : ALWAYS use for questions about the user's own documents - their
                notes, records, resumes, contracts, saved plans. Never answer
                those from memory; they are not something you were trained on.
- read_file   : files in the working directory
- add         : arithmetic
- no tool     : general knowledge you already have

Examples:
  "what did the lease say about termination?"  -> answer_docs
  "what did the interview notes say?"          -> answer_docs
  "what is in config.json?"                   -> read_file
  "10 + 2"                                    -> add
  "who wrote Dune?"                           -> no tool, just answer"""
MAX_STEPS = 10
_qdrant = None


def tool(name, desc, props, optional=None):
    """OpenAI tool schema. `props` are required string args; `optional` maps
    name -> JSON schema for optional args."""
    properties = {k: {"type": "string"} for k in props}
    for k, schema in (optional or {}).items():
        properties[k] = schema
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(props),
            },
        },
    }


LOCAL_TOOLS = [
    tool("bash", "Run a shell command and return stdout/stderr.", ["command"]),
    tool("read_file", "Read a text file.", ["path"]),
    tool("write_file", "Write text to a file, overwriting it.", ["path", "content"]),
]
LOCAL_NAMES = {t["function"]["name"] for t in LOCAL_TOOLS}

# Only added to the model's tool list when the corpus is actually there, so it is never
# handed a tool that cannot work.
DOCS_TOOL = tool(
    "answer_docs",
    "Answer a question about the user's OWN indexed documents: their notes, records, "
    "resumes, contracts. Use ONLY when the question asks about that specific private "
    "corpus. Do NOT use for arithmetic, general knowledge, or anything already on disk "
    "— use add, or just answer directly. Returns a cited answer, or says plainly when "
    "the documents do not cover the question.",
    ["query"],
)

# Everything the harness dispatches itself. Anything outside this set goes to MCP.
LOCAL_NAMES |= {DOCS_TOOL["function"]["name"]}


class McpBridge:
    """Keeps one MCP Client alive on a background event loop.

    The REPL is synchronous but MCP is async, and reconnecting per tool call would
    repeat the handshake every time. So the connection lives in a daemon thread
    and the REPL submits coroutines to it.
    """

    def __init__(self, url, timeout=10.0):
        self.url = url
        self.timeout = timeout
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
            async with Client(self.url, read_timeout_seconds=self.timeout) as client:
                self._client = client
                self.tools = [_to_openai_tool(t) for t in (await client.list_tools()).tools]
                self._ready.set()
                while True:  # park; the REPL drives us via _submit
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


def _to_openai_tool(t):
    """MCP tool definition -> OpenAI tool schema. input_schema is already JSON Schema."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or t.name,
            "parameters": t.input_schema,
        },
    }


def _result_to_text(result):
    """Flatten an MCP CallToolResult into the string the model reads back."""
    parts = [b.text for b in result.content if getattr(b, "text", None)]
    text = "\n".join(parts) if parts else json.dumps(result.structured_content)
    if result.is_error:
        return f"[tool error] {text}"
    return text


def get_qdrant():
    """Lazily connect, so the agent still starts when Qdrant is down."""
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient

        _qdrant = QdrantClient(url=QDRANT_URL, timeout=15)
    return _qdrant


def docs_available():
    """True when Qdrant is reachable and the collection has been built."""
    try:
        return get_qdrant().collection_exists(COLLECTION)
    except Exception:
        return False


def retrieve(query, k=TOP_K):
    """Embed the query and return the chunks that clear MIN_SCORE."""
    emb = client.embeddings.create(model=EMBED_MODEL, input=[SEARCH_PREFIX + query])
    points = (
        get_qdrant()
        .query_points(COLLECTION, query=emb.data[0].embedding, limit=k, with_payload=True)
        .points
    )
    return [p for p in points if p.score >= MIN_SCORE]


def answer_docs(query):
    """Ground an answer in the indexed documents, or admit nothing was found.

    Generation is deliberately kept inside this tool: the agent receives a finished,
    cited answer rather than raw chunks, so it cannot blend irrelevant passages into
    a confident-sounding reply.
    """
    try:
        hits = retrieve(query)
    except Exception as e:
        return f"Document search unavailable: {e}"

    if not hits:
        return (
            f"No document matched {query!r} above the relevance threshold "
            f"(MIN_SCORE={MIN_SCORE}). Either the corpus does not cover it, or the "
            "threshold is too high for this embedding model."
        )

    context = "\n\n".join(
        f"[{i + 1}] ({h.payload['file']})\n{h.payload['text']}" for i, h in enumerate(hits)
    )
    messages = [
        {
            "role": "system",
            "content": "Answer using ONLY the context below. Cite sources like [1]. "
            "If the context doesn't contain the answer, say you don't know.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    sources = "\n".join(f"  [{i + 1}] {h.payload['file']} (score {h.score:.2f})" for i, h in enumerate(hits))
    return f"{resp.choices[0].message.content}\n\nSources:\n{sources}"


def trim_history(messages, limit=MAX_HISTORY):
    """Keep the conversation bounded; drop oldest turns, never orphan a tool result."""
    if len(messages) <= limit:
        return messages
    start = len(messages) - limit
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1  # a tool result without its assistant call breaks the API contract
    return messages[:1] + messages[start:]


def run_local_tool(name, args):
    try:
        if name == "answer_docs":
            print(f"\n? {args['query']}")
            return answer_docs(args["query"])
        if name == "bash":
            print(f"\n$ {args['command']}")
            if input("run? [y/N] ").strip().lower() != "y":
                return "User denied this command."
            r = subprocess.run(
                args["command"], shell=True, capture_output=True, text=True, timeout=60
            )
            return (r.stdout + r.stderr)[-10000:] or "(no output)"
        if name == "read_file":
            with open(args["path"]) as f:
                return f.read()[-20000:]
        if name == "write_file":
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return f"Wrote {len(args['content'])} chars to {args['path']}"
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


def connect_mcp():
    """Return (bridge, tools), or (None, []) if the server is unreachable.

    A missing MCP server is not fatal: the agent still works with its local tools.
    """
    try:
        bridge = McpBridge(MCP_URL)
    except Exception as e:
        print(f"MCP unavailable ({e}); continuing with local tools only.")
        return None, []

    tools = []
    for t in bridge.tools:
        name = t["function"]["name"]
        if name in LOCAL_NAMES:
            print(f"  skipped MCP tool {name!r}: name collides with a local tool")
        else:
            tools.append(t)
    return bridge, tools


def run_turn(messages, tools, bridge):
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            print(f"\n{msg.content}")
        if not msg.tool_calls:
            return

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            name = tc.function.name
            source = "local" if name in LOCAL_NAMES else "mcp"
            print(f"\n[tool:{source}] {name}({json.dumps(args)[:200]})")

            if name in LOCAL_NAMES:
                output = run_local_tool(name, args)
            elif bridge is not None:
                try:
                    output = _result_to_text(bridge.call(name, args))
                except Exception as e:
                    output = f"Error calling MCP tool {name}: {e}"
            else:
                output = f"Unknown tool: {name}"

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(output)[:20000]}
            )
    print("\n[stopped: hit MAX_STEPS]")


def main():
    bridge, mcp_tools = connect_mcp()
    tools = list(LOCAL_TOOLS)

    if docs_available():
        tools.append(DOCS_TOOL)
    else:
        print(f"  no document corpus: {COLLECTION} not found at {QDRANT_URL}")
        print("  index it first:  docker compose --profile rag run --rm rag")

    messages = [{"role": "system", "content": SYSTEM}]
    print(f"Agent ready ({MODEL}).")
    if mcp_tools:
        names = ", ".join(t["function"]["name"] for t in mcp_tools)
        print(f"MCP tools from {MCP_URL}: {names}")
    print(f"Local tools: {', '.join(t['function']['name'] for t in tools)}")
    print("Ctrl+C to quit.")
    while True:
        try:
            user = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        messages.append({"role": "user", "content": user})
        run_turn(messages, tools, bridge)
        messages[:] = trim_history(messages)  # bound context growth across turns


if __name__ == "__main__":
    main()
