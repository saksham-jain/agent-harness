#!/usr/bin/env python3
"""Minimal agent harness for any OpenAI-compatible endpoint (default: local Ollama).

Local tools (bash, read_file, write_file) are defined in-process. Tools from an MCP
server are discovered at startup and exposed to the model alongside them, so the
agent can call either without knowing the difference.
"""
import asyncio
import json
import os
import subprocess
import threading

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
SYSTEM = "You are a helpful coding agent working in the user's current directory. Use tools when needed."
MAX_STEPS = 10


def tool(name, desc, props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in props},
                "required": props,
            },
        },
    }


LOCAL_TOOLS = [
    tool("bash", "Run a shell command and return stdout/stderr.", ["command"]),
    tool("read_file", "Read a text file.", ["path"]),
    tool("write_file", "Write text to a file, overwriting it.", ["path", "content"]),
]
LOCAL_NAMES = {t["function"]["name"] for t in LOCAL_TOOLS}


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


def run_local_tool(name, args):
    try:
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
    tools = LOCAL_TOOLS + mcp_tools

    messages = [{"role": "system", "content": SYSTEM}]
    print(f"Agent ready ({MODEL}).")
    if mcp_tools:
        names = ", ".join(t["function"]["name"] for t in mcp_tools)
        print(f"MCP tools from {MCP_URL}: {names}")
    print("Ctrl+C to quit.")
    while True:
        try:
            user = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        messages.append({"role": "user", "content": user})
        run_turn(messages, tools, bridge)


if __name__ == "__main__":
    main()
