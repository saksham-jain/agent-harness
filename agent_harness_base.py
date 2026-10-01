#!/usr/bin/env python3
"""AI Harness: an agent loop over an OpenAI-compatible endpoint.

Owns the model, the conversation, and its own file/shell tools. Every other
capability — the document corpus included — arrives over MCP, so this layer never
imports Qdrant or knows a vector store exists.
"""
import json
import os
import subprocess

from openai import OpenAI

from mcp_client import connect, result_to_text

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_STEPS = 10

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


def tool(name, desc, props):
    """OpenAI tool schema for a tool this harness runs itself."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in props},
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


def run_local_tool(name, args):
    try:
        if name == "bash":
            print(f"\n$ {args['command']}")
            if input("run? [y/N] ").strip().lower() != "y":
                return "User denied this command."
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=60)
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


def trim_history(messages, limit=MAX_HISTORY):
    """Keep the conversation bounded; drop oldest turns, never orphan a tool result."""
    if len(messages) <= limit:
        return messages
    start = len(messages) - limit
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1  # a tool result without its assistant call breaks the API contract
    return messages[:1] + messages[start:]


def handle_command(line, tools, bridge):
    """Run a slash command. Returns text to print, or None if it isn't one.

    These bypass the model entirely, so a tool always runs exactly as asked instead
    of the model deciding the task is too simple to need it.
    """
    cmd, _, rest = line.strip().partition(" ")
    cmd = cmd.lower()

    if cmd in ("/help", "/?"):
        return (
            "/tools                 list every tool the agent has\n"
            "/call <tool> <json>    call a tool directly, no model involved\n"
            "                        e.g. /call answer_docs {\"query\": \"how long is the trip?\"}\n"
            "anything else          goes to the model"
        )

    if cmd == "/tools":
        lines = []
        for t in tools:
            f = t["function"]
            props = f["parameters"].get("properties", {})
            sig = ", ".join(f"{k}:{v.get('type', '?')}" for k, v in props.items())
            lines.append(f"  {f['name']:<14} ({sig})")
        return "\n".join(lines)

    if cmd == "/call":
        name, _, raw = rest.partition(" ")
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            return f"Bad JSON arguments: {e}"
        if not any(t["function"]["name"] == name for t in tools):
            return f"No such tool: {name}. Try /tools"
        if name in LOCAL_NAMES:
            try:
                return str(run_local_tool(name, args))
            except Exception as e:
                return f"Error calling {name}: {e}"
        if bridge is None:
            return f"{name} is an MCP tool but the server is not connected."
        try:
            return result_to_text(bridge.call(name, args))
        except Exception as e:
            # A tool that times out or drops must not take the REPL down with it.
            return f"Error calling MCP tool {name}: {e}"

    return None


def run_turn(messages, tools, bridge):
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
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
                    output = result_to_text(bridge.call(name, args))
                except Exception as e:
                    output = f"Error calling MCP tool {name}: {e}"
            else:
                output = f"Unknown tool: {name}"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)[:20000]})
    print("\n[stopped: hit MAX_STEPS]")


def main():
    bridge, mcp_tools = connect(MCP_URL, LOCAL_NAMES)
    tools = LOCAL_TOOLS + mcp_tools

    messages = [{"role": "system", "content": SYSTEM}]
    print(f"Agent ready ({MODEL}).")
    if mcp_tools:
        names = ", ".join(t["function"]["name"] for t in mcp_tools)
        print(f"MCP tools from {MCP_URL}: {names}")
    if not mcp_tools:
        print("  document tools unavailable, so this agent cannot search the corpus")
    print("Slash commands: /tools, /call <tool> <json>, /help   (Ctrl+C to quit)")
    while True:
        try:
            user = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        reply = handle_command(user, tools, bridge) if user.startswith("/") else None
        if reply is not None:
            print(f"\n{reply}")
            continue
        messages.append({"role": "user", "content": user})
        run_turn(messages, tools, bridge)
        messages[:] = trim_history(messages)  # bound context growth across turns


if __name__ == "__main__":
    main()
