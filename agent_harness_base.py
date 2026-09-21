#!/usr/bin/env python3
"""Minimal agent harness for any OpenAI-compatible endpoint (default: local Ollama)."""
import json
import os
import subprocess

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
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


TOOLS = [
    tool("bash", "Run a shell command and return stdout/stderr.", ["command"]),
    tool("read_file", "Read a text file.", ["path"]),
    tool("write_file", "Write text to a file, overwriting it.", ["path", "content"]),
]


def run_tool(name, args):
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


def run_turn(messages):
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": run_tool(tc.function.name, args),
                }
            )
    print("\n[stopped: hit MAX_STEPS]")


def main():
    messages = [{"role": "system", "content": SYSTEM}]
    print(f"Agent ready ({MODEL}). Ctrl+C to quit.")
    while True:
        try:
            user = input("\n> ")
        except (KeyboardInterrupt, EOFError):
            break
        messages.append({"role": "user", "content": user})
        run_turn(messages)


if __name__ == "__main__":
    main()
