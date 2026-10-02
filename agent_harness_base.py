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
from skills import catalog, list_skills, load as load_skill_body

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),  # Ollama ignores the key
)
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_STEPS = 10

# A small model decides, once per message, whether the corpus is needed at all. The
# 7B has to choose between nine tools at once and misroutes; a binary decision is a
# much easier ask. Measured warm at ~0.7s, against 37s for one answer_docs call.
#
# qwen2.5:0.5b was tried first and scored 4/8 on a labelled set - coin-flip accuracy,
# and it routed arithmetic to the corpus. 1.5b scored 6/8 with no false negatives,
# which is the property that matters here: see select_tools(). Set ROUTER_MODEL=""
# to turn routing off.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen2.5:1.5b")
ROUTER_KEEP_ALIVE = os.getenv("ROUTER_KEEP_ALIVE", "30m")
DOCS_TOOL_NAME = "answer_docs"

ROUTER_PROMPT = """Decide whether answering the message needs facts from the user's own \
indexed documents - their notes, records, resumes, contracts, saved plans.

Answer yes only for questions about those specific documents or their contents.
Answer no for general knowledge, arithmetic, coding, and anything about files in \
the working directory.

Reply with exactly one word, yes or no.

Message: """

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


def tool(name, desc, props, optional=None):
    """OpenAI tool schema for a tool this harness runs itself.

    `props` are required string args; `optional` maps name -> JSON schema.
    """
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
    tool(
        "load_skill",
        "Load the full playbook for one of the available skills. Use when a task matches "
        "a skill description. Returns the instructions to follow.",
        ["name"],
    ),
]
LOCAL_NAMES = {t["function"]["name"] for t in LOCAL_TOOLS}


def run_local_tool(name, args):
    try:
        if name == "load_skill":
            body = load_skill_body(args["name"])
            if body is None:
                available = ", ".join(s["name"] for s in list_skills()) or "none"
                return f"No skill named {args['name']!r}. Available: {available}"
            return f"# skill: {args['name']}\n\n{body}"
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
            "/skills                list available skill playbooks\n"
            "/call <tool> <json>    call a tool directly, no model involved\n"
            "/<skill-name>          run a skill's playbook\n"
            "anything else          goes to the model"
        )

    if cmd == "/skills":
        found = list_skills()
        if not found:
            return f"No skills found in $SKILLS_DIR ({os.getenv('SKILLS_DIR', 'skills')!r})"
        lines = [f"  {s['name']:<14} {s['description']}" for s in found]
        return "Available skills:\n" + "\n".join(lines)

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


router = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)


def needs_documents(prompt):
    """One binary decision from the small model. True if the corpus is likely needed.

    A failure here must not break the turn, so anything unexpected means "let the main
    model decide" -- routing is an optimisation, not a gate.
    """
    if not ROUTER_MODEL:
        return None
    try:
        resp = router.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[{"role": "user", "content": ROUTER_PROMPT + prompt}],
            max_tokens=3,
            extra_body={"keep_alive": ROUTER_KEEP_ALIVE},
        )
        return "yes" in (resp.choices[0].message.content or "").strip().lower()
    except Exception as e:
        print(f"  router unavailable ({e}); letting the main model decide")
        return None


def select_tools(tools, wants_docs):
    """Constrain the tool list by the routing decision.

    The two failure directions are not symmetric, which is what makes this worth doing:

    - A false negative (needs docs, routed no) hides answer_docs, so the model cannot
      search and will answer from memory instead. That is the damaging one.
    - A false positive (routed yes, did not need docs) only adds a tool back. The main
      model still decides whether to call it, so nothing is wasted.

    So this is deliberately permissive: when in doubt, say yes. It never forces a call.
    """
    if wants_docs is None:
        return tools
    local = [t for t in tools if t["function"]["name"] in LOCAL_NAMES]
    docs = [t for t in tools if t["function"]["name"] == DOCS_TOOL_NAME]
    return local + docs if wants_docs else local


def run_turn(messages, tools, bridge, prompt):
    messages.append({"role": "user", "content": prompt})
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            print(f"\n{msg.content}")
        elif not msg.tool_calls:
            # A small model sometimes stops without saying anything at all. Say so
            # rather than leaving the user staring at an apparently dead agent.
            print("\n[model returned nothing - try rephrasing, or call a tool with /call]")
            return
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

    messages = [{"role": "system", "content": SYSTEM + catalog()}]
    print(f"Agent ready ({MODEL}).")
    if mcp_tools:
        names = ", ".join(t["function"]["name"] for t in mcp_tools)
        print(f"MCP tools from {MCP_URL}: {names}")
    if not mcp_tools:
        print("  document tools unavailable, so this agent cannot search the corpus")
    found = list_skills()
    print(f"Skills: {', '.join(s['name'] for s in found) if found else 'none'}")
    print("Slash commands: /tools, /skills, /call, /<skill>   (Ctrl+C to quit)")

    while True:
        try:
            user = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            continue  # a bare newline should not cost a router call and a generation

        prompt = None
        if user.startswith("/"):
            reply = handle_command(user, tools, bridge)
            if reply is not None:
                print(f"\n{reply}")
                continue
            # Not a known command: maybe it names a skill.
            name = user[1:].strip()
            body = load_skill_body(name)
            if body is None:
                print("\nUnknown command. Try /help, /tools or /skills.")
                continue
            print(f"\n[skill] {name}")
            prompt = f"# skill: {name}\n\n{body}"
        else:
            prompt = user

        if prompt is not None:
            wants_docs = needs_documents(prompt)
            if wants_docs is not None:
                print(f"[route] {'documents' if wants_docs else 'no documents'}")
            run_turn(messages, select_tools(tools, wants_docs), bridge, prompt)

        messages[:] = trim_history(messages)  # bound context growth across turns


if __name__ == "__main__":
    main()
