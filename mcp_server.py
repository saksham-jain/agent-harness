#!/usr/bin/env python3
"""Basic MCP server exposing a few demo tools over Streamable HTTP.

Run:  python mcp_server.py
Serves: http://<MCP_HOST>:<MCP_PORT><MCP_PATH>   (0.0.0.0:8000/mcp in Docker)

Clients connect with a plain URL:
    Client("http://localhost:8000/mcp")
"""
import functools
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))
PATH = os.getenv("MCP_PATH", "/mcp")
SERVER_NAME = "agent-harness-demo"

logging.basicConfig(
    level=os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("mcp.tools")


def logged(fn):
    """Log every tool call with its arguments, duration, and result.

    Applied under @mcp.tool() so functools.wraps keeps the type hints that the
    server turns into the input schema.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args_text = ", ".join(
            [repr(a)[:80] for a in args] + [f"{k}={v!r}"[:80] for k, v in kwargs.items()]
        )
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            log.warning("%s(%s) failed after %.0fms: %s", fn.__name__, args_text,
                        (time.perf_counter() - started) * 1000, e)
            raise
        log.info("%s(%s) -> %s in %.0fms", fn.__name__, args_text,
                 repr(result)[:120], (time.perf_counter() - started) * 1000)
        return result

    return wrapper


mcp = MCPServer(
    SERVER_NAME,
    instructions=(
        "Demo tool server for the agent-harness stack: arithmetic, text utilities, and the "
        "clock. Use these to sanity-check that the MCP wiring between client and server works."
    ),
)


def _parse_tz(tz: str) -> timezone:
    """Accept 'UTC', 'Z', or a fixed offset like '+05:30' / '-08:00'."""
    if tz.upper() in ("UTC", "Z"):
        return timezone.utc
    try:
        sign = -1 if tz[0] == "-" else 1
        hours, minutes = tz[1:].split(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
    except Exception:
        raise ToolError(f"Unrecognised timezone {tz!r}. Use 'UTC' or a fixed offset like '+05:30'.")


@mcp.tool(title="Add two integers")
@logged
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@mcp.tool(title="Echo text back")
@logged
def echo(text: str, times: int = 1) -> str:
    """Echo `text` back, repeated `times` times and space-separated."""
    if times < 1:
        raise ToolError("times must be >= 1")
    return " ".join([text] * times)


class Counts(BaseModel):
    words: int
    characters: int


@mcp.tool(title="Count words")
@logged
def word_count(text: str) -> Counts:
    """Count the words and characters in `text`."""
    return Counts(words=len(text.split()), characters=len(text))


@mcp.tool(title="Square root")
@logged
def sqrt(x: float) -> float:
    """Return the square root of a non-negative number."""
    if x < 0:
        # ToolError messages reach the model, so it can read the reason and retry.
        # Any other exception would surface as a bare "Error executing tool sqrt".
        raise ToolError(f"Cannot take the square root of a negative number: {x}")
    return x**0.5


@mcp.tool(title="Current time")
@logged
def now(tz: str = "UTC") -> str:
    """Return the current time. `tz` is 'UTC' or a fixed offset such as '+05:30'."""
    return datetime.now(_parse_tz(tz)).isoformat(timespec="seconds")


@mcp.resource("demo://status")
def status() -> str:
    """A one-line health string for the demo server."""
    return f"{SERVER_NAME} ok at {datetime.now(timezone.utc).isoformat(timespec='seconds')}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=HOST, port=PORT, streamable_http_path=PATH)
