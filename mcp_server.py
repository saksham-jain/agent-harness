#!/usr/bin/env python3
"""MCP Server: serves the document tools over Streamable HTTP.

This layer owns protocol concerns only. What a tool actually does lives in
`rag_service`, so retrieval logic is testable without a client and reusable by
anything that can speak MCP.

Run:  python mcp_server.py
Serves: http://<MCP_HOST>:<MCP_PORT><MCP_PATH>   (0.0.0.0:8000/mcp in Docker)

Clients connect with a plain URL:  Client("http://localhost:8000/mcp")
"""
import asyncio
import functools
import inspect
import logging
import os
import time
from datetime import datetime, timezone

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

import auth
import rag_service

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))
PATH = os.getenv("MCP_PATH", "/mcp")
SERVER_NAME = "agent-harness-docs"

# DNS-rebinding protection. The SDK defaults this OFF for backwards compatibility, which
# leaves any browser page a user visits able to talk to an instance on their machine.
# Empty list means disabled, so local development is unaffected.
ALLOWED_HOSTS = [h for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h]

# TLS. Both must be set to serve https; unset means plain http, which is the status quo
# and still the right thing on a loopback-only setup. Generate with `mkcert localhost`.
TLS_CERT = os.getenv("MCP_TLS_CERT", "")
TLS_KEY = os.getenv("MCP_TLS_KEY", "")
LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")

logging.basicConfig(
    level=os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("mcp.tools")


def logged(fn):
    """Log every tool call with its arguments, duration, and result.

    Applied under @mcp.tool() so functools.wraps keeps the type hints that the
    server turns into the input schema.

    Async tools need an async wrapper. A sync one returns the coroutine object instead of
    awaiting it, and the schema then describes the tool as returning a string while the
    body actually hands back a coroutine — which fails at the first string operation, far
    from the cause.
    """

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            args_text = ", ".join(
                [repr(a)[:80] for a in args] + [f"{k}={v!r}"[:80] for k, v in kwargs.items()]
            )
            started = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                log.warning("%s(%s) failed after %.0fms: %s", fn.__name__, args_text,
                            (time.perf_counter() - started) * 1000, e)
                raise
            log.info("%s(%s) -> %s in %.0fms", fn.__name__, args_text,
                     repr(result)[:120], (time.perf_counter() - started) * 1000)
            return result

        return wrapper

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


TOKEN_VERIFIER, AUTH_SETTINGS = auth.build()

mcp = MCPServer(
    SERVER_NAME,
    instructions=(
        "Tools over the user's indexed document corpus: answer questions from it with "
        "citations, see what is in it, and refresh it after files change. Use these for "
        "anything about the user's own documents."
    ),
    # These two travel together. Passing one without the other is a ValueError here,
    # at construction, before the server ever serves a request.
    token_verifier=TOKEN_VERIFIER,
    auth=AUTH_SETTINGS,
)


class DocInfo(BaseModel):
    file: str
    chunks: int


@mcp.tool(title="Answer from the documents")
@logged
async def answer_docs(query: str, ctx: Context) -> str:
    """Answer a question using the indexed documents. Returns a cited answer, or says
    plainly that the documents do not cover it. Use this for anything about the user's
    own files rather than guessing.

    Takes a while on first use: generation is a local model. If your client shows
    progress, tokens arrive as they are produced."""
    # Retrieval is quick and generation is not, so the two phases report separately. A
    # client that asked for progress gets a moving indicator instead of a frozen call.
    def on_token(piece):
        nonlocal emitted
        emitted += 1
        # Progress rather than the text itself: the final result already carries the whole
        # answer, and shipping every fragment twice would double the payload for no gain.
        return ctx.report_progress(float(emitted), None, piece.strip()[:60] or None)

    emitted = 0
    loop = asyncio.get_running_loop()
    try:
        await ctx.report_progress(0.0, None, "searching the corpus")
        hits = await loop.run_in_executor(None, rag_service.retrieve, query)
        if not hits:
            return rag_service.abstention(query)
        await ctx.report_progress(0.0, None, f"reading {len(hits)} passages, generating")

        def sync_on_token(piece):
            # report_progress is async; the generator runs in a worker thread.
            asyncio.run_coroutine_threadsafe(on_token(piece), loop).result()

        text = await loop.run_in_executor(None, lambda: rag_service.answer_from(query, hits, sync_on_token))
    except Exception as e:
        if isinstance(e, ToolError):
            raise
        return rag_service.answer_error(query, e)
    return text


@mcp.tool(title="List indexed documents")
@logged
def list_docs() -> list[DocInfo]:
    """List the documents currently in the index, with how many chunks each has."""
    if not rag_service.available():
        raise ToolError(f"No index found (collection {rag_service.COLLECTION!r}). Run the indexer first.")
    counts, sources = rag_service.list_docs()
    return [DocInfo(file=sources.get(d, d), chunks=n) for d, n in counts.items()]


@mcp.tool(title="Re-index the documents")
@logged
def refresh_index() -> str:
    """Re-scan the document directory and embed anything that changed. Call after
    creating or editing files that should become searchable."""
    return rag_service.index()


@mcp.tool(title="Index status")
@logged
def index_status() -> str:
    """Whether a document index exists, plus where it is looking."""
    ready = rag_service.available()
    return (
        f"ready: collection={rag_service.COLLECTION!r} at {rag_service.QDRANT_URL} "
        f"({rag_service.EMBED_MODEL}), docs dir={rag_service.DOCS_DIR!r}"
        if ready
        else f"not indexed: no collection {rag_service.COLLECTION!r} at {rag_service.QDRANT_URL}. "
        "Run the indexer, then call refresh_index()."
    )


@mcp.tool(title="Who am I")
@logged
def whoami() -> str:
    """Report which authenticated caller is on this request, and its scopes."""
    token = auth.whoami()
    if token is None:
        return "anonymous: no token on this request (auth is off, or this is not HTTP)"
    return f"subject={token.subject} client_id={token.client_id} scopes={','.join(token.scopes)}"


@mcp.resource("docs://status")
@logged
def status() -> str:
    """A one-line health string for the document service."""
    state = "ready" if rag_service.available() else "not indexed"
    return f"{SERVER_NAME} {state} at {datetime.now(timezone.utc).isoformat(timespec='seconds')}"


if __name__ == "__main__":
    transport_security = TransportSecuritySettings(
        # Enabling the check with an empty allow list would lock out every request,
        # including local ones. Only turn it on with hosts named.
        enable_dns_rebinding_protection=bool(ALLOWED_HOSTS),
        allowed_hosts=ALLOWED_HOSTS,
    )

    if TLS_CERT and TLS_KEY:
        # mcp.run() builds its own uvicorn.Config with host, port and log level only,
        # so it cannot serve TLS. Take the ASGI app it would have served and hand it
        # to uvicorn directly. Both paths stay available: with no cert configured this
        # falls through to plain HTTP, unchanged.
        import uvicorn

        app = mcp.streamable_http_app(
            streamable_http_path=PATH, transport_security=transport_security
        )
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level=LOG_LEVEL.lower(),
            ssl_certfile=TLS_CERT,
            ssl_keyfile=TLS_KEY,
        )
    else:
        # transport_security is a run() argument, not a constructor argument.
        mcp.run(
            transport="streamable-http",
            host=HOST,
            port=PORT,
            streamable_http_path=PATH,
            transport_security=transport_security,
        )
