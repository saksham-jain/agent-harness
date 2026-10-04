#!/usr/bin/env python3
"""Token verification for the MCP server.

MCP servers over Streamable HTTP are OAuth 2.1 **resource servers**: they verify a
bearer token on every request, they never issue one. The SDK owns the OAuth-shaped parts
-- the 401, the `WWW-Authenticate` pointer, the RFC 9728 discovery document -- and asks
this module for exactly one thing: is this token good, and who is it?

`TokenVerifier` is a protocol with one async method, so replacing the static table below
with real JWT signature checks or RFC 7662 introspection touches nothing else in the
server.

Transport note: `Authorization` is an HTTP header. stdio has none, and the in-process
`Client(mcp)` used in tests connects to the object directly and skips the HTTP layer
entirely. A test that passes with `Client(mcp)` proves nothing about authentication --
it has to go over HTTP.
"""
import json
import os

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

# Off by default so the local docker-compose flow keeps working with no setup. Turn on
# with MCP_AUTH=1 when the server is reachable by anything other than you.
AUTH_ENABLED = os.getenv("MCP_AUTH", "0") not in ("0", "", "false", "False")

# The exact URL clients connect to. It names which resource a token is for, so a token
# issued for another resource must not be accepted here.
RESOURCE_URL = os.getenv("MCP_RESOURCE_URL", "http://localhost:8000/mcp")
ISSUER_URL = os.getenv("MCP_ISSUER_URL", "https://auth.example.com/")
REQUIRED_SCOPES = [s for s in os.getenv("MCP_REQUIRED_SCOPES", "docs:read").split(",") if s]
TOKENS_FILE = os.getenv("MCP_TOKENS_FILE", "tokens.json")


class StaticTokenVerifier(TokenVerifier):
    """Looks a token up in a table. Each entry says who it belongs to and what scopes
    it carries; the SDK hands that straight back via get_access_token()."""

    def __init__(self, tokens):
        self._tokens = tokens

    async def verify_token(self, token: str) -> AccessToken | None:
        entry = self._tokens.get(token)
        if entry is None:
            return None
        return AccessToken(
            token=token,
            client_id=entry.get("client_id", "claude-code"),
            scopes=entry.get("scopes", REQUIRED_SCOPES),
            resource=RESOURCE_URL,
            subject=entry.get("subject", "unknown"),
            expires_at=entry.get("expires_at"),
        )


def load_tokens():
    """Read the token table.

    MCP_TOKENS as inline JSON wins, so a one-user pilot needs no file at all:

        MCP_TOKENS='{"tok_alice":{"subject":"alice"}}'

    Otherwise read MCP_TOKENS_FILE. Nothing here is hashed or encrypted -- it is a
    lookup table, and the file must not be world-readable in production.
    """
    raw = os.getenv("MCP_TOKENS")
    if raw:
        return json.loads(raw)
    if os.path.isfile(TOKENS_FILE):
        with open(TOKENS_FILE) as f:
            return json.load(f)
    return {}


def build():
    """Return (token_verifier, auth) for MCPServer, or (None, None) when disabled.

    The two arguments travel together: passing one without the other is a ValueError
    at construction, before the server ever serves a request.
    """
    if not AUTH_ENABLED:
        return None, None

    tokens = load_tokens()
    if not tokens:
        raise RuntimeError(
            f"MCP_AUTH is on but no tokens were found. Set MCP_TOKENS to inline JSON or "
            f"point MCP_TOKENS_FILE at a file (looked for {TOKENS_FILE!r})."
        )

    from pydantic import AnyHttpUrl

    verifier = StaticTokenVerifier(tokens)
    auth = AuthSettings(
        issuer_url=AnyHttpUrl(ISSUER_URL),
        resource_server_url=AnyHttpUrl(RESOURCE_URL),
        required_scopes=REQUIRED_SCOPES,
        # Explicit rather than left unset: unset warns today and flips to True in 3.0,
        # which would reject tokens we are currently accepting.
        validate_token_resource=True,
    )
    return verifier, auth


def whoami():
    """The AccessToken for this request, or None outside an authenticated HTTP call."""
    from mcp.server.auth.middleware.auth_context import get_access_token

    return get_access_token()
