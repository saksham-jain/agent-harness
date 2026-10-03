# Auth + TLS

Working notes for the `feat/auth-tls` branch. Nothing here is implemented yet.

## Why this is needed

The MCP server is meant to be reachable by users' own Claude Code, over the network.
Today anything that can reach port 8000 gets full access. Verified, not assumed:

```
no credentials of any kind -> ['answer_docs', 'list_docs', 'refresh_index', 'index_status']
list_docs returned         -> { "file": "/docs/Saksham-Jain-Resume.pdf", "chunks": 5 }
```

Zero credentials retrieved the corpus. Four concrete exposures:

| # | Problem | Consequence |
| --- | --- | --- |
| 1 | **No authentication** | Anyone reads the corpus, spends inference budget, or calls `refresh_index` and mutates Qdrant |
| 2 | **Write tool exposed** | `refresh_index` is a mutation with no auth. An unauthenticated caller can rewrite the index |
| 3 | **Plain HTTP** | Bearer tokens would cross the network in cleartext. `claude mcp add` needs https for a remote host |
| 4 | **DNS-rebinding protection off** | The SDK defaults `TransportSecuritySettings(enable_dns_rebinding_protection=False)` for backwards compatibility. A browser on a user's machine could be made to talk to the local instance |

Docker publishes `8000:8000` on all interfaces (`0.0.0.0`), so this is exposed as soon
as there is a port forward, a tunnel, or a public host.

Problem 2 is the one that is not obvious: **read-only is not the default here.** Two of
the four tools mutate or enumerate. Auth is not just about confidentiality.

## What the SDK gives us

Checked against the installed `mcp` 2.2.0, not just the docs.

Over Streamable HTTP an MCP server is an OAuth 2.1 **resource server**. It verifies
tokens, it never issues them. Two constructor arguments:

| Argument | Default | Notes |
| --- | --- | --- |
| `token_verifier` | `None` | Protocol: one async method |
| `auth` | `None` | `AuthSettings` |
| `transport_security` | not a constructor arg | Lives on `run()`, not `MCPServer()` |

**`token_verifier` and `auth` travel together.** Passing one without the other raises
`ValueError` at construction, before serving anything.

```python
class TokenVerifier(Protocol):
    async def verify_token(self, token: str) -> AccessToken | None: ...
```

`AccessToken` carries `token`, `client_id`, `scopes`, `expires_at`, `resource`,
`subject`, `claims`.

`AuthSettings` requires exactly two fields:

| Field | Required | Default |
| --- | --- | --- |
| `issuer_url` | **yes** | — |
| `resource_server_url` | **yes** | — |
| `required_scopes` | no | `None` |
| `validate_token_resource` | no | `None` → warns, behaves `False`; `True` in 3.0 |

The SDK also serves RFC 9728 metadata at
`/.well-known/oauth-protected-resource/mcp` and answers unauthenticated requests with a
`401` whose `WWW-Authenticate` points at it. That is the whole discovery dance —
401 → metadata → authorization server → token → retry — and we write none of it.

Inside any tool, `get_access_token()` returns the `AccessToken` for the current request.
That is the hook per-user isolation hangs off.

`TransportSecuritySettings` is passed to `run()`, not the constructor — an easy thing to
get wrong.

## Design

### 1. Token verifier

Start with a static table keyed by token, holding one `AccessToken` per user. It proves
the shape; it is not the end state.

```python
USERS = {
    "tok_alice": {"subject": "alice", "collection": "docs_alice"},
    "tok_bob":   {"subject": "bob",   "collection": "docs_bob"},
}
```

Why a table first:

- a few lines, so the wiring is reviewable
- no external dependency to deploy before auth works at all
- a real JWT verifier can replace it without touching a caller, since the SDK only ever
  calls `verify_token`

Swap later for JWT verification, or introspection against a real authorization server
(`IntrospectionTokenVerifier` in the SDK's `examples/servers/simple-auth/`).

### 2. Per-tenant collection

Qdrant gives no magic here — isolation is a payload filter you remember to write down,
and one forgotten `filter=` leaks. One collection per user makes a leak structurally
impossible, at the cost of more collections. Overkill for the first cut; **the wrong
omission is a shared collection with a filter that gets dropped.**

### 3. Document identity — must be fixed first

`rag_service` keys documents by **absolute filesystem path**:

```python
payload={"file": path, ...}   # path is the delete/update filter key
```

Two tenants who both upload `notes.md` collide, and `refresh_index` would delete each
other's chunks. Under a local mount this is cosmetic; with two users it is a
cross-tenant data loss bug. Introduce a `doc_id` that is stable and tenant-scoped, and
stop using paths as keys.

### 4. TLS

TLS terminates at a reverse proxy, not in the Python process. Caddy or nginx in front,
app bound to `127.0.0.1`, so plaintext never leaves the host.

```
:443 ──TLS──> caddy ──http──> mcp-server (127.0.0.1:8000)
```

This is why `resource_server_url` must be the public https URL exactly as clients
connect: it names which resource a token is for, and where discovery lives.

### 5. Host allowlist

Turn on what is currently off:

```python
transport_security=TransportSecuritySettings(allowed_hosts=["docs.example.com", "localhost:*"])
```

Only a few minutes' work, and it closes problem 4 above.

## Try it locally

Auth is transport-independent, so the whole flow verifies over plain http before any TLS
exists. `.env` in the repo root turns it on:

```bash
docker compose up -d mcp-server                      # reads .env, MCP_AUTH=1
```

Without a token:

```bash
curl -i http://localhost:8000/mcp -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
# HTTP/1.1 401 Unauthorized
# www-authenticate: Bearer ... resource_metadata=".../.well-known/oauth-protected-resource/mcp"
```

Discovery needs no auth:

```bash
curl -s http://localhost:8000/.well-known/oauth-protected-resource/mcp
```

With a real token:

```bash
export MCP_BEARER_TOKEN=$(python3 -c "
import json
t = json.load(open('tokens.json'))
print(next(k for k, v in t.items() if v['subject'] == 'saksham'))")
docker compose --profile client run --rm mcp-client     # agent authenticates
```

Ask for a token by subject rather than by position. `list(...)[0]` works too and Python
preserves file order, but it returns whichever key comes first, so the command reads the
same while meaning something different.

Claude Code:

```bash
claude mcp remove mydocs -s local
claude mcp add --transport http mydocs http://localhost:8000/mcp \
  --header "Authorization: Bearer $MCP_BEARER_TOKEN"
```

Turn it off again with `MCP_AUTH=0` in `.env`.

`tokens.json` is gitignored and generated locally; `tokens.example.json` is the committed
template. Tokens are stored in plaintext — it is a lookup table, not a hash — so the file
is `chmod 600`.

## Order of work

| # | Step | Blocks everything after? |
| --- | --- | --- |
| 1 | TLS proxy + public https URL | yes — `resource_server_url` depends on it |
| 2 | Host allowlist | no |
| 3 | Static `TokenVerifier` + `AuthSettings` | no |
| 4 | Verify: 401 without token, discovery document, token works | — |
| 5 | `get_access_token()` → tenant resolution | yes, for multi-user |
| 6 | `doc_id` replacing path keys | yes, for multi-user |
| 7 | Scope checks per tool — separate read from write | no |
| 8 | Real JWT / introspection verifier | no |

Steps 1–4 make the server safe to expose to one trusted user. Steps 5–7 are what make
it safe for many.

## What is deliberately not planned

- **A login UI.** The resource server does not sign anyone in; that is the
  authorization server's job.
- **Billing.** Step 8 in `ENHANCEMENTS.md`: let users bring their own inference key, so
  hosting is retrieval only and the abuse vector mostly disappears.
- **Per-user rate limiting.** Needed before real money, not before a pilot.

## Verify it

```bash
# no token -> 401 with a WWW-Authenticate pointer
curl -i https://docs.example.com/mcp -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# discovery document, no auth needed
curl -s https://docs.example.com/.well-known/oauth-protected-resource/mcp

# with a token -> tools list
curl -s https://docs.example.com/mcp -X POST \
  -H 'Authorization: Bearer tok_alice' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# client side
claude mcp add --transport http mydocs https://docs.example.com/mcp \
  --header "Authorization: Bearer tok_alice"
```

`Authorization` is an HTTP header, so **stdio and the in-process `Client(mcp)` used in
tests never see any of this.** A test that passes locally proves nothing about auth; it
has to go over HTTP.
