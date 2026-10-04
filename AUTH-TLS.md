# Auth + TLS

Working notes for the `feat/auth-tls` branch. Nothing here is implemented yet.

## Why this is needed

The MCP server is meant to be reachable by MCP clients over the network. Today anything
that can reach port 8000 gets full access. Verified, not assumed:

```
no credentials of any kind -> ['answer_docs', 'list_docs', 'refresh_index', 'index_status']
list_docs returned         -> { "file": "/docs/Saksham-Jain-Resume.pdf", "chunks": 5 }
```

Zero credentials retrieved the corpus. Four concrete exposures:

| # | Problem | Consequence |
| --- | --- | --- |
| 1 | **No authentication** | Anyone reads the corpus, spends inference budget, or calls `refresh_index` and mutates Qdrant |
| 2 | **Write tool exposed** | `refresh_index` is a mutation with no auth. An unauthenticated caller can rewrite the index |
| 3 | **Plain HTTP** | Bearer tokens would cross the network in cleartext. Most MCP clients require https for a remote host |
| 4 | **DNS-rebinding protection off** | The SDK defaults `TransportSecuritySettings(enable_dns_rebinding_protection=False)` for backwards compatibility. A browser on a user's machine could be made to talk to the local instance |

Docker publishes `8000:8000` on all interfaces (`0.0.0.0`), so this is exposed as soon
as there is a port forward, a tunnel, or a public host.

Problem 2 is the one that is not obvious: **read-only is not the default here.** Two of
the four tools mutate or enumerate. Auth is not just about confidentiality.

## How TLS works, and why it needs a hostname

### The problem it solves

The server speaks plain HTTP. Every byte crosses the wire in the clear. Anyone who can
see the traffic between a client and the server can read the bearer token going past
and then use it — at which point they are the user. That is the whole problem.

### It does two things, not one

Encryption is the obvious half. The half people miss is authentication:

| | |
| --- | --- |
| **Encrypt** | nobody in the middle can read or alter the traffic |
| **Authenticate** | you are certain you are talking to the real server |

Both are required, and encryption alone is worse than useless. When you connect, the
other end offers you a key. An attacker in the middle can offer you *their* key
instead, and you would happily encrypt to them believing you were private — a secret
conversation with the wrong person. So encryption needs a way to ask: **prove you are
the server I asked for.**

### The certificate is that proof

A certificate is a signed document saying: *I am `docs.example.com`. I did not write
this myself — a Certificate Authority did, and it checked that whoever asked actually
controls that name.*

The CA is the trust anchor. The OS ships with a list of CAs it trusts, so the client
can check the signature without asking anyone.

### Why a hostname specifically

**The hostname is written into the certificate.** Not metadata — the subject of the
document. Before sending any real data, the client does this:

1. Server presents its certificate
2. Is it signed by a CA my OS trusts?
3. **Does this certificate name the host I just asked for?**
4. Not expired, not revoked?
5. All pass → encrypt and talk

Step 3 is where a hostname is mandatory. A certificate for `docs.example.com` is
worthless when you connect to `something-else.com` — that is exactly the attack it
exists to stop. An attacker cannot obtain a certificate for your hostname, nor present
one for *their* domain as though it were you.

And you cannot get one without proving you control the name. That is step 3 working in
your favour: the CA contacts the domain and checks the requester controls it. Nobody is
handed a certificate for someone else's name. **Which is why a hostname comes before a
certificate, and a certificate before HTTPS.**

### What it means here

| Question | Answer |
| --- | --- |
| Why is localhost fine? | No certificate exists and none is needed — there is no path to intercept. This is the current state. |
| Why does it break on a network? | Now there is a path, and a token crossing it in cleartext |
| Why not self-sign? | Possible, but nothing trusts it. Every client needs a manual "trust this certificate" step, and anyone who skips it gets no protection. |

### Why a tunnel solves it

Doing this yourself means: buy a name, point DNS at the machine, forward ports on the
router, keep it running, renew certs every 90 days.

A tunnel skips the hard parts. **The tunnel operator already owns a domain and already
holds the certificate** — it just carries your traffic. Same cryptography, someone else
holding the name. Not a shortcut around security; security with the operational burden
handed off.

### What actually changes

Auth is unaffected. TLS is only the outer envelope around it:

```
now:    client ──plaintext token──> mcp-server :8000    localhost only
later:  client ──encrypted──> tunnel ──> mcp-server     safe anywhere
```

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
    "tok_saksham": {"subject": "saksham", "collection": "docs_saksham"},
    "tok_bob":     {"subject": "bob",     "collection": "docs_bob"},
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

Two stages, because they solve different problems and only the first is worth doing now.

#### Stage 1 — encrypted localhost (do this next)

`mkcert` generates a local Certificate Authority and installs it in the macOS keychain,
so everything on that machine trusts it automatically. Real encryption, no domain, no
DNS, no purchase.

```bash
brew install mkcert && mkcert -install && mkcert localhost
```

Two implementation details that matter:

- **`MCPServer.run()` does not expose SSL options.** It builds its own
  `uvicorn.Config` with host, port and log level only. Serving TLS from the app means
  calling `mcp.streamable_http_app(...)` and handing the returned Starlette app to
  `uvicorn.run()` with `ssl_certfile` / `ssl_keyfile`. Both paths stay available: no cert
  configured means today's behaviour, unchanged.
- **The MCP client validates through `httpx2`, which trusts the OS store via
  `truststore`, not certifi.** So the mkcert CA should be picked up with no client
  change. If it is not — a minimal container with no CA store is the usual cause — the
  escape hatch is `SSL_CERT_FILE`/`SSL_CERT_DIR`, or passing `verify=` to the httpx2
  client that `transport_for()` already builds. Worth asserting rather than assuming.

What this actually buys: loopback traffic is already unreachable from the network, so
the realistic exposure is *other processes on this machine*. Useful, but it does not
change the answer for anyone else connecting.

#### Stage 2 — a real hostname (only when someone else connects)

Stage 1 is not a substitute once the server is reachable by anyone but you, because the
mkcert CA only exists on your machine. Other clients get an untrusted certificate.

The options, and only these three:

| Source | Cost to you |
| --- | --- |
| Public CA (Caddy/Let's Encrypt) | Own a real domain, prove control via DNS, forward ports, renew every 90 days |
| A tunnel (Tailscale Funnel, Cloudflare) | They hold the certificate, you point at their URL |
| Your own CA, distributed | Same as `mkcert` but every client must install the CA first |

A tunnel is usually the best value: a stable hostname for free, a valid certificate, and
no router to configure. Caddy is only needed on the public-CA path — with a tunnel, TLS
is already terminated before it reaches this container, so the `Caddyfile` in this repo
becomes unnecessary.

```text
:443 ──TLS──> caddy ──http──> mcp-server (127.0.0.1:8000)     public-CA path
             tunnel
:443 ──TLS───────────> mcp-server (:8000)                      tunnel path
```

Whichever route, `MCP_RESOURCE_URL` must be the exact https URL clients connect to: it
names which resource a token is for, and where discovery lives.

### 5. Host allowlist

Turn on what is currently off:

```python
transport_security=TransportSecuritySettings(allowed_hosts=["localhost:*", "127.0.0.1:*"])
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

Our own client:

```bash
export MCP_BEARER_TOKEN=...          # see above
docker compose --profile client run --rm mcp-client
```

Or probe directly, which is what a test should do:

```bash
MCP_BEARER_TOKEN=$TOKEN python3 -c "
import anyio, mcp_client
from mcp import Client
async def m():
    async with Client(mcp_client.transport_for('http://localhost:8000/mcp')) as c:
        print([t.name for t in (await c.list_tools()).tools])
anyio.run(m)"
```

Turn it off again with `MCP_AUTH=0` in `.env`.

`tokens.json` is gitignored and generated locally; `tokens.example.json` is the committed
template. Tokens are stored in plaintext — it is a lookup table, not a hash — so the file
is `chmod 600`.

## Order of work

| # | Step | State |
| --- | --- | --- |
| 1 | Static `TokenVerifier` + `AuthSettings` | **done** — 401, identity, scopes all verified |
| 2 | Host allowlist wiring | **done** — rejects `evil.example.com`; off unless hosts named |
| 3 | Encrypted localhost via `mkcert` | next, see design section 4 |
| 4 | Unpublish port 8000, or bind to `127.0.0.1` | **live exposure**, do this regardless of TLS |
| 5 | A hostname + real certificate | blocks anyone else connecting |
| 6 | `get_access_token()` → tenant resolution | blocks multi-user |
| 7 | `doc_id` replacing path keys | blocks multi-user |
| 8 | Scope checks per tool — split read from write | no |

Steps 1–3 leave the server safe for you alone on one machine. Steps 6–7 are what make it
safe for many users. Step 4 is worth doing now and on its own: a published port is a way
*around* TLS, so TLS does not close it.

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
  -H 'Authorization: Bearer tok_saksham' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# client side: MCP_BEARER_TOKEN is read by mcp_client.transport_for()
export MCP_BEARER_TOKEN=tok_saksham
python3 agent_harness_base.py
```

`Authorization` is an HTTP header, so **stdio and the in-process `Client(mcp)` used in
tests never see any of this.** A test that passes locally proves nothing about auth; it
has to go over HTTP.
