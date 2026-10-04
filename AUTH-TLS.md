# Auth + TLS

Design and working notes. **Auth, the hostname, and the LAN exposure fix are implemented
and verified.** What remains is the auth *model*, not the transport: a static token table
with no expiry, and no per-tool scopes.

The current shape:

| | |
| --- | --- |
| Public name | `https://sakshams-macbook-air.tailf61e07.ts.net/mcp` |
| Certificate | Let's Encrypt, auto-renewed, 90 days |
| Transport | Tailscale Funnel; TLS terminates at the edge |
| Origin | plain HTTP on `127.0.0.1:8000`, no LAN route |
| Auth | static bearer tokens, `docs:read` |

## Why this was needed

Originally, anything that could reach port 8000 got full access. Verified, not assumed:

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

Docker now publishes `127.0.0.1:8000:8000` — loopback only. The public path is the
Tailscale Funnel, which dials loopback from the host, so the LAN has no route in.

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

#### Stage 1 — encrypted localhost (done, then retired)

`mkcert` generates a local Certificate Authority and installs it in the macOS keychain,
so everything on that machine trusts it automatically. Real encryption, no domain, no
DNS, no purchase.

```bash
brew install mkcert && mkcert -install && mkcert localhost
```

**Retired.** The Funnel holds a real public certificate, so there is one URL for local
and remote clients and no CA to install. `MCP_TLS_CERT` / `MCP_TLS_KEY` are now unset and
the origin is plain HTTP on loopback. Set both to bring this path back for a pure-local
setup with no tunnel — the code still supports it. The handshake below is kept because it
is the clearest statement of what the TLS path actually does.

#### What the handshake actually looks like

Measured while this path was live, with `openssl s_client -connect localhost:8443`. It is
kept as the clearest statement of what the in-container TLS branch does — which still
exists, and is what you get back by setting `MCP_TLS_CERT` and `MCP_TLS_KEY`. The
8443 mapping itself is gone; the current public path is in **Stage 2** below.
This is the real exchange, not a textbook sketch:

```
Client                                                     Server
  │                                                            │
  │ 1. TCP connect  ─────────────────────────────────────────►│
  │                                                            │
  │ 2. ClientHello                                            │  "I speak TLS 1.3, and
  │    + SNI: localhost                                       │   here is the name I
  │    + ALPN: h2, http/1.1                                   │   think I'm dialling"
  │◄─────────────────────────────────────────────────────────│
  │                                                            │
  │                              ServerHello                   │  chosen: TLSv1.3
  │                              Certificate  ────────────────►│  "I am localhost"
  │                              CertificateFinished           │  "and I hold the key"
  │◄─────────────────────────────────────────────────────────│
  │                                                            │
  │ 3. Validate:                                              │
  │    · signed by a CA in my trust store?  → mkcert CA, yes  │  the keychain, from
  │    · does it name "localhost"?          → yes             │  mkcert -install
  │    · not expired?                       → Jan 2029        │
  │                                                            │
  │ 4. Derive session keys, Finished ─────────────────────────►│
  │                                                            │
  │ 5. Everything below is now ENCRYPTED ─────────────────────►│
  │    HTTP request + Authorization: Bearer <token>            │
```

| | |
| --- | --- |
| Protocol | TLSv1.3 |
| Certificate | `O=mkcert development certificate, OU=<you>` |
| Issuer | `mkcert development CA` |
| Expiry | 2029 |

Two details that surprise people:

- **The server sends only its own certificate, not the CA.** `openssl s_client` reports
  `Verify return code: 21 (unable to verify the first certificate)` and that is
  *correct* — the client already trusts the CA through the keychain, so the chain does not
  need sending. `curl` and `httpx2` both verify, because they consult the OS store.
- **Where TLS terminates is a deployment choice, not a property of the server.** It used to
  be inside the container, with Docker mapping host `8443` to container `8000` and uvicorn
  doing the handshake. The bearer token in step 5 was encrypted before it left the
  container — but that left a *second*, plain-HTTP connection on host port `8000` with no
  TLS at all, which is exactly why a published port was a way around the encryption.

  With the Funnel, TLS terminates at Tailscale and the origin is plain HTTP — safe only
  because it is published as `127.0.0.1:8000` and has no route off the machine. The lesson
  is not "put TLS in the container"; it is that **a listener is an exposure regardless of
  what protects the traffic reaching it.**

Two implementation details that mattered:

- **`MCPServer.run()` does not expose SSL options.** It builds its own
  `uvicorn.Config` with host, port and log level only. Serving TLS means calling
  `mcp.streamable_http_app(...)` and handing the returned Starlette app to
  `uvicorn.run()` with `ssl_certfile` / `ssl_keyfile`. Both paths stay available: no cert
  configured means today's behaviour, unchanged.
- **The MCP client validates through `httpx2`, which trusts the OS store via
  `truststore`, not certifi.** So the mkcert CA is picked up with no client change —
  but only once `mkcert -install` has actually run. Verified rather than assumed: before
  the CA was installed the default path failed with *"certificate is not trusted"*, and
  passed an explicit `verify=` only. After installing, the default path returns 200.

What this buys: loopback traffic is already unreachable from the network, so the
realistic exposure is *other processes on this machine*. Useful, but it does not change
the answer for anyone else connecting.

#### Stage 2 — a real hostname (done)

Stage 1 could never be enough: the mkcert CA exists only on this machine, so any other
client got an untrusted certificate.

The options were a public CA, a tunnel, or distributing your own CA. **A tunnel won**,
because TLS is terminated before traffic reaches the container, which also made stage 1
unnecessary.

```bash
brew install tailscale
sudo brew services start tailscale      # RunAtLoad + KeepAlive
tailscale up                            # browser login
tailscale funnel --bg --https=443 http://127.0.0.1:8000
```

**Free on every Tailscale plan, including Personal at $0** — that decided it. Cloudflare's
named tunnel needs a domain (~$10/yr) and its quick tunnel hands out a *random hostname
that changes on every restart*, which is unusable as a token resource.

Two things cost time:

- **The port is a flag, not a positional.** `--https=443`, not `funnel --bg 443 http://…`.
- **It stores nothing until you click a link.** The first run prints
  `login.tailscale.com/f/funnel?node=…` and `funnel status` keeps reporting
  `No serve config` until that page is visited.

On macOS this needs the Homebrew CLI — the App Store app does not ship the binary Funnel
requires.

**The `Caddyfile` was deleted.** It only ever applied to the public-CA path.

##### What the tunnel verified

```
subject=CN=sakshams-macbook-air.tailf61e07.ts.net
issuer=C=US, O=Let's Encrypt, CN=YE1
notBefore=Oct  4 14:57:07 2026 GMT
notAfter=Jan  2 14:57:06 2027 GMT
```

| Check | Result |
| --- | --- |
| Real certificate for the public name | Let's Encrypt, 90 days, auto-renewed |
| No token | `401` |
| Valid token | all five tools |
| Discovery document | `resource` matches `MCP_RESOURCE_URL` exactly |
| `answer_docs` over the tunnel | **3.3s** warm, grounded answer, score 0.74 |
| Does the Funnel preserve `Host`? | **Yes** — proved by allowing only the public name |
| Direct loopback with a valid token | `421`, DNS-rebinding protection working |
| Survives `brew services restart` | **Yes** — config is daemon state, not a process |

That `answer_docs` number is the one worth having. The relay did not truncate a
multi-second generation, which was the open risk.

##### The URL is stable — with one dependency

`LocalHostName` gives the first label, the tailnet fixes the rest:

```
sakshams-macbook-air . tailf61e07 . ts.net
└── this Mac's name     └── fixed at tailnet creation
```

Renaming the Mac changes the URL, which breaks every client *and* invalidates tokens
issued for the old resource URL, because `MCP_RESOURCE_URL` names the resource.

Whichever route is ever used, `MCP_RESOURCE_URL` must be the exact https URL clients
connect to: it names which resource a token is for, and where discovery lives.

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
| 3 | Encrypted localhost via `mkcert` | **done** — TLS served by uvicorn from the ASGI app |
| 4 | Unpublish port 8000, or bind to `127.0.0.1` | **done** — `127.0.0.1:8000:8000`; `192.168.1.3:8000` refuses |
| 5 | A hostname + real certificate | **done** — Tailscale Funnel, Let's Encrypt cert |
| 6 | `get_access_token()` → tenant resolution | blocks multi-user |
| 7 | `doc_id` replacing path keys | **done** — relative to the docs directory |
| 8 | Scope checks per tool — split read from write | no |

Steps 1–5 leave the server safe on one machine and reachable by name. Steps 6 and 8 are
what make it safe for many users. Two of the four original problems were only ever
*partly* fixed and should not be read as closed: `refresh_index` is still a mutation
guarded only by a scope that every current token holds, and the token table still has no
expiry.

## What is deliberately not planned

- **A login UI.** The resource server does not sign anyone in; that is the
  authorization server's job.
- **Billing.** Step 8 in `ENHANCEMENTS.md`: let users bring their own inference key, so
  hosting is retrieval only and the abuse vector mostly disappears.
- **Per-user rate limiting.** Needed before real money, not before a pilot.

## Verify it

Ask for a token by subject, not by position, then use the real client — Streamable HTTP
needs an `initialize` handshake before `tools/list`, so a hand-written `curl` returns
`Missing session ID` and looks like a failure when it is not.

```bash
export MCP_BEARER_TOKEN=$(../.venv/bin/python3 -c "
import json; t = json.load(open('tokens.json'))
print(next(k for k, v in t.items() if v['subject'] == 'saksham'))")
export MCP_URL=https://sakshams-macbook-air.tailf61e07.ts.net/mcp

# no token -> 401 with a WWW-Authenticate pointer
curl -s -o /dev/null -w '%{http_code}\n' -X POST $MCP_URL \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# discovery, no auth needed; `resource` must equal MCP_RESOURCE_URL exactly
curl -s $HOST/.well-known/oauth-protected-resource/mcp

# the real path: handshake, tools, and a grounded answer
MCP_BEARER_TOKEN=$MCP_BEARER_TOKEN ../.venv/bin/python3 -c "
import anyio, mcp_client, time, os
from mcp import Client
async def main():
    async with Client(mcp_client.transport_for(os.environ['MCP_URL'])) as c:
        print([t.name for t in (await c.list_tools()).tools])
        t0 = time.perf_counter()
        r = await c.call_tool('answer_docs', {'query': 'how long is the hotel booked for?'})
        print(f'{time.perf_counter() - t0:.1f}s', r.content[0].text[:80])
anyio.run(main)"
```

Also worth checking after a reboot, since the Funnel config lives in daemon state rather
than in a process:

```bash
sudo brew services restart tailscale && sleep 5
tailscale funnel status          # config still there?
```

`curl` to `192.168.1.3:8000` must refuse — that is the loopback binding doing its job.

`Authorization` is an HTTP header, so **stdio and the in-process `Client(mcp)` used in
tests never see any of this.** A test that passes locally proves nothing about auth; it
has to go over HTTP.
