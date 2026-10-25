# Home network

## Router

**192.168.1.1**, admin password is in the keychain under `router-admin`. It is the ISP
default and should be changed, which is on the list and off the list every month.

## SSIDs

- `fernside-5g` — the good one. Use this.
- `fernside-guest` — visitors, rate limited to 2 Mbps, expires at midnight.
- `fernside-legacy` — the 2.4GHz one that reaches the back bedroom. Slower but it is the
  only thing that works out there.

## Fixed IP

The NAS is pinned to **192.168.1.30** because the router kept handing the address to
something else. Do not use DHCP for it.

## Known problem

The printer drops off roughly weekly. The fix is to power-cycle the router, which is not a
fix, it is a ritual.
