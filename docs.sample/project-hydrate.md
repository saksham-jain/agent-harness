# Project Hydrate — working notes

## What it is

An internal service that predicts water-usage anomalies for 400 buildings. Replaces a
spreadsheet that two people maintained by hand.

## Status as of 14 October

- Ingestion from 11 of 12 buildings. **Aldgate Row is blocked** waiting on a meter that
  the landlord has not replaced.
- Anomaly detection works on all 11. False positive rate around **4%**.
- Nobody has run it in production for more than six hours without a crash.

## Decisions

- **SQLite, not Postgres.** 400 buildings is not large, and a single file is one fewer
  thing to page someone about at 3am.
- **Daily batch, not streaming.** The dashboard is checked once a morning. Streaming would
  be complexity bought for nothing.
- Keep the raw readings for **13 months**, because billing disputes arrive that late.

## Open questions

- What is the alerting channel? Nobody has decided between Slack and email, and both are
  wrong for someone who is not at a desk.
- Who owns this after the contractor's contract ends in March?
