"""Deployment notes, kept next to the code because they are only true here.

Run order matters: migrations before the app, because the new column is NOT NULL
and the old code will crash on start without it.
"""
MIGRATE_FIRST = True
SERVICES = ["qdrant", "mcp-server"]
# The index service is deliberately NOT in the list: it is a one-shot and starting
# it during a deploy will re-embed the whole corpus mid-release.
ONE_SHOT = ["index"]

ROLLBACK = "git revert, then re-run migrations in reverse. There is no down migration."

# 2026-10-14: rollback of the alerting change broke because the migration was not
# reversible. Write downs or do not write the migration.
