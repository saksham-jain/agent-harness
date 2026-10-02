---
name: commit
description: Write and create a git commit for the current changes. Use when the user asks to commit, or asks for a commit message.
---

Run `git status --short` and `git diff` to see what changed. Include untracked files in
the review so nothing gets committed blind.

Write the commit message to a temp file with `git commit -F`, never `-m` inline, so
multi-line bodies survive:

- Subject: imperative mood, under 72 characters, no trailing period.
- Body: explain WHY, not what. Wrap at 72 columns. Skip it entirely if the subject
  says everything.
- List files by bare path (`corpus.py`), not the full path.

Before committing, stop and report instead of committing if any of these are staged:

- anything that looks like a credential, key, or `.env` file
- caches, virtualenvs, or build output that should be gitignored instead
- a large binary with no obvious reason to be tracked

Add missing ignores to `.gitignore` rather than committing them. Never use
`git commit -a`, `git push`, or amend a commit unless the user asks.

After committing, report the commit hash and subject, and mention anything left
unstaged.
