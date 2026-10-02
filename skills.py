#!/usr/bin/env python3
"""Skill discovery: reads `skills/<name>/SKILL.md`.

Follows the Agent Skills open format (https://agentskills.io): YAML frontmatter
followed by a markdown playbook. Only `name` and `description` are read at startup;
the body is loaded when a skill is invoked, so a long playbook costs nothing until
something needs it.

The frontmatter parser is deliberately tiny rather than a yaml dependency. It handles
`key: value` plus indented continuation lines, which is all the spec's own examples
use. Anything richer belongs in a real parser.
"""
import os

SKILLS_DIR = os.getenv("SKILLS_DIR", "skills")


def _parse_frontmatter(text):
    """Return (fields, body). A file with no `---` header is all body."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw, body = text[3:end], text[end + 4 :].lstrip("\n")

    fields, key = {}, None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1] not in (" ", "\t") and ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            fields[key] = value.strip("'\"")
        elif key:  # indented continuation, e.g. a wrapped description
            fields[key] = (fields[key] + " " + line.strip()).strip()
    return fields, body


def list_skills(docs_dir=None):
    """[{name, description, path}] sorted by name. Bad files are reported, not fatal."""
    root = docs_dir or SKILLS_DIR
    if not os.path.isdir(root):
        return []

    found = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry, "SKILL.md")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                fields, _ = _parse_frontmatter(f.read())
        except OSError as e:
            print(f"  skipped skill {entry!r}: {e}")
            continue
        found.append(
            {
                "name": fields.get("name") or entry,
                "description": fields.get("description", ""),
                "path": path,
            }
        )
    return found


def load(name, docs_dir=None):
    """The playbook body of one skill, frontmatter stripped. None if not found."""
    for skill in list_skills(docs_dir):
        if skill["name"] == name:
            with open(skill["path"]) as f:
                _, body = _parse_frontmatter(f.read())
            return body
    return None


def catalog(docs_dir=None):
    """Compact name+description block for the system prompt."""
    found = list_skills(docs_dir)
    if not found:
        return ""
    lines = "\n".join(f"- {s['name']}: {s['description']}" for s in found)
    return (
        "\n\nPlaybooks you can load when a task matches one "
        "(load_skill(name), or ask the user to type /<name>):\n" + lines
    )
