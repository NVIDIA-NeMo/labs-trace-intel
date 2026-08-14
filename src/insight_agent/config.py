"""Environment configuration for the Analyst stage.

The Analyst is the only component that needs credentials, so this is the only
place that reads them. Values are resolved in one order, most explicit first:

1. an explicit CLI flag
2. a variable already in the process environment
3. a variable in a ``.env`` file next to the corpus, the working directory, or
   the repo root

``.env`` is gitignored; ``.env.example`` is the committed template. Nothing here
imports litellm, and nothing fails when the file is absent — a fully
deterministic run never touches this module.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ENV_API_BASE",
    "ENV_API_KEY",
    "ENV_MODEL",
    "find_dotenv",
    "load_dotenv",
    "resolve",
]

#: Repo-specific names, so a `.env` can carry Analyst settings without
#: colliding with whatever provider variables litellm resolves on its own.
ENV_MODEL = "INSIGHT_AGENT_MODEL"
ENV_API_BASE = "INSIGHT_AGENT_API_BASE"
ENV_API_KEY = "INSIGHT_AGENT_API_KEY"

#: Fallbacks, so an environment already configured for litellm's OpenAI-
#: compatible path keeps working without a repo-specific `.env`.
_FALLBACKS = {
    ENV_API_BASE: ("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    ENV_API_KEY: ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
}


def _parse(text: str) -> dict[str, str]:
    """Minimal ``.env`` parser.

    Deliberately dependency-free: python-dotenv arrives with litellm, but the
    deterministic half of this package must not acquire a dependency through
    the back door. Handles ``KEY=value``, ``export KEY=value``, ``#`` comments,
    blank lines, and single or double quoted values.
    """

    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def find_dotenv(start: Path | str | None = None) -> Path | None:
    """Locate a ``.env``: beside ``start``, then the cwd, then upward to a repo root."""

    candidates: list[Path] = []
    if start is not None:
        start = Path(start)
        candidates.append((start.parent if start.is_file() else start) / ".env")

    here = Path.cwd()
    candidates.append(here / ".env")
    for parent in here.parents:
        candidates.append(parent / ".env")
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load a ``.env`` into ``os.environ`` and return what it set.

    The real process environment wins by default: a variable already exported
    is not silently replaced by a stale file.
    """

    resolved = Path(path) if path is not None else find_dotenv()
    if resolved is None or not resolved.is_file():
        return {}

    applied: dict[str, str] = {}
    for key, value in _parse(resolved.read_text(encoding="utf-8")).items():
        if not value:
            # An unfilled template line. Leaving it out means a real
            # environment variable of the same name still applies.
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def resolve(name: str, default: str | None = None) -> str | None:
    """Read one setting, falling back to the provider-standard variables."""

    value = os.environ.get(name)
    if value:
        return value
    for fallback in _FALLBACKS.get(name, ()):
        value = os.environ.get(fallback)
        if value:
            return value
    return default
