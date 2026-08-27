"""Venue-specific configuration shared by evidence streams.

A "venue" is one source of traces. The research code was measured against a
handful of internal venues, and a few of their local conventions ended up as
literals in the algorithms: the tool that runs code was called
``CodeExecutionTool``, one tool answered with ``[NO_ACTIVE_SESSION]``, and the
trajectory vocabulary used ``planning`` / ``agent`` / ``evaluation``. None of
that generalises, but all of it is *measured* behaviour that must not change by
accident.

So every such literal moves here, and every default is exactly the value that
was hardcoded before. ``DEFAULT_PROFILE`` reproduces the original behaviour
byte for byte; a venue that names things differently supplies its own profile
instead of patching the algorithms.

This module imports nothing from the engines, so there are no cycles: the
engines import ``venue``, never the other way around.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_AGENT_STEP_TYPES",
    "DEFAULT_CODE_EXECUTION_TOOLS",
    "DEFAULT_PROFILE",
    "DEFAULT_STATE_PATTERNS",
    "VenueProfile",
    "load_profile",
]

# Defaults used by the tool-issue evidence stream. These detect a tool
# refusing to act because a precondition was not met. They are English-only and
# the first one is openly venue-specific, which is exactly why they are
# overridable.
DEFAULT_STATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "no_active_session",
        r"(?i)(?:\[NO_ACTIVE_SESSION\]|no active [a-z0-9 _-]{1,40}session)",
    ),
    (
        "initialize_before_use",
        r"(?i)(?:not initialized|please .{0,80}\bbefore (?:running|using|calling))",
    ),
    (
        "required_type_or_target",
        r"(?i)(?:must be (?:a|an|the) [a-z][a-z0-9 _-]{1,50}"
        r"|cannot be created without (?:a|an|the) [a-z][a-z0-9 _-]{1,50})",
    ),
    (
        "required_first_step",
        r"(?i)(?:must|need(?:s)? to|require(?:s|d)?) .{0,100}\bfirst\b",
    ),
)

DEFAULT_CODE_EXECUTION_TOOLS = frozenset({"CodeExecutionTool"})
DEFAULT_AGENT_STEP_TYPES = frozenset({"agent", "agent_step", "planning"})


@lru_cache(maxsize=32)
def _compile(patterns: tuple[tuple[str, str], ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((name, re.compile(source)) for name, source in patterns)


@dataclass(frozen=True)
class VenueProfile:
    """Names and conventions that vary between trace sources.

    Every field defaults to the value the research code hardcoded, so
    ``VenueProfile()`` is behaviour-preserving.
    """

    name: str = "default"

    #: Tools whose output may be decoded as a Python traceback, and whose share
    #: of the call budget becomes the ``code_execution_share`` feature. Gating
    #: on the tool name is what stops an ordinary tool that happens to echo a
    #: traceback from being scored as a code-execution failure.
    code_execution_tools: frozenset[str] = DEFAULT_CODE_EXECUTION_TOOLS

    #: Step types whose ``content`` counts as agent reasoning.
    agent_step_types: frozenset[str] = DEFAULT_AGENT_STEP_TYPES

    #: Step type marking a terminal evaluation. Collapsed to a single token
    #: during trajectory grouping so that per-run verdict text does not
    #: fragment the clusters.
    evaluation_step_type: str = "evaluation"

    #: Key inside a tool result mapping whose ``False`` value means "the call
    #: succeeded but came back empty".
    returned_data_key: str = "returned_data"

    #: ``(mechanism_name, regex_source)`` pairs for prerequisite/state refusals.
    state_patterns: tuple[tuple[str, str], ...] = DEFAULT_STATE_PATTERNS

    #: Extra free-form notes, carried into run metadata for provenance.
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise container types so profiles built from JSON compare and
        # hash exactly like profiles built in Python.
        object.__setattr__(self, "code_execution_tools", frozenset(self.code_execution_tools))
        object.__setattr__(self, "agent_step_types", frozenset(self.agent_step_types))
        object.__setattr__(
            self, "state_patterns", tuple((str(n), str(p)) for n, p in self.state_patterns)
        )
        for name, source in self.state_patterns:
            try:
                re.compile(source)
            except re.error as exc:
                raise ValueError(
                    f"venue profile state pattern {name!r} is not a valid regex: {exc}"
                ) from exc

    def compiled_state_patterns(self) -> tuple[tuple[str, re.Pattern[str]], ...]:
        """Compiled ``state_patterns``, cached across calls."""
        return _compile(self.state_patterns)

    def is_code_execution_tool(self, tool_name: str) -> bool:
        return tool_name in self.code_execution_tools

    def with_overrides(self, **changes: Any) -> VenueProfile:
        return replace(self, **changes)

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code_execution_tools": sorted(self.code_execution_tools),
            "agent_step_types": sorted(self.agent_step_types),
            "evaluation_step_type": self.evaluation_step_type,
            "returned_data_key": self.returned_data_key,
            "state_patterns": [list(pair) for pair in self.state_patterns],
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VenueProfile:
        known = {
            "name",
            "code_execution_tools",
            "agent_step_types",
            "evaluation_step_type",
            "returned_data_key",
            "state_patterns",
            "notes",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown venue profile field(s): {', '.join(unknown)}. "
                f"Known fields: {', '.join(sorted(known))}"
            )
        kwargs: dict[str, Any] = {}
        if "name" in data:
            kwargs["name"] = str(data["name"])
        if "code_execution_tools" in data:
            kwargs["code_execution_tools"] = frozenset(data["code_execution_tools"])
        if "agent_step_types" in data:
            kwargs["agent_step_types"] = frozenset(data["agent_step_types"])
        if "evaluation_step_type" in data:
            kwargs["evaluation_step_type"] = str(data["evaluation_step_type"])
        if "returned_data_key" in data:
            kwargs["returned_data_key"] = str(data["returned_data_key"])
        if "state_patterns" in data:
            kwargs["state_patterns"] = tuple(
                (str(pair[0]), str(pair[1])) for pair in data["state_patterns"]
            )
        if "notes" in data:
            kwargs["notes"] = dict(data["notes"])
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


DEFAULT_PROFILE = VenueProfile()


def load_profile(source: str | Path | Mapping[str, Any] | None) -> VenueProfile:
    """Load a venue profile from a path, a mapping, or ``None``.

    ``None`` yields ``DEFAULT_PROFILE``, so callers can pass an optional
    ``--profile`` argument straight through.
    """

    if source is None:
        return DEFAULT_PROFILE
    if isinstance(source, VenueProfile):
        return source
    if isinstance(source, Mapping):
        return VenueProfile.from_dict(source)
    path = Path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"venue profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"venue profile {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"venue profile {path} must contain a JSON object")
    return VenueProfile.from_dict(data)
