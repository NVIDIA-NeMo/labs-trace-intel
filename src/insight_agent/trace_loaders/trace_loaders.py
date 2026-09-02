"""Trace source boundary."""

from __future__ import annotations

from typing import Protocol, TypedDict

from insight_agent.traces import TraceSnapshot


class TraceDescription(TypedDict):
    """Source-independent corpus facts recorded with analysis results."""

    source: str
    trace_count: int
    call_count: int
    steps_present: bool
    steps_partially_present: bool
    distinct_logical_cases: int


class TraceLoader(Protocol):
    """Load one source into a normalized, reiterable trace snapshot."""

    def load(self) -> TraceSnapshot:
        """Return the configured source as normalized traces."""
        ...

    def describe(self) -> TraceDescription:
        """Describe the configured source and most recently loaded corpus."""
        ...
