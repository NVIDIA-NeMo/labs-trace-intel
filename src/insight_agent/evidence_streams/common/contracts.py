"""Shared trace source boundary for evidence streams."""

from __future__ import annotations

from typing import Protocol

from insight_agent.traces import TraceSnapshot


class TraceLoader(Protocol):
    """Load one source into a normalized, reiterable trace snapshot."""

    def load(self) -> TraceSnapshot:
        """Return the configured source as normalized traces."""
        ...
