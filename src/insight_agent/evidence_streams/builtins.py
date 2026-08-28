"""Construction and registration of the repository's built-in evidence streams."""

from __future__ import annotations

from .anomaly_and_patterns import (
    AnomalyAndPatternsConfig,
    AnomalyAndPatternsEvidenceStream,
)
from .registry import EvidenceStreamRegistry
from .tool_issues import ToolIssueConfig, ToolIssueEvidenceStream
from .venue import VenueProfile

ANOMALY_AND_PATTERNS = AnomalyAndPatternsEvidenceStream.name
TOOL_ISSUES = ToolIssueEvidenceStream.name
BUILTIN_STREAM_NAMES = (ANOMALY_AND_PATTERNS, TOOL_ISSUES)


def registered_builtin_streams(
    *,
    profile: VenueProfile,
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None,
    tool_issues: ToolIssueConfig | None = None,
) -> EvidenceStreamRegistry:
    """Construct and register the built-ins with supplied typed configuration."""

    registry = EvidenceStreamRegistry()
    if anomaly_and_patterns is not None:
        registry.register(
            AnomalyAndPatternsEvidenceStream(
                config=anomaly_and_patterns,
                profile=profile,
            )
        )
    if tool_issues is not None:
        registry.register(ToolIssueEvidenceStream(config=tool_issues, profile=profile))
    return registry


__all__ = [
    "ANOMALY_AND_PATTERNS",
    "BUILTIN_STREAM_NAMES",
    "TOOL_ISSUES",
    "registered_builtin_streams",
]
