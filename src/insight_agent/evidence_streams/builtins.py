"""Construction and registration of the repository's built-in evidence streams."""

from __future__ import annotations

from insight_agent.evidence_streams.anomaly_and_patterns import (
    AnomalyAndPatternsConfig,
    AnomalyAndPatternsEvidenceStream,
)
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.evidence_streams.tool_issues import ToolIssueConfig, ToolIssueEvidenceStream

ANOMALY_AND_PATTERNS = AnomalyAndPatternsEvidenceStream.name
TOOL_ISSUES = ToolIssueEvidenceStream.name
BUILTIN_STREAM_NAMES = (ANOMALY_AND_PATTERNS, TOOL_ISSUES)


def registered_builtin_streams(
    *,
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None,
    tool_issues: ToolIssueConfig | None = None,
) -> EvidenceStreamRegistry:
    """Construct and register the built-ins with supplied typed configuration."""

    registry = EvidenceStreamRegistry()
    if anomaly_and_patterns is not None:
        registry.register(AnomalyAndPatternsEvidenceStream(config=anomaly_and_patterns))
    if tool_issues is not None:
        registry.register(ToolIssueEvidenceStream(config=tool_issues))
    return registry


__all__ = [
    "ANOMALY_AND_PATTERNS",
    "BUILTIN_STREAM_NAMES",
    "TOOL_ISSUES",
    "registered_builtin_streams",
]
