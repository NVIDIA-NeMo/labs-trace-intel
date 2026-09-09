# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction and registration of the repository's built-in evidence streams."""

from __future__ import annotations

from nooa.unifiedllm import CompletionClient

from insight_agent.evidence_streams.anomaly_and_patterns.stream import (
    AnomalyAndPatternsConfig,
    AnomalyAndPatternsEvidenceStream,
)
from insight_agent.evidence_streams.ethos_divergence.ethos_divergence_detector import (
    EthosDivergenceConfig,
    EthosDivergenceEvidenceStream,
)
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.evidence_streams.tool_issues.stream import (
    ToolIssueConfig,
    ToolIssueEvidenceStream,
)

ANOMALY_AND_PATTERNS = AnomalyAndPatternsEvidenceStream.name
TOOL_ISSUES = ToolIssueEvidenceStream.name
ETHOS_DIVERGENCE = EthosDivergenceEvidenceStream.name
BUILTIN_STREAM_NAMES = (ANOMALY_AND_PATTERNS, TOOL_ISSUES, ETHOS_DIVERGENCE)


def registered_builtin_streams(
    *,
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None,
    tool_issues: ToolIssueConfig | None = None,
    ethos_divergence: EthosDivergenceConfig | None = None,
    llm: CompletionClient | None = None,
) -> EvidenceStreamRegistry:
    """Construct and register the built-ins with supplied typed configuration."""

    registry = EvidenceStreamRegistry()
    if anomaly_and_patterns is not None:
        registry.register(AnomalyAndPatternsEvidenceStream(config=anomaly_and_patterns))
    if tool_issues is not None:
        registry.register(ToolIssueEvidenceStream(config=tool_issues))
    if ethos_divergence is not None:
        if llm is None:
            raise ValueError("ethos-divergence requires an LLM client")
        registry.register(EthosDivergenceEvidenceStream(config=ethos_divergence, llm=llm))
    return registry


__all__ = [
    "ANOMALY_AND_PATTERNS",
    "BUILTIN_STREAM_NAMES",
    "ETHOS_DIVERGENCE",
    "TOOL_ISSUES",
    "registered_builtin_streams",
]
