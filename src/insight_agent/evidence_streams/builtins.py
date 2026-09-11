# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction and registration of the repository's built-in evidence streams."""

from __future__ import annotations

from collections.abc import Callable

from nooa.unifiedllm import UnifiedLLM

from insight_agent.evidence_streams.anomaly_and_patterns.stream import (
    AnomalyAndPatternsConfig,
    AnomalyAndPatternsEvidenceStream,
)
from insight_agent.evidence_streams.ethos_divergence.ethos_divergence_detector import (
    EthosDivergenceConfig,
    EthosDivergenceEvidenceStream,
)
from insight_agent.evidence_streams.eval_failure_patterns import (
    EvalFailurePatternsConfig,
    EvalFailurePatternsEvidenceStream,
)
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.evidence_streams.tool_issues.stream import (
    ToolIssueConfig,
    ToolIssueEvidenceStream,
)
from insight_agent.evidence_streams.user_dissatisfaction.stream import (
    UserDissatisfactionConfig,
    UserDissatisfactionEvidenceStream,
)

USER_DISSATISFACTION = UserDissatisfactionEvidenceStream.name
ANOMALY_AND_PATTERNS = AnomalyAndPatternsEvidenceStream.name
TOOL_ISSUES = ToolIssueEvidenceStream.name
ETHOS_DIVERGENCE = EthosDivergenceEvidenceStream.name
EVAL_FAILURE_PATTERNS = EvalFailurePatternsEvidenceStream.name
BUILTIN_STREAM_NAMES = (
    ANOMALY_AND_PATTERNS,
    TOOL_ISSUES,
    ETHOS_DIVERGENCE,
    EVAL_FAILURE_PATTERNS,
    USER_DISSATISFACTION,
)


def registered_builtin_streams(
    *,
    anomaly_and_patterns: AnomalyAndPatternsConfig | None = None,
    tool_issues: ToolIssueConfig | None = None,
    ethos_divergence: EthosDivergenceConfig | None = None,
    eval_failure_patterns: EvalFailurePatternsConfig | None = None,
    user_dissatisfaction: UserDissatisfactionConfig | None = None,
    llm_factory: Callable[[], UnifiedLLM] | None = None,
) -> EvidenceStreamRegistry:
    """Construct and register the built-ins with supplied typed configuration."""

    registry = EvidenceStreamRegistry()
    if anomaly_and_patterns is not None:
        registry.register(AnomalyAndPatternsEvidenceStream(config=anomaly_and_patterns))
    if tool_issues is not None:
        registry.register(ToolIssueEvidenceStream(config=tool_issues))
    if ethos_divergence is not None:
        if llm_factory is None:
            raise ValueError("ethos-divergence requires an LLM client factory")
        registry.register(EthosDivergenceEvidenceStream(config=ethos_divergence, llm=llm_factory()))
    if eval_failure_patterns is not None:
        if llm_factory is None:
            raise ValueError("eval-failure-patterns requires an LLM client factory")
        registry.register(
            EvalFailurePatternsEvidenceStream(llm=llm_factory(), config=eval_failure_patterns)
        )
    if user_dissatisfaction is not None:
        if llm_factory is None:
            raise ValueError("user-dissatisfaction requires an LLM client factory")
        registry.register(
            UserDissatisfactionEvidenceStream(config=user_dissatisfaction, llm=llm_factory())
        )
    return registry


__all__ = [
    "ANOMALY_AND_PATTERNS",
    "BUILTIN_STREAM_NAMES",
    "ETHOS_DIVERGENCE",
    "EVAL_FAILURE_PATTERNS",
    "TOOL_ISSUES",
    "USER_DISSATISFACTION",
    "registered_builtin_streams",
]
