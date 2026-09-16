# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings
from unittest.mock import Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

from insight_agent.cli.main import _run_evidence_streams
from insight_agent.config import EvidenceStreamsConfig
from insight_agent.evidence_streams.anomaly_and_patterns.stream import (
    NormalizedCall,
    NormalizedTrace,
    run_anomaly_and_patterns,
)
from insight_agent.evidence_streams.user_sentiment import stream as sentiment
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.mark.parametrize("count,patterns", [(0, 1), (2, 2), (10, 1), (10, 2)])
def test_clustering_handles_small_and_repetitive_corpora(count, patterns):
    traces = [
        NormalizedTrace(str(i), [NormalizedCall(str(i), 0, f"tool-{i % patterns}")])
        for i in range(count)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = run_anomaly_and_patterns(traces)
    groups = result.trajectory_groups
    assert groups is not None
    if count >= 3 and patterns > 1:
        assert groups["selected_k"] == 2
        assert len(groups["assignments"]) == count
    else:
        assert groups["status"] == "not_evaluable"


def test_missing_prerequisites_skip_analysis_without_llm_calls(monkeypatch):
    monkeypatch.setattr(
        sentiment, "validate_embedding_dependencies", Mock(side_effect=ValueError("unavailable"))
    )
    snapshot = TraceSnapshot([Trace(id="one", root_spans=[], aggregate=TraceAggregate())])
    llm = FakeLLMClient()
    progress = []
    results = asyncio.run(
        _run_evidence_streams(EvidenceStreamsConfig(), snapshot, lambda: llm, progress.append)
    )
    assert {item.stream_name for item in results if item.skip_reason} == {
        "tool-issues",
        "ethos-divergence",
        "eval-failure-patterns",
        "user-sentiment",
    }
    assert set().union(*map(set, progress)) == {"anomaly-and-patterns"}
    assert llm.call_count == 0
