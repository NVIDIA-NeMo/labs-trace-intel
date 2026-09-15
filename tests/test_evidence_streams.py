# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import warnings

import pytest
from nooa.unifiedllm import FakeLLMClient

from insight_agent.cli.main import _run_evidence_streams
from insight_agent.config import EvidenceStreamsConfig
from insight_agent.evidence_streams.anomaly_and_patterns.stream import (
    AnomalyAndPatternsEvidenceStream,
    TraceFeatures,
    group_trajectories,
)
from insight_agent.evidence_streams.tool_issues.stream import ToolIssueEvidenceStream
from insight_agent.evidence_streams.user_sentiment import stream as sentiment
from insight_agent.traces import Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


def snapshot(count, patterns):
    return TraceSnapshot(
        [
            Trace(
                id=str(i),
                aggregate=TraceAggregate(),
                root_spans=[
                    Span(
                        id=f"call-{i}",
                        kind=SpanKind.TOOL,
                        tool_name=f"tool-{i % patterns}",
                        input={},
                        output={},
                    )
                ],
            )
            for i in range(count)
        ]
    )


@pytest.mark.parametrize("count,patterns", [(0, 1), (1, 1), (2, 2), (10, 1), (10, 2)])
def test_clustering_handles_small_and_duplicate_corpora_without_warnings(count, patterns):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = asyncio.run(AnomalyAndPatternsEvidenceStream().analyze(snapshot(count, patterns)))
    groups = result.artifacts.result.trajectory_groups
    if count >= 3 and patterns > 1:
        assert groups["selected_k"] == 2
        assert len(groups["clusters"]) == 2
        assert len(groups["assignments"]) == count
        assert not result.skipped_checks
    else:
        assert groups["status"] == "not_evaluable"
        assert result.skipped_checks
    assert result.problems == ()


def test_clustering_counts_vectors_after_feature_selection():
    records = [
        TraceFeatures(trace_id=str(i), numeric={}, sequence_tokens=("shared", f"unique-{i}"))
        for i in range(10)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = group_trajectories(records, max_features=1, on_insufficient_traces="skip")
    assert result["status"] == "not_evaluable"
    assert "identical trajectories" in result["reason"]


def test_defaults_report_missing_prerequisites_without_llm_calls(monkeypatch):
    def missing_embeddings():
        raise ValueError("missing optional dependencies")

    monkeypatch.setattr(sentiment, "validate_embedding_dependencies", missing_embeddings)
    llms = []

    def llm_factory():
        llm = FakeLLMClient()
        llms.append(llm)
        return llm

    results = asyncio.run(
        _run_evidence_streams(EvidenceStreamsConfig(), snapshot(10, 2), llm_factory)
    )
    assert len(results) == 5
    skipped = {result.stream_name: result.skipped_checks for result in results}
    assert skipped["ethos-divergence"] == ("no ethos document",)
    assert skipped["eval-failure-patterns"] == ("no evaluator results",)
    assert "embedding backend unavailable" in skipped["user-sentiment"][0]
    assert all(llm.call_count == 0 for llm in llms)


def test_tool_observations_survive_candidate_filtering_and_missing_tool_names():
    traces = TraceSnapshot(
        [
            Trace(
                id="trace-1",
                aggregate=TraceAggregate(),
                root_spans=[
                    Span(id="call", kind=SpanKind.TOOL, input="invalid arguments", error="timeout")
                ],
            )
        ]
    )
    result = asyncio.run(ToolIssueEvidenceStream().analyze(traces))
    assert result.finding_count > 0
    assert result.problems == ()
    assert "1 of 1 tool calls lack usable results" in result.limited_checks
    assert "1 of 1 traces lack tool schemas" in result.limited_checks
