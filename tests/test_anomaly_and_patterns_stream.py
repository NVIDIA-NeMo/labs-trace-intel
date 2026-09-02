"""IA2 projection and evidence-stream parity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from insight_agent.evidence_streams.anomaly_and_patterns import (
    AnomalyAndPatternsArtifacts,
    AnomalyAndPatternsEvidenceStream,
    to_ia2_trace,
)
from insight_agent.trace_loaders import InsightTraceLoader
from insight_agent.traces import Span, SpanKind, TokenCounts, Trace, TraceAggregate

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
DUPLICATE_CALL_ID_WARNING = r"WARNING\[duplicate_call_id\].*docops-instrumentation"
NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
TOKENS = TokenCounts(input_tokens=0, cached_input_tokens=0, output_tokens=0)


def test_input_normalization_preserves_every_field_ia2_uses():
    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        loader = InsightTraceLoader.from_path(CORPUS)
    record = loader.records[0]
    trace = next(iter(loader.load()))
    projected = to_ia2_trace(trace)

    assert projected.trace_id == record["trace_id"]
    assert [call.call_id for call in projected.calls] == [
        call["call_id"] for call in record["calls"]
    ]
    assert [call.arguments for call in projected.calls] == [
        call["arguments"] for call in record["calls"]
    ]
    assert [call.result for call in projected.calls] == [call["result"] for call in record["calls"]]
    assert [step.step_type for step in projected.steps] == [
        step["step_type"] for step in record["steps"]
    ]
    assert [step.content for step in projected.steps] == [
        step.get("content", "") for step in record["steps"]
    ]
    assert projected.source_pointer == record["source_pointer"]
    assert projected.observed_verdict == record["observed_verdict"]
    assert projected.cost == record["cost"]
    assert projected.metrics == record["metrics"]


def test_tool_calls_are_a_valid_trajectory_when_no_other_spans_exist():
    record = {
        "schema_version": "insight-trace/v1",
        "trace_id": "no-steps",
        "calls": [
            {
                "call_id": "call-1",
                "call_index": 0,
                "tool_name": "search",
                "arguments": {"query": "report"},
                "result": {"content": "found"},
            }
        ],
    }

    trace = next(iter(InsightTraceLoader.from_records([record]).load()))
    projected = to_ia2_trace(trace)
    assert len(projected.calls) == 1
    assert [(step.step_type, step.name) for step in projected.steps] == [("tool", "search")]


def test_native_projection_uses_canonical_spans_for_steps_and_tool_calls():
    tool_span = Span(
        id="call-1",
        kind=SpanKind.TOOL,
        children=[],
        start_time=NOW,
        end_time=NOW,
        input={"query": "report"},
        output={"content": "found"},
        cost_usd=0.0,
        token_counts=TOKENS,
        model=None,
        tool_name="search",
    )
    llm_span = Span(
        id="llm",
        kind=SpanKind.LLM,
        children=[tool_span],
        start_time=NOW,
        end_time=NOW,
        input=None,
        output={"messages": [{"role": "assistant", "content": "calling search"}]},
        cost_usd=0.0,
        token_counts=TOKENS,
        model=None,
    )
    trace = Trace(
        id="native",
        root_spans=[
            Span(
                id="agent",
                kind=SpanKind.AGENT,
                children=[llm_span],
                start_time=NOW,
                end_time=NOW,
                input=None,
                output="planning",
                cost_usd=0.0,
                token_counts=TOKENS,
                model=None,
            )
        ],
        aggregate=TraceAggregate(cost_usd=0.25, latency_ms=0.0, token_counts=TOKENS),
    )

    projected = to_ia2_trace(trace)
    assert projected.trace_id == "native"
    assert [step.step_type for step in projected.steps] == ["agent", "llm", "tool"]
    assert [step.content for step in projected.steps] == [
        "planning",
        "calling search",
        "found",
    ]
    assert len(projected.calls) == 1
    assert projected.calls[0].call_id == "call-1"
    assert projected.calls[0].call_index == 0
    assert projected.calls[0].arguments == {"query": "report"}
    assert projected.calls[0].result == {"content": "found"}
    assert projected.calls[0].source_pointer == {
        "trace_id": "native",
        "span_id": "call-1",
        "span_path": [0, 0, 0],
    }
    assert projected.observed_verdict is None
    assert projected.metrics == {}
    assert projected.cost == 0.25


def test_ia2_stream_runs_the_engine_from_a_snapshot():
    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        loader = InsightTraceLoader.from_path(CORPUS)
    stream = AnomalyAndPatternsEvidenceStream()
    snapshot = loader.load()
    actual = stream.analyze(snapshot)

    assert isinstance(actual.artifacts, AnomalyAndPatternsArtifacts)
    assert "## Unusual traces" in actual.artifacts.result.digest
    assert "docops-outlier" in actual.artifacts.result.digest
    assert actual.problems
    assert any("statistical outliers" in problem.description for problem in actual.problems)
    assert "docops-outlier" in {
        trace_id for problem in actual.problems for trace_id in problem.supporting_trace_ids
    }
    assert [trace.id for trace in snapshot] == [record["trace_id"] for record in loader.records]
