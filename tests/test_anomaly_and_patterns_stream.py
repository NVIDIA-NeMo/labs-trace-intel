"""Anomaly-and-pattern projection and evidence-stream parity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from insight_agent.evidence_streams._trace import walk_spans
from insight_agent.evidence_streams.anomaly_and_patterns import (
    AnomalyAndPatternsArtifacts,
    AnomalyAndPatternsEvidenceStream,
    extract_trace_features,
    to_ia2_trace,
)
from insight_agent.trace_loaders import FSDataLoader
from insight_agent.traces import Span, SpanKind, TokenCounts, ToolCall, Trace, TraceAggregate

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
TOKENS = TokenCounts(input_tokens=0, cached_input_tokens=0, output_tokens=0)


def test_input_normalization_preserves_every_field_ia2_uses():
    loader = FSDataLoader(CORPUS)
    trace = next(iter(loader.load()))
    projected = to_ia2_trace(trace)
    visits = tuple(walk_spans(trace))
    calls = [visit.span for visit in visits if visit.span.kind is SpanKind.TOOL]

    assert projected.trace_id == trace.id
    assert [call.call_id for call in projected.calls] == [
        span.tool_call.call_id for span in calls if span.tool_call is not None
    ]
    assert [call.arguments for call in projected.calls] == [span.input for span in calls]
    assert [call.result for call in projected.calls] == [span.output for span in calls]
    assert [step.step_type for step in projected.steps] == [
        visit.span.attributes["subtype"] for visit in visits
    ]
    assert [step.content for step in projected.steps] == [
        visit.span.attributes["summary"] for visit in visits
    ]
    assert projected.source_pointer == trace.attributes["source_pointer"]
    assert projected.observed_verdict == trace.attributes["observed_verdict"]
    assert projected.cost == trace.aggregate.cost_usd
    assert projected.metrics == trace.attributes["metrics"]


def test_tool_calls_are_a_valid_trajectory_when_no_other_spans_exist():
    trace = Trace(
        id="no-steps",
        root_spans=[
            Span(
                id="call-1",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={"query": "report"},
                output={"content": "found"},
                tool_call=ToolCall(call_id="call-1", index=0),
            )
        ],
        aggregate=TraceAggregate(),
    )
    projected = to_ia2_trace(trace)
    assert len(projected.calls) == 1
    assert [(step.step_type, step.name) for step in projected.steps] == [("tool", "search")]


def test_returned_data_is_projected_into_mapping_results():
    trace = Trace(
        id="returned-data",
        root_spans=[
            Span(
                id="call-1",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={},
                output={"content": "no matches"},
                tool_call=ToolCall(call_id="call-1", returned_data=False),
            )
        ],
        aggregate=TraceAggregate(),
    )

    projected = to_ia2_trace(trace)

    assert projected.calls[0].result == {"content": "no matches", "returned_data": False}
    assert extract_trace_features(projected).features.numeric["returned_data_false_rate"] == 1.0


def test_returned_data_does_not_rewrite_string_results():
    trace = Trace(
        id="string-result",
        root_spans=[
            Span(
                id="call-1",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={},
                output="no matches",
                tool_call=ToolCall(call_id="call-1", returned_data=False),
            )
        ],
        aggregate=TraceAggregate(),
    )

    with pytest.warns(UserWarning, match="result is not a JSON object"):
        projected = to_ia2_trace(trace)

    assert projected.calls[0].result == "no matches"


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
    loader = FSDataLoader(CORPUS)
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
    assert [trace.id for trace in snapshot] == list(snapshot.traces_by_id)
