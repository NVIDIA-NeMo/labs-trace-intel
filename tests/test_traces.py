"""Normalized Trace, Span, and TraceSnapshot contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from insight_agent.evidence_streams._trace import walk_spans
from insight_agent.trace_loaders import InsightTraceLoader
from insight_agent.traces import (
    UNSET,
    Span,
    SpanKind,
    TokenCounts,
    ToolCall,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)

NOW = datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc)
TOKENS = TokenCounts(input_tokens=1, cached_input_tokens=0, output_tokens=2)


def _span(
    span_id: str,
    *,
    children: list[Span] | None = None,
    seconds: int = 0,
    kind: SpanKind = SpanKind.AGENT,
    input: object = None,
    output: object = None,
    **values: object,
) -> Span:
    started = NOW + timedelta(seconds=seconds)
    return Span(
        id=span_id,
        kind=kind,
        children=children or [],
        start_time=started,
        end_time=started + timedelta(milliseconds=500),
        input=input,
        output=output,
        cost_usd=0.0,
        token_counts=TOKENS,
        model=None,
        **values,
    )


def _trace(trace_id: str, root_spans: list[Span]) -> Trace:
    return Trace(
        id=trace_id,
        root_spans=root_spans,
        aggregate=TraceAggregate(cost_usd=0.0, latency_ms=0.0, token_counts=TOKENS),
    )


def test_nested_span_topology_is_valid_and_serializable():
    trace = _trace(
        "trace-1",
        [
            _span(
                "agent-root",
                children=[
                    _span(
                        "chain",
                        seconds=1,
                        kind=SpanKind.CHAIN,
                        children=[
                            _span(
                                "agent-child",
                                seconds=2,
                                children=[_span("llm", seconds=3, kind=SpanKind.LLM)],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    dumped = trace.model_dump(mode="json")
    assert dumped["root_spans"][0]["children"][0]["kind"] == "CHAIN"
    assert [visit.span.id for visit in walk_spans(trace)] == [
        "agent-root",
        "chain",
        "agent-child",
        "llm",
    ]


def test_shared_traversal_exposes_stable_depth_first_paths():
    trace = _trace(
        "trace-paths",
        [
            _span("root-0", children=[_span("child-0"), _span("child-1")]),
            _span("root-1"),
        ],
    )

    visits = list(walk_spans(trace))
    assert [visit.index for visit in visits] == [0, 1, 2, 3]
    assert [visit.path for visit in visits] == [(0,), (0, 0), (0, 1), (1,)]
    assert visits[1].source_pointer == {
        "trace_id": "trace-paths",
        "span_id": "child-0",
        "span_path": [0, 0],
    }


def test_full_llm_messages_remain_structured():
    messages = {
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": [{"type": "text", "text": "Find it"}]},
        ]
    }
    response = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "name": "search", "arguments": {"query": "it"}}],
            }
        ]
    }
    trace = _trace(
        "trace-messages",
        [_span("llm-1", kind=SpanKind.LLM, input=messages, output=response)],
    )

    dumped = trace.model_dump(mode="json")
    assert dumped["root_spans"][0]["input"] == messages
    assert dumped["root_spans"][0]["output"] == response


def test_duplicate_span_ids_fail_loudly_across_the_tree():
    with pytest.raises(ValidationError, match="duplicate span id"):
        _trace("invalid", [_span("duplicate", children=[_span("duplicate")])])


def test_end_must_not_precede_start():
    with pytest.raises(ValidationError, match="ends before it starts"):
        Span(
            id="invalid",
            kind=SpanKind.AGENT,
            children=[],
            start_time=NOW,
            end_time=NOW - timedelta(seconds=1),
            input=None,
            output=None,
            cost_usd=0.0,
            token_counts=TOKENS,
            model=None,
        )


def test_tool_call_metadata_requires_tool_kind():
    with pytest.raises(ValidationError, match="tool_call metadata requires kind=TOOL"):
        _span("invalid", tool_call=ToolCall(call_id="call-1"))


def test_snapshot_iteration_is_stable_and_independent():
    traces = tuple(_trace(f"trace-{index}", []) for index in range(3))
    snapshot = TraceSnapshot(traces)

    first = iter(snapshot)
    second = iter(snapshot)
    assert next(first) is traces[0]
    assert next(first) is traces[1]
    assert next(second) is traces[0]
    assert [trace.id for trace in first] == ["trace-2"]
    assert [trace.id for trace in second] == ["trace-1", "trace-2"]
    assert snapshot.trace_count == 3


def test_snapshot_rejects_duplicate_trace_ids():
    with pytest.raises(ValueError, match="duplicate trace id"):
        TraceSnapshot((_trace("same", []), _trace("same", [])))


def test_timestamp_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="timestamp must include a timezone"):
        Span(
            id="s",
            kind=SpanKind.AGENT,
            children=[],
            start_time=datetime(2026, 8, 26),
            end_time=datetime(2026, 8, 26),
            input=None,
            output=None,
            cost_usd=0.0,
            token_counts=TOKENS,
            model=None,
        )


def test_input_normalization_preserves_missing_null_and_duplicate_source_ids():
    with pytest.warns(UserWarning, match=r"WARNING\[duplicate_call_id\].*duplicate-source-ids"):
        trace = next(
            iter(
                InsightTraceLoader.from_records(
                    [
                        {
                            "schema_version": "insight-trace/v1",
                            "trace_id": "duplicate-source-ids",
                            "calls": [
                                {
                                    "call_id": "duplicate",
                                    "call_index": 0,
                                    "tool_name": "search",
                                    "arguments": {"query": "first"},
                                },
                                {
                                    "call_id": "duplicate",
                                    "call_index": 1,
                                    "tool_name": "search",
                                    "arguments": {"query": "second"},
                                    "result": None,
                                },
                            ],
                        }
                    ]
                ).load()
            )
        )

    tool_spans = [visit.span for visit in walk_spans(trace) if visit.span.kind is SpanKind.TOOL]
    assert [span.id for span in tool_spans] == ["duplicate", "duplicate#2"]
    assert [span.tool_call.call_id for span in tool_spans if span.tool_call] == [
        "duplicate",
        "duplicate",
    ]
    assert [span.tool_call.result_count for span in tool_spans if span.tool_call] == [0, 1]
    assert [span.output for span in tool_spans] == [UNSET, None]
