"""Normalized Trace, Span, and TraceSnapshot contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from insight_agent.trace_loaders import InsightTraceLoader
from insight_agent.traces import (
    UNSET,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    TraceSnapshot,
)

NOW = datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc)


def _span(
    span_id: str,
    *,
    parent: str | None = None,
    seconds: int = 0,
    kind: SpanKind = SpanKind.AGENT,
) -> Span:
    started = NOW + timedelta(seconds=seconds)
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        kind=kind,
        status=SpanStatus.SUCCESS,
        started_at=started,
        ended_at=started + timedelta(milliseconds=500),
    )


def test_deep_flat_span_topology_is_valid_and_serializable():
    trace = Trace(
        id="trace-1",
        input=[{"role": "user", "content": "Find the report"}],
        spans=(
            _span("agent-root"),
            _span("chain", parent="agent-root", seconds=1, kind=SpanKind.CHAIN),
            _span("agent-child", parent="chain", seconds=2),
            _span("llm", parent="agent-child", seconds=3, kind=SpanKind.LLM),
        ),
    )

    assert [span.parent_span_id for span in trace.spans] == [
        None,
        "agent-root",
        "chain",
        "agent-child",
    ]
    dumped = trace.model_dump(mode="json")
    assert dumped["input"] == [{"role": "user", "content": "Find the report"}]
    assert dumped["spans"][3]["kind"] == "LLM"


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
    trace = Trace(
        id="trace-messages",
        spans=(
            Span(
                span_id="llm-1",
                kind=SpanKind.LLM,
                input=messages,
                output=response,
            ),
        ),
    )

    dumped = trace.model_dump(mode="json")
    assert dumped["spans"][0]["input"] == messages
    assert dumped["spans"][0]["output"] == response


def test_absent_and_explicit_null_are_distinct():
    absent = Span(span_id="absent", kind=SpanKind.TOOL)
    explicit_null = Span(span_id="null", kind=SpanKind.TOOL, output=None)

    assert absent.output is UNSET
    assert explicit_null.output is None
    assert "output" not in absent.model_dump(mode="json")
    assert explicit_null.model_dump(mode="json")["output"] is None


@pytest.mark.parametrize(
    ("spans", "message"),
    [
        (
            (
                _span("duplicate"),
                _span("duplicate", seconds=1),
            ),
            "duplicate span_id",
        ),
        (
            (_span("child", parent="missing"),),
            "references missing parent",
        ),
        (
            (
                _span("child", parent="parent"),
                _span("parent", seconds=1),
            ),
            "parent 'parent' must precede child 'child'",
        ),
        (
            (
                _span("a", parent="b"),
                _span("b", parent="a", seconds=1),
            ),
            "parent cycle",
        ),
    ],
)
def test_invalid_span_graphs_fail_loudly(spans, message):
    with pytest.raises(ValidationError, match=message):
        Trace(id="invalid", spans=spans)


def test_spans_must_be_in_temporal_order_when_timestamps_exist():
    with pytest.raises(ValidationError, match="canonical temporal order"):
        Trace(
            id="invalid-time",
            spans=(
                _span("later", seconds=2),
                _span("earlier", seconds=1),
            ),
        )


def test_models_reject_unknown_fields_and_field_reassignment():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Span(span_id="s", kind=SpanKind.AGENT, made_up=True)

    span = Span(span_id="s", kind=SpanKind.AGENT, source_pointer={"nested": {"value": 1}})
    with pytest.raises(ValidationError, match="Instance is frozen"):
        span.source_pointer = {}  # type: ignore[misc]


def test_snapshot_scans_are_stable_and_independent():
    traces = tuple(Trace(id=f"trace-{index}", spans=()) for index in range(3))
    snapshot = TraceSnapshot.from_traces(traces, source="fixture")

    first = snapshot.scan()
    second = snapshot.scan()
    assert next(first) is traces[0]
    assert next(first) is traces[1]
    assert next(second) is traces[0]
    assert [trace.id for trace in first] == ["trace-2"]
    assert [trace.id for trace in second] == ["trace-1", "trace-2"]
    assert snapshot.trace_count == 3
    assert snapshot.source == "fixture"


def test_snapshot_rejects_duplicate_trace_ids():
    traces = (Trace(id="same", spans=()), Trace(id="same", spans=()))
    with pytest.raises(ValidationError, match="duplicate trace id"):
        TraceSnapshot.from_traces(traces)


def test_timestamp_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="timestamp must include a timezone"):
        Span(span_id="s", kind=SpanKind.AGENT, started_at=datetime(2026, 8, 26))


def test_input_normalization_preserves_missing_null_and_duplicate_source_ids():
    with pytest.warns(UserWarning, match=r"WARNING\[duplicate_call_id\].*duplicate-source-ids"):
        trace = next(
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
                        "steps": [
                            {
                                "step_index": 0,
                                "step_type": "planning",
                                "content": "search twice",
                            },
                            {"step_index": 1, "step_type": "tool", "name": "search"},
                            {"step_index": 2, "step_type": "tool", "name": "search"},
                        ],
                        "tool_catalog": {"search": {"type": "object"}},
                    }
                ]
            )
            .load()
            .scan()
        )

    tool_spans = [span for span in trace.spans if span.kind is SpanKind.TOOL]
    assert [span.span_id for span in tool_spans] == ["duplicate", "duplicate#2"]
    assert [span.tool_call.call_id for span in tool_spans if span.tool_call] == [
        "duplicate",
        "duplicate",
    ]
    assert tool_spans[0].output is UNSET
    assert tool_spans[1].output is None
    assert trace.tool_catalog == {"search": {"type": "object"}}
