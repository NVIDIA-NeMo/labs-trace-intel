"""IA2 projection and evidence-stream parity."""

from __future__ import annotations

from pathlib import Path

from insight_agent.loader import LoadOptions, load_corpus, to_trace
from insight_agent.streams import IA2EvidenceArtifacts, IA2EvidenceStream, to_ia2_trace
from insight_agent.traces import Span, SpanKind, Trace

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"


def test_input_normalization_preserves_every_field_ia2_uses():
    corpus = load_corpus(CORPUS, LoadOptions())
    record = corpus.records[0]
    projected = to_ia2_trace(to_trace(record), profile=corpus.options.profile)

    assert projected.trace_id == record["trace_id"]
    assert [call.call_id for call in projected.calls] == [
        call["call_id"] for call in record["calls"]
    ]
    assert [call.arguments for call in projected.calls] == [
        call["arguments"] for call in record["calls"]
    ]
    assert [call.result for call in projected.calls] == [
        call["result"] for call in record["calls"]
    ]
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

    projected = to_ia2_trace(to_trace(record))
    assert len(projected.calls) == 1
    assert [(step.step_type, step.name) for step in projected.steps] == [("tool", "search")]


def test_native_projection_uses_canonical_spans_for_steps_and_tool_calls():
    trace = Trace(
        id="native",
        source_pointer={"source": "fixture"},
        observed_verdict="completed",
        metrics={"turns": 2},
        cost_usd=0.25,
        spans=(
            Span(span_id="agent", kind=SpanKind.AGENT, output="planning"),
            Span(
                span_id="llm",
                parent_span_id="agent",
                kind=SpanKind.LLM,
                output={"messages": [{"role": "assistant", "content": "calling search"}]},
            ),
            Span(
                span_id="call-1",
                parent_span_id="agent",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={"query": "report"},
                output={"content": "found"},
                source_pointer={"span": 2},
            ),
        ),
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
    assert projected.calls[0].source_pointer == {"span": 2}
    assert projected.observed_verdict == "completed"
    assert projected.metrics == {"turns": 2.0}
    assert projected.cost == 0.25


def test_ia2_stream_runs_the_engine_from_a_snapshot():
    corpus = load_corpus(CORPUS, LoadOptions())
    stream = IA2EvidenceStream(profile=corpus.options.profile)
    snapshot = corpus.snapshot()
    actual = stream.analyze(snapshot)

    assert actual.status == "completed"
    assert actual.coverage.traces_examined == len(corpus)
    assert isinstance(actual.payload, IA2EvidenceArtifacts)
    assert "## Unusual traces" in actual.payload.result["digest"]
    assert "docops-outlier" in actual.payload.result["digest"]
    assert [trace.id for trace in snapshot.scan()] == [
        record["trace_id"] for record in corpus.records
    ]
