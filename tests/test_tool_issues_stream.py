"""IA3 projection and evidence-stream behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from insight_agent.evidence_streams.tool_issues import (
    FINDING_TYPES,
    MISSING,
    ToolIssueConfig,
    ToolIssueEvidenceArtifacts,
    ToolIssueEvidenceStream,
    to_ia3_trace,
)
from insight_agent.trace_loaders import FSDataLoader
from insight_agent.traces import Span, SpanKind, TokenCounts, ToolCall, Trace, TraceAggregate

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
TOKENS = TokenCounts(input_tokens=0, cached_input_tokens=0, output_tokens=0)


def test_input_normalization_preserves_every_field_ia3_uses():
    trace = Trace(
        id="trace-1",
        root_spans=[
            Span(
                id="call-1",
                kind=SpanKind.TOOL,
                tool_name="search_alias",
                input={"document_id": "DOC-1"},
                output={"content": "Error: not found"},
                error="not_found",
                attributes={
                    "explicit_error": True,
                    "outcome_marker": "not_found",
                    "source_pointer": {"call": 4},
                },
                tool_call=ToolCall(
                    call_id="call-1",
                    index=4,
                    result_id="result-1",
                    result_count=2,
                    instrumentation_alias_of="search",
                    prior_user_text="Use DOC-1",
                ),
            )
        ],
        aggregate=TraceAggregate(),
        attributes={
            "logical_case_id": "case-1",
            "task_text": "Find document DOC-1",
            "complete_provenance_context": True,
            "tool_catalog": {"search": {"type": "object"}},
            "orphan_results": [{"result_id": "orphan"}],
        },
    )
    projected = to_ia3_trace(trace)
    call = projected.calls[0]
    assert projected.trace_id == "trace-1"
    assert projected.logical_case_id == "case-1"
    assert projected.tool_catalog == {"search": {"type": "object"}}
    assert projected.orphan_results == ({"result_id": "orphan"},)
    assert projected.complete_provenance_context is True
    assert call.call_index == 4
    assert call.call_id == "call-1"
    assert call.tool_name == "search_alias"
    assert call.arguments == {"document_id": "DOC-1"}
    assert call.result == {"content": "Error: not found"}
    assert call.result_id == "result-1"
    assert call.result_count == 2
    assert call.explicit_error is True
    assert call.outcome_marker == "not_found"
    assert call.instrumentation_alias_of == "search"
    assert call.prior_user_text == "Use DOC-1"
    assert call.source_pointer == {"call": 4}


def test_missing_output_uses_ia3_singleton_but_json_null_remains_none():
    trace = Trace(
        id="missing-null",
        root_spans=[
            Span(
                id="missing",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={},
                tool_call=ToolCall(call_id="missing", index=0, result_count=0),
            ),
            Span(
                id="null",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={},
                output=None,
                tool_call=ToolCall(call_id="null", index=1),
            ),
        ],
        aggregate=TraceAggregate(),
    )

    projected = to_ia3_trace(trace)
    assert projected.calls[0].result is MISSING
    assert projected.calls[1].result is None


def test_full_trace_input_supplies_prior_user_context():
    trace = Trace(
        id="messages",
        root_spans=[
            Span(
                id="agent",
                kind=SpanKind.AGENT,
                children=[
                    Span(
                        id="call-1",
                        kind=SpanKind.TOOL,
                        children=[],
                        start_time=NOW,
                        end_time=NOW,
                        input={"document_id": "DOC-9"},
                        output=None,
                        cost_usd=0.0,
                        token_counts=TOKENS,
                        model=None,
                        tool_name="search",
                    )
                ],
                start_time=NOW,
                end_time=NOW,
                input={
                    "messages": [
                        {"role": "system", "content": "Use tools"},
                        {"role": "user", "content": "Find DOC-9"},
                    ]
                },
                output=None,
                cost_usd=0.0,
                token_counts=TOKENS,
                model=None,
            )
        ],
        aggregate=TraceAggregate(cost_usd=0.0, latency_ms=0.0, token_counts=TOKENS),
    )

    assert to_ia3_trace(trace).calls[0].prior_user_text == "Find DOC-9"


def test_tool_issue_stream_retains_all_findings_and_cards():
    loader = FSDataLoader(CORPUS)
    snapshot = loader.load()
    evidence = ToolIssueEvidenceStream().analyze(snapshot)

    assert isinstance(evidence.artifacts, ToolIssueEvidenceArtifacts)
    assert {finding["issue_type"] for finding in evidence.artifacts.findings} == set(FINDING_TYPES)
    assert any(card.eligible_for_analyst for card in evidence.artifacts.cards)
    assert set(evidence.artifacts.catalog_coverage) == set(FINDING_TYPES)
    assert len(evidence.problems) == sum(
        card.eligible_for_analyst for card in evidence.artifacts.cards
    )
    assert [trace.id for trace in snapshot] == list(snapshot.traces_by_id)


def test_tool_issue_stream_can_expose_audit_cards_as_problems():
    snapshot = FSDataLoader(CORPUS).load()
    evidence = ToolIssueEvidenceStream(config=ToolIssueConfig(include_audit_problems=True)).analyze(
        snapshot
    )

    assert isinstance(evidence.artifacts, ToolIssueEvidenceArtifacts)
    assert len(evidence.problems) == len(evidence.artifacts.cards)
