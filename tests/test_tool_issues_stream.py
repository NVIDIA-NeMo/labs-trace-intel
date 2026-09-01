"""IA3 projection and evidence-stream behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.evidence_streams.tool_issues import (
    FINDING_TYPES,
    MISSING,
    ToolIssueConfig,
    ToolIssueEvidenceArtifacts,
    ToolIssueEvidenceStream,
    to_ia3_trace,
)
from insight_agent.trace_loaders import InsightTraceLoader
from insight_agent.traces import Span, SpanKind, Trace

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
DUPLICATE_CALL_ID_WARNING = r"WARNING\[duplicate_call_id\].*docops-instrumentation"


def test_input_normalization_preserves_every_field_ia3_uses():
    record = {
        "schema_version": "insight-trace/v1",
        "trace_id": "trace-1",
        "logical_case_id": "case-1",
        "task_text": "Find document DOC-1",
        "complete_provenance_context": True,
        "tool_catalog": {"search": {"type": "object"}},
        "orphan_results": [{"result_id": "orphan"}],
        "calls": [
            {
                "call_id": "call-1",
                "call_index": 4,
                "tool_name": "search_alias",
                "arguments": {"document_id": "DOC-1"},
                "result": {"content": "Error: not found"},
                "result_id": "result-1",
                "result_count": 2,
                "explicit_error": True,
                "outcome_marker": "not_found",
                "instrumentation_alias_of": "search",
                "prior_user_text": "Use DOC-1",
                "source_pointer": {"call": 4},
            }
        ],
    }

    with pytest.warns(UserWarning, match=r"WARNING\[result_id_mostly_mismatched\]"):
        trace = next(InsightTraceLoader.from_records([record]).load().scan())
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
    trace = next(
        InsightTraceLoader.from_records(
            [
                {
                    "schema_version": "insight-trace/v1",
                    "trace_id": "missing-null",
                    "calls": [
                        {
                            "call_id": "missing",
                            "call_index": 0,
                            "tool_name": "search",
                            "arguments": {},
                        },
                        {
                            "call_id": "null",
                            "call_index": 1,
                            "tool_name": "search",
                            "arguments": {},
                            "result": None,
                        },
                    ],
                }
            ]
        )
        .load()
        .scan()
    )

    projected = to_ia3_trace(trace)
    assert projected.calls[0].result is MISSING
    assert projected.calls[1].result is None


def test_full_trace_input_supplies_prior_user_context():
    trace = Trace(
        id="messages",
        input={
            "messages": [
                {"role": "system", "content": "Use tools"},
                {"role": "user", "content": "Find DOC-9"},
            ]
        },
        spans=(
            Span(
                span_id="call-1",
                kind=SpanKind.TOOL,
                tool_name="search",
                input={"document_id": "DOC-9"},
            ),
        ),
    )

    assert to_ia3_trace(trace).calls[0].prior_user_text == "Find DOC-9"


def test_tool_issue_stream_retains_all_findings_and_cards():
    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        loader = InsightTraceLoader.from_path(CORPUS)
    snapshot = loader.load()
    evidence = ToolIssueEvidenceStream().analyze(snapshot)

    assert isinstance(evidence.artifacts, ToolIssueEvidenceArtifacts)
    assert {finding["issue_type"] for finding in evidence.artifacts.findings} == set(FINDING_TYPES)
    assert any(card.eligible_for_analyst for card in evidence.artifacts.cards)
    assert set(evidence.artifacts.catalog_coverage) == set(FINDING_TYPES)
    assert len(evidence.problems) == sum(
        card.eligible_for_analyst for card in evidence.artifacts.cards
    )
    assert [trace.id for trace in snapshot.scan()] == [
        record["trace_id"] for record in loader.records
    ]


def test_tool_issue_stream_can_expose_audit_cards_as_problems():
    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        snapshot = InsightTraceLoader.from_path(CORPUS).load()
    evidence = ToolIssueEvidenceStream(config=ToolIssueConfig(include_audit_problems=True)).analyze(
        snapshot
    )

    assert isinstance(evidence.artifacts, ToolIssueEvidenceArtifacts)
    assert len(evidence.problems) == len(evidence.artifacts.cards)
