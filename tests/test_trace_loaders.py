"""Filesystem loading of canonical Trace JSONL."""

from __future__ import annotations

import json

import pytest

from insight_agent.evidence_streams.tool_issues import MISSING, to_ia3_trace
from insight_agent.trace_loaders import FSDataLoader, FSDataLoadError
from insight_agent.traces import UNSET, Span, SpanKind, ToolCall, Trace, TraceAggregate


def trace(trace_id: str = "t1", *, output=UNSET, logical_case_id: str | None = None) -> Trace:
    span_fields = {
        "id": f"{trace_id}:call",
        "kind": SpanKind.TOOL,
        "tool_name": "search",
        "input": {"query": "x"},
        "tool_call": ToolCall(
            call_id=f"{trace_id}:call",
            index=0,
            result_count=0 if output is UNSET else 1,
        ),
    }
    if output is not UNSET:
        span_fields["output"] = output
    attributes = {"logical_case_id": logical_case_id} if logical_case_id else {}
    return Trace(
        id=trace_id,
        root_spans=[Span(**span_fields)],
        aggregate=TraceAggregate(cost_usd=0.1, latency_ms=2.0),
        attributes=attributes,
    )


def write_jsonl(path, traces: list[Trace]) -> None:
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json", exclude_unset=True)) + "\n" for item in traces
        ),
        encoding="utf-8",
    )


def test_loader_parses_each_jsonl_record_directly_as_a_trace(tmp_path):
    path = tmp_path / "traces.jsonl"
    expected = [trace("t1", output={"content": "found"}), trace("t2", output=None)]
    write_jsonl(path, expected)

    actual = list(FSDataLoader(path).load())

    assert actual == expected


def test_missing_output_and_explicit_null_remain_distinct(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [trace("missing"), trace("null", output=None)])

    projected = [to_ia3_trace(item).calls[0].result for item in FSDataLoader(path).load()]

    assert projected[0] is MISSING
    assert projected[1] is None


def test_loader_preserves_nested_spans(tmp_path):
    nested = trace("nested", output="ok")
    child = nested.root_spans.pop()
    nested.root_spans.append(Span(id="agent", kind=SpanKind.AGENT, children=[child]))
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [nested])

    loaded = next(iter(FSDataLoader(path).load()))

    assert loaded.root_spans[0].children[0].id == "nested:call"


def test_snapshot_is_cached_and_reiterable(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [trace("t1"), trace("t2")])
    loader = FSDataLoader(path)

    first = loader.load()

    assert loader.load() is first
    assert [item.id for item in first] == ["t1", "t2"]
    assert [item.id for item in first] == ["t1", "t2"]


def test_describe_reports_typed_corpus_facts(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_jsonl(
        path,
        [
            trace("t1", logical_case_id="case-a"),
            trace("t2", logical_case_id="case-a"),
            trace("t3", logical_case_id="case-b"),
        ],
    )

    description = FSDataLoader(path).describe()

    assert description["trace_count"] == 3
    assert description["call_count"] == 3
    assert description["distinct_logical_cases"] == 2
    assert description["source"] == f"fs:{path.resolve()}"


def test_loader_rejects_malformed_json_with_a_line_number(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n{oops\n", encoding="utf-8")

    with pytest.raises(FSDataLoadError, match=r":2: malformed JSON"):
        FSDataLoader(path).load()


def test_loader_preserves_extra_trace_and_span_fields(tmp_path):
    path = tmp_path / "traces.jsonl"
    value = trace().model_dump(mode="json", exclude_unset=True)
    value["future_trace_field"] = {"version": 3}
    value["root_spans"][0]["future_span_field"] = [1, 2]
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    loaded = next(iter(FSDataLoader(path).load())).model_dump(mode="json", exclude_unset=True)

    assert loaded["future_trace_field"] == {"version": 3}
    assert loaded["root_spans"][0]["future_span_field"] == [1, 2]


def test_loader_rejects_duplicate_trace_ids(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [trace(), trace()])

    with pytest.raises(FSDataLoadError, match="duplicate trace id 't1'"):
        FSDataLoader(path).load()


def test_loader_rejects_an_empty_corpus(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(FSDataLoadError, match="contains no traces"):
        FSDataLoader(path).load()
