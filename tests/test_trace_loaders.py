"""Input loading and normalization."""

from __future__ import annotations

import copy
import json

import pytest

from insight_agent.evidence_streams.anomaly_and_patterns import to_ia2_trace
from insight_agent.evidence_streams.tool_issues import MISSING, detect, to_ia3_trace
from insight_agent.trace_loaders import (
    InsightTraceV1Loader,
    InsightTraceV1Options,
    TraceLoadError,
)
from insight_agent.validate import CANONICAL_VERSION
from insight_agent.venue import DEFAULT_PROFILE

BASE = {
    "schema_version": CANONICAL_VERSION,
    "trace_id": "t1",
    "calls": [
        {
            "call_id": "c0",
            "call_index": 0,
            "tool_name": "FileSearchTool",
            "arguments": {"query": "x"},
            "result": {"content": "2 matches"},
        }
    ],
}


def record(**overrides):
    return {**copy.deepcopy(BASE), **overrides}


def one_call(**call_overrides):
    rec = copy.deepcopy(BASE)
    rec["calls"][0].update(call_overrides)
    return rec


def ia2_trace(rec, *, profile=DEFAULT_PROFILE, tool_catalog=None):
    options = InsightTraceV1Options(tool_catalog=tool_catalog)
    trace = next(InsightTraceV1Loader.from_records([rec], options).load().scan())
    return to_ia2_trace(
        trace,
        profile=profile,
    )


def ia3_trace(rec, *, tool_catalog=None):
    options = InsightTraceV1Options(tool_catalog=tool_catalog)
    trace = next(InsightTraceV1Loader.from_records([rec], options).load().scan())
    return to_ia3_trace(trace)


def first_ia3_call(rec):
    return ia3_trace(rec).calls[0]


# -- MISSING invariants ----------------------------------------------------


def test_absent_result_key_becomes_the_missing_sentinel():
    rec = copy.deepcopy(BASE)
    del rec["calls"][0]["result"]
    assert first_ia3_call(rec).result is MISSING


def test_explicit_json_null_is_none_and_is_not_missing():
    """ "result": null means the tool genuinely returned null."""
    call = first_ia3_call(one_call(result=None))
    assert call.result is None
    assert call.result is not MISSING


def test_result_missing_flag_forces_the_sentinel():
    assert first_ia3_call(one_call(result_missing=True)).result is MISSING


def test_result_count_zero_forces_the_sentinel():
    assert first_ia3_call(one_call(result_count=0)).result is MISSING


def test_result_count_zero_does_not_leak_into_the_duplicate_check():
    """0 means 'missing', not 'a count worth reporting'."""
    call = first_ia3_call(one_call(result_count=0))
    assert call.result_count == 1


def test_missing_sentinel_reaches_the_engine_and_fires_missing_tool_result():
    rec = copy.deepcopy(BASE)
    del rec["calls"][0]["result"]
    findings = detect([ia3_trace(rec)])
    assert "missing_tool_result" in {f["issue_type"] for f in findings}


def test_explicit_null_result_does_not_fire_missing_tool_result():
    findings = detect([ia3_trace(one_call(result=None))])
    assert "missing_tool_result" not in {f["issue_type"] for f in findings}


# -- IA2 conversion --------------------------------------------------------


def test_ia2_conversion_carries_every_canonical_field():
    rec = record(
        source_pointer={"uri": "s3://x"},
        observed_verdict="completed",
        cost=1.25,
        metrics={"turn_count": 7.0},
        steps=[{"step_index": 0, "step_type": "planning", "name": "plan", "content": "go"}],
    )
    trace = ia2_trace(rec)

    assert trace.trace_id == "t1"
    assert trace.source_pointer == {"uri": "s3://x"}
    assert trace.observed_verdict == "completed"
    assert trace.cost == 1.25
    assert trace.metrics == {"turn_count": 7.0}
    assert [s.step_type for s in trace.steps] == ["planning", "tool"]

    call = trace.calls[0]
    assert (call.call_id, call.call_index, call.tool_name) == ("c0", 0, "FileSearchTool")
    assert call.arguments == {"query": "x"}


def test_ia2_sees_none_for_an_unrecorded_result():
    """IA2 has no MISSING concept; it must not receive the sentinel."""
    rec = copy.deepcopy(BASE)
    del rec["calls"][0]["result"]
    assert ia2_trace(rec).calls[0].result is None


def test_calls_are_ordered_by_call_index_regardless_of_input_order():
    rec = record(
        calls=[
            {"call_id": "c2", "call_index": 2, "tool_name": "B", "arguments": {}},
            {"call_id": "c0", "call_index": 0, "tool_name": "A", "arguments": {}},
            {"call_id": "c1", "call_index": 1, "tool_name": "C", "arguments": {}},
        ]
    )
    assert [c.call_index for c in ia2_trace(rec).calls] == [0, 1, 2]
    assert [c.call_index for c in ia3_trace(rec).calls] == [0, 1, 2]


# -- steps vs calls --------------------------------------------------------


def test_with_steps_ia2_tokens_follow_the_trajectory():
    from insight_agent.evidence_streams.anomaly_and_patterns import extract_trace_features

    rec = record(
        steps=[
            {"step_index": 0, "step_type": "planning", "name": "plan"},
            {"step_index": 1, "step_type": "tool", "name": "FileSearchTool"},
            {"step_index": 2, "step_type": "evaluation", "name": "boundary", "content": "done"},
        ]
    )
    features = extract_trace_features(ia2_trace(rec)).features
    assert list(features.sequence_tokens) == [
        "planning:plan",
        "tool:FileSearchTool",
        "evaluation:boundary",
    ]
    assert features.numeric["trajectory_step_count"] == 3.0


def test_without_steps_ia2_falls_back_to_one_token_per_call():
    from insight_agent.evidence_streams.anomaly_and_patterns import extract_trace_features

    rec = record(
        calls=[
            {"call_id": "c0", "call_index": 0, "tool_name": "Alpha", "arguments": {}},
            {"call_id": "c1", "call_index": 1, "tool_name": "Beta", "arguments": {}},
        ]
    )
    features = extract_trace_features(ia2_trace(rec)).features
    assert list(features.sequence_tokens) == ["tool:Alpha", "tool:Beta"]
    assert features.numeric["trajectory_step_count"] == 2.0


# -- IA3 specifics ---------------------------------------------------------


def test_prior_user_text_falls_back_to_task_text():
    rec = record(task_text="the original request")
    assert first_ia3_call(rec).prior_user_text == "the original request"

    rec = record(task_text="trace level")
    rec["calls"][0]["prior_user_text"] = "call level"
    assert first_ia3_call(rec).prior_user_text == "call level"


def test_tool_catalog_precedence_record_then_option_then_none():
    corpus_wide = {"FileSearchTool": {"type": "object"}}
    record_level = {"FileSearchTool": {"type": "object", "required": ["query"]}}

    assert ia3_trace(record()).tool_catalog is None
    assert ia3_trace(record(), tool_catalog=corpus_wide).tool_catalog == corpus_wide
    assert (
        ia3_trace(record(tool_catalog=record_level), tool_catalog=corpus_wide).tool_catalog
        == record_level
    )


def test_dropping_the_tool_catalog_makes_the_contract_rules_abstain():
    catalog = {
        "FileSearchTool": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
    }
    bad = one_call(arguments={"query": "x", "recursive": True})

    with_catalog = detect([ia3_trace({**bad, "tool_catalog": catalog})])
    without = detect([ia3_trace(bad)])

    assert "unknown_argument" in {f["issue_type"] for f in with_catalog}
    assert "unknown_argument" not in {f["issue_type"] for f in without}


def test_arguments_are_passed_through_raw_including_non_objects():
    """A string argument blob is evidence, and IA3 reports it as malformed."""
    rec = one_call(arguments="query=foo")
    assert ia3_trace(rec).calls[0].arguments == "query=foo"
    # IA2's contract wants a Mapping, so it degrades to {} rather than crashing.
    assert ia2_trace(rec).calls[0].arguments == {}


def test_orphan_results_and_provenance_flag_are_forwarded():
    rec = record(
        orphan_results=[{"result_id": "r1", "tool_name": "X"}],
        complete_provenance_context=True,
    )
    trace = ia3_trace(rec)
    assert trace.complete_provenance_context is True
    assert trace.orphan_results[0]["result_id"] == "r1"
    assert "orphan_tool_result" in {f["issue_type"] for f in detect([trace])}


def test_logical_case_id_defaults_to_none_so_the_engine_can_fall_back():
    assert ia3_trace(record()).logical_case_id is None
    assert ia3_trace(record(logical_case_id="case-9")).logical_case_id == "case-9"


# -- returned_data injection ----------------------------------------------


def test_returned_data_is_injected_into_a_mapping_result():
    from insight_agent.evidence_streams.anomaly_and_patterns import extract_trace_features

    rec = one_call(result={"content": "none"}, returned_data=False)
    trace = ia2_trace(rec)
    assert trace.calls[0].result["returned_data"] is False
    assert extract_trace_features(trace).features.numeric["returned_data_false_rate"] == 1.0


def test_returned_data_uses_the_profile_key():
    custom = DEFAULT_PROFILE.with_overrides(returned_data_key="had_rows")
    rec = one_call(result={"content": "none"}, returned_data=False)
    assert ia2_trace(rec, profile=custom).calls[0].result["had_rows"] is False


def test_returned_data_on_a_string_result_warns_and_does_not_wrap():
    """Wrapping would change output-size features and break the error anchor."""
    rec = one_call(result="plain text", returned_data=False)
    with pytest.warns(UserWarning, match="not a JSON object"):
        trace = ia2_trace(rec)
    assert trace.calls[0].result == "plain text"


def test_existing_returned_data_key_is_not_overwritten():
    rec = one_call(result={"content": "x", "returned_data": True}, returned_data=False)
    assert ia2_trace(rec).calls[0].result["returned_data"] is True


# -- corpus loading --------------------------------------------------------


def test_load_records_validates_and_raises_in_strict_mode():
    with pytest.raises(TraceLoadError) as excinfo:
        InsightTraceV1Loader.from_records([{"schema_version": CANONICAL_VERSION, "trace_id": "t"}])
    assert excinfo.value.report.errors


def test_metric_shadowing_is_an_error_unless_allowed():
    bad = record(metrics={"tool_call_count": 5.0})
    with pytest.raises(TraceLoadError):
        InsightTraceV1Loader.from_records([bad])

    loader = InsightTraceV1Loader.from_records(
        [bad], InsightTraceV1Options(allow_metric_shadowing=True)
    )
    assert len(loader) == 1


def test_non_strict_mode_drops_bad_records_and_keeps_the_rest():
    good = record()
    bad = {"schema_version": CANONICAL_VERSION, "trace_id": "t2"}  # no calls
    with pytest.warns(UserWarning):
        loader = InsightTraceV1Loader.from_records([good, bad], InsightTraceV1Options(strict=False))
    assert [r["trace_id"] for r in loader.records] == ["t1"]
    assert loader.report.errors


def test_loader_builds_a_reiterable_snapshot():
    loader = InsightTraceV1Loader.from_records([record(trace_id=f"t{i}") for i in range(3)])
    snapshot = loader.load()
    assert snapshot.trace_count == 3
    assert [trace.id for trace in snapshot.scan()] == ["t0", "t1", "t2"]
    assert [trace.id for trace in snapshot.scan()] == ["t0", "t1", "t2"]


def test_loader_describe_reports_provenance_relevant_facts():
    loader = InsightTraceV1Loader.from_records(
        [
            record(trace_id="t1", logical_case_id="case-a"),
            record(trace_id="t2", logical_case_id="case-a"),
            record(trace_id="t3", logical_case_id="case-b"),
        ]
    )
    described = loader.describe()
    assert described["trace_count"] == 3
    assert described["distinct_logical_cases"] == 2
    assert described["steps_present"] is False


def test_loader_reads_jsonl_from_disk(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps(record(trace_id=f"t{i}")) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    loader = InsightTraceV1Loader.from_path(path)
    assert len(loader) == 3
    assert loader.source == str(path)


def test_loader_rejects_malformed_json_lines(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(record()) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(TraceLoadError, match="malformed JSON"):
        InsightTraceV1Loader.from_path(path)


def test_duplicate_trace_ids_are_rejected():
    with pytest.raises(TraceLoadError):
        InsightTraceV1Loader.from_records([record(), record()])


# -- structural round trip -------------------------------------------------


def test_round_trip_preserves_identity_fields():
    rec = record(
        calls=[
            {
                "call_id": "c0",
                "call_index": 0,
                "tool_name": "Alpha",
                "arguments": {"a": 1, "nested": {"b": [1, 2]}},
                "result": {"content": "ok"},
            },
            {
                "call_id": "c1",
                "call_index": 1,
                "tool_name": "Beta",
                "arguments": {},
            },
        ]
    )
    trace = next(InsightTraceV1Loader.from_records([rec]).load().scan())
    ia2 = to_ia2_trace(trace)
    ia3 = to_ia3_trace(trace)

    for source, converted2, converted3 in zip(rec["calls"], ia2.calls, ia3.calls, strict=False):
        assert source["call_id"] == converted2.call_id == converted3.call_id
        assert source["call_index"] == converted2.call_index == converted3.call_index
        assert source["tool_name"] == converted2.tool_name == converted3.tool_name
        assert source["arguments"] == converted2.arguments == converted3.arguments
