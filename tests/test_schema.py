"""Structural validation of the canonical trace schema."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from insight_agent.evidence_streams.anomaly_and_patterns import DEFAULT_FEATURES
from insight_agent.trace_loaders.insight_trace.validation import (
    BUILTIN_FEATURE_NAMES,
    CANONICAL_VERSION,
    trace_schema,
    validate_corpus,
    validate_record,
    validate_records,
)

MINIMAL = {
    "schema_version": CANONICAL_VERSION,
    "trace_id": "trace-001",
    "calls": [
        {
            "call_id": "call-0",
            "call_index": 0,
            "tool_name": "FileSearchTool",
            "arguments": {"query": "example"},
        }
    ],
}


def codes(record, **kwargs):
    return [d.code for d in validate_record(record, **kwargs)]


def test_schema_is_itself_a_valid_draft_2020_12_schema():
    Draft202012Validator.check_schema(trace_schema())


def test_minimal_record_validates():
    assert validate_record(MINIMAL) == []


def test_only_four_top_level_and_four_call_fields_are_required():
    schema = trace_schema()
    assert set(schema["required"]) == {"schema_version", "trace_id", "calls"}
    assert set(schema["$defs"]["call"]["required"]) == {
        "call_id",
        "call_index",
        "tool_name",
        "arguments",
    }


@pytest.mark.parametrize("field", ["schema_version", "trace_id", "calls"])
def test_missing_required_top_level_field_is_rejected(field):
    record = {k: v for k, v in MINIMAL.items() if k != field}
    assert "schema_violation" in codes(record)


def test_wrong_schema_version_is_rejected():
    assert codes({**MINIMAL, "schema_version": "insight-trace/v2"}) == ["schema_violation"]


def test_call_index_must_be_an_integer_not_a_string():
    record = copy.deepcopy(MINIMAL)
    record["calls"][0]["call_index"] = "3"
    diagnostics = validate_record(record, line=7)
    assert [d.code for d in diagnostics] == ["schema_violation"]
    assert diagnostics[0].path == "calls[0].call_index"
    assert diagnostics[0].line == 7
    assert "line 7" in diagnostics[0].format()


def test_a_typo_in_an_optional_field_fails_loudly():
    """The whole point of additionalProperties:false.

    Without it, ``tool_catelog`` would silently disable seven IA3 rules and the
    adapter author would see an empty findings list with no explanation.
    """
    diagnostics = validate_record({**MINIMAL, "tool_catelog": {"X": None}})
    assert [d.code for d in diagnostics] == ["unknown_field"]
    assert "extra" in diagnostics[0].message


def test_unknown_call_level_field_is_rejected():
    record = copy.deepcopy(MINIMAL)
    record["calls"][0]["resault"] = {"content": "typo"}
    assert codes(record) == ["unknown_field"]


def test_extra_object_accepts_arbitrary_vendor_data():
    record = {**MINIMAL, "extra": {"venue": "example", "nested": {"anything": [1, 2, 3]}}}
    assert validate_record(record) == []


def test_arguments_may_be_any_json_value_including_a_bare_string():
    """A non-object argument blob is legitimate evidence, not a schema error."""
    record = copy.deepcopy(MINIMAL)
    record["calls"][0]["arguments"] = "query=foo"
    assert validate_record(record) == []


def test_result_may_be_null_and_may_be_absent():
    absent = copy.deepcopy(MINIMAL)
    explicit_null = copy.deepcopy(MINIMAL)
    explicit_null["calls"][0]["result"] = None
    assert validate_record(absent) == []
    assert validate_record(explicit_null) == []


def test_tool_catalog_accepts_a_schema_or_null_per_tool():
    record = {
        **MINIMAL,
        "tool_catalog": {
            "FileSearchTool": {"type": "object", "properties": {"query": {"type": "string"}}},
            "SessionTool": None,
        },
    }
    assert validate_record(record) == []


def test_orphan_result_requires_a_result_id():
    assert codes({**MINIMAL, "orphan_results": [{"tool_name": "X"}]}) == ["schema_violation"]
    assert validate_record({**MINIMAL, "orphan_results": [{"result_id": "r1"}]}) == []


def test_step_requires_index_and_type():
    ok = {**MINIMAL, "steps": [{"step_index": 0, "step_type": "planning", "name": "plan"}]}
    assert validate_record(ok) == []
    assert codes({**MINIMAL, "steps": [{"step_index": 0}]}) == ["schema_violation"]


def test_metrics_values_must_be_numeric():
    assert codes({**MINIMAL, "metrics": {"turn_count": "many"}}) == ["schema_violation"]


def test_non_object_line_is_reported_clearly():
    assert [d.code for d in validate_record([1, 2, 3], line=2)] == ["not_an_object"]


def test_builtin_feature_name_list_matches_the_engine():
    """Loader validation duplicates the feature list to stay import-light."""
    assert BUILTIN_FEATURE_NAMES == set(DEFAULT_FEATURES)


def test_validate_corpus_reports_every_bad_line_not_just_the_first(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(MINIMAL),
                "{not json",
                json.dumps({**MINIMAL, "trace_id": "trace-002", "nope": 1}),
                "",
                "also not json",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_corpus(path)
    assert not report.ok
    assert sorted({d.code for d in report.errors}) == ["invalid_json", "unknown_field"]
    assert [d.line for d in report.errors if d.code == "invalid_json"] == [2, 5]


def test_report_serialises_to_json():
    report = validate_records([(1, MINIMAL)])
    payload = report.to_dict()
    assert payload["ok"] is True
    assert payload["record_count"] == 1
    json.dumps(payload)  # must not raise
