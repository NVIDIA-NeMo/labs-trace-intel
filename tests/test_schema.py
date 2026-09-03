"""The canonical JSON Schema comes directly from the Trace model."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from insight_agent.traces import Trace


def test_trace_schema_is_a_valid_draft_2020_12_schema():
    Draft202012Validator.check_schema(Trace.model_json_schema())


def test_trace_schema_requires_the_public_contract_fields():
    schema = Trace.model_json_schema()

    assert set(schema["required"]) == {"id", "root_spans", "aggregate"}
    assert set(schema["$defs"]["Span"]["required"]) == {"id", "kind"}


def test_trace_schema_is_recursive():
    children = Trace.model_json_schema()["$defs"]["Span"]["properties"]["children"]

    assert children["items"]["$ref"] == "#/$defs/Span"


def test_trace_schema_allows_extra_json_fields():
    schema = Trace.model_json_schema()
    record = {"id": "t", "root_spans": [], "aggregate": {}, "future_field": {"value": 1}}

    assert Draft202012Validator(schema).is_valid(record)
    assert schema["additionalProperties"] == {"$ref": "#/$defs/JsonValue"}
    assert schema["$defs"]["Span"]["additionalProperties"] == {"$ref": "#/$defs/JsonValue"}


def test_missing_capable_fields_are_not_required():
    span_schema = Trace.model_json_schema()["$defs"]["Span"]

    assert "input" not in span_schema["required"]
    assert "output" not in span_schema["required"]


def test_schema_documents_source_normalization_requirements():
    schema = Trace.model_json_schema()
    trace_fields = schema["properties"]
    span_fields = schema["$defs"]["Span"]["properties"]
    tool_call_fields = schema["$defs"]["ToolCall"]["properties"]

    assert "unique within the loaded corpus" in trace_fields["id"]["description"]
    assert "source execution order" in trace_fields["root_spans"]["description"]
    assert "Omit when absent" in span_fields["input"]["description"]
    assert "explicit JSON null" in span_fields["output"]["description"]
    assert "use zero when none exists" in tool_call_fields["result_count"]["description"]
