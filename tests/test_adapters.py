"""The reference adapter.

These tests double as the specification the skill points people at: each one
names a decision an adapter author has to get right, and what breaks otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from insight_agent.adapters.messages import (
    adapt_conversation,
    adapt_file,
    adapt_many,
    detect_format,
)
from insight_agent.ia3_tid import MISSING, detect
from insight_agent.loader import load_records, to_trace_record
from insight_agent.validate import validate_record

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
ANTHROPIC = DATA_DIR / "anthropic_messages.json"
OPENAI = DATA_DIR / "openai_messages.json"


def by_id(records):
    return {r["trace_id"]: r for r in records}


@pytest.fixture(scope="module")
def anthropic_records():
    return by_id(adapt_file(ANTHROPIC))


@pytest.fixture(scope="module")
def openai_records():
    return by_id(adapt_file(OPENAI))


def issue_types(record):
    return {f["issue_type"] for f in detect([to_trace_record(record)])}


# -- format detection ------------------------------------------------------


def test_format_detection(anthropic_records, openai_records):
    assert detect_format(json.loads(ANTHROPIC.read_text(encoding="utf-8"))) == "anthropic"
    assert detect_format(json.loads(OPENAI.read_text(encoding="utf-8"))) == "openai"


def test_both_sample_files_produce_valid_canonical_records(anthropic_records, openai_records):
    for record in list(anthropic_records.values()) + list(openai_records.values()):
        assert validate_record(record) == [], record["trace_id"]


def test_adapted_corpora_load_without_lint_errors(anthropic_records, openai_records):
    assert len(load_records(list(anthropic_records.values()))) == 4
    assert len(load_records(list(openai_records.values()))) == 4


# -- the call/result join --------------------------------------------------


def test_anthropic_links_a_call_to_a_result_in_a_later_message(anthropic_records):
    record = anthropic_records["anthropic-clean"]
    assert [c["tool_name"] for c in record["calls"]] == ["FileSearchTool", "FileReadTool"]
    assert record["calls"][0]["result"]["content"].startswith("1 match")


def test_openai_links_tool_calls_to_role_tool_replies(openai_records):
    record = openai_records["openai-clean"]
    assert record["calls"][0]["call_id"] == "call_aa01"
    assert record["calls"][0]["result"]["content"] == "42"


@pytest.mark.parametrize(
    "fixture,trace_id",
    [("anthropic_records", "anthropic-missing-result"), ("openai_records", "openai-missing-result")],
)
def test_an_unanswered_call_omits_the_result_key(request, fixture, trace_id):
    """Absence, not null: emitting None would claim the tool returned null."""
    record = request.getfixturevalue(fixture)[trace_id]
    call = record["calls"][0]

    assert "result" not in call
    assert to_trace_record(record).calls[0].result is MISSING
    assert "missing_tool_result" in issue_types(record)


def test_a_stray_result_becomes_an_orphan_not_a_dropped_record(anthropic_records):
    record = anthropic_records["anthropic-orphan-result"]
    assert [o["result_id"] for o in record["orphan_results"]] == ["toolu_01DAghost"]
    assert "orphan_tool_result" in issue_types(record)


# -- leaving evidence raw --------------------------------------------------


def test_unparsable_openai_arguments_survive_as_a_string(openai_records):
    """The defect is the evidence; parsing or dropping it would hide the bug."""
    record = openai_records["openai-malformed-arguments"]
    arguments = record["calls"][0]["arguments"]

    assert isinstance(arguments, str)
    assert arguments.startswith("{")
    assert "malformed_tool_call" in issue_types(record)


def test_valid_openai_arguments_are_parsed_into_an_object(openai_records):
    arguments = openai_records["openai-clean"]["calls"][0]["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["sql"].startswith("SELECT count(*)")


def test_a_type_mismatch_reaches_the_contract_rules(openai_records):
    record = openai_records["openai-type-mismatch"]
    assert record["calls"][0]["arguments"]["limit"] == "three"
    assert "argument_type_mismatch" in issue_types(record)


# -- not inventing signals -------------------------------------------------


def test_explicit_error_is_set_only_when_the_source_says_so(anthropic_records):
    """`explicit_error: false` would disable all text-based failure decoding."""
    failing = anthropic_records["anthropic-error-flag"]["calls"][0]
    assert failing["explicit_error"] is True

    for record in anthropic_records.values():
        for call in record["calls"]:
            assert call.get("explicit_error") is not False, (
                f"{record['trace_id']}/{call['call_id']} asserts success, which silences "
                "explicit_tool_failure for that call"
            )


def test_the_error_flag_becomes_an_explicit_tool_failure(anthropic_records):
    assert "explicit_tool_failure" in issue_types(anthropic_records["anthropic-error-flag"])


def test_no_logical_case_id_is_invented(anthropic_records, openai_records):
    """Inventing one would overstate independent-case counts and card eligibility."""
    for record in list(anthropic_records.values()) + list(openai_records.values()):
        assert "logical_case_id" not in record


def test_result_id_is_never_populated_from_the_wrong_namespace(openai_records):
    for record in openai_records.values():
        for call in record["calls"]:
            assert "result_id" not in call


# -- text placement --------------------------------------------------------


def test_textual_results_land_under_content(anthropic_records, openai_records):
    """IA3 unwraps `content` and nothing else."""
    for record in list(anthropic_records.values()) + list(openai_records.values()):
        for call in record["calls"]:
            if "result" in call and isinstance(call["result"], dict):
                assert "content" in call["result"], f"{record['trace_id']}/{call['call_id']}"


def test_the_two_engines_agree_on_every_adapted_call(anthropic_records, openai_records):
    """The content-vs-output trap would show up here as a disagreement."""
    from insight_agent.ia2_pipeline import decode_explicit_failure
    from insight_agent.ia3_tid import strict_failure
    from insight_agent.loader import to_normalized_trace

    for record in list(anthropic_records.values()) + list(openai_records.values()):
        ia2 = {c.call_id: c for c in to_normalized_trace(record).calls}
        for call in to_trace_record(record).calls:
            ia3_failed, _ = strict_failure(call)
            ia2_failed, _, _ = decode_explicit_failure(
                ia2[call.call_id].tool_name, ia2[call.call_id].result
            )
            assert ia2_failed == ia3_failed, f"{record['trace_id']}/{call.call_id}"


# -- catalog extraction ----------------------------------------------------


def test_tool_catalog_is_extracted_from_declared_tools(anthropic_records, openai_records):
    """The single highest-value field: without it seven rules abstain."""
    anthropic = anthropic_records["anthropic-clean"]["tool_catalog"]
    assert set(anthropic) == {"FileSearchTool", "FileReadTool"}
    assert anthropic["FileSearchTool"]["required"] == ["query"]

    openai = openai_records["openai-clean"]["tool_catalog"]
    assert set(openai) == {"DatabaseQueryTool", "FileSearchTool"}


def test_a_conversation_without_tools_has_no_catalog(anthropic_records):
    assert "tool_catalog" not in anthropic_records["anthropic-missing-result"]


def test_an_external_catalog_can_be_supplied(tmp_path):
    catalog = {"FileReadTool": {"type": "object", "required": ["path"]}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    records = by_id(adapt_file(ANTHROPIC, tool_catalog_path=path))
    # Only fills the gap; a conversation that declared its own keeps it.
    assert records["anthropic-missing-result"]["tool_catalog"] == catalog
    assert "FileSearchTool" in records["anthropic-clean"]["tool_catalog"]


# -- trajectory and provenance --------------------------------------------


def test_steps_are_emitted_for_user_agent_and_tool_turns(anthropic_records):
    steps = anthropic_records["anthropic-clean"]["steps"]
    assert [s["step_type"] for s in steps][:3] == ["user", "agent", "tool"]
    assert [s["step_index"] for s in steps] == list(range(len(steps)))


def test_source_pointers_locate_the_original_block(anthropic_records):
    pointer = anthropic_records["anthropic-clean"]["calls"][0]["source_pointer"]
    assert {"message_index", "block_index", "tool_use_id"} <= set(pointer)


def test_prior_user_text_is_captured_for_provenance(openai_records):
    call = openai_records["openai-clean"]["calls"][0]
    assert "open tickets" in call["prior_user_text"]


# -- API surface -----------------------------------------------------------


def test_adapt_conversation_generates_ids_when_the_source_has_none():
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "X", "input": {}},
                ],
            }
        ]
    }
    record = adapt_conversation(conversation, index=7, trace_id_prefix="run")
    assert record["trace_id"] == "run-00007"
    assert record["calls"][0]["call_id"] == "run-00007#0"


def test_adapt_many_assigns_distinct_trace_ids():
    conversations = [{"messages": []}, {"messages": []}]
    assert len({r["trace_id"] for r in adapt_many(conversations)}) == 2


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown message format"):
        adapt_conversation({"messages": []}, fmt="langsmith")
