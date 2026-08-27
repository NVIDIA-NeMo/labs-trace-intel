"""IA3 tool-issue detection over the bundled sample corpus.

The headline test is the parametrised one: every single one of the nineteen
finding types must fire on shipped data. That is what makes the sample corpus a
usable reference — someone writing an adapter can compare their own coverage
against a corpus that is known to exercise everything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from insight_agent.evidence_streams.tool_issues import (
    FINDING_TYPES,
    MISSING,
    RepresentativeEvidence,
    ToolIssueCard,
    build_cards,
    catalog_coverage,
    detect,
    to_ia3_trace,
)
from insight_agent.trace_loaders import InsightTraceV1Loader

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"
DUPLICATE_CALL_ID_WARNING = r"WARNING\[duplicate_call_id\].*docops-instrumentation"

CONTRACT_TYPES = {
    "unknown_tool",
    "missing_required_argument",
    "unknown_argument",
    "argument_type_mismatch",
    "argument_enum_violation",
    "json_schema_violation",
}


@pytest.fixture(scope="module")
def loader():
    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        return InsightTraceV1Loader.from_path(CORPUS)


@pytest.fixture(scope="module")
def findings(loader):
    return detect(to_ia3_trace(trace) for trace in loader.load().scan())


def test_there_are_exactly_nineteen_finding_types():
    assert len(FINDING_TYPES) == 19


@pytest.mark.parametrize("issue_type", sorted(FINDING_TYPES))
def test_every_finding_type_fires_on_the_sample_corpus(findings, issue_type):
    fired = {f["issue_type"] for f in findings}
    assert issue_type in fired, (
        f"{issue_type} never fires on the bundled corpus, so nobody can use the sample to "
        "check their own adapter's coverage of this rule. Extend tools/make_sample_corpus.py."
    )


def test_findings_are_fully_attributed(findings):
    for finding in findings:
        assert finding["trace_id"]
        assert finding["issue_family"]
        assert finding["attribution"] in {
            "agent_call",
            "instrumentation",
            "tool_or_environment",
            "agent_recovery",
        }
        assert finding["detector_version"] == "tid-v1"
        assert len(finding["issue_id"]) == 24


def test_findings_are_deterministic_within_a_process(loader):
    first = detect(to_ia3_trace(trace) for trace in loader.load().scan())
    second = detect(to_ia3_trace(trace) for trace in loader.load().scan())
    assert first == second


def test_issue_ids_are_stable_across_processes():
    """issue_id is a SHA-256 of a stable payload, so it must survive a restart.

    Cards are keyed on these ids, so instability would silently fragment
    recurrence counting across runs.
    """
    script = (
        "import json;"
        "from insight_agent.trace_loaders import InsightTraceV1Loader;"
        "from insight_agent.evidence_streams.tool_issues import detect,to_ia3_trace;"
        f"loader=InsightTraceV1Loader.from_path({str(CORPUS)!r});"
        "print(json.dumps(sorted(f['issue_id'] for f in "
        "detect(to_ia3_trace(t) for t in loader.load().scan()))))"
    )
    runs = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True
            ).stdout
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    assert len(runs[0]) > 20


def test_catalog_coverage_reports_all_nineteen_keys(findings):
    coverage = catalog_coverage(findings)
    assert set(coverage) == set(FINDING_TYPES)
    assert sum(coverage.values()) == len(findings)


# -- abstention ------------------------------------------------------------


def test_dropping_the_catalog_silences_exactly_the_contract_rules(loader, findings):
    stripped = [{k: v for k, v in r.items() if k != "tool_catalog"} for r in loader.records]

    with pytest.warns(UserWarning, match=DUPLICATE_CALL_ID_WARNING):
        snapshot = InsightTraceV1Loader.from_records(stripped).load()
    without = detect(to_ia3_trace(trace) for trace in snapshot.scan())

    before = {f["issue_type"] for f in findings}
    after = {f["issue_type"] for f in without}

    assert CONTRACT_TYPES <= before
    assert not (CONTRACT_TYPES & after)
    # Everything else keeps working; abstention is targeted, not global.
    assert (before - CONTRACT_TYPES) - after == set()


def test_a_null_schema_enables_unknown_tool_but_not_argument_checks():
    record = {
        "schema_version": "insight-trace/v1",
        "trace_id": "t",
        "tool_catalog": {"SessionTool": None},
        "calls": [
            {
                "call_id": "c0",
                "call_index": 0,
                "tool_name": "SessionTool",
                "arguments": {"anything": 1},
            },
            {"call_id": "c1", "call_index": 1, "tool_name": "GhostTool", "arguments": {}},
        ],
    }
    trace = next(InsightTraceV1Loader.from_records([record]).load().scan())
    fired = {f["issue_type"] for f in detect([to_ia3_trace(trace)])}
    assert "unknown_tool" in fired
    assert not (fired & (CONTRACT_TYPES - {"unknown_tool"}))


# -- cards -----------------------------------------------------------------


def test_cards_are_promoted_only_at_three_independent_cases(findings):
    cards = build_cards(findings, minimum_independent_cases=3)
    assert cards
    assert all(isinstance(card, ToolIssueCard) for card in cards)
    assert all(
        isinstance(evidence, RepresentativeEvidence)
        for card in cards
        for evidence in card.representative_evidence
    )
    for card in cards:
        assert card.eligible_for_analyst == (card.independent_case_count >= 3)
    assert any(card.eligible_for_analyst for card in cards)


def test_card_eligibility_counts_cases_not_traces(loader, findings):
    """Two traces share a logical case, so the counts must differ."""
    cards = {card.card_id: card for card in build_cards(findings)}
    card = cards["tid:explicit_tool_failure:error_prefix"]
    assert card.finding_count > card.independent_case_count


def test_cards_never_claim_impact(findings):
    for card in build_cards(findings):
        assert card.impact_status == "not_established"


def test_raising_the_threshold_disqualifies_everything(findings):
    cards = build_cards(findings, minimum_independent_cases=999)
    assert cards and not any(card.eligible_for_analyst for card in cards)


# -- parameters ------------------------------------------------------------


def test_retry_threshold_changes_repeat_detection(loader):
    def repeats(**kwargs):
        return [
            f
            for f in detect((to_ia3_trace(trace) for trace in loader.load().scan()), **kwargs)
            if f["issue_type"] == "repeated_identical_failed_call"
        ]

    assert repeats()
    assert not repeats(retry_threshold=99)


# -- serialisation ---------------------------------------------------------


def test_findings_serialise_without_a_default_encoder(findings):
    """A bare object() in the payload would raise here, not silently stringify."""
    round_tripped = json.loads(json.dumps(findings))
    assert len(round_tripped) == len(findings)


def test_serialize_renders_the_sentinel_rather_than_crashing():
    from insight_agent.cli.artifacts import jsonable

    assert jsonable({"result": MISSING}) == {"result": "<missing>"}
    json.dumps(jsonable({"result": MISSING}))
