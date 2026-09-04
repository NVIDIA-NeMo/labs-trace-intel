"""Phase 0 smoke tests: the package imports and both engines run end to end.

The import test is not ceremonial. The original handoff shipped the modules in a
directory named ``code/``, which shadows a stdlib module. The src layout is what
fixes it, and this test is what keeps it fixed.
"""

from __future__ import annotations

import code
import importlib

import baseline_corpus
import pytest

import insight_agent
from insight_agent.evidence_streams.anomaly_and_patterns import (
    group_trajectories,
    prepare_traces,
    run_anomaly_and_patterns,
)
from insight_agent.evidence_streams.tool_issues import FINDING_TYPES, build_cards, detect


def test_package_imports_without_shadowing_stdlib():
    assert insight_agent.__version__
    assert code.__name__ == "code"


def test_engine_modules_are_importable_by_dotted_path():
    for name in ("anomaly_and_patterns", "tool_issues"):
        assert importlib.import_module(f"insight_agent.evidence_streams.{name}")


def test_run_anomaly_and_patterns_produces_a_digest():
    result = run_anomaly_and_patterns(
        baseline_corpus.anomaly_and_patterns_traces(), minimum_independent_traces=3
    )

    assert set(type(result).model_fields) >= {
        "prepared",
        "failure_events",
        "anomalies",
        "trajectory_groups",
        "verdict_groups",
        "failure_groups",
        "cross_tool_failure_groups",
        "digest",
    }
    assert result.digest.strip()
    assert len(result.prepared) == len(baseline_corpus.anomaly_and_patterns_traces())


def test_run_anomaly_and_patterns_is_deterministic_in_process():
    traces = baseline_corpus.anomaly_and_patterns_traces()
    first = run_anomaly_and_patterns(traces, minimum_independent_traces=3).digest
    second = run_anomaly_and_patterns(traces, minimum_independent_traces=3).digest
    assert first == second


def test_detect_produces_findings_from_the_known_catalog():
    findings = detect(baseline_corpus.tool_issue_traces())

    assert findings
    assert {f["issue_type"] for f in findings} <= set(FINDING_TYPES)
    assert all(f["issue_id"] for f in findings)
    assert all(f["detector_version"] == "tid-v1" for f in findings)


def test_build_cards_requires_three_independent_cases():
    findings = detect(baseline_corpus.tool_issue_traces())
    cards = build_cards(findings, minimum_independent_cases=3)

    assert cards
    assert any(card.eligible_for_analyst for card in cards)
    for card in cards:
        expected = card.independent_case_count >= 3
        assert card.eligible_for_analyst is expected


def test_build_cards_at_a_higher_threshold_disqualifies_everything():
    findings = detect(baseline_corpus.tool_issue_traces())
    cards = build_cards(findings, minimum_independent_cases=99)

    assert cards
    assert not any(card.eligible_for_analyst for card in cards)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("n_traces", [1, 2])
def test_run_anomaly_and_patterns_abstains_from_grouping_corpora_too_small_to_cluster(n_traces):
    """Below three traces ``run_anomaly_and_patterns`` must abstain, not crash.

    ``group_trajectories`` itself raises in this situation (it needs some k with
    ``2 <= k < n_traces``); ``run_anomaly_and_patterns`` guards the call and reports ``None``,
    which the digest renders as an explicit abstention.
    """
    traces = baseline_corpus.anomaly_and_patterns_traces()[:n_traces]
    result = run_anomaly_and_patterns(traces, minimum_independent_traces=3)

    assert result.trajectory_groups is None
    assert result.digest.strip()


def test_group_trajectories_raises_directly_on_small_corpora():
    """The guard lives in ``run_anomaly_and_patterns``, so direct callers still hit the raise."""
    prepared, _ = prepare_traces(baseline_corpus.anomaly_and_patterns_traces()[:2])
    with pytest.raises(ValueError, match="at least three traces"):
        group_trajectories([item.features for item in prepared])
