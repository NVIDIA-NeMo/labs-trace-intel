"""IA2 evidence preprocessing over the bundled sample corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.evidence_streams.anomaly_and_patterns import (
    DEFAULT_FEATURES,
    RECURRENCE_THRESHOLD,
    Anomaly,
    AnomalyAndPatternsAnalysis,
    FailureGroup,
    _rank_top_terms,
    normalize_error_template,
    run_ia2,
    to_ia2_trace,
)
from insight_agent.trace_loaders import FSDataLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "insight_agent" / "data"
CORPUS = DATA_DIR / "sample_corpus.jsonl"

DIGEST_SECTIONS = (
    "## Reader contract",
    "## Inventory",
    "## Unusual traces",
    "## Recurring ordered trajectory patterns",
    "## Explicit terminal-verdict patterns",
    "## Recurring strict tool-failure signatures",
    "## Insight compilation rules",
)


@pytest.fixture(scope="module")
def traces():
    loader = FSDataLoader(CORPUS)
    return [to_ia2_trace(trace) for trace in loader.load()]


@pytest.fixture(scope="module")
def result(traces):
    return run_ia2(traces, minimum_independent_traces=3)


def test_digest_contains_every_section(result):
    assert isinstance(result, AnomalyAndPatternsAnalysis)
    assert all(isinstance(anomaly, Anomaly) for anomaly in result.anomalies)
    assert all(isinstance(group, FailureGroup) for group in result.failure_groups)
    for section in DIGEST_SECTIONS:
        assert section in result.digest, section


def test_digest_states_the_reader_contract(result):
    """IA2 must never be read as authoring conclusions."""
    digest = result.digest
    assert "does not author an Insight" in digest
    assert "turn an anomaly into an error" in digest


def test_digest_is_byte_identical_across_runs(traces):
    first = run_ia2(traces, minimum_independent_traces=3).digest
    second = run_ia2(traces, minimum_independent_traces=3).digest
    assert first == second


def test_the_deliberate_outlier_is_flagged(result):
    flagged = [anomaly.trace_id for anomaly in result.anomalies if anomaly.is_anomaly]
    assert "docops-outlier" in flagged


def test_every_anomaly_carries_interpretable_reasons(result):
    for anomaly in result.anomalies:
        if anomaly.is_anomaly:
            assert anomaly.anomaly_reasons, anomaly.trace_id
            assert anomaly.source_pointer


def test_outcome_labels_never_enter_anomaly_selection(traces):
    """The algorithm boundary: verdicts may be grouped, never fitted on."""
    stripped = [
        type(t)(
            trace_id=t.trace_id,
            calls=t.calls,
            steps=t.steps,
            source_pointer=t.source_pointer,
            observed_verdict=None,
            cost=t.cost,
            metrics=t.metrics,
        )
        for t in traces
    ]
    with_verdicts = run_ia2(traces, minimum_independent_traces=3).anomalies
    without = run_ia2(stripped, minimum_independent_traces=3).anomalies

    assert [a.is_anomaly for a in with_verdicts] == [a.is_anomaly for a in without]
    assert [a.anomaly_score for a in with_verdicts] == [a.anomaly_score for a in without]


def test_failure_signatures_are_normalised(result):
    """Volatile numbers and paths must fold, or recurrence never accumulates."""
    assert result.failure_groups
    for group in result.failure_groups:
        assert group.independent_trace_count >= RECURRENCE_THRESHOLD
        assert "30s" not in group.signature, group.signature
        assert "<n>" in group.signature or not any(ch.isdigit() for ch in group.signature)


def test_normalize_error_template_folds_volatile_detail():
    template = normalize_error_template("Failed to open /var/run/abc123/file.txt after 42 tries")
    assert "<path>" in template
    assert "42" not in template


def test_cross_tool_grouping_spans_tools(result):
    groups = {group.message_signature: group for group in result.cross_tool_failure_groups}
    timeout = next(g for k, g in groups.items() if "timed out" in k)
    assert set(timeout.tool_names) == {"FileSearchTool", "DatabaseQueryTool"}


def test_verdict_groups_require_independent_traces(result):
    assert result.verdict_groups
    for group in result.verdict_groups:
        assert group["independent_trace_count"] >= RECURRENCE_THRESHOLD


def test_trajectory_clustering_separates_shapes(result):
    assert result.trajectory_groups is not None
    clusters = result.trajectory_groups["clusters"]
    assert len(clusters) >= 2
    assigned = result.trajectory_groups["assignments"]
    # The search traces share a shape and should not be scattered one per cluster.
    search = {assigned[t] for t in assigned if t.startswith("docops-search-")}
    assert len(search) < 4


def test_trajectory_term_ranking_is_stable_across_insignificant_float_differences():
    terms = _rank_top_terms(
        [0.5, 0.5 + 1e-14, 0.4],
        ["alpha", "zeta", "middle"],
    )

    assert terms == ["zeta", "alpha", "middle"]


def test_features_cover_the_documented_defaults(result):
    for prepared in result.prepared:
        assert set(DEFAULT_FEATURES) <= set(prepared.features.numeric)


def test_custom_metrics_become_usable_features(traces):
    result = run_ia2(
        traces,
        feature_names=(*DEFAULT_FEATURES, "turn_count"),
        minimum_independent_traces=3,
    )
    assert result.anomalies


def test_a_missing_custom_feature_errors_informatively(traces):
    with pytest.raises(ValueError) as excinfo:
        run_ia2(
            traces, feature_names=("tool_call_count", "not_logged"), minimum_independent_traces=3
        )
    message = str(excinfo.value)
    assert "not_logged" in message
    assert "metrics" in message
    assert "docops-" in message  # names the offending traces


def test_contamination_controls_how_many_traces_are_flagged(traces):
    few = run_ia2(traces, contamination=0.02, minimum_independent_traces=3).anomalies
    many = run_ia2(traces, contamination=0.30, minimum_independent_traces=3).anomalies
    assert sum(a.is_anomaly for a in many) > sum(a.is_anomaly for a in few)
