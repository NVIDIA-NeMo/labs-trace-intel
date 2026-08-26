"""The venue-profile refactor must not change any measured behaviour.

Every venue-specific literal in the engines moved into ``VenueProfile`` with its
original value as the default. The golden artefacts in ``tests/data/`` were
generated from ``baseline_corpus`` *before* that refactor, so these tests are
what prove the defaults are faithful.
"""

from __future__ import annotations

import json
import pathlib

import baseline_corpus
import pytest

from insight_agent.evidence_streams.anomaly_and_patterns import run_ia2
from insight_agent.evidence_streams.tool_issues import detect
from insight_agent.venue import DEFAULT_PROFILE, VenueProfile, load_profile

DATA = pathlib.Path(__file__).parent / "data"


def _jsonable(value):
    """Normalise tuples to lists so golden JSON compares structurally."""
    return json.loads(json.dumps(value))


# -- behaviour preservation ------------------------------------------------


def test_ia2_digest_matches_the_pre_refactor_golden():
    digest = run_ia2(baseline_corpus.ia2_traces(), minimum_independent_traces=3).digest
    assert digest == (DATA / "baseline_digest.md").read_text(encoding="utf-8")


def test_ia2_default_profile_is_identical_to_passing_none():
    traces = baseline_corpus.ia2_traces()
    implicit = run_ia2(traces, minimum_independent_traces=3).digest
    explicit = run_ia2(traces, minimum_independent_traces=3, profile=DEFAULT_PROFILE).digest
    assert implicit == explicit


def test_ia3_findings_match_the_pre_refactor_golden():
    golden = json.loads((DATA / "baseline_findings.json").read_text(encoding="utf-8"))
    assert _jsonable(detect(baseline_corpus.ia3_traces())) == golden


def test_ia3_default_profile_is_identical_to_passing_none():
    traces = baseline_corpus.ia3_traces()
    assert _jsonable(detect(traces)) == _jsonable(detect(traces, profile=DEFAULT_PROFILE))


# -- the knobs actually do something ---------------------------------------


def _traceback_markers(findings):
    return [f for f in findings if f.get("mechanism_key") == "python_traceback"]


def test_renaming_the_code_execution_tool_disables_traceback_decoding_in_ia3():
    traces = baseline_corpus.ia3_traces()
    assert _traceback_markers(detect(traces)), "baseline must contain a traceback finding"

    renamed = DEFAULT_PROFILE.with_overrides(code_execution_tools=frozenset({"PythonSandbox"}))
    assert not _traceback_markers(detect(traces, profile=renamed))


def test_ia2_traceback_gate_follows_the_profile_symmetrically():
    from insight_agent.evidence_streams.anomaly_and_patterns import decode_explicit_failure

    text = {"content": "Traceback (most recent call last):\nValueError: boom"}

    failed, marker, _ = decode_explicit_failure("CodeExecutionTool", text)
    assert (failed, marker) == (True, "python_traceback")

    failed, marker, _ = decode_explicit_failure("PythonSandbox", text)
    assert (failed, marker) == (False, None)

    renamed = DEFAULT_PROFILE.with_overrides(code_execution_tools=frozenset({"PythonSandbox"}))
    failed, marker, _ = decode_explicit_failure("PythonSandbox", text, profile=renamed)
    assert (failed, marker) == (True, "python_traceback")


def test_custom_state_patterns_replace_the_venue_specific_defaults():
    traces = baseline_corpus.ia3_traces()
    state_type = "explicit_prerequisite_or_state_failure"
    assert [f for f in detect(traces) if f["issue_type"] == state_type]

    # A profile whose patterns cannot match the sample text must abstain.
    quiet = DEFAULT_PROFILE.with_overrides(state_patterns=(("never", r"(?i)\bzzzz-not-present\b"),))
    assert not [f for f in detect(traces, profile=quiet) if f["issue_type"] == state_type]


def test_code_execution_share_counts_every_configured_tool():
    from insight_agent.evidence_streams.anomaly_and_patterns import extract_trace_features

    traces = {t.trace_id: t for t in baseline_corpus.ia2_traces()}
    trace = traces["base-code-0"]  # 2 calls, both CodeExecutionTool

    assert extract_trace_features(trace).features.numeric["code_execution_share"] == 1.0

    renamed = DEFAULT_PROFILE.with_overrides(code_execution_tools=frozenset({"OtherTool"}))
    assert extract_trace_features(trace, profile=renamed).features.numeric["code_execution_share"] == 0.0


def test_returned_data_key_is_configurable():
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        NormalizedCall,
        NormalizedTrace,
        extract_trace_features,
    )

    trace = NormalizedTrace(
        trace_id="t",
        calls=(
            NormalizedCall(
                call_id="c0",
                call_index=0,
                tool_name="Search",
                arguments={},
                result={"content": "none", "had_rows": False},
            ),
        ),
    )
    assert extract_trace_features(trace).features.numeric["returned_data_false_rate"] == 0.0

    custom = DEFAULT_PROFILE.with_overrides(returned_data_key="had_rows")
    features = extract_trace_features(trace, profile=custom).features
    assert features.numeric["returned_data_false_rate"] == 1.0


def test_retry_threshold_is_a_parameter_not_a_module_global():
    """`repeated_identical_failed_call` fired at a hardcoded 3 before Phase 1."""
    from insight_agent.evidence_streams.tool_issues import CallRecord, TraceRecord

    calls = tuple(
        CallRecord(
            trace_id="t",
            call_index=i,
            call_id=f"c{i}",
            tool_name="DatabaseQueryTool",
            arguments={"q": "same"},
            result={"content": "Error: connection refused"},
        )
        for i in range(3)
    )
    trace = TraceRecord(trace_id="t", calls=calls)

    def repeats(**kwargs):
        return [
            f
            for f in detect([trace], **kwargs)
            if f["issue_type"] == "repeated_identical_failed_call"
        ]

    assert repeats()
    assert not repeats(retry_threshold=4)


# -- profile serialisation -------------------------------------------------


def test_profile_json_round_trip():
    profile = VenueProfile(
        name="example-venue",
        code_execution_tools=frozenset({"PythonSandbox", "ShellTool"}),
        agent_step_types=frozenset({"think"}),
        evaluation_step_type="verdict",
        returned_data_key="had_rows",
        state_patterns=(("needs_login", r"(?i)please log in"),),
        notes={"owner": "someone"},
    )
    assert VenueProfile.from_dict(json.loads(profile.to_json())) == profile


def test_load_profile_accepts_none_a_mapping_and_a_path(tmp_path):
    assert load_profile(None) is DEFAULT_PROFILE
    assert load_profile({"name": "x"}).name == "x"

    path = tmp_path / "venue.json"
    path.write_text(VenueProfile(name="from-disk").to_json(), encoding="utf-8")
    assert load_profile(path).name == "from-disk"


def test_profile_rejects_unknown_fields_and_bad_regexes():
    with pytest.raises(ValueError, match="unknown venue profile field"):
        VenueProfile.from_dict({"code_exec_tools": ["X"]})

    with pytest.raises(ValueError, match="not a valid regex"):
        VenueProfile(state_patterns=(("broken", "(unclosed"),))


# -- robustness fixes ------------------------------------------------------


def test_metric_shadowing_a_builtin_feature_warns():
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        NormalizedCall,
        NormalizedTrace,
        extract_trace_features,
    )

    trace = NormalizedTrace(
        trace_id="t",
        calls=(NormalizedCall(call_id="c0", call_index=0, tool_name="Search", arguments={}),),
        metrics={"tool_call_count": 999.0},
    )
    with pytest.warns(UserWarning, match="shadows a built-in IA2 feature"):
        features = extract_trace_features(trace).features
    # The overwrite itself is preserved; only the silence was removed.
    assert features.numeric["tool_call_count"] == 999.0


def test_group_trajectories_can_abstain_instead_of_raising():
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        group_trajectories,
        prepare_traces,
    )

    prepared, _ = prepare_traces(baseline_corpus.ia2_traces()[:2])
    records = [item.features for item in prepared]

    with pytest.raises(ValueError, match="at least three traces"):
        group_trajectories(records)

    result = group_trajectories(records, on_insufficient_traces="skip")
    assert result["status"] == "not_evaluable"
    assert result["clusters"] == []
    assert "2 trace(s)" in result["reason"]


def test_digest_trace_citation_guard_survives_python_O():
    """The guard was a bare `assert`, which vanishes under `python -O`."""
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        DEFAULT_FEATURES,
        build_evidence_digest,
        prepare_traces,
        select_anomalies,
    )

    prepared, _ = prepare_traces(baseline_corpus.ia2_traces())
    anomalies = select_anomalies([p.features for p in prepared], DEFAULT_FEATURES)
    # Fabricate a flagged row citing a trace that was never prepared.
    doctored = [dict(row) for row in anomalies]
    doctored[0] = {**doctored[0], "trace_id": "ghost-trace", "is_anomaly": True}

    with pytest.raises(RuntimeError, match="ghost-trace"):
        build_evidence_digest(prepared, doctored, None, [], [], [])


def test_missing_feature_error_names_the_offending_traces():
    from insight_agent.evidence_streams.anomaly_and_patterns import (
        prepare_traces,
        select_anomalies,
    )

    prepared, _ = prepare_traces(baseline_corpus.ia2_traces())
    with pytest.raises(ValueError, match="no_such_feature") as excinfo:
        select_anomalies([p.features for p in prepared], ["no_such_feature"])
    assert "base-" in str(excinfo.value)
