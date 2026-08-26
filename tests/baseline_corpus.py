"""A small hand-written corpus used as the pre-refactor behavioural baseline.

Phase 1 makes every venue-specific literal configurable through a
``VenueProfile``. The whole point of that refactor is that the *defaults*
reproduce the measured behaviour exactly, so this module builds a fixed corpus
and the golden artefacts in ``tests/data/`` are generated from it **before** the
refactor. ``tests/test_venue_profile.py`` then asserts the post-refactor output
is byte-identical.

Deliberately written against the raw engine dataclasses rather than the
canonical JSONL loader: the loader does not exist yet at Phase 0, and keeping
the baseline independent of it means a loader bug can never silently rewrite the
thing we are comparing against.
"""

from __future__ import annotations

from insight_agent.evidence_streams.anomaly_and_patterns import (
    NormalizedCall,
    NormalizedStep,
    NormalizedTrace,
)
from insight_agent.evidence_streams.tool_issues import CallRecord, TraceRecord

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
    "additionalProperties": False,
}

EXEC_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "mode": {"type": "string", "enum": ["python", "shell"]},
    },
    "required": ["code"],
    "additionalProperties": False,
}

TOOL_CATALOG = {
    "FileSearchTool": SEARCH_SCHEMA,
    "CodeExecutionTool": EXEC_SCHEMA,
    "SessionTool": None,
}


def _call(index, tool, args, result, *, duration=100.0, trace="t"):
    return NormalizedCall(
        call_id=f"{trace}-c{index}",
        call_index=index,
        tool_name=tool,
        arguments=args,
        result=result,
        duration_ms=duration,
        source_pointer={"trace_id": trace, "call_index": index},
    )


def _steps(kinds, trace):
    return tuple(
        NormalizedStep(
            step_index=i,
            step_type=kind,
            name=name,
            content=content,
            source_pointer={"trace_id": trace, "step_index": i},
        )
        for i, (kind, name, content) in enumerate(kinds)
    )


def ia2_traces() -> list[NormalizedTrace]:
    """Four traces: two search-shaped, one code-shaped, one deliberate outlier."""

    traces = []

    for n in range(2):
        trace = f"base-search-{n}"
        traces.append(
            NormalizedTrace(
                trace_id=trace,
                calls=(
                    _call(0, "FileSearchTool", {"query": "alpha"},
                          {"content": "2 matches", "returned_data": True}, trace=trace),
                    _call(1, "FileSearchTool", {"query": "beta"},
                          {"content": "Error: connection timed out after 30s",
                           "returned_data": False}, duration=4200.0, trace=trace),
                    _call(2, "SessionTool", {"document_id": "DOC-1"},
                          {"content": "[NO_ACTIVE_SESSION] open a session first"}, trace=trace),
                ),
                steps=_steps(
                    [
                        ("planning", "plan", "search then read"),
                        ("tool", "FileSearchTool", ""),
                        ("tool", "FileSearchTool", ""),
                        ("tool", "SessionTool", ""),
                        ("evaluation", "boundary", "The task completed successfully."),
                    ],
                    trace,
                ),
                source_pointer={"trace_id": trace},
                observed_verdict="dead_end: no artifact produced",
                cost=0.42 + n,
                metrics={"turn_count": 6.0 + n},
            )
        )

    trace = "base-code-0"
    traces.append(
        NormalizedTrace(
            trace_id=trace,
            calls=(
                _call(0, "CodeExecutionTool", {"code": "import mesh", "mode": "python"},
                      {"content": 'Traceback (most recent call last):\n  File "<stdin>", '
                                  "line 1\nModuleNotFoundError: No module named 'mesh'"},
                      duration=2210.0, trace=trace),
                _call(1, "CodeExecutionTool", {"code": "print(1)", "mode": "python"},
                      {"content": "1", "returned_data": True}, trace=trace),
            ),
            steps=_steps(
                [
                    ("agent", "reflect", "try running the check"),
                    ("tool", "CodeExecutionTool", ""),
                    ("tool", "CodeExecutionTool", ""),
                    ("evaluation", "boundary", "Done."),
                ],
                trace,
            ),
            source_pointer={"trace_id": trace},
            observed_verdict="completed",
            cost=1.10,
            metrics={"turn_count": 4.0},
        )
    )

    # One deliberate outlier so the 2% contamination default has something to
    # flag on a corpus this small.
    trace = "base-outlier-0"
    traces.append(
        NormalizedTrace(
            trace_id=trace,
            calls=tuple(
                _call(i, "FileSearchTool", {"query": "same"},
                      {"content": "x" * 40_000, "returned_data": False},
                      duration=9000.0, trace=trace)
                for i in range(24)
            ),
            steps=_steps(
                [("tool", "FileSearchTool", "")] * 24 + [("evaluation", "boundary", "Gave up.")],
                trace,
            ),
            source_pointer={"trace_id": trace},
            observed_verdict="dead_end: no artifact produced",
            cost=9.9,
            metrics={"turn_count": 24.0},
        )
    )

    return traces


def ia3_traces() -> list[TraceRecord]:
    """Exercises the traceback gate, the state patterns and the retry rules."""

    from insight_agent.evidence_streams.tool_issues import MISSING

    records = []
    for n in range(3):
        trace = f"base-tid-{n}"
        records.append(
            TraceRecord(
                trace_id=trace,
                logical_case_id=f"case-{n}",
                complete_provenance_context=True,
                tool_catalog=TOOL_CATALOG,
                calls=(
                    CallRecord(
                        trace_id=trace,
                        call_index=0,
                        call_id=f"{trace}-c0",
                        tool_name="FileSearchTool",
                        arguments={"query": "alpha", "limit": "five"},
                        result={"content": "Error: limit must be an integer"},
                        source_pointer={"trace_id": trace, "call_index": 0},
                        prior_user_text="find alpha and run the check",
                    ),
                    CallRecord(
                        trace_id=trace,
                        call_index=1,
                        call_id=f"{trace}-c1",
                        tool_name="CodeExecutionTool",
                        arguments={"code": "import mesh", "mode": "python"},
                        result={"content": "Traceback (most recent call last):\n"
                                           "ModuleNotFoundError: No module named 'mesh'"},
                        source_pointer={"trace_id": trace, "call_index": 1},
                        prior_user_text="find alpha and run the check",
                    ),
                    CallRecord(
                        trace_id=trace,
                        call_index=2,
                        call_id=f"{trace}-c2",
                        tool_name="SessionTool",
                        arguments={"document_id": "DOC-99213"},
                        result={"content": "[NO_ACTIVE_SESSION] invalid document "
                                           "DOC-99213: not found"},
                        source_pointer={"trace_id": trace, "call_index": 2},
                        prior_user_text="find alpha and run the check",
                    ),
                    CallRecord(
                        trace_id=trace,
                        call_index=3,
                        call_id=f"{trace}-c3",
                        tool_name="FileSearchTool",
                        arguments={"query": "gamma"},
                        result=MISSING,
                        source_pointer={"trace_id": trace, "call_index": 3},
                        prior_user_text="find alpha and run the check",
                    ),
                ),
            )
        )
    return records
