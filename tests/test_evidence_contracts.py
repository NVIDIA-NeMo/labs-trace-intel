"""Shared evidence-stream handoff contracts."""

import pytest
from pydantic import ValidationError

from insight_agent.evidence_streams.contracts import EvidenceStreamResult, Problem


def test_problem_requires_a_description_and_supporting_trace():
    with pytest.raises(ValidationError):
        Problem(description="", supporting_trace_ids=("trace-1",))
    with pytest.raises(ValidationError):
        Problem(description="candidate", supporting_trace_ids=())


def test_evidence_result_keeps_problems_separate_from_native_artifacts():
    problem = Problem(description="candidate", supporting_trace_ids=("trace-1",))
    result = EvidenceStreamResult(
        stream_name="example",
        problems=(problem,),
        artifacts={"native": "detail"},
    )

    assert result.problems == (problem,)
    assert result.artifacts == {"native": "detail"}
