"""Shared evidence-stream handoff contracts."""

import pytest
from pydantic import ValidationError

from insight_agent.evidence_streams.contracts import (
    EvidenceCoverage,
    EvidenceStreamResult,
    Problem,
)


def test_problem_requires_a_description_and_supporting_trace():
    with pytest.raises(ValidationError):
        Problem(description="", supporting_trace_ids=("trace-1",))
    with pytest.raises(ValidationError):
        Problem(description="candidate", supporting_trace_ids=())


def test_evidence_result_keeps_problems_separate_from_native_artifacts():
    problem = Problem(description="candidate", supporting_trace_ids=("trace-1",))
    result = EvidenceStreamResult(
        stream_name="example",
        stream_version="1",
        status="completed",
        coverage=EvidenceCoverage(
            traces_available=1,
            traces_examined=1,
            traces_evaluable=1,
        ),
        problems=(problem,),
        artifacts={"native": "detail"},
    )

    assert result.problems == (problem,)
    assert result.artifacts == {"native": "detail"}


def test_abstained_stream_cannot_return_problems():
    with pytest.raises(ValidationError, match="cannot return problems"):
        EvidenceStreamResult(
            stream_name="example",
            stream_version="1",
            status="abstained",
            coverage=EvidenceCoverage(
                traces_available=1,
                traces_examined=0,
                traces_evaluable=0,
            ),
            problems=(
                Problem(description="candidate", supporting_trace_ids=("trace-1",)),
            ),
        )
