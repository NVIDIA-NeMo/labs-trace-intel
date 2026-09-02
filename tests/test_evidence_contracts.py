"""Shared evidence-stream handoff contracts."""

import pytest
from pydantic import ValidationError

from insight_agent.evidence_streams.anomaly_and_patterns import (
    AnomalyAndPatternsConfig,
    AnomalyAndPatternsEvidenceStream,
)
from insight_agent.evidence_streams.builtins import (
    ANOMALY_AND_PATTERNS,
    TOOL_ISSUES,
    registered_builtin_streams,
)
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.evidence_streams.tool_issues import ToolIssueConfig
from insight_agent.traces import TraceSnapshot


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


class _ExampleStream:
    def __init__(self, name: str = "example", *, invalid: bool = False) -> None:
        self.name = name
        self.invalid = invalid
        self.validated = False

    def validate_configuration(self) -> None:
        self.validated = True
        if self.invalid:
            raise ValueError("invalid example configuration")

    def analyze(self, snapshot):
        return EvidenceStreamResult(stream_name=self.name, problems=())


def test_registry_validates_streams_when_they_are_explicitly_registered():
    registry = EvidenceStreamRegistry()
    stream = _ExampleStream()

    registry.register(stream)

    assert stream.validated is True
    assert registry.names == ("example",)


def test_registry_rejects_invalid_configuration_and_duplicate_names():
    registry = EvidenceStreamRegistry()

    with pytest.raises(ValueError, match="invalid example configuration"):
        registry.register(_ExampleStream(invalid=True))
    assert registry.names == ()

    registry.register(_ExampleStream())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_ExampleStream())


def test_registry_analyzes_registered_streams_in_registration_order():
    registry = EvidenceStreamRegistry()
    registry.register(_ExampleStream("first"))
    registry.register(_ExampleStream("second"))
    snapshot = TraceSnapshot(())

    results = registry.analyze_all(snapshot)

    assert tuple(result.stream_name for result in results) == ("first", "second")


def test_builtin_streams_accept_typed_configuration_and_are_explicitly_registered():
    anomaly_config = AnomalyAndPatternsConfig(
        contamination=0.1,
        cluster_candidates=(2, 4),
    )
    tool_config = ToolIssueConfig(
        minimum_independent_cases=5,
        retry_threshold=2,
    )
    registry = registered_builtin_streams(
        anomaly_and_patterns=anomaly_config,
        tool_issues=tool_config,
    )

    assert registry.names == (ANOMALY_AND_PATTERNS, TOOL_ISSUES)


def test_typed_stream_configuration_rejects_invalid_values():
    with pytest.raises(ValidationError, match="contamination must be between"):
        AnomalyAndPatternsConfig(contamination=0.9)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ToolIssueConfig(retry_threshold=0)


def test_registration_requires_the_streams_typed_configuration():
    registry = EvidenceStreamRegistry()

    stream = AnomalyAndPatternsEvidenceStream(config={"contamination": 0.1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="requires AnomalyAndPatternsConfig"):
        registry.register(stream)

    assert registry.names == ()
