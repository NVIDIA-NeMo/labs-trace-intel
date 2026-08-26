"""Shared evidence-stream contracts."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from ..traces import ContractModel, TraceSnapshot


class EvidenceCoverage(ContractModel):
    traces_available: int = Field(ge=0)
    traces_examined: int = Field(ge=0)
    traces_evaluable: int = Field(ge=0)
    abstention_reasons: tuple[str, ...] = ()


class Problem(ContractModel):
    """A potential issue surfaced by one evidence stream."""

    description: str = Field(min_length=1)
    supporting_trace_ids: tuple[str, ...] = Field(min_length=1)


class EvidenceStreamResult(ContractModel):
    stream_name: str = Field(min_length=1)
    stream_version: str = Field(min_length=1)
    status: Literal["completed", "abstained"]
    coverage: EvidenceCoverage
    problems: tuple[Problem, ...]
    artifacts: Any = None
    withheld_problem_count: int = Field(default=0, ge=0)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_abstention(self) -> EvidenceStreamResult:
        if self.status == "abstained" and self.problems:
            raise ValueError("an abstained evidence stream cannot return problems")
        return self


class EvidenceStream(Protocol):
    name: str
    version: str

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult: ...


__all__ = ["EvidenceCoverage", "EvidenceStream", "EvidenceStreamResult", "Problem"]
