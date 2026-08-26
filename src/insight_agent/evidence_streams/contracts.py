"""Shared evidence-stream contracts."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field, JsonValue

from ..traces import ContractModel, TraceSnapshot


class EvidenceCoverage(ContractModel):
    traces_available: int = Field(ge=0)
    traces_examined: int = Field(ge=0)
    traces_evaluable: int = Field(ge=0)
    abstention_reasons: tuple[str, ...] = ()


class EvidenceStreamResult(ContractModel):
    stream_name: str
    stream_version: str
    status: Literal["completed", "abstained"]
    coverage: EvidenceCoverage
    payload: Any
    metrics: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceStream(Protocol):
    name: str
    version: str

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult: ...


__all__ = ["EvidenceCoverage", "EvidenceStream", "EvidenceStreamResult"]
