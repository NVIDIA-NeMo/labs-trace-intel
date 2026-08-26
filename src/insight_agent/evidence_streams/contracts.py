"""Shared evidence-stream contracts."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field

from ..traces import ContractModel, TraceSnapshot


class Problem(ContractModel):
    """A potential issue surfaced by one evidence stream."""

    description: str = Field(min_length=1)
    supporting_trace_ids: tuple[str, ...] = Field(min_length=1)


class EvidenceStreamResult(ContractModel):
    stream_name: str = Field(min_length=1)
    problems: tuple[Problem, ...]
    artifacts: Any = None


class EvidenceStream(Protocol):
    name: str

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult: ...


__all__ = ["EvidenceStream", "EvidenceStreamResult", "Problem"]
