# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence-stream interfaces and models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from insight_agent.traces import TraceSnapshot


class Problem(BaseModel):
    """A potential issue surfaced by one evidence stream."""

    description: str = Field(min_length=1)
    supporting_trace_ids: tuple[str, ...] = Field(min_length=1)


class EvidenceStreamResult(BaseModel):
    stream_name: str = Field(min_length=1)
    problems: tuple[Problem, ...]
    artifacts: Any = None
    finding_count: int = Field(
        default=0, ge=0, description="Observations before candidate filtering"
    )
    skip_reason: str | None = None
    limitations: tuple[str, ...] = ()


class EvidenceStream(Protocol):
    @property
    def name(self) -> str: ...

    def validate_configuration(self) -> None:
        """Raise when the stream cannot run with its current configuration."""

    async def analyze(
        self, snapshot: TraceSnapshot, *, on_start: Callable[[], None] | None = None
    ) -> EvidenceStreamResult:
        """Call on_start only after prerequisites pass; return a skip reason otherwise."""
        ...


__all__ = ["EvidenceStream", "EvidenceStreamResult", "Problem"]
