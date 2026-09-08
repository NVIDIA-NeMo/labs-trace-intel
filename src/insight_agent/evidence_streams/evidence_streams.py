# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence-stream interfaces and models."""

from __future__ import annotations

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


class EvidenceStream(Protocol):
    name: str

    def validate_configuration(self) -> None:
        """Raise when the stream cannot run with its current configuration."""

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult: ...


__all__ = ["EvidenceStream", "EvidenceStreamResult", "Problem"]
