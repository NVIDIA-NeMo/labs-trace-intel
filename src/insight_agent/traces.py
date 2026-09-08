# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized trace contracts shared by evidence streams and the Analyst."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.experimental.missing_sentinel import MISSING

# Pydantic's missing sentinel is omitted by ``model_dump`` and JSON Schema, while
# an explicitly supplied ``None`` remains JSON null. Tool-result analysis depends
# on this distinction.
UNSET = MISSING


class _TraceModel(BaseModel):
    """Base configuration for forward-compatible canonical trace models."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


class SpanKind(str, Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    EVALUATOR = "EVALUATOR"
    GUARDRAIL = "GUARDRAIL"
    UNKNOWN = "UNKNOWN"


class ToolCall(_TraceModel):
    """Source metadata that identifies and joins one tool invocation."""

    call_id: str | None = Field(
        default=None,
        description=(
            "The source's tool-call identifier. Preserve repeated IDs so duplicate "
            "instrumentation remains observable."
        ),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based tool-call execution order when the source records it.",
    )
    result_id: str | None = Field(
        default=None,
        description="The source result's genuine reference to this call, when available.",
    )
    result_count: int = Field(
        default=1,
        ge=0,
        description="Number of source results matched to this call; use zero when none exists.",
    )
    instrumentation_alias_of: str | None = Field(
        default=None,
        description="Canonical tool name when the observed name is an instrumentation alias.",
    )
    prior_user_text: str | None = Field(
        default=None,
        description="Nearest preceding user text, only when captured from the source.",
    )
    returned_data: bool | UNSET = Field(
        default=UNSET,
        description="Whether the result returned usable data; omit when the source cannot say.",
    )


class TokenCounts(_TraceModel):
    """Non-negative token counts using the source's authoritative accounting."""

    input_tokens: int = Field(ge=0, description="Uncached input tokens.")
    cached_input_tokens: int = Field(ge=0, description="Cached input tokens.")
    output_tokens: int = Field(ge=0, description="Generated output tokens.")


class Span(_TraceModel):
    """One canonical, recursively nested unit of work in an agent trace."""

    id: str = Field(
        min_length=1,
        description="Stable span identifier, unique within its containing trace.",
    )
    kind: SpanKind = Field(description="Normalized span category represented by this source span.")
    children: list[Span] = Field(
        default_factory=list,
        description="Direct child spans in deterministic source execution order.",
    )

    start_time: datetime | None = Field(
        default=None,
        description="Timezone-aware source start time, when recorded.",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Timezone-aware source end time, when recorded.",
    )
    # UNSET means the value was not recorded; None means it was explicitly
    # recorded as JSON null. Evidence streams use this to detect missing results.
    input: JsonValue | UNSET = Field(
        default=UNSET,
        description="Raw recorded input. Omit when absent; explicit JSON null remains null.",
    )
    output: JsonValue | UNSET = Field(
        default=UNSET,
        description="Raw recorded output. Omit when absent; explicit JSON null remains null.",
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Source-reported span cost in US dollars, when available.",
    )
    token_counts: TokenCounts | None = Field(
        default=None,
        description="Source-reported token counts for this span, when available.",
    )
    model: str | None = Field(
        default=None,
        description="Source-reported model identifier, when applicable.",
    )

    tool_name: str | None = Field(
        default=None,
        description="Observed tool name for a TOOL span; preserve the source value.",
    )
    tool_call: ToolCall | None = Field(
        default=None,
        description="Call/result join metadata; valid only for a TOOL span.",
    )

    error: str | None = Field(
        default=None,
        description="Authoritative source error text or status; do not infer one from output text.",
    )
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless source metadata not represented by a typed field.",
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def timestamps_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> Span:
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError(f"span {self.id!r} ends before it starts")
        if self.tool_call is not None and self.kind is not SpanKind.TOOL:
            raise ValueError("tool_call metadata requires kind=TOOL")
        return self


class TraceAggregate(_TraceModel):
    """Source-reported or deterministically derived whole-trace measurements."""

    cost_usd: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Whole-trace cost in US dollars, when available.",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Whole-trace latency in milliseconds, when available.",
    )
    token_counts: TokenCounts | None = Field(
        default=None,
        description="Whole-trace token counts, when available.",
    )


class Trace(_TraceModel):
    """Canonical representation of one complete end-to-end agent run.

    Source loaders deterministically construct this model before any evidence
    stream runs. Provider-specific parsing or compatibility logic does not
    belong downstream of this boundary.
    """

    id: str = Field(
        min_length=1,
        description="Stable trace identifier, unique within the loaded corpus.",
    )
    root_spans: list[Span] = Field(
        description="Complete root span trees in deterministic source execution order."
    )
    aggregate: TraceAggregate = Field(
        description="Whole-trace measurements; use an empty object when none are available."
    )
    attributes: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional lossless trace-level source metadata; do not invent values.",
    )

    @model_validator(mode="after")
    def span_ids_are_unique(self) -> Trace:
        seen: set[str] = set()
        pending = list(reversed(self.root_spans))
        while pending:
            span = pending.pop()
            if span.id in seen:
                raise ValueError(f"trace {self.id!r} contains duplicate span id {span.id!r}")
            seen.add(span.id)
            pending.extend(reversed(span.children))
        return self


class TraceSnapshot:
    """A reiterable in-memory view of normalized traces, indexed by ID."""

    def __init__(self, traces: Iterable[Trace]):
        traces_by_id: dict[str, Trace] = {}
        for trace in traces:
            if trace.id in traces_by_id:
                raise ValueError(f"snapshot contains duplicate trace id {trace.id!r}")
            traces_by_id[trace.id] = trace
        self.traces_by_id = traces_by_id

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces_by_id.values())

    def __len__(self) -> int:
        return len(self.traces_by_id)

    def get_trace_by_id(self, trace_id: str) -> Trace:
        return self.traces_by_id[trace_id]

    @property
    def trace_count(self) -> int:
        return len(self)


__all__ = [
    "SpanKind",
    "Span",
    "TokenCounts",
    "TraceAggregate",
    "TraceSnapshot",
    "Trace",
    "ToolCall",
    "UNSET",
]
