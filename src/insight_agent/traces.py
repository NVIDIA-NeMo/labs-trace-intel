"""Normalized trace contracts shared by evidence streams and the Analyst."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator
from pydantic.experimental.missing_sentinel import MISSING

# Pydantic's missing sentinel is omitted by ``model_dump`` and JSON Schema, while
# an explicitly supplied ``None`` remains JSON null. Tool-result analysis depends
# on this distinction.
UNSET = MISSING


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


class ToolCall(BaseModel):
    """Normalized metadata specific to a tool invocation."""

    call_id: str | None = None
    index: int | None = Field(default=None, ge=0)
    result_id: str | None = None
    result_count: int = Field(default=1, ge=0)
    instrumentation_alias_of: str | None = None
    prior_user_text: str | None = None
    returned_data: bool | UNSET = UNSET


class TokenCounts(BaseModel):
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Span(BaseModel):
    """One nested unit of work in an agent trace."""

    id: str = Field(min_length=1)
    kind: SpanKind
    children: list[Span] = Field(default_factory=list)

    start_time: datetime | None = None
    end_time: datetime | None = None
    # UNSET means the value was not recorded; None means it was explicitly
    # recorded as JSON null. Evidence streams use this to detect missing results.
    input: JsonValue | UNSET = UNSET
    output: JsonValue | UNSET = UNSET
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    token_counts: TokenCounts | None = None
    model: str | None = None

    tool_name: str | None = None
    tool_call: ToolCall | None = None

    error: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

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


class TraceAggregate(BaseModel):
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    token_counts: TokenCounts | None = None


class Trace(BaseModel):
    """One normalized end-to-end agent run."""

    id: str = Field(min_length=1)
    root_spans: list[Span]
    aggregate: TraceAggregate
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

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
