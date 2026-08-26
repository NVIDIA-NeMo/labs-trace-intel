"""Normalized trace contracts shared by evidence streams and the Analyst."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    field_validator,
    model_validator,
)
from pydantic.experimental.missing_sentinel import MISSING

# Pydantic's missing sentinel is omitted by ``model_dump`` and JSON Schema, while
# an explicitly supplied ``None`` remains JSON null. IA3 depends on this exact
# distinction for tool results.
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


class SpanStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ContractModel(BaseModel):
    """Strict, frozen base for normalized public contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(ContractModel):
    """Normalized metadata specific to a tool invocation."""

    call_id: str | None = None
    index: int | None = Field(default=None, ge=0)
    result_id: str | None = None
    result_count: int = Field(default=1, ge=0)
    instrumentation_alias_of: str | None = None
    prior_user_text: str | None = None
    returned_data: bool | UNSET = UNSET


class Span(ContractModel):
    """One unit of work in an agent trace."""

    span_id: str = Field(min_length=1)
    kind: SpanKind
    status: SpanStatus = SpanStatus.UNKNOWN
    # Source records can preserve order without recording wall-clock timestamps.
    started_at: datetime | None = None
    parent_span_id: str | None = None
    name: str | None = None
    subtype: str | None = None
    summary: str | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    input: JsonValue | UNSET = UNSET
    output: JsonValue | UNSET = UNSET
    provider: str | None = None
    model: str | None = None
    tool_name: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    error_type: str | None = None
    error_message: str | None = None
    tool_call: ToolCall | None = None
    source_pointer: dict[str, JsonValue] = Field(default_factory=dict)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at")
    @classmethod
    def timestamps_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def end_does_not_precede_start(self) -> Span:
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError(f"span {self.span_id!r} ends before it starts")
        if self.tool_call is not None and self.kind is not SpanKind.TOOL:
            raise ValueError("tool_call metadata requires kind=TOOL")
        return self


class Trace(ContractModel):
    """One normalized end-to-end agent run."""

    id: str = Field(min_length=1)
    spans: tuple[Span, ...]
    schema_version: Literal["trace/v1"] = "trace/v1"
    root_span_id: str | None = None
    session_id: str | None = None
    name: str | None = None
    input: JsonValue | UNSET = UNSET
    output: JsonValue | UNSET = UNSET
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: SpanStatus = SpanStatus.UNKNOWN
    agent_name: str | None = None
    agent_version: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    models: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    tool_catalog: dict[str, JsonValue | None] | None = None
    logical_case_id: str | None = None
    observed_verdict: str | None = None
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    complete_provenance_context: bool = False
    orphan_results: tuple[dict[str, JsonValue], ...] = ()
    evaluation_context: dict[str, JsonValue] = Field(default_factory=dict)
    source_pointer: dict[str, JsonValue] = Field(default_factory=dict)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "ended_at")
    @classmethod
    def timestamps_include_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_trace(self) -> Trace:
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError(f"trace {self.id!r} ends before it starts")
        self._validate_span_graph()
        return self

    def _validate_span_graph(self) -> None:
        by_id: dict[str, Span] = {}
        positions: dict[str, int] = {}
        for position, span in enumerate(self.spans):
            if span.span_id in by_id:
                raise ValueError(
                    f"trace {self.id!r} contains duplicate span_id {span.span_id!r}"
                )
            by_id[span.span_id] = span
            positions[span.span_id] = position

        if self.root_span_id is not None:
            root = by_id.get(self.root_span_id)
            if root is None:
                raise ValueError(
                    f"trace {self.id!r} root_span_id {self.root_span_id!r} does not resolve"
                )
            if root.parent_span_id is not None:
                raise ValueError(f"trace {self.id!r} root span must not have a parent")

        partial = self.attributes.get("trace.partial") is True
        for span in self.spans:
            parent_id = span.parent_span_id
            if parent_id is not None and parent_id not in by_id and not partial:
                raise ValueError(
                    f"trace {self.id!r} span {span.span_id!r} references missing parent "
                    f"{parent_id!r}"
                )

        # Detect cycles before reporting order so a cycle has one precise error.
        for span in self.spans:
            seen = {span.span_id}
            parent_id = span.parent_span_id
            while parent_id is not None and parent_id in by_id:
                if parent_id in seen:
                    raise ValueError(f"trace {self.id!r} contains a parent cycle")
                seen.add(parent_id)
                parent_id = by_id[parent_id].parent_span_id

        for span in self.spans:
            parent_id = span.parent_span_id
            if parent_id in positions and positions[parent_id] >= positions[span.span_id]:
                raise ValueError(
                    f"trace {self.id!r} is not in canonical order: parent {parent_id!r} "
                    f"must precede child {span.span_id!r}"
                )

        previous: Span | None = None
        for span in self.spans:
            if span.started_at is None:
                continue
            if (
                previous is not None
                and previous.started_at is not None
                and span.started_at < previous.started_at
            ):
                raise ValueError(
                    f"trace {self.id!r} is not in canonical temporal order: "
                    f"span {span.span_id!r} starts before {previous.span_id!r}"
                )
            previous = span


class TraceSnapshot(ContractModel):
    """An in-memory corpus with a fresh iterator for every scan."""

    snapshot_id: str = Field(min_length=1)
    source: str
    traces: tuple[Trace, ...] = Field(repr=False, exclude=True)

    @model_validator(mode="after")
    def trace_ids_are_unique(self) -> TraceSnapshot:
        seen: set[str] = set()
        for trace in self.traces:
            if trace.id in seen:
                raise ValueError(f"snapshot contains duplicate trace id {trace.id!r}")
            seen.add(trace.id)
        return self

    @classmethod
    def from_traces(cls, traces: Iterable[Trace], *, source: str = "<memory>") -> TraceSnapshot:
        materialized = tuple(traces)
        payload = json.dumps(
            [trace.model_dump(mode="json") for trace in materialized],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        snapshot_id = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        return cls(snapshot_id=snapshot_id, source=source, traces=materialized)

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    def scan(self) -> Iterator[Trace]:
        return iter(self.traces)


__all__ = [
    "ContractModel",
    "JsonValue",
    "Span",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "TraceSnapshot",
    "ToolCall",
    "UNSET",
]
