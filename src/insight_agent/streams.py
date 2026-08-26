"""Evidence-stream contracts and built-in IA2/IA3 streams."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import Field, JsonValue

from .ia2_pipeline import NormalizedCall, NormalizedStep, NormalizedTrace, run_ia2
from .ia3_tid import MISSING, CallRecord, TraceRecord, build_cards, catalog_coverage, detect
from .traces import (
    UNSET,
    ContractModel,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    TraceSnapshot,
)
from .venue import DEFAULT_PROFILE, VenueProfile


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


@dataclass(frozen=True)
class IA2EvidenceArtifacts:
    result: Mapping[str, Any]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolIssueEvidenceArtifacts:
    findings: tuple[Mapping[str, Any], ...]
    cards: tuple[Mapping[str, Any], ...]
    catalog_coverage: Mapping[str, int]


def _duration_ms(span: Span) -> float | None:
    if span.duration_ms is not None:
        return span.duration_ms
    if span.started_at is None or span.ended_at is None:
        return None
    return (span.ended_at - span.started_at).total_seconds() * 1_000


def _span_content(span: Span) -> str:
    value = span.output
    if value is UNSET or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
        messages = value.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            for message in reversed(messages):
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return str(message["content"])
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _native_step_type(span: Span) -> str:
    return {
        SpanKind.TOOL: "tool",
        SpanKind.AGENT: "agent",
        SpanKind.EVALUATOR: "evaluation",
    }.get(span.kind, span.kind.value.lower())


def to_ia2_trace(
    trace: Trace, *, profile: VenueProfile = DEFAULT_PROFILE
) -> NormalizedTrace:
    """Project one normalized trace into IA2's analysis model."""

    calls: list[NormalizedCall] = []
    tool_spans = [span for span in trace.spans if span.kind is SpanKind.TOOL]

    for call_index, span in enumerate(tool_spans):
        details = span.tool_call
        result = None if span.output is UNSET else span.output
        returned_data = details.returned_data if details is not None else UNSET
        if returned_data is not UNSET:
            if isinstance(result, Mapping):
                if profile.returned_data_key not in result:
                    result = {**result, profile.returned_data_key: returned_data}
            else:
                warnings.warn(
                    f"call {(details.call_id if details else span.span_id)!r} in trace "
                    f"{trace.id!r} sets 'returned_data' but its result is not a JSON object, "
                    "so IA2 cannot read it; returned_data_false_rate will stay 0 for this "
                    'call. Wrap the result as {"content": ...} to make it count.',
                    UserWarning,
                    stacklevel=2,
                )
        arguments = span.input if isinstance(span.input, Mapping) else {}
        calls.append(
            NormalizedCall(
                call_id=str(details.call_id if details and details.call_id else span.span_id),
                call_index=(
                    details.index if details is not None and details.index is not None else call_index
                ),
                tool_name=str(span.tool_name or span.name or ""),
                arguments=arguments,
                result=result,
                duration_ms=_duration_ms(span),
                source_pointer=dict(span.source_pointer),
            )
        )

    steps = tuple(
        NormalizedStep(
            step_index=index,
            step_type=span.subtype or _native_step_type(span),
            name=str(span.name or span.tool_name or ""),
            content=span.summary if span.summary is not None else _span_content(span),
            source_pointer=dict(span.source_pointer),
        )
        for index, span in enumerate(trace.spans)
    )
    return NormalizedTrace(
        trace_id=trace.id,
        calls=tuple(calls),
        steps=steps,
        source_pointer=dict(trace.source_pointer),
        observed_verdict=trace.observed_verdict,
        cost=trace.cost_usd,
        metrics=dict(trace.metrics),
    )


def _trace_input_text(trace: Trace) -> str:
    if isinstance(trace.input, str):
        return trace.input

    values: Sequence[Any]
    if isinstance(trace.input, Mapping):
        messages = trace.input.get("messages")
        values = messages if isinstance(messages, Sequence) else ()
    elif isinstance(trace.input, Sequence) and trace.input is not UNSET:
        values = trace.input
    else:
        values = ()
    for message in reversed(values):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def to_ia3_trace(trace: Trace) -> TraceRecord:
    """Project one normalized trace into IA3's analysis model."""

    task_text = _trace_input_text(trace)
    tool_spans = [span for span in trace.spans if span.kind is SpanKind.TOOL]
    calls: list[CallRecord] = []
    for index, span in enumerate(tool_spans):
        details = span.tool_call
        result_count = details.result_count if details is not None else 1
        if result_count < 1:
            result_count = 1
        explicit_error = (
            True
            if span.status is SpanStatus.ERROR
            else False
            if span.status is SpanStatus.SUCCESS
            else None
        )
        prior_user_text = details.prior_user_text if details is not None else None
        calls.append(
            CallRecord(
                trace_id=trace.id,
                call_index=(
                    details.index if details is not None and details.index is not None else index
                ),
                call_id=str(details.call_id if details and details.call_id else span.span_id),
                tool_name=str(span.tool_name or span.name or ""),
                arguments=None if span.input is UNSET else span.input,
                result=MISSING if span.output is UNSET else span.output,
                source_pointer=dict(span.source_pointer),
                result_id=details.result_id if details is not None else None,
                result_count=result_count,
                explicit_error=explicit_error,
                outcome_marker=span.error_type,
                instrumentation_alias_of=(
                    details.instrumentation_alias_of if details is not None else None
                ),
                prior_user_text=str(prior_user_text or task_text),
            )
        )

    return TraceRecord(
        trace_id=trace.id,
        calls=tuple(calls),
        logical_case_id=trace.logical_case_id,
        tool_catalog=trace.tool_catalog,
        orphan_results=tuple(dict(item) for item in trace.orphan_results),
        complete_provenance_context=trace.complete_provenance_context,
    )


@dataclass(frozen=True)
class IA2EvidenceStream:
    name = "ia2"
    version = "1"

    contamination: float = 0.02
    input_scaling: Literal["none", "robust"] = "none"
    cluster_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    minimum_independent_traces: int = 3
    feature_names: tuple[str, ...] | None = None
    profile: VenueProfile = DEFAULT_PROFILE

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult:
        kwargs: dict[str, Any] = {
            "contamination": self.contamination,
            "cluster_candidates": self.cluster_candidates,
            "minimum_independent_traces": self.minimum_independent_traces,
            "input_scaling": self.input_scaling,
            "profile": self.profile,
        }
        if self.feature_names:
            kwargs["feature_names"] = self.feature_names
        traces = [to_ia2_trace(trace, profile=self.profile) for trace in snapshot.scan()]
        result = run_ia2(traces, **kwargs)
        return EvidenceStreamResult(
            stream_name=self.name,
            stream_version=self.version,
            status="completed",
            coverage=EvidenceCoverage(
                traces_available=snapshot.trace_count,
                traces_examined=len(traces),
                traces_evaluable=len(traces),
            ),
            payload=IA2EvidenceArtifacts(
                result=result,
                parameters=kwargs
                | {
                    "feature_names": (
                        list(self.feature_names) if self.feature_names else "default"
                    )
                },
            ),
            metrics={
                "anomaly_count": sum(
                    1 for row in result["anomalies"] if row["is_anomaly"]
                ),
            },
        )


def _tool_issue_abstentions(traces: Sequence[TraceRecord]) -> tuple[str, ...]:
    calls = [call for trace in traces for call in trace.calls]
    checks = (
        (any(trace.tool_catalog is not None for trace in traces), "no tool catalog is available"),
        (any(call.result is MISSING for call in calls), "no call has an unobserved result"),
        (any(call.result_id is not None for call in calls), "no call records a result id"),
        (any(call.result_count > 1 for call in calls), "no call records multiple results"),
        (any(trace.orphan_results for trace in traces), "no trace records orphan results"),
        (
            any(call.instrumentation_alias_of for call in calls),
            "no call records an instrumentation alias",
        ),
        (
            any(trace.complete_provenance_context for trace in traces),
            "no trace asserts complete provenance context",
        ),
    )
    return tuple(reason for evaluable, reason in checks if not evaluable)


@dataclass(frozen=True)
class ToolIssueEvidenceStream:
    name = "tool-issues"
    version = "1"

    minimum_independent_cases: int = 3
    retry_threshold: int = 3
    profile: VenueProfile = DEFAULT_PROFILE

    def analyze(self, snapshot: TraceSnapshot) -> EvidenceStreamResult:
        traces = tuple(to_ia3_trace(trace) for trace in snapshot.scan())
        findings = detect(
            traces,
            profile=self.profile,
            retry_threshold=self.retry_threshold,
        )
        cards = build_cards(
            findings,
            minimum_independent_cases=self.minimum_independent_cases,
        )
        return EvidenceStreamResult(
            stream_name=self.name,
            stream_version=self.version,
            status="completed",
            coverage=EvidenceCoverage(
                traces_available=snapshot.trace_count,
                traces_examined=len(traces),
                traces_evaluable=len(traces),
                abstention_reasons=_tool_issue_abstentions(traces),
            ),
            payload=ToolIssueEvidenceArtifacts(
                findings=tuple(findings),
                cards=tuple(cards),
                catalog_coverage=catalog_coverage(findings),
            ),
            metrics={
                "finding_count": len(findings),
                "card_count": len(cards),
            },
        )


__all__ = [
    "EvidenceCoverage",
    "EvidenceStream",
    "EvidenceStreamResult",
    "IA2EvidenceArtifacts",
    "IA2EvidenceStream",
    "ToolIssueEvidenceArtifacts",
    "ToolIssueEvidenceStream",
    "to_ia2_trace",
    "to_ia3_trace",
]
