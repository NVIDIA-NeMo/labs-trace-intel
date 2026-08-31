"""Load shared ``insight-trace/v1`` records into normalized traces."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.evidence_streams.common.insight_trace.validation import (
    Diagnostic,
    ValidationReport,
    iter_jsonl,
    validate_records,
)
from insight_agent.traces import UNSET, Span, SpanKind, SpanStatus, ToolCall, Trace, TraceSnapshot

__all__ = [
    "InsightTraceLoader",
    "InsightTraceOptions",
    "TraceLoadError",
    "load_tool_catalog",
]


class TraceLoadError(ValueError):
    """Raised when a trace source cannot be loaded.

    Carries the full diagnostic list so callers can render every problem at
    once rather than one per run.
    """

    def __init__(self, message: str, report: ValidationReport):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class InsightTraceOptions:
    """Options for loading ``insight-trace/v1`` records."""

    #: Source-wide fallback catalog, used for any record that has none of its
    #: own. Record-level catalogs always win.
    tool_catalog: Mapping[str, Mapping[str, Any] | None] | None = None

    #: Treat validation errors as fatal. Warnings are always surfaced.
    strict: bool = True

    allow_metric_shadowing: bool = False


# -- helpers ---------------------------------------------------------------


def _result_is_missing(call: Mapping[str, Any]) -> bool:
    """Decide whether this source record reports an unobserved result.

    Key absence is the primary signal; ``result_missing`` and
    ``result_count: 0`` are explicit escape hatches for emitters that cannot
    omit a key (many ORMs and protobuf-to-JSON bridges fill defaults).
    """

    return (
        "result" not in call or call.get("result_missing") is True or call.get("result_count") == 0
    )


def _sorted_calls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = [c for c in record.get("calls", []) if isinstance(c, Mapping)]
    return sorted(calls, key=lambda c: (c.get("call_index", 0), str(c.get("call_id", ""))))


def _sorted_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = [s for s in record.get("steps", []) if isinstance(s, Mapping)]
    return sorted(steps, key=lambda s: s.get("step_index", 0))


# -- normalized Trace ------------------------------------------------------


def _span_id(source_id: str, used: set[str]) -> str:
    """Return a unique span id while retaining source identity in ToolCall."""

    if source_id not in used:
        used.add(source_id)
        return source_id
    occurrence = 2
    while f"{source_id}#{occurrence}" in used:
        occurrence += 1
    span_id = f"{source_id}#{occurrence}"
    used.add(span_id)
    return span_id


def _tool_span(
    call: Mapping[str, Any],
    *,
    used_span_ids: set[str],
    step: Mapping[str, Any] | None = None,
) -> Span:
    call_id = str(call["call_id"])
    explicit_error = call.get("explicit_error")
    status = (
        SpanStatus.ERROR
        if explicit_error is True
        else SpanStatus.SUCCESS
        if explicit_error is False
        else SpanStatus.UNKNOWN
    )
    output = UNSET if _result_is_missing(call) else call.get("result")
    returned_data = call.get("returned_data", UNSET)
    return Span(
        span_id=_span_id(call_id, used_span_ids),
        kind=SpanKind.TOOL,
        status=status,
        name=str(call["tool_name"]),
        subtype=str(step["step_type"]) if step is not None else "tool",
        summary=str(step.get("content") or "") if step is not None else None,
        duration_ms=call.get("duration_ms"),
        tool_name=str(call["tool_name"]),
        input=call.get("arguments"),
        output=output,
        error_type=call.get("outcome_marker"),
        tool_call=ToolCall(
            call_id=call_id,
            index=int(call["call_index"]),
            result_id=call.get("result_id"),
            result_count=int(call.get("result_count", 1)),
            instrumentation_alias_of=call.get("instrumentation_alias_of"),
            prior_user_text=call.get("prior_user_text"),
            returned_data=returned_data,
        ),
        source_pointer=dict(call.get("source_pointer") or {}),
    )


def _step_span(
    trace_id: str,
    step: Mapping[str, Any],
    *,
    used_span_ids: set[str],
) -> Span:
    step_index = int(step["step_index"])
    step_type = str(step["step_type"])
    if step_type == "evaluation":
        kind = SpanKind.EVALUATOR
    elif step_type in {"agent", "agent_step", "planning", "user"}:
        kind = SpanKind.AGENT
    else:
        kind = SpanKind.UNKNOWN
    span_id = _span_id(f"{trace_id}#step-{step_index}", used_span_ids)
    content = step.get("content")
    return Span(
        span_id=span_id,
        kind=kind,
        name=str(step.get("name") or "") or None,
        subtype=step_type,
        summary=str(content or ""),
        output=content if content is not None else UNSET,
        source_pointer=dict(step.get("source_pointer") or {}),
    )


def _normalize_trace(
    record: Mapping[str, Any],
    *,
    tool_catalog: Mapping[str, Any] | None = None,
) -> Trace:
    """Normalize one validated input record."""

    calls = _sorted_calls(record)
    steps = _sorted_steps(record)
    used_span_ids: set[str] = set()
    used_calls: set[int] = set()
    spans: list[Span] = []
    next_call = 0

    for step in steps:
        if step.get("step_type") == "tool" and next_call < len(calls):
            call = calls[next_call]
            used_calls.add(next_call)
            next_call += 1
            spans.append(
                _tool_span(
                    call,
                    used_span_ids=used_span_ids,
                    step=step,
                )
            )
        else:
            spans.append(_step_span(str(record["trace_id"]), step, used_span_ids=used_span_ids))

    for index, call in enumerate(calls):
        if index not in used_calls:
            spans.append(
                _tool_span(
                    call,
                    used_span_ids=used_span_ids,
                )
            )

    catalog = record.get("tool_catalog")
    if catalog is None:
        catalog = tool_catalog

    return Trace(
        id=str(record["trace_id"]),
        spans=tuple(spans),
        input=record["task_text"] if "task_text" in record else UNSET,
        cost_usd=record.get("cost"),
        tool_catalog=dict(catalog) if isinstance(catalog, Mapping) else None,
        logical_case_id=record.get("logical_case_id"),
        observed_verdict=record.get("observed_verdict"),
        metrics=dict(record.get("metrics") or {}),
        complete_provenance_context=bool(record.get("complete_provenance_context", False)),
        orphan_results=tuple(
            dict(item) for item in record.get("orphan_results", ()) if isinstance(item, Mapping)
        ),
        source_pointer=dict(record.get("source_pointer") or {}),
    )


# -- loader ----------------------------------------------------------------


@dataclass(frozen=True)
class InsightTraceLoader:
    """Validate and normalize one ``insight-trace/v1`` source."""

    records: tuple[Mapping[str, Any], ...]
    options: InsightTraceOptions = field(default_factory=InsightTraceOptions)
    report: ValidationReport = field(default_factory=ValidationReport)
    source: str = "<memory>"

    def __len__(self) -> int:
        return len(self.records)

    def load(self) -> TraceSnapshot:
        """Normalize the validated records into a reiterable snapshot."""

        return TraceSnapshot.from_traces(
            (
                _normalize_trace(
                    record,
                    tool_catalog=self.options.tool_catalog,
                )
                for record in self.records
            ),
            source=self.source,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        options: InsightTraceOptions | None = None,
        *,
        source: str = "<memory>",
    ) -> InsightTraceLoader:
        """Create a loader from already-parsed records."""

        options = options or InsightTraceOptions()
        numbered = [(index + 1, record) for index, record in enumerate(records)]
        report = validate_records(numbered, allow_metric_shadowing=options.allow_metric_shadowing)

        if report.errors and options.strict:
            raise TraceLoadError(
                f"{source}: {len(report.errors)} validation error(s); "
                "run `insight-agent validate` for the full list",
                report,
            )
        for diagnostic in report.warnings:
            warnings.warn(diagnostic.format(), UserWarning, stacklevel=2)

        valid = _drop_invalid(numbered, report)
        dropped = len(numbered) - len(valid)
        if dropped:
            warnings.warn(
                f"{source}: dropped {dropped} of {len(numbered)} record(s) that failed "
                "structural validation; results below cover only the remaining "
                f"{len(valid)}. Run `insight-agent validate` to see why.",
                UserWarning,
                stacklevel=2,
            )
        return cls(records=tuple(valid), options=options, report=report, source=source)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        options: InsightTraceOptions | None = None,
    ) -> InsightTraceLoader:
        """Create a loader from an ``insight-trace/v1`` JSONL file."""

        options = options or InsightTraceOptions()
        path = Path(path)
        records: list[Mapping[str, Any]] = []
        parse_errors: list[Diagnostic] = []

        for line, record, error in iter_jsonl(path):
            if error is not None:
                parse_errors.append(Diagnostic("error", "invalid_json", error, line=line))
                continue
            records.append(record)

        if parse_errors and options.strict:
            report = ValidationReport(diagnostics=parse_errors, record_count=len(records))
            raise TraceLoadError(f"{path}: {len(parse_errors)} malformed JSON line(s)", report)

        loader = cls.from_records(records, options, source=str(path))
        loader.report.diagnostics[:0] = parse_errors
        return loader

    # -- provenance helpers used by `run.json` and `coverage` --------------

    @property
    def any_steps_present(self) -> bool:
        return any(r.get("steps") for r in self.records)

    @property
    def all_steps_present(self) -> bool:
        return all(r.get("steps") for r in self.records)

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "trace_count": len(self.records),
            "call_count": sum(len(r.get("calls", [])) for r in self.records),
            "steps_present": self.all_steps_present,
            "steps_partially_present": self.any_steps_present and not self.all_steps_present,
            "distinct_logical_cases": len(
                {r.get("logical_case_id") or r["trace_id"] for r in self.records}
            ),
        }


def _drop_invalid(
    numbered: Sequence[tuple[int, Any]], report: ValidationReport
) -> list[Mapping[str, Any]]:
    """In non-strict mode, keep only records that validated structurally."""

    bad_lines = {d.line for d in report.errors if d.line is not None}
    return [r for line, r in numbered if line not in bad_lines and isinstance(r, Mapping)]


def load_tool_catalog(path: str | Path | None) -> Mapping[str, Any] | None:
    """Load a standalone corpus-wide tool catalog."""

    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(
            f"tool catalog {path} must contain a JSON object mapping tool name to schema"
        )
    return data
