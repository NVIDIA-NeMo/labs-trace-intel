"""Fan one canonical record out into the engine contracts.

IA2 and IA3 were developed independently and each has its own input
dataclass. Left alone, that means anyone onboarding a new trace source has to
write two adapters against two contracts with different field names and
different sentinels. This module is the unification layer: an adapter emits
canonical JSONL in any language, and everything below is handled here.

The most load-bearing detail in the whole package lives here. ``ia3_tid``
distinguishes "the tool returned null" from "no result was ever recorded" using
a module-level ``MISSING = object()`` compared **by identity**. So:

* absent ``result`` key (or ``result_missing``/``result_count: 0``) -> ``MISSING``
* ``"result": null`` -> Python ``None``, which is *not* ``MISSING``

and ``MISSING`` must be imported from ``ia3_tid``, never re-created — a locally
declared ``object()`` would compare unequal, silently suppressing
``missing_tool_result`` and making every call look like it produced a result.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ia3_tid import MISSING, CallRecord, TraceRecord
from .validate import (
    CANONICAL_VERSION,
    Diagnostic,
    ValidationReport,
    iter_jsonl,
    validate_records,
)
from .venue import DEFAULT_PROFILE, VenueProfile

__all__ = [
    "CANONICAL_VERSION",
    "Corpus",
    "CorpusError",
    "LoadOptions",
    "load_corpus",
    "load_records",
    "read_jsonl",
    "to_normalized_trace",
    "to_trace_record",
]


class CorpusError(ValueError):
    """Raised when a corpus cannot be loaded.

    Carries the full diagnostic list so callers can render every problem at
    once rather than one per run.
    """

    def __init__(self, message: str, report: ValidationReport):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class LoadOptions:
    """Knobs for turning canonical records into engine inputs."""

    #: Corpus-wide fallback catalog, used for any record that has none of its
    #: own. Record-level catalogs always win.
    tool_catalog: Mapping[str, Mapping[str, Any] | None] | None = None

    profile: VenueProfile = DEFAULT_PROFILE

    #: Treat validation errors as fatal. Warnings are always surfaced.
    strict: bool = True

    allow_metric_shadowing: bool = False


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for each non-blank line.

    Raises on the first malformed line. Use ``validate_corpus`` when you want
    every problem reported instead of just the first.
    """

    for line, record, error in iter_jsonl(path):
        if error is not None:
            raise CorpusError(f"{path}: {error}", ValidationReport())
        yield line, record


# -- helpers ---------------------------------------------------------------


def _result_is_missing(call: Mapping[str, Any]) -> bool:
    """Decide whether IA3 should see the MISSING sentinel for this call.

    Key absence is the primary signal; ``result_missing`` and
    ``result_count: 0`` are explicit escape hatches for emitters that cannot
    omit a key (many ORMs and protobuf-to-JSON bridges fill defaults).
    """

    return (
        "result" not in call
        or call.get("result_missing") is True
        or call.get("result_count") == 0
    )


def _sorted_calls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = [c for c in record.get("calls", []) if isinstance(c, Mapping)]
    return sorted(calls, key=lambda c: (c.get("call_index", 0), str(c.get("call_id", ""))))


def _sorted_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = [s for s in record.get("steps", []) if isinstance(s, Mapping)]
    return sorted(steps, key=lambda s: s.get("step_index", 0))


# -- IA2 -------------------------------------------------------------------


def to_normalized_trace(
    record: Mapping[str, Any], *, profile: VenueProfile = DEFAULT_PROFILE
):
    """Convert one canonical record into IA2's ``NormalizedTrace``."""

    from .ia2_pipeline import NormalizedCall, NormalizedStep, NormalizedTrace

    calls = []
    for call in _sorted_calls(record):
        # IA2 has no MISSING concept: an unrecorded result is simply None.
        result = None if _result_is_missing(call) else call.get("result")

        # `returned_data` is a call-level field in the canonical format, but
        # IA2 reads it from inside the result mapping. Inject it only when that
        # is lossless: wrapping a string result would change the output-size
        # features and break the anchored error-prefix match.
        if "returned_data" in call:
            if isinstance(result, Mapping):
                if profile.returned_data_key not in result:
                    result = {**result, profile.returned_data_key: call["returned_data"]}
            else:
                warnings.warn(
                    f"call {call.get('call_id')!r} in trace {record.get('trace_id')!r} sets "
                    "'returned_data' but its result is not a JSON object, so IA2 cannot read "
                    "it; returned_data_false_rate will stay 0 for this call. Wrap the result "
                    'as {"content": ...} to make it count.',
                    UserWarning,
                    stacklevel=2,
                )

        calls.append(
            NormalizedCall(
                call_id=str(call["call_id"]),
                call_index=int(call["call_index"]),
                tool_name=str(call["tool_name"]),
                arguments=call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {},
                result=result,
                duration_ms=call.get("duration_ms"),
                source_pointer=dict(call.get("source_pointer") or {}),
            )
        )

    steps = tuple(
        NormalizedStep(
            step_index=int(step["step_index"]),
            step_type=str(step["step_type"]),
            name=str(step.get("name") or ""),
            content=str(step.get("content") or ""),
            source_pointer=dict(step.get("source_pointer") or {}),
        )
        for step in _sorted_steps(record)
    )

    metrics = {
        str(k): float(v)
        for k, v in (record.get("metrics") or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    return NormalizedTrace(
        trace_id=str(record["trace_id"]),
        calls=tuple(calls),
        steps=steps,
        source_pointer=dict(record.get("source_pointer") or {}),
        observed_verdict=record.get("observed_verdict"),
        cost=record.get("cost"),
        metrics=metrics,
    )


# -- IA3 -------------------------------------------------------------------


def to_trace_record(
    record: Mapping[str, Any],
    *,
    tool_catalog: Mapping[str, Any] | None = None,
    profile: VenueProfile = DEFAULT_PROFILE,
) -> TraceRecord:
    """Convert one canonical record into IA3's ``TraceRecord``."""

    task_text = str(record.get("task_text") or "")
    calls = []
    for call in _sorted_calls(record):
        result = MISSING if _result_is_missing(call) else call.get("result")

        # result_count 0 already became MISSING above; normalise it back to 1
        # so the `result_count > 1` duplicate check is unaffected.
        result_count = call.get("result_count", 1)
        result_count = 1 if not isinstance(result_count, int) or result_count < 1 else result_count

        calls.append(
            CallRecord(
                trace_id=str(record["trace_id"]),
                call_index=int(call["call_index"]),
                call_id=str(call["call_id"]),
                tool_name=str(call["tool_name"]),
                arguments=call.get("arguments"),
                result=result,
                source_pointer=dict(call.get("source_pointer") or {}),
                result_id=call.get("result_id"),
                result_count=result_count,
                explicit_error=call.get("explicit_error"),
                outcome_marker=call.get("outcome_marker"),
                instrumentation_alias_of=call.get("instrumentation_alias_of"),
                prior_user_text=str(call.get("prior_user_text") or task_text),
            )
        )

    # Record-level catalog wins over the corpus-wide fallback; absent both, the
    # six catalog-gated rules abstain rather than guess.
    catalog = record.get("tool_catalog")
    if catalog is None:
        catalog = tool_catalog

    return TraceRecord(
        trace_id=str(record["trace_id"]),
        calls=tuple(calls),
        logical_case_id=record.get("logical_case_id"),
        tool_catalog=catalog,
        orphan_results=tuple(dict(o) for o in record.get("orphan_results", []) if isinstance(o, Mapping)),
        complete_provenance_context=bool(record.get("complete_provenance_context", False)),
    )


# -- corpus ----------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """A validated set of canonical records, convertible to any engine input."""

    records: tuple[Mapping[str, Any], ...]
    options: LoadOptions = field(default_factory=LoadOptions)
    report: ValidationReport = field(default_factory=ValidationReport)
    source: str = "<memory>"

    def __len__(self) -> int:
        return len(self.records)

    def ia2(self) -> list[Any]:
        return [to_normalized_trace(r, profile=self.options.profile) for r in self.records]

    def ia3(self) -> list[TraceRecord]:
        return [
            to_trace_record(
                r, tool_catalog=self.options.tool_catalog, profile=self.options.profile
            )
            for r in self.records
        ]

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
            "venue_profile": self.options.profile.name,
        }


def load_records(
    records: Iterable[Mapping[str, Any]],
    options: LoadOptions | None = None,
    *,
    source: str = "<memory>",
) -> Corpus:
    """Validate and wrap already-parsed canonical records."""

    options = options or LoadOptions()
    numbered = [(index + 1, record) for index, record in enumerate(records)]
    report = validate_records(
        numbered, allow_metric_shadowing=options.allow_metric_shadowing
    )

    if report.errors and options.strict:
        raise CorpusError(
            f"{source}: {len(report.errors)} validation error(s); "
            "run `insight-agent validate` for the full list",
            report,
        )
    for diagnostic in report.warnings:
        warnings.warn(diagnostic.format(), UserWarning, stacklevel=2)

    valid = _drop_invalid(numbered, report)
    dropped = len(numbered) - len(valid)
    if dropped:
        # Non-strict mode still must not drop records quietly: a corpus that
        # silently shrinks looks exactly like a corpus that was fully analysed.
        warnings.warn(
            f"{source}: dropped {dropped} of {len(numbered)} record(s) that failed structural "
            "validation; results below cover only the remaining "
            f"{len(valid)}. Run `insight-agent validate` to see why.",
            UserWarning,
            stacklevel=2,
        )
    return Corpus(records=tuple(valid), options=options, report=report, source=source)


def _drop_invalid(
    numbered: Sequence[tuple[int, Any]], report: ValidationReport
) -> list[Mapping[str, Any]]:
    """In non-strict mode, keep only records that validated structurally."""

    bad_lines = {d.line for d in report.errors if d.line is not None}
    return [r for line, r in numbered if line not in bad_lines and isinstance(r, Mapping)]


def load_corpus(path: str | Path, options: LoadOptions | None = None) -> Corpus:
    """Load and validate a canonical JSONL corpus from disk."""

    options = options or LoadOptions()
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
        raise CorpusError(f"{path}: {len(parse_errors)} malformed JSON line(s)", report)

    corpus = load_records(records, options, source=str(path))
    corpus.report.diagnostics[:0] = parse_errors
    return corpus


def load_tool_catalog(path: str | Path | None) -> Mapping[str, Any] | None:
    """Load a standalone corpus-wide tool catalog."""

    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"tool catalog {path} must contain a JSON object mapping tool name to schema")
    return data
