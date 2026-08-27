"""Validation for the ``insight-trace/v1`` source format.

Two layers, deliberately separated:

**Structural** — JSON Schema. Catches wrong types, missing required fields and
misspelled keys. A typo like ``tool_catelog`` is a hard error rather than six
silently disabled rules, which is the single most useful property of the schema
for anyone writing an adapter.

**Semantic** — lints that the schema cannot express. These encode the traps
that make an adapter *validate cleanly and still produce garbage*: a result
wrapped under ``output`` instead of ``content`` is invisible to IA3's failure
decoder; ``explicit_error: false`` on every call silently disables text-based
failure decoding; a ``result_id`` drawn from the wrong id namespace fires a
mismatch on every call. Each lint names the symptom the author will otherwise
observe downstream.

Import-light on purpose: no numpy, no scikit-learn. ``insight-agent validate``
should be instant.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "CANONICAL_VERSION",
    "Diagnostic",
    "ValidationReport",
    "iter_jsonl",
    "lint_records",
    "trace_schema",
    "trace_validator",
    "validate_corpus",
    "validate_record",
]

CANONICAL_VERSION = "insight-trace/v1"

SCHEMA_FILENAME = "insight_trace_v1.schema.json"

#: The eleven built-in IA2 features. Duplicated here rather than imported so
#: that validation stays free of the scikit-learn import chain; a test asserts
#: the two lists agree.
BUILTIN_FEATURE_NAMES = frozenset(
    {
        "tool_call_count",
        "distinct_tool_count",
        "trajectory_step_count",
        "dominant_tool_share",
        "repeated_identical_call_rate",
        "explicit_failure_rate",
        "returned_data_false_rate",
        "code_execution_share",
        "output_kb_per_call",
        "largest_output_kb",
        "tool_duration_sec_total",
    }
)

#: Keys IA2's text extractor understands but IA3's does not.
IA2_ONLY_TEXT_KEYS = ("output", "message", "summary")

SEVERITIES = ("error", "warning", "info")


@dataclass(frozen=True)
class Diagnostic:
    """One validation or lint result."""

    severity: str
    code: str
    message: str
    line: int | None = None
    path: str = ""
    trace_id: str = ""

    def format(self) -> str:
        where = []
        if self.line is not None:
            where.append(f"line {self.line}")
        if self.trace_id:
            where.append(f"trace {self.trace_id}")
        if self.path:
            where.append(self.path)
        prefix = ", ".join(where)
        head = f"{self.severity.upper()}[{self.code}]"
        return f"{head} {prefix}: {self.message}" if prefix else f"{head} {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "path": self.path,
            "trace_id": self.trace_id,
        }


@dataclass
class ValidationReport:
    """Diagnostics for a whole corpus, plus the records that parsed."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    record_count: int = 0

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    def of(self, severity: str) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == severity]

    @property
    def errors(self) -> list[Diagnostic]:
        return self.of("error")

    @property
    def warnings(self) -> list[Diagnostic]:
        return self.of("warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "ok": self.ok,
            "counts": {s: len(self.of(s)) for s in SEVERITIES},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# -- schema access ---------------------------------------------------------


@lru_cache(maxsize=1)
def trace_schema() -> dict[str, Any]:
    """The canonical JSON Schema as a dict.

    Read through ``importlib.resources`` so it works from a wheel, an editable
    install and a zipapp alike.
    """

    from importlib.resources import files

    # Resolved through the parent package because `schemas/` has no
    # __init__.py and is therefore a namespace package.
    text = files("insight_agent").joinpath("schemas", SCHEMA_FILENAME).read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def trace_validator():
    """A cached Draft 2020-12 validator for the canonical schema."""

    from jsonschema import Draft202012Validator

    schema = trace_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _json_path(error) -> str:
    parts = []
    for token in error.absolute_path:
        parts.append(f"[{token}]" if isinstance(token, int) else f".{token}")
    return "".join(parts).lstrip(".") or "<root>"


# -- reading ---------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(line_number, parsed_or_None, error_or_None)`` for each line.

    Blank lines are skipped. A malformed line yields an error string rather
    than raising, so validation can report every bad line in one pass instead
    of stopping at the first.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield line_number, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_number, None, f"line is not valid JSON: {exc}"


# -- structural validation -------------------------------------------------


def validate_record(record: Any, *, line: int | None = None) -> list[Diagnostic]:
    """Structurally validate one canonical record."""

    if not isinstance(record, Mapping):
        return [
            Diagnostic(
                "error",
                "not_an_object",
                f"each JSONL line must be a JSON object, got {type(record).__name__}",
                line=line,
            )
        ]

    trace_id = str(record.get("trace_id", "")) if isinstance(record.get("trace_id"), str) else ""
    diagnostics = []
    for error in sorted(trace_validator().iter_errors(record), key=lambda e: list(e.absolute_path)):
        path = _json_path(error)
        code = "schema_violation"
        message = error.message
        # additionalProperties failures are the typo case; make the advice explicit.
        if error.validator == "additionalProperties":
            code = "unknown_field"
            message = (
                f"{error.message} Unrecognised keys are rejected so that typos fail loudly; "
                "put vendor-specific data under 'extra'."
            )
        diagnostics.append(
            Diagnostic("error", code, message, line=line, path=path, trace_id=trace_id)
        )
    return diagnostics


# -- semantic lints --------------------------------------------------------


def _lint_tool_catalog(record: Mapping, line, trace_id) -> Iterator[Diagnostic]:
    catalog = record.get("tool_catalog")
    if not isinstance(catalog, Mapping):
        return
    from jsonschema import SchemaError, validators

    for tool_name, schema in catalog.items():
        if schema is None:
            continue
        try:
            validators.validator_for(schema).check_schema(schema)
        except SchemaError as exc:
            yield Diagnostic(
                "error",
                "invalid_tool_schema",
                f"tool_catalog[{tool_name!r}] is not a valid JSON Schema: {exc.message}. "
                "IA3 would raise partway through detection.",
                line=line,
                path=f"tool_catalog.{tool_name}",
                trace_id=trace_id,
            )


def _lint_metrics(
    record: Mapping, line, trace_id, *, allow_shadowing: bool
) -> Iterator[Diagnostic]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return
    for name, value in metrics.items():
        if name in BUILTIN_FEATURE_NAMES and not allow_shadowing:
            yield Diagnostic(
                "error",
                "metric_shadows_builtin",
                f"metrics[{name!r}] shadows a built-in IA2 feature and would silently replace "
                "the measured value, corrupting anomaly detection. Rename it, or pass "
                "--allow-metric-shadowing if this is deliberate.",
                line=line,
                path=f"metrics.{name}",
                trace_id=trace_id,
            )
        if isinstance(value, (int, float)) and not (float("-inf") < float(value) < float("inf")):
            yield Diagnostic(
                "error",
                "non_finite_metric",
                f"metrics[{name!r}] is {value!r}; IA2 drops non-finite metrics, which produces "
                "a ragged feature matrix when custom feature names are used.",
                line=line,
                path=f"metrics.{name}",
                trace_id=trace_id,
            )


def _lint_calls(record: Mapping, line, trace_id) -> Iterator[Diagnostic]:
    calls = record.get("calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return

    seen_ids: dict[str, int] = {}
    seen_indexes: dict[int, int] = {}
    result_id_present = 0
    result_id_mismatched = 0
    explicit_false = 0
    considered = 0

    for position, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        considered += 1
        path = f"calls[{position}]"

        call_id = call.get("call_id")
        if isinstance(call_id, str):
            if call_id in seen_ids:
                yield Diagnostic(
                    "warning",
                    "duplicate_call_id",
                    f"call_id {call_id!r} repeats (also at calls[{seen_ids[call_id]}]). This "
                    "fires duplicate_call_id for every repeat. Keep it if the source really "
                    "reuses ids; fix your id synthesis if it does not — '{trace_id}#{index}' "
                    "is a safe scheme.",
                    line=line,
                    path=f"{path}.call_id",
                    trace_id=trace_id,
                )
            else:
                seen_ids[call_id] = position

        call_index = call.get("call_index")
        if isinstance(call_index, int):
            if call_index in seen_indexes:
                yield Diagnostic(
                    "warning",
                    "duplicate_call_index",
                    f"call_index {call_index} repeats (also at calls[{seen_indexes[call_index]}]). "
                    "IA2 and IA3 break the ordering tie differently, so the two engines can "
                    "disagree about call order.",
                    line=line,
                    path=f"{path}.call_index",
                    trace_id=trace_id,
                )
            else:
                seen_indexes[call_index] = position

        # The content-vs-output trap.
        result = call.get("result")
        if isinstance(result, Mapping) and "content" not in result:
            offending = [k for k in IA2_ONLY_TEXT_KEYS if isinstance(result.get(k), str)]
            if offending:
                yield Diagnostic(
                    "warning",
                    "text_under_non_content_key",
                    f"result carries text under {offending[0]!r} but has no 'content' key. IA2 "
                    f"unwraps {offending[0]!r}, IA3 does not — it will see serialised JSON, so "
                    "error prefixes and tracebacks will not be decoded and the two engines will "
                    "disagree about this call. Move the text to 'content'.",
                    line=line,
                    path=f"{path}.result",
                    trace_id=trace_id,
                )

        if call.get("result_id") is not None:
            result_id_present += 1
            if call.get("result_id") != call.get("call_id"):
                result_id_mismatched += 1

        if call.get("explicit_error") is False:
            explicit_false += 1

    if considered:
        if result_id_present == considered and result_id_mismatched == 0:
            yield Diagnostic(
                "info",
                "result_id_always_equal",
                "result_id equals call_id on every call, so call_result_id_mismatch can never "
                "fire. Consider omitting result_id unless the source has an independent "
                "result-to-call reference.",
                line=line,
                trace_id=trace_id,
            )
        if result_id_present and result_id_mismatched > considered * 0.5:
            yield Diagnostic(
                "warning",
                "result_id_mostly_mismatched",
                f"result_id differs from call_id on {result_id_mismatched}/{considered} calls. "
                "That is usually an id-namespace mismatch in the adapter rather than real "
                "instrumentation corruption, and it fires call_result_id_mismatch on nearly "
                "every call. Populate result_id only from a genuine result-to-call reference.",
                line=line,
                trace_id=trace_id,
            )
        if explicit_false > considered * 0.9:
            yield Diagnostic(
                "warning",
                "explicit_error_false_everywhere",
                f"explicit_error is false on {explicit_false}/{considered} calls. That forces "
                "success and disables ALL text-based failure decoding, so explicit_tool_failure "
                "will be silent for this trace. Use null (or omit the field) unless the source "
                "carries a genuinely reliable success flag.",
                line=line,
                trace_id=trace_id,
            )


def lint_records(
    records: Sequence[tuple[int | None, Mapping[str, Any]]],
    *,
    allow_metric_shadowing: bool = False,
) -> list[Diagnostic]:
    """Run the semantic lints over already-parsed, structurally valid records."""

    diagnostics: list[Diagnostic] = []
    seen_trace_ids: dict[str, int | None] = {}
    catalog_schemas: dict[str, tuple[str, int | None]] = {}

    for line, record in records:
        if not isinstance(record, Mapping):
            continue
        trace_id = record.get("trace_id") if isinstance(record.get("trace_id"), str) else ""

        if trace_id:
            if trace_id in seen_trace_ids:
                diagnostics.append(
                    Diagnostic(
                        "error",
                        "duplicate_trace_id",
                        f"trace_id {trace_id!r} already appeared on line "
                        f"{seen_trace_ids[trace_id]}. IA2 de-duplicates by trace id, so the "
                        "digest would become incoherent.",
                        line=line,
                        trace_id=trace_id,
                    )
                )
            else:
                seen_trace_ids[trace_id] = line

        diagnostics.extend(_lint_tool_catalog(record, line, trace_id))
        diagnostics.extend(
            _lint_metrics(record, line, trace_id, allow_shadowing=allow_metric_shadowing)
        )
        diagnostics.extend(_lint_calls(record, line, trace_id))

        # Cross-record: the same tool must not carry two different schemas.
        catalog = record.get("tool_catalog")
        if isinstance(catalog, Mapping):
            for tool_name, schema in catalog.items():
                if schema is None:
                    continue
                fingerprint = json.dumps(schema, sort_keys=True)
                previous = catalog_schemas.get(tool_name)
                if previous is None:
                    catalog_schemas[tool_name] = (fingerprint, line)
                elif previous[0] != fingerprint:
                    diagnostics.append(
                        Diagnostic(
                            "warning",
                            "tool_catalog_disagreement",
                            f"tool {tool_name!r} has a different schema here than on line "
                            f"{previous[1]}. If the tool contract genuinely changed mid-corpus "
                            "this is real evidence worth keeping; if not, the adapter is "
                            "emitting inconsistent catalogs.",
                            line=line,
                            trace_id=trace_id,
                        )
                    )
                    catalog_schemas[tool_name] = (fingerprint, line)

    return diagnostics


# -- top level -------------------------------------------------------------


def validate_records(
    records: Sequence[tuple[int | None, Any]],
    *,
    allow_metric_shadowing: bool = False,
    run_lints: bool = True,
) -> ValidationReport:
    report = ValidationReport(record_count=len(records))
    structurally_valid: list[tuple[int | None, Mapping[str, Any]]] = []

    for line, record in records:
        errors = validate_record(record, line=line)
        report.extend(errors)
        if not errors and isinstance(record, Mapping):
            structurally_valid.append((line, record))

    if run_lints:
        report.extend(
            lint_records(structurally_valid, allow_metric_shadowing=allow_metric_shadowing)
        )
    return report


def validate_corpus(
    path: str | Path,
    *,
    allow_metric_shadowing: bool = False,
    run_lints: bool = True,
) -> ValidationReport:
    """Validate a canonical JSONL corpus on disk."""

    records: list[tuple[int | None, Any]] = []
    report = ValidationReport()

    for line, record, error in iter_jsonl(path):
        if error is not None:
            report.add(Diagnostic("error", "invalid_json", error, line=line))
            continue
        records.append((line, record))

    body = validate_records(
        records, allow_metric_shadowing=allow_metric_shadowing, run_lints=run_lints
    )
    report.record_count = body.record_count
    report.extend(body.diagnostics)
    return report
