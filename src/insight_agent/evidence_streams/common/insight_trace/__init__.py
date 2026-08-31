"""Canonical input contract shared by the built-in evidence streams."""

from insight_agent.evidence_streams.common.insight_trace.loader import (
    InsightTraceLoader,
    InsightTraceOptions,
    TraceLoadError,
    load_tool_catalog,
)
from insight_agent.evidence_streams.common.insight_trace.validation import (
    CANONICAL_VERSION,
    Diagnostic,
    ValidationReport,
    trace_schema,
    validate_corpus,
    validate_record,
)

__all__ = [
    "CANONICAL_VERSION",
    "Diagnostic",
    "InsightTraceLoader",
    "InsightTraceOptions",
    "TraceLoadError",
    "ValidationReport",
    "load_tool_catalog",
    "trace_schema",
    "validate_corpus",
    "validate_record",
]
