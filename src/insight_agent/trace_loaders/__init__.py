"""Trace loaders normalize source-specific data into ``TraceSnapshot``."""

from .contracts import TraceLoader
from .insight_trace_v1 import (
    InsightTraceV1Loader,
    InsightTraceV1Options,
    TraceLoadError,
    load_tool_catalog,
)
from .validation import (
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
    "InsightTraceV1Loader",
    "InsightTraceV1Options",
    "TraceLoadError",
    "TraceLoader",
    "ValidationReport",
    "load_tool_catalog",
    "trace_schema",
    "validate_corpus",
    "validate_record",
]
