# Shared evidence-stream infrastructure

This package contains contracts and configuration owned jointly by the built-in evidence
streams. It is not a third stream.

- [`insight_trace/`](insight_trace/) owns the canonical JSONL input contract, schema,
  validation, and normalization into `TraceSnapshot`.
- [`failure-decoders.md`](failure-decoders.md) documents the intentional differences between the
  two streams' measured failure semantics.
- `contracts.py` defines the common `TraceLoader` boundary.

The normalized `Trace`, `Span`, and `TraceSnapshot` domain models remain in
`insight_agent.traces` because they are also consumed by Insights generation.
