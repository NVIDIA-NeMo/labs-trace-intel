# Tool-issue evidence stream

This stream performs deterministic, capability-gated auditing of tool calls. It projects the
shared `TraceSnapshot` into IA3 records, evaluates nineteen finding types, groups recurring
findings into evidence cards, and returns recurrence-qualified `Problem` values for Insights
generation.

Missing evidence causes a rule to abstain; it does not create a speculative finding. Every
finding retains the original trace, call identifier, and source pointer.

## Detection and promotion

The nineteen rules cover tool contracts and arguments, call/result integrity, explicit outcomes,
retry behavior, trace instrumentation, argument provenance, and prerequisite or state failures.
`detect()` emits individual findings. `build_cards()` groups findings by issue type and mechanism,
selects representative evidence, and marks a card eligible for the Analyst after it appears in
the configured number of independent logical cases.

`problems_from_cards()` excludes audit-only cards by default. This keeps one-off observations in
the native artifacts without presenting them as recurring problems.

## Configuration

`ToolIssueConfig` owns the stream configuration:

| Field | Default | Purpose |
|---|---:|---|
| `minimum_independent_cases` | `3` | Cases required before a card is eligible for the Analyst |
| `retry_threshold` | `3` | Failed calls required for retry-pattern findings |
| `include_audit_problems` | `false` | Include cards below the recurrence threshold in projected problems |

## Run it

```bash
uv run insight-agent run-ia3 traces.jsonl -o out
```

The CLI writes individual findings, cards, finding-type coverage, projected problems, rendered
card Markdown, and run metadata under `out/ia3/`.

The adjacent `coverage.py` reports which rules the supplied corpus has enough evidence to
evaluate. This distinguishes a clean corpus from a detector that is silent because required
fields were not captured.

## Implementation boundary

- `to_ia3_trace()` owns projection from the shared normalized trace model.
- `detect()` owns individual rule evaluation.
- `build_cards()` owns recurrence grouping and promotion.
- `ToolIssueEvidenceStream.analyze()` owns the shared evidence-stream handoff.
- `strict_failure()` is intentionally narrower than the anomaly stream's decoder; their measured
  differences are documented in [the decoder comparison](../common/failure-decoders.md).
