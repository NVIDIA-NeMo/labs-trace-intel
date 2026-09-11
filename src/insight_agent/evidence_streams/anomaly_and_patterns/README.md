<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Anomaly and pattern evidence stream

This stream finds statistically unusual traces and recurring behavioral patterns. It projects
the shared `TraceSnapshot` into its native model, performs deterministic feature extraction,
and returns both native analysis artifacts and generic `Problem` values for Insights generation.

## Analysis

The stream:

1. extracts the built-in trace features plus requested numeric metrics;
2. scores unusual traces with `IsolationForest`;
3. clusters trajectory shapes when at least three traces are available;
4. groups observed verdicts and recurring strict failures; and
5. renders an evidence digest with supporting trace identifiers and source pointers.

An anomaly is evidence for investigation, not proof of a defect. Only statistical anomalies and
recurring same-tool or cross-tool failures become `Problem` values; the complete native analysis
remains available in `AnomalyAndPatternsArtifacts`.

## Configuration

`AnomalyAndPatternsConfig` owns the stream configuration:

| Field | Default | Purpose |
|---|---:|---|
| `contamination` | `0.02` | Expected fraction of statistical outliers; must be between 0 and 0.5 |
| `input_scaling` | `"none"` | Feature scaling mode; `"robust"` is also supported |
| `cluster_candidates` | `2` through `8` | Candidate trajectory-cluster counts |
| `minimum_independent_traces` | `3` | Recurrence threshold for grouped evidence |
| `feature_names` | `None` | Uses the built-in feature set unless explicitly overridden |

## Run it

```bash
uv run insight-agent --config trace-analyst-config.yaml
```

Include `evidence_streams.anomaly_and_patterns` in `trace-analyst-config.yaml` and omit the
other stream keys for an anomaly-only run.

The CLI writes `out/anomaly_and_patterns/digest.md`, extracted features, anomalies, trajectory and verdict groups,
failure groups, projected problems, and run metadata.

## Implementation boundary

- `to_anomaly_and_patterns_trace()` owns projection from the shared normalized trace model.
- `walk_spans()` is the private shared depth-first traversal of nested spans.
- `run_anomaly_and_patterns()` owns the native analysis.
- `AnomalyAndPatternsEvidenceStream.analyze()` owns the shared evidence-stream handoff.
- `decode_explicit_failure()` is intentionally independent from the tool-issue decoder; their
  measured differences are documented in [the decoder comparison](../../../../docs/failure-decoders.md).
