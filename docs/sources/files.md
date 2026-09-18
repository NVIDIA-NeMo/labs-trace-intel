<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Analyze trace files

Use this guide for canonical JSONL, ATIF trajectories, or data from a custom source.
For native exports, follow the [LangSmith](langsmith.md#use-an-export),
[Langfuse](langfuse.md#use-an-export), or [MLflow](mlflow.md#use-an-export) guide.

## Run a file

With [uv and Git installed](../../README.md#start-here), install the CLI:

```bash
uv tool install \
  'insight-agent @ git+https://github.com/NVIDIA-NeMo/labs-trace-intel.git@main'
```

[Configure your inference model and key](../model-access.md#choose-a-model), then run:

```bash
insight-agent --trace.filesystem.path traces.jsonl --max-tokens 16384
```

For ATIF, use `--trace.atif.path trajectories.jsonl` instead.
Both loaders read the whole file and do not support `trace.max_traces`.
Paths resolve from your working directory.

[Read your results](../results.md) when the run finishes.

## Canonical JSONL

Each line must contain one complete [Trace](../../packages/trace-ingest/src/trace_ingest/models.py).
This small example shows a recorded tool failure and evaluation score:

```json
{"id":"run-12","root_spans":[{"id":"tool-1","kind":"TOOL","tool_name":"lookup_order","input":{"order_id":"unknown"},"output":{"error":"Order not found"},"error":"Order not found"}],"aggregate":{},"evaluator_results":{"task_success":0}}
```

Preserve stable trace and span IDs, nested spans, tool inputs and results, user messages,
and recorded evaluation signals. Omit values that were not recorded; `null` means the source
explicitly recorded null. Repeated problems need supporting evidence across multiple traces.

## ATIF

Each nonblank line must contain one complete ATIF trajectory. The loader accepts schema
versions `ATIF-v1.0` through `ATIF-v1.8`. Keep tool calls, observations, and nested subagent
trajectories in the exported record.

## A different trace system

Map your records to the canonical model above, then use the filesystem loader.
For a reusable integration, the repository includes a
[trace-loader skill](../../.agents/skills/trace-loader/SKILL.md) and the
[TraceLoader interface](../../packages/trace-ingest/src/trace_ingest/loaders/trace_loaders.py).
