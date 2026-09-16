<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Read your results

Start with the saved insights. Each describes a recurring problem and lists traces
you can inspect in your source platform or export.

An illustrative entry in `insights.yml`:

```yaml
- name: Retries repeat an invalid order lookup
  description: >-
    After lookup_order reports an unknown order ID, the agent repeats
    the same request without asking the customer to correct the ID.
  trace_refs:
    - run-12
    - run-38
```

Use the trace IDs to review the supporting behavior before deciding what to change.

## Terminal report

A run might report:

```text
Produced 1 insight from 100 traces.

Completed
  Tool issues          3 candidate issues
  Evaluation failures  No findings

Skipped
  Ethos divergence     No ethos document
```

A **candidate issue** is a problem found during analysis. Further review may merge or
discard it, so candidate counts can exceed the number of saved insights.

**No findings** means a check ran and found nothing to report in the available data.
**Skipped** means it could not run. A limitation beside a completed check describes
missing coverage; it does not mean the whole check was skipped.

## A check was skipped

Open [Add context and checks](checks.md) for the prerequisite and a setup example.
To hide a check you don’t need, [disable it](checks.md#hide-a-check) in your configuration.
Disabled checks are omitted from the report.

If no traces were loaded, check your source, filters, and time window first.

## No insights were produced

Read the completed checks and their limitations. The run may have found no recurring
problems, or its candidates may not have passed review. This result alone does not
establish that your agent is free of problems.

## Saved output

The output is a YAML list with `name`, `description`, and `trace_refs` for each insight.
Each insight has at least two trace references.

By default, a non-empty collection is written to `insights.yml`.
**An empty result leaves any existing output file untouched.** Use a fresh output path
per run, or capture stdout to receive the current result, including `[]`:

```bash
uv run --no-sync insight-agent --config config.yaml --output-path - > run-insights.yml
```

Progress and the report go to stderr. Piped stdout contains YAML.
The default output file is still written unless `--output-path -` is selected.
A successful run, including one with skipped checks or no insights, exits with code 0.
Missing required environment settings exit with code 2; other errors exit nonzero.

To carry findings forward, see [repeat a run](configuration.md#repeat-a-run).
