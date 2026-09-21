<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Read your results

Start with the saved insights. Each highlights your agent’s behavior and links it to
supporting traces you can inspect in your source platform or export.

An illustrative entry in `insights.yml`:

```yaml
- name: Retries repeat an invalid order lookup
  description: >-
    After lookup_order reports an unknown order ID, the agent repeats
    the same request without asking the customer to correct the ID.
  trace_refs:
    - run-12
    - run-38
  trace_links:
    run-12: https://observability.example.com/traces/run-12
    run-38: https://observability.example.com/traces/run-38
```

Open a trace link to review the supporting behavior before deciding what to change.
The terminal also shows these URLs, clickable in terminals that support hyperlinks.
`trace_refs` retains stable IDs; `trace_links` maps those IDs to source locations and is
omitted when no links are available. URLs come from the loader, not the inference model.

Live LangSmith and Langfuse traces use their provider-returned UI locations; live MLflow
traces use the HTTP tracking server’s UI. Braintrust links require your
[organization name](sources/braintrust.md#links-to-braintrust). Intake traces currently
retain trace IDs without viewer links.

Local inputs link to the original file on the machine that ran the analysis. Canonical
JSONL, ATIF, Gym JSONL, and LangSmith exports include a `#L<number>` fragment for the
physical source line. Opening the file at that line depends on your viewer’s support.
Other native exports link to the containing file. Canonical file links always point
to the input file; user-supplied URLs are not retained.

When a provider does not supply a usable URL, or its location cannot be determined
(for example, an MLflow `databricks` or local tracking URI), the ID remains available.
Previously saved links survive reconciliation for traces absent from the current run;
links for traces loaded in this run are resolved from the current source.

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

**No findings** means an evidence stream ran and found nothing to report in the available data.
**Skipped** means it could not run. A limitation beside a completed evidence stream describes
missing coverage; it does not mean the whole evidence stream was skipped.

## An evidence stream was skipped

Open [Evidence streams](evidence-streams.md) for the prerequisite and a setup example.
[Disable evidence streams](evidence-streams.md#disable-an-evidence-stream) you don’t need in your configuration.
Disabled evidence streams are omitted from the report.

If no traces were loaded, check your source, filters, and time window first.

## No insights were produced

Read the completed evidence streams and their limitations. The run may have found no actionable
patterns, or its candidates may not have passed review. This result alone does not
establish that your agent is free of problems.

## Saved output

The output is a YAML list with `name`, `description`, `trace_refs`, and optional
`trace_links` for each insight.
Each insight has at least two trace references.

By default, a non-empty collection is written to `insights.yml`.
**An empty result leaves any existing output file untouched.** Use a fresh output path
per run, or capture stdout to receive the current result, including `[]`:

```bash
insight-agent --config config.yaml --output-path - > run-insights.yml
```

Progress and the report go to stderr. Piped stdout contains YAML.
The default output file is still written unless `--output-path -` is selected.
A successful run, including one with skipped evidence streams or no insights, exits with code 0.
Missing required environment settings exit with code 2; other errors exit nonzero.

To carry findings forward, see [repeat a run](configuration.md#repeat-a-run).
