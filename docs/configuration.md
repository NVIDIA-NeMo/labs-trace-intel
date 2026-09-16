<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration reference

For your first run, choose a [source guide](../README.md#start-here).
Use this page when you want to adjust an existing setup.

## Settings

Save reusable settings in YAML and keep credentials in the environment or `.env`.

| Setting | Purpose |
| --- | --- |
| `trace` | Exactly one source; see the [source guides](../README.md#start-here). |
| `trace.max_traces` | Limit complete traces from a provider or native export. Unsupported for canonical JSONL and ATIF. |
| `output_path` | YAML output file; defaults to `insights.yml`. Use `-` for stdout. |
| `model`, `api_base`, `max_tokens` | [Inference settings](model-access.md#choose-a-model). |
| `evidence_streams` | [Checks and their prerequisites](checks.md). All five are enabled by default. |
| `code_base` | [Local agent source](checks.md#check-findings-against-code) to consult during validation. |
| `existing_insights` | Previous JSON or YAML insight collection to reconcile with this run. |

Relative paths resolve from the directory where you run the command.

## One-run overrides

CLI options override individual YAML values. YAML uses underscores; CLI options use hyphens:

```bash
uv run --no-sync insight-agent --config config.yaml \
  --trace.max-traces 25 \
  --output-path investigation.yml
```

You can also run without YAML:

```bash
uv run --no-sync insight-agent --trace.filesystem.path traces.jsonl --max-tokens 16384
```

Use `uv run --no-sync insight-agent --help-all` for every option and its default.
Explicit CLI and YAML model settings take priority over [environment defaults](model-access.md#credentials).

## Repeat a run

Keep the configuration and a complete trace export to analyze the same input again.
Model-generated findings can vary between runs. Give each run its own `output_path`
when you want to compare results.

To update a previous collection, add these settings to your configuration:

```yaml
existing_insights: previous-insights.yml
output_path: updated-insights.yml
```

Trace Analyst reconciles old and new findings into a complete collection.
See [output behavior](results.md#saved-output) before using the files in a pipeline.
