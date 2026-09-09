<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run configuration

The Insights Analyst can be configured with YAML, generated CLI options, or a
combination of both. A YAML file is optional, but every run must select exactly
one trace source and at least one evidence stream.

```yaml
# insight-analyst.yaml
trace:
  filesystem:
    path: traces.jsonl

output_path: insights.yml

evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}

# Optional LLM settings. Credentials remain in the environment.
# model: openai/azure/openai/gpt-5.6-luna
# api_base: https://gateway.example/v1

# Optionally reconcile with a previous JSON or YAML Insight collection.
# existing_insights: previous-insights.yml
```

Run it with:

```bash
uv run insight-agent --config insight-analyst.yaml
```

## Trace sources

Select one trace loader. For a canonical JSONL file:

```yaml
trace:
  filesystem:
    path: traces.jsonl
```

For a live MLflow experiment:

```yaml
trace:
  max_traces: 500
  mlflow_experiment:
    experiment: customer-support-agent
    tracking_uri: https://mlflow.example
    filter: "trace.status = 'ERROR'"
```

For a native MLflow trace export:

```yaml
trace:
  max_traces: 500
  mlflow_export:
    path: exports/traces.json
```

Relative paths currently resolve from the directory where `insight-agent` is
run, including paths supplied by YAML.

## Evidence streams

The presence of a stream selects it for the run. Omit a stream when it should
not run. An empty mapping selects the stream with its defaults:

```yaml
evidence_streams:
  anomaly_and_patterns: {}
```

Ethos divergence checks observed agent behavior against a business-purpose document.
Select it by providing an existing, non-empty UTF-8 Markdown file:

```yaml
evidence_streams:
  ethos_divergence:
    ethos_path: /path/to/ethos.md
```

It can run alone or alongside the other streams and uses the same model and
credentials as insight compilation. Relative paths resolve from the working directory.
For a CLI-only run:

```bash
uv run insight-agent \
  --trace.filesystem.path traces.jsonl \
  --evidence-streams.ethos-divergence.ethos-path /path/to/ethos.md
```

Run `uv run insight-agent --help` to see the generated options and configurable
stream settings.

## CLI-only configuration and overrides

YAML is not required. A complete run can be configured through generated CLI
options:

```bash
uv run insight-agent \
  --trace.filesystem.path traces.jsonl \
  --evidence-streams.anomaly-and-patterns.contamination 0.02
```

When YAML is used, explicit CLI values take priority and override only the
specified nested value:

```bash
uv run insight-agent --config insight-analyst.yaml \
  --trace.filesystem.path replacement-traces.jsonl \
  --output-path experiment-insights.yml
```

## Continue from a previous run

Set `existing_insights` to a previous JSON or YAML Insight collection. Existing
Insights are retained, semantic duplicates are merged, and newly supported
trace references can be added:

```yaml
existing_insights: previous-insights.yml
output_path: insights.yml
```

The output is a complete collection, so it can become `existing_insights` for
the next run.

## Credentials

Insight compilation requires `INSIGHT_AGENT_API_KEY`. The CLI also recognizes
the provider-standard `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` fallbacks. Model
and OpenAI-compatible endpoint settings can come from `INSIGHT_AGENT_MODEL` and
`INSIGHT_AGENT_API_BASE`, or from the non-secret `model` and `api_base`
configuration fields.

The CLI loads an optional local `.env` file. Keep credentials out of YAML and
source control.
