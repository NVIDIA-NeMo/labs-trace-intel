<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run configuration

`insight-agent --config` executes the trace source and evidence streams
selected in a YAML file. Explicit CLI options can still override individual
settings for a one-off experiment.

```yaml
# analyst.yaml
trace:
  # Paths are relative to this YAML file.
  filesystem:
    path: traces.jsonl

output:
  directory: out

evidence_streams:
  anomaly_and_patterns:
    contamination: 0.02
    input_scaling: robust
    cluster_candidates: [2, 3, 4, 5]
    minimum_independent_traces: 3
  tool_issues:
    minimum_independent_cases: 3
    retry_threshold: 3

analyst:
  enabled: true
  model: anthropic/claude-opus-5
  # api_base: https://gateway.example/v1
  # env_file: .env
```

The same config file can select either MLflow source supported by the CLI:

```yaml
# A live MLflow experiment.
trace:
  max_traces: 500 # optional limit supported by the MLflow loaders
  mlflow_experiment:
    experiment: customer-support-agent
    tracking_uri: https://mlflow.example
    filter: "trace.status = 'ERROR'" # optional
```

```yaml
# A native MLflow trace export on disk.
trace:
  max_traces: 500 # optional limit supported by the MLflow loaders
  mlflow_export:
    path: exports/traces.json
```

Run it with:

```bash
uv run insight-agent --config analyst.yaml
```

## Continue from a previous run

To carry Insights across runs, point `analyst.existing_insights` at an
`insights.json` artifact from a previous run. The Insight compilation agent
keeps the existing collection, merges semantic duplicates, and adds newly
supported trace references. The current run writes the complete reconciled
collection to `<output.directory>/analyst/insights.json`.

```yaml
analyst:
  enabled: true
  existing_insights: previous-run/analyst/insights.json
```

When the input file is that same path, the CLI reads it before replacing it, so
the file acts as the latest Insight collection and older versions are not kept.
To retain every version, use a new output directory for each run and update
`analyst.existing_insights` to point at the preceding run's artifact.

The presence of a stream selects it for the run. Omit a stream when it should
not run; at least one evidence stream must be configured:

```yaml
evidence_streams:
  anomaly_and_patterns: {}
```

The precedence order is: built-in defaults, YAML, then explicit CLI flags.
The YAML schema and these nested CLI options come from the same Pydantic
models, so their names, defaults, and validation stay aligned. Run
`uv run insight-agent --help` to inspect every generated override.
For example, this retains the YAML setup but changes the output directory and
one anomaly-and-pattern setting for a one-off experiment:

```bash
uv run insight-agent --config analyst.yaml \
  --output.directory experiment-out \
  --evidence-streams.anomaly-and-patterns.minimum-independent-traces 5
```

The generated option names mirror their YAML paths, so the filesystem path can
also be overridden directly:

```bash
uv run insight-agent --config analyst.yaml \
  --trace.filesystem.path replacement-traces.jsonl
```

Use `validate` to validate and inspect the resolved config before a run:

```bash
uv run insight-agent validate --config analyst.yaml
```

To validate a canonical trace corpus independently of a run configuration,
select the other explicit validation target:

```bash
uv run insight-agent validate --traces traces.jsonl
```

Utility commands are generated from typed command models in the same way. For
example, `uv run insight-agent run-analyst --help` shows its composed trace,
output, evidence-stream, and Analyst settings. Existing concise options such
as `-o` and `--model` remain aliases of their generated model paths.

## Credentials

Keep credentials out of YAML configuration files. Configure providers through
their normal environment variables or credential mechanism; do not commit
local `.env` files.
