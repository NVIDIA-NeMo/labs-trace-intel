<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run configuration

The Insights Analyst can be configured with YAML, generated CLI options, or a
combination of both. A YAML file is optional, but every run must select exactly
one trace source and at least one evidence stream.

Use YAML for reusable, non-secret run settings. Explicit CLI options override
individual YAML fields for one run. Environment variables provide credentials
and provider defaults without putting secrets in YAML or shell history.

```yaml
# insight-analyst.yaml
trace:
  filesystem:
    path: traces.jsonl

output_path: insights.yml

evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
  # Select when traces contain evaluator_results.
  # eval_failure_patterns: {max_tool_rounds: 72}

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

Select exactly one trace loader.

### Filesystem

For canonical JSONL with one serialized `Trace` per line:

```yaml
trace:
  filesystem:
    path: traces.jsonl
```

### Intake

Select a bounded NeMo Platform Intake query using the shared trace limit:

```yaml
trace:
  max_traces: 500
  intake:
    base_url: https://platform.example.com
    workspace: example-workspace
    query:
      started_at_gte: 2026-08-28T00:00:00Z
      started_at_lte: 2026-08-29T00:00:00Z
      sort: -started_at
```

Use `--trace.max-traces` to override the shared limit from the CLI. An explicit
shared limit takes precedence over `trace.intake.query.max_traces`.
The loader returns an in-memory `TraceSnapshot`; it does not write snapshot or
manifest files.

Intake reads optional bearer authentication from `NMP_ACCESS_TOKEN`. For an
authenticated deployment, export the token before running; with the NeMo Platform
CLI installed, use `nemo auth login --base-url https://platform.example.com` and
`export NMP_ACCESS_TOKEN="$(nemo auth token)"`.

The loader fetches evaluator results for the selected sessions and attaches only
results targeting spans in each trace. Select `eval_failure_patterns: {}` under
`evidence_streams` to use those results in analysis. It uses the same model and
credentials as Insight compilation.

### LangSmith

For the LangSmith Trace Loader, configure the API project to query:

```yaml
trace:
  max_traces: 100
  langsmith:
    project: customer-support-agent
    start_time: 2026-09-01T00:00:00Z
    filter: 'eq(status, "error")'
    tree_filter: 'eq(run_type, "tool")'
```

`filter` applies to root Runs. `tree_filter` selects a complete trace when any
Run in its tree matches. The example therefore selects failed root Runs whose
trees contain a tool Run. See LangSmith's [trace query
guide](https://docs.langchain.com/langsmith/export-traces) for supported syntax.

The equivalent one-off overrides are generated from the same configuration
model:

```bash
uv run insight-agent --config insight-analyst.yaml \
  --trace.langsmith.filter 'eq(status, "success")' \
  --trace.max-traces 25
```

The LangSmith SDK reads these environment variables:

| Variable | Purpose | YAML/CLI equivalent |
| --- | --- | --- |
| `LANGSMITH_API_KEY` | Authenticates API requests | None; credentials stay outside run configuration |
| `LANGSMITH_ENDPOINT` | Selects a regional or self-hosted API | `trace.langsmith.api_url` / `--trace.langsmith.api-url` |
| `LANGSMITH_WORKSPACE_ID` | Selects a workspace when a key can access multiple workspaces | None |

An explicit `api_url` CLI value overrides YAML, which overrides
`LANGSMITH_ENDPOINT` or the SDK profile. Project names, filters, time windows,
and limits are run settings rather than environment variables. The loader
defaults to 100 complete traces.

For the LangSmith Trace Export File Loader, use `langsmith trace export --full`
and select its output file or directory:

```yaml
trace:
  max_traces: 100
  langsmith_trace_export_file:
    path: exports/langsmith-traces
```

The LangSmith Trace Loader supports the v1 query API as tested against
self-hosted LangSmith 0.15. It does not yet support the SmithDB-backed v2 query
API; that migration will wait until a newer self-hosted version is available
for testing.

Both LangSmith sources support `eval_failure_patterns` using recorded feedback
aggregates from root and child runs.

### MLflow

For a live MLflow experiment:

```yaml
trace:
  max_traces: 500
  mlflow_experiment:
    experiment: customer-support-agent
    tracking_uri: https://mlflow.example
    filter: "trace.status = 'ERROR'"
```

The corresponding CLI fields use the generated names
`--trace.mlflow-experiment.experiment`, `--trace.mlflow-experiment.tracking-uri`,
and `--trace.mlflow-experiment.filter`. `--trace.max-traces` sets the final
complete-trace limit.

The MLflow SDK reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` for HTTP Basic authentication, or
`MLFLOW_TRACKING_TOKEN` for bearer authentication. An explicit `tracking_uri`
CLI value overrides YAML, which overrides `MLFLOW_TRACKING_URI`. Credentials do
not have YAML or CLI equivalents.

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

User dissatisfaction extracts initiating user messages with NeMo OO, screens each
message locally with Nemotron 3 Nano 4B Q4_K_M, then uses the existing hosted issue
detector to verify and group complaints. A single `complaint` flags the trace,
even if a later message expresses satisfaction. The only classification labels
are `complaint` and `no_complaint`; reasoning and explanations are disabled.
Repeated span histories do not count as additional user messages.
Extraction includes later user turns identified by source actor metadata and
decodes original event bodies from embedded conversation histories. Generated
workflow documents in a model's `user` role are not user feedback.

Provide a [llama.cpp server executable](https://github.com/ggml-org/llama.cpp/releases/tag/b10793)
and the [NVIDIA GGUF model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/tree/ba223d14e45525f7fae81db77ea8cabeb2fc6c25).
The tested runtime is llama.cpp b10793 on Apple M5 (24 GiB), with the model file
`NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` (2.84 GB), revision
`ba223d14e45525f7fae81db77ea8cabeb2fc6c25`. Its SHA-256 is
`be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2`.
No PyTorch, Transformers, or optional Python extra is needed.

```yaml
evidence_streams:
  user_dissatisfaction:
    model_path: /path/to/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
    llama_server: /path/to/llama-server  # Defaults to llama-server on PATH.
```

```bash
uv run insight-agent --config insight-analyst.yaml
```

The stream starts one local server for the screening phase, loads the model once,
and stops the server before hosted verification. It binds only to localhost,
uses one inference slot, and requests GPU offloading. Each request contains only
the classification instructions and one user message. Relative paths resolve
from the working directory; `model_path` also expands `~`.

The classifier receives only the first 20,000 characters of each user message.
This is an approximate input limit; the local server still has an 8,192-token
context and a 32-token output allowance. If classification fails, including a
context-limit error or invalid response, the message is not flagged and screening
continues with the next message. Empty input is recorded as `no_user_messages`.
The original extracted messages remain intact in the evidence artifacts.
Extraction coverage and ordered per-message labels and token counts are retained
in `EvidenceStreamResult.artifacts`. Classification is evaluated on English text.

The opt-in model evaluation checks ten synthetic complaints in two groups, three
neutral controls, and extraction of repeated histories and later user turns:

```bash
INSIGHT_AGENT_EVAL_MODEL_PATH=/path/to/model.gguf \
INSIGHT_AGENT_EVAL_LLAMA_SERVER=/path/to/llama-server \
uv run --locked pytest tests/evals/test_dissatisfaction_models.py -s
```

It uses the configured inference credentials for hosted extraction and grouping,
and writes classification and grouping results to `tmp/user-dissatisfaction-mini-eval.json`.
`INSIGHT_AGENT_EVAL_MODEL` optionally overrides the hosted model, which defaults to
`openai/azure/openai/gpt-5.6-luna`; it does not change the local classifier.
These model evaluations are skipped in the ordinary test suite unless the model
path environment variable is set.

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

Insight compilation requires an API key for the inference provider that runs
the final LLM step. Set that credential as `INSIGHT_AGENT_API_KEY`; it is not a
LangSmith credential. The exact provider is selected by `INSIGHT_AGENT_MODEL`,
so the value might be an OpenAI, Anthropic, or gateway key. The CLI also
recognizes the provider-standard `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
fallbacks. Model and OpenAI-compatible endpoint settings can come from
`INSIGHT_AGENT_MODEL` and `INSIGHT_AGENT_API_BASE`, or from the non-secret
`model` and `api_base` configuration fields.

The CLI loads an optional local `.env` file. Keep credentials out of YAML and
source control.
