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

# Optionally validate trace-derived problems against the agent's local codebase.
# code_base: ../my-agent

# Optional LLM settings. Credentials remain in the environment.
# model: openai/azure/openai/gpt-5.6-luna
# max_tokens: 32768  # Choose a limit supported by the selected model.
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

#### Live project

Configure the LangSmith project and an optional lower time bound:

```yaml
trace:
  max_traces: 100
  langsmith:
    project: customer-support-agent
    start_time: 2026-09-01T00:00:00Z
    filter: 'eq(status, "error")'
    tree_filter: 'eq(run_type, "tool")'
```

`filter` applies to root Runs. `tree_filter` selects a complete trace when any Run in its tree
matches. The example therefore selects failed roots whose trees contain a tool Run. See
LangSmith's [trace query guide](https://docs.langchain.com/langsmith/export-traces) for the query
syntax.

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

An explicit `api_url` CLI value overrides YAML, which overrides `LANGSMITH_ENDPOINT` or the SDK
profile. Project names, filters, time windows, and limits are run settings rather than credentials.
The loader defaults to 100 complete traces.

#### Native exports

The LangSmith CLI is separate from the Python SDK installed by the `langsmith` extra. Install the
[official CLI](https://docs.langchain.com/langsmith/langsmith-cli) (for example,
`brew install langchain-ai/tap/langsmith-cli`), then create a complete export:

```bash
langsmith trace export exports/langsmith \
  --project customer-support-agent --limit 100 --full
```

Then select the exported `.jsonl` file or its containing directory:

```yaml
trace:
  max_traces: 100
  langsmith_trace_export_file:
    path: exports/langsmith
```

The loader reads all immediate `.jsonl` files in a directory, validates the complete export, and
then applies `max_traces`. Both live and export sources support `eval_failure_patterns` using
recorded feedback aggregates from root and child Runs.

The live loader supports the v1 query API as tested against self-hosted LangSmith 0.15. It does not
yet support the SmithDB-backed v2 query API.

### Langfuse

#### Live project

Select a required, timezone-aware trace window. The lower bound is inclusive and the upper bound
is exclusive:

```yaml
trace:
  max_traces: 100
  langfuse:
    from_timestamp: 2026-08-01T00:00:00Z
    to_timestamp: 2026-08-02T00:00:00Z
    # Optional Langfuse advanced filter, encoded as a JSON array.
    filter: >-
      [{"type":"string","column":"environment","operator":"=","value":"production"}]
    # Optional; takes precedence over LANGFUSE_BASE_URL.
    # base_url: https://langfuse.example.com
```

Langfuse API keys identify the project, so there is no project field:

| Variable | Purpose | YAML/CLI equivalent |
| --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Identifies the project | None |
| `LANGFUSE_SECRET_KEY` | Authenticates API requests | None |
| `LANGFUSE_BASE_URL` | Selects the deployment | `trace.langfuse.base_url` / `--trace.langfuse.base-url` |

The loader reads these variables from the shell or `.env`. A configured `base_url` overrides the
environment. The loader pages through matching summaries and fetches each selected trace with its
complete observation tree, so `max_traces` never splits a trace.

`filter` accepts Langfuse's JSON-encoded advanced filter conditions for trace fields such as
environment, name, session, tags, metadata, cost, and error counts. The loader combines the filter
with the required time window.

The equivalent CLI settings are `--trace.langfuse.from-timestamp`,
`--trace.langfuse.to-timestamp`, `--trace.langfuse.filter`, and
`--trace.langfuse.base-url`. For example:

```bash
uv run insight-agent \
  --trace.langfuse.from-timestamp 2026-09-01T00:00:00Z \
  --trace.langfuse.to-timestamp 2026-09-02T00:00:00Z \
  --trace.max-traces 100 \
  --evidence-streams.tool-issues '{}'
```

Explicit CLI values override YAML fields. The configured base URL overrides
`LANGFUSE_BASE_URL`; credentials have no YAML or CLI equivalents.

#### Native exports

The offline loader needs complete v3 trace-detail records rather than trace-list summaries or UI
CSV exports. With the Langfuse environment variables set, this script exports all traces from one
UTC day:

```bash
uv run --locked --extra langfuse python - <<'PY'
from datetime import UTC, datetime
from itertools import count
from langfuse import Langfuse

client = Langfuse(tracing_enabled=False)
start = datetime(2026, 9, 1, tzinfo=UTC)
end = datetime(2026, 9, 2, tzinfo=UTC)

with open("langfuse-traces.jsonl", "x", encoding="utf-8") as output:
    for page in count(1):
        result = client.api.trace.list(
            from_timestamp=start, to_timestamp=end,
            page=page, limit=100, order_by="timestamp.asc",
        )
        for summary in result.data:
            output.write(client.api.trace.get(summary.id).json(by_alias=True) + "\n")
        if page >= result.meta.total_pages:
            break
PY
```

Analyze the export without Langfuse credentials or network access:

```yaml
trace:
  max_traces: 100
  langfuse_export:
    path: langfuse-traces.jsonl
```

The loader also accepts raw v3 `GET /api/public/traces/{id}` bodies, Langfuse CLI response
envelopes, or a directory of immediate `.json` and `.jsonl` files. It validates every record before
selecting the newest `max_traces` traces and rejects duplicate IDs or incomplete observation trees.

For catalog-aware tool checks, record definitions in `input.tools` on `GENERATION` observations.
All explicit generation catalogs must agree; missing or conflicting catalogs make schema-dependent
rules abstain. A root `AGENT` catalog is a compatibility fallback only when a trace has no generation
observations.

This research-preview adapter is validated against self-hosted Langfuse 3.205.1 and Python SDK 3.15.
It supports the v3 API contract only, not Langfuse v4.

### MLflow

#### Live experiment

Select the experiment by name and optionally narrow it with an MLflow trace filter:

```yaml
trace:
  max_traces: 500
  mlflow_experiment:
    experiment: customer-support-agent
    tracking_uri: https://mlflow.example
    filter: "trace.status = 'ERROR'"
```

The corresponding CLI fields are `--trace.mlflow-experiment.experiment`,
`--trace.mlflow-experiment.tracking-uri`, and `--trace.mlflow-experiment.filter`.
`--trace.max-traces` sets the final complete-trace limit.

| Variable | Purpose | YAML/CLI equivalent |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | Selects the tracking server | `trace.mlflow_experiment.tracking_uri` / `--trace.mlflow-experiment.tracking-uri` |
| `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` | HTTP Basic authentication | None |
| `MLFLOW_TRACKING_TOKEN` | Bearer authentication | None |

An explicit `tracking_uri` overrides `MLFLOW_TRACKING_URI`. Credentials do not have YAML or CLI
equivalents. See MLflow's [trace search guide](https://mlflow.org/docs/latest/genai/tracing/search-traces/)
for filters over timestamps, names, span types, tags, and metadata.

The loader defaults to 10,000 traces and requests sequential pages of at most 500. Large or
span-heavy corpora can exhaust local memory before that count, so use a coherent time window plus
stable tags or metadata instead of an unbounded production query.

#### Native exports

Export complete traces, including spans, with the MLflow CLI:

```bash
uv run mlflow traces search \
  --experiment-id <experiment-id> --max-results 500 --output json > mlflow-traces.json
```

Then select the native export:

```yaml
trace:
  max_traces: 500
  mlflow_export:
    path: mlflow-traces.json
```

The loader also accepts `mlflow traces get` output, `Trace.to_json()` or `Trace.to_dict()` output,
arrays of native trace objects, and JSONL with one native trace per line. Exports must include spans;
metadata-only results are rejected. Export-time selection is authoritative: the loader does not
apply filters or follow continuation tokens, and it validates the complete file before applying
`max_traces`.

Relative export paths resolve from the directory where `insight-agent` is run.

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

## Code-aware validation

Set `code_base` to a local copy of the agent's source when you want candidate
Problems checked against its implementation before Insight compilation:

```yaml
code_base: ../my-agent
```

The validator can search and read non-sensitive text files inside that directory,
but it cannot edit files or run arbitrary shell commands. Each candidate is
checked only against its supporting traces and relevant code. Candidates that
the code contradicts are removed before Insight compilation; candidates that
the repository cannot adjudicate, such as externally managed deployment or
RAG-content problems, are retained. Omit `code_base` to retain the trace-only
workflow.

Code excerpts selected during validation are sent to the configured LLM provider.

## CLI-only configuration and overrides

YAML is not required. A complete run can be configured through generated CLI
options:

```bash
uv run insight-agent \
  --trace.filesystem.path traces.jsonl \
  --evidence-streams.anomaly-and-patterns.contamination 0.02 \
  --code-base ../my-agent
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
