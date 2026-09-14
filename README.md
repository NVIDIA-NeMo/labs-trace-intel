<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace Analyst Research Preview

Trace Analyst analyzes agent execution traces to find recurring, actionable problems,
then produces YAML insights with supporting trace references. It reads live traces or native
exports from LangSmith, Langfuse, and MLflow, live NeMo Platform Intake traces, and canonical JSONL.

This research preview explores a more scalable approach than the
[earlier implementation](https://github.com/NVIDIA-NeMo/nemo-platform/blob/f57bb6ca64d84e505742e73f6234fd31e27e7a4a/plugins/nemo-insights/README.md):
preprocess the trace corpus to identify promising evidence, then use an LLM to investigate
and synthesize findings. It is shared for collaboration, not production use or broad distribution.

## What it can find

Choose one or more evidence streams:

- **Anomalies and patterns:** unusual traces and recurring behavior across the corpus.
- **Tool issues:** tool-calling problems detected by deterministic checks.
- **Ethos divergence:** behavior that conflicts with a supplied business-purpose document.
- **Evaluation failure patterns:** recurring behavior associated with recorded evaluation results.

The agent can also check candidate problems against your local codebase and reconcile findings
with a previous Insight collection. It reports problems and supporting evidence; it does not
modify your agent's code.

See [run configuration](docs/configuration.md) for stream selection and optional settings,
or [architecture](docs/architecture.md) for the pipeline design.

## Run Trace Analyst on Example Traces
This repo includes example traces derived from the
[Tau benchmark](https://github.com/sierra-research/tau-bench), which is licensed under the MIT
License. The complete Tau Bench license is preserved in
[`third_party/tau-bench-LICENSE.txt`](third_party/tau-bench-LICENSE.txt).

Configure inference before the first run. Copy the environment template, then set
`INSIGHT_AGENT_API_KEY` for the selected model provider:

```bash
cp .env.example .env
```

If you use an OpenAI-compatible gateway, also set `INSIGHT_AGENT_MODEL` and
`INSIGHT_AGENT_API_BASE` in `.env`. The API base commonly ends in `/v1`, but this is
provider-specific: use the base URL documented by your inference provider rather than guessing the
suffix or supplying the full `/chat/completions` endpoint. Direct providers such as OpenAI or
Anthropic normally do not need `INSIGHT_AGENT_API_BASE`.

Then run Trace Analyst:

```bash
uv sync --locked

uv run insight-agent --config examples/trace-analyst-config.yaml
```

The command prints the complete Insight collection as YAML and writes it to
`insights.yml`. Insight compilation requires an API key.

### Install the local complaint classifier from a wheel

The wheel includes the trained projection and classification head (about 4 MB).
Install its optional encoder dependencies along with the supplied wheel:

```bash
uv pip install './insight_agent-0.1.0rc1-py3-none-any.whl[local-embedding]'
```

Embeddings run locally by default, or remotely through [LiteLLM configuration](docs/configuration.md).
Both paths require **`Qwen/Qwen3-Embedding-8B` (4,096 dimensions)**, the specific
model used to train the bundled projection and classifier.

The first local classification downloads the pinned Qwen3-Embedding-8B encoder weights
(about 16 GB) from Hugging Face. Later runs reuse the Hugging Face cache. No inference
server is required; PyTorch selects CUDA, Apple MPS, or CPU. The encoder weights are
separate from the small trained classifier bundled in the wheel.

For offline deployments, prepare a cache on a connected build machine using the
installed package's exact model revision:

```bash
export HF_HOME="$PWD/model-cache"
python -c 'from insight_agent.evidence_streams.user_embedding.embedding import UserEmbeddingProjection; from huggingface_hub import snapshot_download; e = UserEmbeddingProjection().metadata["encoder"]; snapshot_download(e["model"], revision=e["revision"], allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])'
```

Ship the entire `model-cache` directory with your application or container, preserving
its directory structure and symlinks. Set `HF_HOME` to that directory and
`HF_HUB_OFFLINE=1` before starting the classifier. This uses Transformers' native
offline cache support without putting multi-gigabyte weights in the Python wheel.
Only the local classifier works offline; the full analyst still needs its configured
LLM service. A result with empty `scores` means scoring failed or was skipped, rather
than a successful negative prediction.

### Reading the output

The CLI prints and writes a YAML list of final Insights. Each Insight contains
a name, description, and the trace references that support it.

Progress is written to stderr while the final YAML is printed to stdout and saved to
`output_path` (by default, `insights.yml`).

### Reusable run configuration

The default command can load its trace source, evidence-stream selection,
output path, and non-secret model settings from YAML. YAML is optional, and
explicit CLI options override individual file values. See
[run configuration](docs/configuration.md) for the full schema and examples.

## Analyze your own traces

Trace Analyst can read live projects or native exports from LangSmith, Langfuse, and MLflow,
and live traces from NeMo Platform Intake. Every
source is normalized before the same evidence streams run. Choose one integration below. Each YAML
snippet is a complete `trace-analyst-config.yaml` file. Set up inference as described above,
then choose your endpoint, project, and a time window containing your traces where applicable.
Platform credentials are separate from the inference API key.

### LangSmith

Supports the v1 query API, tested against self-hosted LangSmith 0.15, with Python SDK
`>=0.12,<0.13`. The SmithDB-backed v2 query API is not supported yet.

```bash
uv sync --locked --extra langsmith
export LANGSMITH_API_KEY=<langsmith-api-key>
```

```yaml
# trace-analyst-config.yaml
trace:
  max_traces: 100
  langsmith:
    project: my-agent
    api_url: https://api.smith.langchain.com
    start_time: 2026-09-01T00:00:00Z
    # filter: 'eq(status, "error")'
    # tree_filter: 'eq(run_type, "tool")'

evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config trace-analyst-config.yaml
```

Trace Analyst also reads complete trace exports from the LangSmith CLI using
`langsmith_trace_export_file`. See the detailed [LangSmith configuration docs](docs/configuration.md#langsmith)
for export instructions, filters, workspace selection, and supported versions.

### Langfuse

Supports the v3 API, tested against self-hosted Langfuse 3.205.1 and Python SDK 3.15.
The SDK range is `>=3.15,<4`; Langfuse v4 is not supported.

```bash
uv sync --locked --extra langfuse
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

```yaml
# trace-analyst-config.yaml
trace:
  max_traces: 100
  langfuse:
    base_url: https://langfuse.example.com
    from_timestamp: 2026-09-01T00:00:00Z
    to_timestamp: 2026-09-02T00:00:00Z
    # filter: >-
    #   [{"type":"string","column":"environment","operator":"=","value":"production"}]

evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config trace-analyst-config.yaml
```

Langfuse API keys identify the project, so no project name is needed. Trace Analyst also reads
complete traces saved from the Langfuse CLI, including their observations, using `langfuse_export`.
See the detailed [Langfuse configuration docs](docs/configuration.md#langfuse) for supported export
formats, advanced filters, tool catalogs, and supported v3 server and SDK versions.

### MLflow

Supports Python SDK `>=3.6,<4` via `mlflow-skinny`. Native exports must be readable by
the installed SDK; compatibility with every tracking-server version is not guaranteed.

```bash
uv sync --locked --extra mlflow
# Optional HTTP Basic authentication:
# export MLFLOW_TRACKING_USERNAME=<username>
# export MLFLOW_TRACKING_PASSWORD=<password>
# Optional bearer authentication instead:
# export MLFLOW_TRACKING_TOKEN=<token>
```

```yaml
# trace-analyst-config.yaml
trace:
  max_traces: 100
  mlflow_experiment:
    experiment: my-agent
    tracking_uri: https://mlflow.example.com
    # filter: "trace.status = 'ERROR'"

evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config trace-analyst-config.yaml
```

Trace Analyst also reads complete trace exports from the MLflow CLI using `mlflow_export`.
See the detailed [MLflow configuration docs](docs/configuration.md#mlflow) for export instructions,
authentication, search filters, and pagination limits.

### NeMo Platform Intake

```bash
uv sync --locked
# For authenticated deployments:
# export NMP_ACCESS_TOKEN=<access-token>
```

```yaml
# trace-analyst-config.yaml
trace:
  max_traces: 100
  intake:
    base_url: https://platform.example.com
    workspace: my-workspace
    # page_size: 100
    # timeout_seconds: 30.0
    query:
      # Both bounds are required. Select a UTC window containing your traces.
      started_at_gte: 2026-09-01T00:00:00Z
      started_at_lte: 2026-09-02T00:00:00Z
      # agent_name: my-agent
      # evaluation_name: my-evaluation
      # test_case_name: my-test-case
      # session_id: my-session
      # status: error  # success, error, cancelled, or unknown.
      # max_traces: 100  # trace.max_traces takes precedence when set.
      # sort: started_at  # Or -started_at for newest first.
evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

```bash
uv run --no-sync insight-agent --config trace-analyst-config.yaml
```

See the detailed [Intake configuration docs](docs/configuration.md#intake) for authentication and query options.
Intake supports live queries only; there is no native-export configuration.

### Other platforms

Other trace platforms can be analyzed after an adapter maps their data to the public
[`Trace`](src/insight_agent/traces.py) model.

```yaml
# trace-analyst-config.yaml
trace:
  filesystem:
    path: traces.jsonl
evidence_streams:
  anomaly_and_patterns: {}
  tool_issues: {}
```

Run Trace Analyst on the resulting file:

```bash
uv sync --locked
uv run --no-sync insight-agent --config trace-analyst-config.yaml
```

If your source is not supported, the repository includes a
[trace-loader skill](.claude/skills/trace-loader/SKILL.md) for implementing another adapter.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for validation, dependency licensing, and release instructions.

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project is licensed under the [Apache License, Version 2.0](LICENSE). See
[NOTICE](NOTICE) for project attributions,
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for human-readable dependency license
disclosures, and [third_party/licenses.jsonl](third_party/licenses.jsonl) for the machine-readable
inventory.

This project is currently not accepting contributions.
